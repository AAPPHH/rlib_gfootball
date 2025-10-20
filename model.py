import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from gymnasium.spaces import Space
from typing import Optional, Tuple


class SelectiveSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        
        A = torch.randn(d_model, d_state)
        self.A_log = nn.Parameter(torch.log(torch.abs(A) + 1e-4))
        
        self.D = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x):
        B, L, D = x.shape
        
        dt = F.softplus(self.dt_proj(x))
        B_param = self.B_proj(x)
        C_param = self.C_proj(x)
        
        A = -torch.exp(self.A_log)
        
        dt_A = torch.einsum('bld,dn->bldn', dt, A)
        dA = torch.exp(dt_A)
        
        dB = torch.einsum('bld,bln->bldn', dt, B_param)
        
        h = torch.zeros(B, D, self.d_state, device=x.device, dtype=x.dtype)
        outputs = torch.zeros(B, L, D, device=x.device, dtype=x.dtype)
        
        for t in range(L):
            h = dA[:, t] * h + torch.einsum('bdn,bd->bdn', dB[:, t], x[:, t])
            outputs[:, t] = torch.einsum('bdn,bn->bd', h, C_param[:, t]) + self.D * x[:, t]
        
        return outputs


class ImprovedMambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        self.ssm = SelectiveSSM(self.d_inner, d_state)
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        x_and_gate = self.in_proj(x)
        x_branch, gate_branch = x_and_gate.chunk(2, dim=-1)
        
        x_conv = x_branch.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :x_branch.size(1)]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        x_ssm = self.ssm(x_conv)
        
        x_out = x_ssm * F.silu(gate_branch)
        x_out = self.out_proj(x_out)
        
        return residual + x_out


class GFootballMamba(TorchModelV2, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space,
                 num_outputs: int, model_config: ModelConfigDict, name: str):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        self.d_model = custom_config.get("d_model", 256)
        self.num_layers = custom_config.get("num_layers", 4)
        self.d_state = custom_config.get("d_state", 16)
        self.d_conv = custom_config.get("d_conv", 4)
        self.expand = custom_config.get("expand", 2)
        self.num_stacked_frames = custom_config.get("num_stacked_frames", 4)
        self.dropout = custom_config.get("dropout", 0.1)
        
        self.use_amp = custom_config.get("use_amp", True)
        self.amp_dtype = torch.float16
        
        if len(obs_space.shape) == 1:
            total_obs_dim = obs_space.shape[0]
        else:
            total_obs_dim = int(np.prod(obs_space.shape))
        
        self.frame_dim = total_obs_dim // self.num_stacked_frames
        
        print(f"📊 Improved Mamba Model:")
        print(f"   Total obs_dim: {total_obs_dim}")
        print(f"   Frames: {self.num_stacked_frames}")
        print(f"   Frame dim: {self.frame_dim}")
        print(f"   d_model: {self.d_model}, layers: {self.num_layers}")
        
        self.frame_embed = nn.Sequential(
            nn.Linear(self.frame_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Mish(),
            nn.Dropout(self.dropout)
        )
        
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_stacked_frames, self.d_model) * 0.02
        )
        
        self.mamba_layers = nn.ModuleList([
            ImprovedMambaBlock(
                d_model=self.d_model,
                d_state=self.d_state,
                d_conv=self.d_conv,
                expand=self.expand
            )
            for _ in range(self.num_layers)
        ])
        
        self.pool = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.Mish()
        )
        
        self.policy_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.Mish(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Mish(),
        )
        self.policy_head = nn.Linear(self.d_model // 2, num_outputs)
        
        self.value_net = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.Mish(),
            nn.Linear(self.d_model // 2, self.d_model // 4),
            nn.Mish(),
        )
        self.value_head = nn.Linear(self.d_model // 4, 1)
        
        self._initialize_weights()
        self._value_out = None
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                if module.out_features in [self.d_model, self.d_model // 2]:
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
        
        normc_initializer(0.01)(self.policy_head.weight)
        nn.init.constant_(self.policy_head.bias, 0)
        
        normc_initializer(1.0)(self.value_head.weight)
        nn.init.constant_(self.value_head.bias, 0)
    
    def _process_observation(self, obs):
        B = obs.shape[0]
        
        if len(obs.shape) > 2:
            obs = obs.reshape(B, -1)
        
        obs_seq = obs.reshape(B, self.num_stacked_frames, self.frame_dim)
        
        x = self.frame_embed(obs_seq)
        
        x = x + self.pos_encoding
        
        return x

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        
        obs = input_dict["obs"].float()
        
        with autocast(device_type=obs.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            x = self._process_observation(obs)
            
            for layer in self.mamba_layers:
                x = layer(x)
            
            features = torch.mean(x, dim=1)
            features = self.pool(features)
            
            policy_features = self.policy_net(features)
            logits = self.policy_head(policy_features)
            
            value_features = self.value_net(features)
            value = self.value_head(value_features).squeeze(-1)
        
        self._value_out = value.float() if self.use_amp else value
        logits_out = logits.float() if self.use_amp else logits
        
        return logits_out, state
    
    def value_function(self) -> TensorType:
        assert self._value_out is not None, "value_function() called before forward()"
        return self._value_out