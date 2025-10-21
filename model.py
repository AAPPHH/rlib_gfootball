import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from gymnasium.spaces import Space
from typing import Tuple, Optional


class SoftClamp(nn.Module):
    def __init__(self, min_val: float, max_val: float, sharpness: float = 1.0):
        super().__init__()
        self.register_buffer('center', torch.tensor((max_val + min_val) / 2))
        self.register_buffer('range', torch.tensor((max_val - min_val) / 2))
        self.sharpness = sharpness
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_normalized = (x - self.center) / (self.range + 1e-8)
        x_clamped = torch.tanh(x_normalized * self.sharpness) / self.sharpness
        return x_clamped * self.range + self.center


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(x.dtype)


class VectorizedSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model) * 0.1)
        
        self.dt_clamp = SoftClamp(0.001, 0.1, sharpness=0.5)
        self.A_log_clamp = SoftClamp(-5.0, 5.0, sharpness=0.5)
        self.dt_A_clamp = SoftClamp(-8.0, 8.0, sharpness=0.5)
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.normal_(self.dt_proj.weight, std=0.02)
        nn.init.constant_(self.dt_proj.bias, -2.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        dt = self.dt_clamp(F.softplus(self.dt_proj(x)))
        B_param = self.B_proj(x)
        C_param = self.C_proj(x)
        
        A = -torch.exp(self.A_log_clamp(self.A_log))
        
        dt_A = self.dt_A_clamp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dA = torch.exp(dt_A)
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)
        
        return self._scan_ssm(dA, dB, x, C_param)
    
    def _scan_ssm(self, dA: torch.Tensor, dB: torch.Tensor, 
                  x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        B, L, D, N = dA.shape
        
        h = torch.zeros(B, D, N, device=x.device, dtype=torch.float32)
        outputs = torch.empty(B, L, D, device=x.device, dtype=x.dtype)
        
        # Cache für wiederholte Berechnungen
        D_param = self.D.float()
        
        for t in range(L):
            # State Update: h = dA * h + dB * x
            # Element-wise Multiplikation mit Broadcasting
            h = dA[:, t].float() * h
            h += dB[:, t].float() * x[:, t].float().unsqueeze(-1)
            
            # Sanftes Clipping nur bei Bedarf (selten aktiv)
            h_norm = h.norm(dim=(1, 2), keepdim=True)
            scale = torch.clamp(20.0 / (h_norm + 1e-8), max=1.0)
            h = h * scale
            
            # Output-Berechnung
            y = torch.einsum('bdn,bn->bd', h, C[:, t].float())
            y = y + D_param * x[:, t].float()
            outputs[:, t] = y.to(x.dtype)
        
        return outputs


class MambaBlock(nn.Module):
    """Optimierter Mamba-Block mit verbesserter Architektur."""
    
    def __init__(self, d_model: int, d_state: int = 16, 
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )
        
        self.ssm = VectorizedSSM(self.d_inner, d_state)
        self.norm = RMSNorm(d_model)
        
        self.residual_scale = nn.Parameter(torch.tensor(0.3))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.in_proj.weight, gain=0.7)
        nn.init.orthogonal_(self.out_proj.weight, gain=0.7)
        nn.init.normal_(self.conv1d.weight, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        
        x_branch, gate_branch = self.in_proj(x).chunk(2, dim=-1)
        
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :x_branch.size(1)]
        x_conv = F.silu(x_conv.transpose(1, 2))
        
        x_ssm = self.ssm(x_conv)
        x_out = x_ssm * F.silu(gate_branch)
        x_out = self.out_proj(x_out)
        
        scale = torch.sigmoid(self.residual_scale) * 0.5
        return residual + scale * x_out


class GFootballMamba(TorchModelV2, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space,
                 num_outputs: int, model_config: ModelConfigDict, name: str):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, 
                              model_config, name)
        nn.Module.__init__(self)

        config = model_config.get("custom_model_config", {})
        self.d_model = config.get("d_model", 256)
        self.num_layers = config.get("num_layers", 4)
        self.d_state = config.get("d_state", 16)
        self.d_conv = config.get("d_conv", 4)
        self.expand = config.get("expand", 2)
        self.num_stacked_frames = config.get("num_stacked_frames", 4)
        self.dropout = config.get("dropout", 0.1)
        self.use_amp = config.get("use_amp", True)
        self.amp_dtype = torch.float16
        
        total_obs_dim = int(np.prod(obs_space.shape))
        self.frame_dim = total_obs_dim // self.num_stacked_frames
        
        self.frame_embed = nn.Sequential(
            nn.Linear(self.frame_dim, self.d_model),
            RMSNorm(self.d_model),
            nn.Mish(),
            nn.Dropout(self.dropout)
        )
        
        self.register_buffer('pos_encoding', 
                           torch.randn(1, self.num_stacked_frames, self.d_model) * 0.01)

        self.mamba_layers = nn.ModuleList([
            MambaBlock(self.d_model, self.d_state, self.d_conv, self.expand)
            for _ in range(self.num_layers)
        ])
        
        self.pool = nn.Sequential(
            RMSNorm(self.d_model),
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
        
        self._init_weights()
        self._value_out: Optional[torch.Tensor] = None
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                if module not in [self.policy_head, self.value_head]:
                    nn.init.orthogonal_(module.weight, gain=0.8)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)

        normc_initializer(0.01)(self.policy_head.weight)
        normc_initializer(1.0)(self.value_head.weight)
    
    def _process_observation(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        obs_flat = obs.view(B, -1)
        obs_seq = obs_flat.view(B, self.num_stacked_frames, self.frame_dim)
        return self.frame_embed(obs_seq) + self.pos_encoding

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        
        obs = input_dict["obs"].float()
        
        with autocast(device_type=obs.device.type, 
                     dtype=self.amp_dtype, 
                     enabled=self.use_amp):
            x = self._process_observation(obs)
            
            for layer in self.mamba_layers:
                x = layer(x)
            
            features = self.pool(x.mean(dim=1))
            
            logits = self.policy_head(self.policy_net(features))
            value = self.value_head(self.value_net(features)).squeeze(-1)
        
        self._value_out = value.float()
        return logits.float(), state
    
    def value_function(self) -> TensorType:
        assert self._value_out is not None, "value_function() wurde vor forward() aufgerufen"
        return self._value_out