import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from gymnasium.spaces import Space
from typing import Optional, Tuple

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)
        
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        x_and_gate = self.in_proj(x)
        x_ssm, gate = x_and_gate.chunk(2, dim=-1)
        
        x_conv = x_ssm.transpose(1, 2)
        x_conv = self.conv1d(x_conv)[:, :, :x_ssm.size(1)]
        x_conv = x_conv.transpose(1, 2)
        x_conv = F.silu(x_conv)
        
        ssm_params = self.x_proj(x_conv)
        A, B = ssm_params.chunk(2, dim=-1)
        
        dt = F.softplus(self.dt_proj(x_conv))
        
        # Simplified SSM: verwende mean pooling für A und B
        A_pooled = A.mean(dim=-1, keepdim=True)
        B_pooled = B.mean(dim=-1, keepdim=True)
        
        x_ssm = x_conv * torch.sigmoid(A_pooled.expand_as(x_conv)) + B_pooled.expand_as(x_conv) * dt
        
        x_out = x_ssm * F.silu(gate)
        x_out = self.out_proj(x_out)
        
        return residual + x_out

class MultiHeadSpatialAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, L, _ = x.shape
        residual = x
        x = self.norm(x)
        
        Q = self.q_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        
        output = self.out_proj(context)
        
        return residual + output

class HybridBlock(nn.Module):
    def __init__(self, d_model: int, use_attention: bool = True, 
                 d_state: int = 16, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.use_attention = use_attention
        
        self.mamba = MambaBlock(d_model, d_state=d_state)
        
        if use_attention:
            self.attention = MultiHeadSpatialAttention(d_model, num_heads, dropout)
        
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        x = self.mamba(x)
        
        if self.use_attention:
            x = self.attention(x, mask)
        
        residual = x
        x = self.ffn(x)
        
        return residual + x

class GFootballMambaHybrid2025(TorchModelV2, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space,
                 num_outputs: int, model_config: ModelConfigDict, name: str):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        self.d_model = custom_config.get("d_model", 256)
        self.num_layers = custom_config.get("num_layers", 4)
        self.d_state = custom_config.get("d_state", 16)
        self.num_heads = custom_config.get("num_heads", 4)
        self.use_attention = custom_config.get("use_attention", True)
        self.num_stacked_frames = custom_config.get("num_stacked_frames", 4)
        self.dropout = custom_config.get("dropout", 0.1)
        
        self.use_amp = custom_config.get("use_amp", True)
        self.amp_dtype = torch.float16
        
        # Handle observation space (simple115v2 gibt flat vector)
        if len(obs_space.shape) == 1:
            obs_dim = obs_space.shape[0]
        else:
            obs_dim = int(np.prod(obs_space.shape))
        
        print(f"📊 Model initialized: obs_dim={obs_dim}, d_model={self.d_model}")
        
        self.input_embed = nn.Sequential(
            nn.Linear(obs_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Mish(),
            nn.Dropout(self.dropout)
        )
        
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.num_stacked_frames, self.d_model) * 0.02
        )
        
        self.layers = nn.ModuleList([
            HybridBlock(
                d_model=self.d_model,
                use_attention=self.use_attention and (i % 2 == 1),
                d_state=self.d_state,
                num_heads=self.num_heads,
                dropout=self.dropout
            )
            for i in range(self.num_layers)
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
                if module.out_features in [self.d_model, self.d_model // 2]:
                    nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        normc_initializer(0.01)(self.policy_head.weight)
        nn.init.constant_(self.policy_head.bias, 0)
        
        normc_initializer(1.0)(self.value_head.weight)
        nn.init.constant_(self.value_head.bias, 0)
    
    def _process_observation(self, obs):
        B = obs.shape[0]
        
        # Flatten if multi-dimensional
        if len(obs.shape) > 2:
            obs = obs.reshape(B, -1)
        
        x = self.input_embed(obs)
        

        x = x.unsqueeze(1)
        
        return x

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        
        obs = input_dict["obs"].float()
        
        with autocast(device_type=obs.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            x = self._process_observation(obs)
            
            for layer in self.layers:
                x = layer(x)
            
            if x.size(1) > 1:
                features = torch.mean(x, dim=1)
            else:
                features = x.squeeze(1)
            
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

class GFootballMambaLite2025(TorchModelV2, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space,
                 num_outputs: int, model_config: ModelConfigDict, name: str):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        custom_config = model_config.get("custom_model_config", {})
        self.d_model = custom_config.get("d_model", 256)
        self.num_layers = custom_config.get("num_layers", 3)
        
        if len(obs_space.shape) == 1:
            obs_dim = obs_space.shape[0]
        else:
            obs_dim = int(np.prod(obs_space.shape))
        
        self.input_proj = nn.Sequential(
            nn.Linear(obs_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.Mish()
        )
        
        self.mamba_layers = nn.ModuleList([
            MambaBlock(self.d_model, d_state=16)
            for _ in range(self.num_layers)
        ])
        
        self.policy_head = nn.Linear(self.d_model, num_outputs)
        self.value_head = nn.Linear(self.d_model, 1)
        
        normc_initializer(0.01)(self.policy_head.weight)
        normc_initializer(1.0)(self.value_head.weight)
        
        self._value_out = None

    def forward(self, input_dict, state, seq_lens):
        x = input_dict["obs"].float()
        
        B = x.shape[0]
        if len(x.shape) > 2:
            x = x.reshape(B, -1)
        
        x = self.input_proj(x).unsqueeze(1)
        
        for mamba in self.mamba_layers:
            x = mamba(x)
        
        features = x.squeeze(1)
        
        logits = self.policy_head(features)
        self._value_out = self.value_head(features).squeeze(-1)
        
        return logits, state
    
    def value_function(self):
        return self._value_out