import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from gymnasium.spaces import Space
from typing import Tuple, Optional, List, Dict

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        padding = (kernel - 1) * dilation
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=kernel, padding=padding, 
                                   dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.trim = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        if self.trim > 0:
            x = x[..., :-self.trim]
        return self.pointwise(x)

class EfficientGroupNorm(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        num_groups = min(32, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(max(1, num_groups), num_channels, eps=eps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class EfficientTCNBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float = 0.05, 
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        expand = max(2, dim // 64)
        inner_dim = dim * expand
        
        self.conv = DepthwiseSeparableConv1d(dim, inner_dim * 2, kernel, dilation)
        self.norm = EfficientGroupNorm(dim, eps=1e-5)
        self.proj = nn.Conv1d(inner_dim, dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_scale = nn.Parameter(torch.ones(1, 1, dim, dtype=torch.float32) * 0.1)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x.transpose(1, 2)).transpose(1, 2)
        conv_out = self.conv(x_norm.transpose(1, 2))
        gate, value = conv_out.chunk(2, dim=1)
        out = value * torch.sigmoid(gate)
        out = self.proj(out).transpose(1, 2)
        out = self.dropout(out)
        return out * self.layer_scale.to(out.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            out = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            out = self._forward_impl(x)
        return x + out


class CompactFrameEncoder(nn.Module):
    def __init__(self, frame_dim: int, d_model: int, dropout: float = 0.05):
        super().__init__()
        self.proj = nn.Linear(frame_dim, d_model, bias=False)
        self.norm = EfficientGroupNorm(d_model, eps=1e-5)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        x = F.gelu(x)
        return self.dropout(x)


class GFootballTCN(TorchModelV2, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, 
                 model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})
        self.d_model = cfg.get("d_model", 128)
        self.kernel = cfg.get("kernel", 3)
        self.dilations = list(cfg.get("dilations", [1, 2, 4, 8]))
        self.dropout = cfg.get("dropout", 0.05)
        self.use_checkpoint = cfg.get("gradient_checkpointing", True)
        self.use_amp = cfg.get("use_amp", True)
        
        if obs_space.shape and len(obs_space.shape) == 2:
            self.num_frames = obs_space.shape[0]
            self.frame_dim = obs_space.shape[1]
        else:
            total = int(np.prod(obs_space.shape))
            self.num_frames = cfg.get("num_frames", 4)
            self.frame_dim = total // self.num_frames

        self.encoder = CompactFrameEncoder(self.frame_dim, self.d_model, self.dropout)
        self.tcn_blocks = nn.ModuleList([
            EfficientTCNBlock(self.d_model, self.kernel, dil, self.dropout, self.use_checkpoint)
            for dil in self.dilations
        ])

        pol_hidden = max(self.d_model // 2, 64)
        val_hidden = max(self.d_model // 4, 32)
        
        self.policy = nn.Sequential(
            nn.Linear(self.d_model * 2, pol_hidden, bias=True),
            nn.GELU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Linear(pol_hidden, num_outputs, bias=True)
        )
        
        self.value = nn.Sequential(
            nn.Linear(self.d_model * 2, val_hidden, bias=True),
            nn.GELU(),
            nn.Linear(val_hidden, 1, bias=True)
        )

        self._init_weights()
        self._value_out: Optional[torch.Tensor] = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
        with torch.no_grad():
            self.policy[-1].weight.mul_(0.01)
            self.policy[-1].bias.zero_()
            self.value[-1].weight.mul_(0.1)
            self.value[-1].bias.zero_()

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], 
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        obs = input_dict["obs"].float()
        
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        
        B = obs.size(0)
        use_amp = self.use_amp and obs.is_cuda
        
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            if obs.ndim == 2:
                try:
                    obs = obs.reshape(B, self.num_frames, self.frame_dim)
                except RuntimeError:
                    obs = obs.contiguous().view(B, self.num_frames, self.frame_dim)
            
            x = self.encoder(obs)
            
            for block in self.tcn_blocks:
                x = block(x)
            
            x_mean = x.mean(dim=1, keepdim=False)
            x_last = x[:, -1]
            x_pooled = torch.cat([x_mean, x_last], dim=-1)
            
            x_pooled = F.layer_norm(x_pooled, (x_pooled.size(-1),))
            
            logits = self.policy(x_pooled)
            value = self.value(x_pooled).squeeze(-1)
        
        self._value_out = value.float()
        return logits.float(), []

    def value_function(self) -> TensorType:
        assert self._value_out is not None
        return self._value_out