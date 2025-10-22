import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from gymnasium.spaces import Space
from typing import Tuple, Optional, List, Dict

class SoftClamp(nn.Module):
    def __init__(self, min_val: float, max_val: float, sharpness: float = 1.0):
        super().__init__()
        self.register_buffer("center", torch.tensor((max_val + min_val) / 2))
        self.register_buffer("range", torch.tensor((max_val - min_val) / 2))
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
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.to(torch.float32)
            var = xf.pow(2).mean(dim=-1, keepdim=True)
            x_norm = xf * torch.rsqrt(var + self.eps)
            w = self.weight.to(x_norm.dtype)
            y = (w * x_norm).to(x.dtype)
        return y

class RecurrentSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8):
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
        N = self.d_state
        device, dtype = x.device, x.dtype

        h = torch.zeros(B, D, N, device=device, dtype=dtype)
        outputs = torch.empty(B, L, D, device=device, dtype=dtype)

        A = -torch.exp(self.A_log_clamp(self.A_log)).to(dtype)
        D_param = self.D.to(dtype)

        for t in range(L):
            x_t = x[:, t]
            dt_t = self.dt_clamp(F.softplus(self.dt_proj(x_t)))
            B_t  = self.B_proj(x_t)
            C_t  = self.C_proj(x_t)

            dt_A_t = self.dt_A_clamp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            dA_t   = torch.exp(dt_A_t)
            dB_t   = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)

            h.mul_(dA_t)
            h.addcmul_(dB_t, x_t.unsqueeze(-1))

            if t % 10 == 0:
                with torch.autocast(device_type=x.device.type, enabled=False):
                    h32 = h.to(torch.float32)
                    hnorm = h32.norm(dim=(1, 2), keepdim=True)
                    scale = torch.clamp(20.0 / (hnorm + 1e-8), max=1.0).to(dtype)
                h.mul_(scale)

            y_t = torch.einsum("bdn,bn->bd", h, C_t) + D_param * x_t
            outputs[:, t] = y_t

        return outputs

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8, d_conv: int = 3, expand: int = 1, dropout: float = 0.0):
        super().__init__()
        self.d_inner = int(d_model * expand)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True,
        )

        self.ssm = RecurrentSSM(self.d_inner, d_state)
        self.norm = RMSNorm(d_model)

        self.residual_scale = nn.Parameter(torch.tensor(0.3))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.in_proj.weight, gain=0.7)
        nn.init.orthogonal_(self.out_proj.weight, gain=0.7)
        nn.init.normal_(self.conv1d.weight, std=0.02)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)

        x_branch, gate_branch = self.in_proj(x).chunk(2, dim=-1)

        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, : x_branch.size(1)]
        x_conv = F.silu(x_conv.transpose(1, 2))

        x_ssm = self.ssm(x_conv)
        x_out = x_ssm * F.silu(gate_branch)
        x_out = self.out_proj(x_out)
        x_out = self.dropout(x_out)

        return x_out

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        residual = x
        if use_checkpoint and self.training:
            x_out = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            x_out = self._forward_impl(x)
        scale = torch.sigmoid(self.residual_scale) * 0.5
        return residual + scale * x_out

class GFootballMamba(TorchModelV2, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, model_config: ModelConfigDict, name: str):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})
        self.d_model = cfg.get("d_model", 128)
        self.num_layers = cfg.get("num_layers", 3)
        self.d_state = cfg.get("d_state", 8)
        self.d_conv = cfg.get("d_conv", 3)
        self.expand = cfg.get("expand", 1)
        self.dropout = cfg.get("dropout", 0.03)
        self.use_amp = cfg.get("use_amp", True)
        self.gradient_checkpointing = cfg.get("gradient_checkpointing", True)
        self.layerdrop = cfg.get("layerdrop", 0.05)
        self.amp_dtype = torch.float16

        if obs_space.shape and len(obs_space.shape) == 2:
            self.num_stacked_frames = obs_space.shape[0]
            self.frame_dim = obs_space.shape[1]
        else:
            total = int(np.prod(obs_space.shape))
            self.num_stacked_frames = cfg.get("num_stacked_frames", 4)
            self.frame_dim = total // self.num_stacked_frames
            if total % self.num_stacked_frames != 0:
                self.frame_dim = 115
                self.num_stacked_frames = total // 115
                if total % 115 != 0:
                    raise ValueError(f"Obs-Shape {obs_space.shape} nicht teilbar.")
        
        embed_dim = self.d_model // 2
        self.frame_embed = nn.Sequential(
            nn.Linear(self.frame_dim, embed_dim, bias=False),
            RMSNorm(embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, self.d_model, bias=False),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
        )

        self.register_buffer("pos_encoding", torch.randn(1, self.num_stacked_frames, self.d_model) * 0.01)

        self.mamba_layers = nn.ModuleList(
            [MambaBlock(self.d_model, self.d_state, self.d_conv, int(self.expand), self.dropout) for _ in range(self.num_layers)]
        )

        self.pooled_dim = self.d_model * 2
        self.pool = nn.Sequential(RMSNorm(self.pooled_dim), nn.Mish())

        se_inner = max(8, self.pooled_dim // 8)
        self.se = nn.Sequential(
            nn.Linear(self.pooled_dim, se_inner, bias=False),
            nn.SiLU(),
            nn.Linear(se_inner, self.pooled_dim, bias=False),
            nn.Sigmoid(),
        )

        policy_hidden = max(self.d_model // 2, 64)
        self.policy_net = nn.Sequential(
            nn.Linear(self.pooled_dim, policy_hidden, bias=False),
            nn.Mish(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
        )
        self.policy_head = nn.Linear(policy_hidden, num_outputs)

        value_hidden = max(self.d_model // 4, 32)
        self.value_net = nn.Sequential(nn.Linear(self.pooled_dim, value_hidden, bias=False), nn.Mish())
        self.value_head = nn.Linear(value_hidden, 1)

        self._init_weights()
        self._value_out: Optional[torch.Tensor] = None

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) and hasattr(m, "weight"):
                if m not in [self.policy_head, self.value_head]:
                    nn.init.orthogonal_(m.weight, gain=0.8)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.zeros_(m.bias)
        normc_initializer(0.01)(self.policy_head.weight)
        normc_initializer(1.0)(self.value_head.weight)

    def _process_observation(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]
        if obs.ndim == 3:
            obs_seq = obs
        elif obs.ndim == 2:
            obs_seq = obs.view(B, self.num_stacked_frames, self.frame_dim)
        else:
            try:
                obs_seq = obs.view(B, self.num_stacked_frames, self.frame_dim)
            except RuntimeError:
                raise ValueError(f"Unerwartete Obs-Shape: {obs.shape}; erwarte (B, L, F) oder (B, L*F).")
        
        x = self.frame_embed(obs_seq)

        L = x.size(1)
        if L <= self.pos_encoding.size(1):
            pe = self.pos_encoding[:, :L].to(x.device, dtype=x.dtype)
        else:
            reps = (L + self.pos_encoding.size(1) - 1) // self.pos_encoding.size(1)
            pe = self.pos_encoding.repeat(1, reps, 1)[:, :L].to(x.device, dtype=x.dtype)
        
        return x + pe

    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType], seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        
        model_device = next(self.parameters()).device
        obs = input_dict["obs"].to(model_device, dtype=torch.float32)
        amp_dtype = self.amp_dtype if model_device.type == "cuda" else torch.bfloat16

        with autocast(device_type=model_device.type, dtype=amp_dtype, enabled=self.use_amp):
            x = self._process_observation(obs)

            for layer in self.mamba_layers:
                if self.training and torch.rand((), device=x.device) < self.layerdrop:
                    continue
                x = layer(x, use_checkpoint=self.gradient_checkpointing)

            x_mean = x.mean(dim=1)
            x_last = x[:, -1]
            x_pooled = torch.cat([x_mean, x_last], dim=-1)

            features = self.pool(x_pooled)
            features = features * self.se(features)

            logits = self.policy_head(self.policy_net(features))
            value = self.value_head(self.value_net(features)).squeeze(-1)

        self._value_out = value.float()
        return logits.float(), state

    def value_function(self) -> TensorType:
        assert self._value_out is not None, "value_function() wurde vor forward() aufgerufen"
        return self._value_out