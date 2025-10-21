import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.misc import normc_initializer
from ray.rllib.utils.typing import Dict, TensorType, List, ModelConfigDict
from gymnasium.spaces import Space
from typing import Tuple, Optional


class SoftClamp(nn.Module):
    """Differenzierbares Soft-Clipping mit Tanh-Funktion."""
    
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
    """Effiziente Root Mean Square Layer Normalization."""
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Nutze torch.rsqrt für effizientere Berechnung
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_normed = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight * x_normed).to(x.dtype)


class VectorizedSSM(nn.Module):
    """Vektorisiertes Selective State Space Model mit optimierter Berechnung."""
    
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # Projektionen
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        
        # Parameter
        self.A_log = nn.Parameter(torch.randn(d_model, d_state) * 0.1)
        self.D = nn.Parameter(torch.ones(d_model) * 0.1)
        
        # Clamps als Buffer registrieren für bessere Serialisierung
        self.dt_clamp = SoftClamp(0.001, 0.1, sharpness=0.5)
        self.A_log_clamp = SoftClamp(-5.0, 5.0, sharpness=0.5)
        self.dt_A_clamp = SoftClamp(-8.0, 8.0, sharpness=0.5)
        
        self._init_parameters()
    
    def _init_parameters(self):
        nn.init.normal_(self.dt_proj.weight, std=0.02)
        nn.init.constant_(self.dt_proj.bias, -2.0)  # softplus(-2) ≈ 0.013
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Vektorisierte Forward-Pass Berechnung.
        
        Args:
            x: Input tensor of shape (batch, length, d_model)
            
        Returns:
            Output tensor of shape (batch, length, d_model)
        """
        B, L, D = x.shape
        
        # Alle Projektionen auf einmal berechnen
        dt = self.dt_clamp(F.softplus(self.dt_proj(x)))  # (B, L, D)
        B_param = self.B_proj(x)  # (B, L, d_state)
        C_param = self.C_proj(x)  # (B, L, d_state)
        
        # A-Matrix vorbereiten
        A = -torch.exp(self.A_log_clamp(self.A_log))  # (D, d_state)
        
        # Diskretisierung - alles vektorisiert
        dt_A = self.dt_A_clamp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, D, d_state)
        dA = torch.exp(dt_A)
        dB = dt.unsqueeze(-1) * B_param.unsqueeze(2)  # (B, L, D, d_state)
        
        # Rekursion - leider sequenziell nötig, aber optimiert
        return self._scan_ssm(dA, dB, x, C_param)
    
    def _scan_ssm(self, dA: torch.Tensor, dB: torch.Tensor, 
                  x: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        """
        Memory-optimierte sequenzielle SSM-Berechnung.
        Minimale Allokationen, maximale In-place Operationen.
        """
        B, L, D, N = dA.shape
        device = x.device
        
        # FP32 für numerische Stabilität (aber nur h State)
        h = torch.zeros(B, D, N, device=device, dtype=torch.float32)
        
        # Output direkt in gewünschtem dtype allokieren
        outputs = torch.empty(B, L, D, device=device, dtype=x.dtype)
        
        # Cache D Parameter
        D_param = self.D.float()
        
        for t in range(L):
            # State Update mit In-place Ops (kein neuer Tensor!)
            h.mul_(dA[:, t].float())  # h *= dA
            h.addcmul_(dB[:, t].float(), x[:, t].float().unsqueeze(-1))  # h += dB * x
            
            # Conditional Clipping (nur wenn nötig - spart Ops)
            if t % 10 == 0:  # Nur jede 10. Iteration checken (meist unnötig)
                h_norm = h.norm(dim=(1, 2), keepdim=True)
                if (h_norm > 20.0).any():
                    h.mul_(torch.clamp(20.0 / (h_norm + 1e-8), max=1.0))
            
            # Output-Berechnung (ohne 'out' Parameter)
            y = torch.einsum('bdn,bn->bd', h, C[:, t].float())
            y.add_(D_param * x[:, t].float())
            outputs[:, t] = y.to(x.dtype)
        
        return outputs


class MambaBlock(nn.Module):
    """Optimierter Mamba-Block mit verbesserter Architektur."""
    
    def __init__(self, d_model: int, d_state: int = 16, 
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_inner = d_model * expand
        
        # Effiziente Projektionen (ohne Bias spart Memory)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Depthwise Convolution (kleinerer Kernel spart Memory)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )
        
        # SSM und Normalisierung
        self.ssm = VectorizedSSM(self.d_inner, d_state)
        self.norm = RMSNorm(d_model)
        
        # Lernbarer Residual-Gewichtung
        self.residual_scale = nn.Parameter(torch.tensor(0.3))
        
        # Optional: Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.in_proj.weight, gain=0.7)
        nn.init.orthogonal_(self.out_proj.weight, gain=0.7)
        nn.init.normal_(self.conv1d.weight, std=0.02)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Interner Forward ohne Residual für Checkpointing."""
        x = self.norm(x)
        
        # Split in Branch und Gate
        x_branch, gate_branch = self.in_proj(x).chunk(2, dim=-1)
        
        # Convolution (mit Padding-Korrektur)
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :x_branch.size(1)]
        x_conv = F.silu(x_conv.transpose(1, 2))
        
        # SSM und Gating
        x_ssm = self.ssm(x_conv)
        x_out = x_ssm * F.silu(gate_branch)
        x_out = self.out_proj(x_out)
        x_out = self.dropout(x_out)
        
        return x_out
    
    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        residual = x
        
        # Gradient Checkpointing für Memory-Einsparung
        if use_checkpoint and self.training:
            x_out = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            x_out = self._forward_impl(x)
        
        # Residual mit lernbarer Gewichtung
        scale = torch.sigmoid(self.residual_scale) * 0.5
        return residual + scale * x_out


class GFootballMamba(TorchModelV2, nn.Module):
    """Optimiertes Mamba-Modell für Google Research Football."""
    
    def __init__(self, obs_space: Space, action_space: Space,
                 num_outputs: int, model_config: ModelConfigDict, name: str):
        
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, 
                              model_config, name)
        nn.Module.__init__(self)

        # Konfiguration extrahieren
        config = model_config.get("custom_model_config", {})
        self.d_model = config.get("d_model", 256)
        self.num_layers = config.get("num_layers", 4)
        self.d_state = config.get("d_state", 16)
        self.d_conv = config.get("d_conv", 4)
        self.expand = config.get("expand", 2)
        self.num_stacked_frames = config.get("num_stacked_frames", 4)
        self.dropout = config.get("dropout", 0.1)
        self.use_amp = config.get("use_amp", True)
        self.gradient_checkpointing = config.get("gradient_checkpointing", True)  # NEU!
        self.amp_dtype = torch.float16
        
        # Dimensionen berechnen
        total_obs_dim = int(np.prod(obs_space.shape))
        self.frame_dim = total_obs_dim // self.num_stacked_frames
        
        # Frame Embedding (kleiner für Memory-Effizienz)
        embed_dim = self.d_model // 2
        self.frame_embed = nn.Sequential(
            nn.Linear(self.frame_dim, embed_dim, bias=False),
            RMSNorm(embed_dim),
            nn.Mish(),
            nn.Linear(embed_dim, self.d_model, bias=False),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        )
        
        # Positional Encoding
        self.register_buffer('pos_encoding', 
                           torch.randn(1, self.num_stacked_frames, self.d_model) * 0.01)
        
        # Mamba Layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(self.d_model, self.d_state, self.d_conv, self.expand, self.dropout)
            for _ in range(self.num_layers)
        ])
        
        # Feature Pooling (ohne extra Linear Layer für Memory)
        self.pool = nn.Sequential(
            RMSNorm(self.d_model),
            nn.Mish()
        )
        
        # Policy Head (kompakter)
        policy_hidden = max(self.d_model // 2, 64)
        self.policy_net = nn.Sequential(
            nn.Linear(self.d_model, policy_hidden, bias=False),
            nn.Mish(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity()
        )
        self.policy_head = nn.Linear(policy_hidden, num_outputs)
        
        # Value Head (kompakter)
        value_hidden = max(self.d_model // 4, 32)
        self.value_net = nn.Sequential(
            nn.Linear(self.d_model, value_hidden, bias=False),
            nn.Mish(),
        )
        self.value_head = nn.Linear(value_hidden, 1)
        
        self._init_weights()
        self._value_out: Optional[torch.Tensor] = None
        
    def _init_weights(self):
        """Initialisierung aller Gewichte."""
        for module in self.modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                if module not in [self.policy_head, self.value_head]:
                    nn.init.orthogonal_(module.weight, gain=0.8)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # Spezielle Initialisierung für Heads
        normc_initializer(0.01)(self.policy_head.weight)
        normc_initializer(1.0)(self.value_head.weight)
    
    def _process_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """Verarbeite Observation zu Sequenz-Embedding."""
        B = obs.shape[0]
        obs_flat = obs.view(B, -1)
        obs_seq = obs_flat.view(B, self.num_stacked_frames, self.frame_dim)
        return self.frame_embed(obs_seq) + self.pos_encoding

    def forward(self, input_dict: Dict[str, TensorType],
                state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:
        """Forward-Pass mit automatischer Mixed Precision."""
        
        obs = input_dict["obs"].float()
        
        with autocast(device_type=obs.device.type, 
                     dtype=self.amp_dtype, 
                     enabled=self.use_amp):
            # Observation verarbeiten
            x = self._process_observation(obs)
            
            # Mamba Layers mit optionalem Gradient Checkpointing
            for layer in self.mamba_layers:
                x = layer(x, use_checkpoint=self.gradient_checkpointing)
            
            # Pooling und Features
            features = self.pool(x.mean(dim=1))
            
            # Policy und Value Heads
            logits = self.policy_head(self.policy_net(features))
            value = self.value_head(self.value_net(features)).squeeze(-1)
        
        # Zurück zu FP32 für numerische Stabilität
        self._value_out = value.float()
        return logits.float(), state
    
    def value_function(self) -> TensorType:
        """Gibt den zuletzt berechneten Value zurück."""
        assert self._value_out is not None, \
            "value_function() wurde vor forward() aufgerufen"
        return self._value_out