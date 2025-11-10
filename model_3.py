import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.jit as jit
from gymnasium.spaces import Space
from typing import Tuple, List, Dict, Optional
import math

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.view_requirement import ViewRequirement

try:
    from torch.cuda.amp import autocast
    AMP_AVAILABLE = True
except ImportError:
    class autocast:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): pass
        def __exit__(self, *args): pass
    AMP_AVAILABLE = False
    print("Warning: torch.cuda.amp not available. AMP disabled.")


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -mu_range, mu_range)
        nn.init.uniform_(self.bias_mu, -mu_range, mu_range)
        nn.init.constant_(self.weight_sigma, self.std_init / math.sqrt(self.in_features))
        nn.init.constant_(self.bias_sigma, self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class CompactS6Layer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.norm = nn.LayerNorm(d_model, eps=1e-5)
        self.in_gate_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        with torch.no_grad():
            nn.init.uniform_(self.A_log, -3.0, -1.0)
    
    def forward(self, x: torch.Tensor, h_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        h = h_prev.view(B, self.d_model, self.d_state)
        x_norm = self.norm(x)
        x_proj = self.in_gate_proj(x_norm)
        x_in, z = x_proj.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(x_in))
        A = -torch.exp(self.A_log)
        B_t = self.B_proj(x_in)
        C_t = self.C_proj(x_in)
        outputs = []
        for t in range(L):
            dt_t = dt[:, t]
            B_t_t = B_t[:, t]
            C_t_t = C_t[:, t]
            x_in_t = x_in[:, t]
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))
            update_term = (dt_t * x_in_t).unsqueeze(-1) * B_t_t.unsqueeze(1)
            h = dA * h + update_term
            y_t = (h * C_t_t.unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t.unsqueeze(1))
        y = torch.cat(outputs, dim=1)
        ssm_out = y * F.silu(z) + x_in * self.D
        out = self.out_proj(ssm_out)
        h_flat = h.reshape(B, -1)
        return x + out, h_flat


class GFootballMamba(TorchRNN, nn.Module):
    
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
        nn.Module.__init__(self)
        cfg = model_config.get("custom_model_config", {})

        self.d_model = cfg.get("d_model", 128)
        self.mamba_state = cfg.get("mamba_state", 8)
        self.num_mamba_layers = cfg.get("num_mamba_layers", 3)

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.view_requirements["prev_actions"] = ViewRequirement(
            data_col="actions", shift=-1, space=self.action_space,
            used_for_compute_actions=True
        )
        self.prev_action_emb_dim = cfg.get("prev_action_emb", 16)
        self.use_checkpoint = cfg.get("gradient_checkpointing", False)

        self.total_obs_dim = int(np.prod(obs_space.shape))
        self.mlp_hidden_dims = cfg.get("mlp_hidden_dims", [256, 128])
        self.mlp_activation = self._get_activation(cfg.get("mlp_activation", "silu"))
        
        self.mlp_encoder = self._build_mlp(
            self.total_obs_dim,
            self.mlp_hidden_dims,
            self.mlp_activation,
            add_final_norm=True,
            layer_cls=nn.Linear
        )
        self.mlp_output_dim = self.mlp_hidden_dims[-1]
        
        self.prev_action_embed = nn.Embedding(action_space.n + 1, self.prev_action_emb_dim)
        self.input_dim = self.mlp_output_dim + self.prev_action_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model), 
            nn.LayerNorm(self.d_model, eps=1e-5), 
            nn.SiLU()
        )
        
        self.mamba_blocks = nn.ModuleList([
            CompactS6Layer(d_model=self.d_model, d_state=self.mamba_state)
            for _ in range(self.num_mamba_layers)
        ])
        
        self.use_noisy = cfg.get("use_noisy", True)
        self.use_distributional = cfg.get("use_distributional", True)
        self.v_min = cfg.get("v_min", -10.0)
        self.v_max = cfg.get("v_max", 10.0)
        self.num_atoms = cfg.get("num_atoms", 51)
        
        self.head_hidden_dims = cfg.get("head_hidden_dims", [128])
        self.head_activation = self._get_activation(cfg.get("head_activation", "silu"))
        self.final_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        
        head_layer_cls = NoisyLinear if self.use_noisy else nn.Linear
        print(f"Using {'NoisyLinear' if self.use_noisy else 'Linear'} for model heads.")

        self.policy_head = self._build_mlp(
            self.d_model,
            self.head_hidden_dims + [num_outputs],
            self.head_activation,
            final_activation=None,
            layer_cls=head_layer_cls
        )
        
        if self.use_distributional:
            self.value_output_dim = self.num_atoms
            self.register_buffer("atoms", torch.linspace(self.v_min, self.v_max, self.num_atoms))
            print(f"Using Distributional Critic (C51) with {self.num_atoms} atoms ({self.v_min} to {self.v_max}).")
        else:
            self.value_output_dim = 1
            print("Using Standard (Scalar) Critic.")

        self.value_head = self._build_mlp(
            self.d_model,
            self.head_hidden_dims + [self.value_output_dim],
            self.head_activation,
            final_activation=None,
            layer_cls=head_layer_cls
        )
        
        self._value_out = None
        self.state_size = self.d_model * self.mamba_state
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[GFootballMamba] Total params: {total_params/1e6:.2f}M, "
              f"NoisyNets={self.use_noisy}, Distributional={self.use_distributional}")

    def _get_activation(self, name: str) -> nn.Module:
        if name == 'silu': return nn.SiLU()
        if name == 'relu': return nn.ReLU()
        if name == 'gelu': return nn.GELU()
        if name == 'tanh': return nn.Tanh()
        return nn.SiLU()

    def _build_mlp(self, in_dim: int, hidden_dims: List[int], 
                   activation: nn.Module, layer_cls: nn.Module,
                   final_activation: Optional[nn.Module] = None,
                   add_final_norm: bool = False) -> nn.Module:
        layers = []
        current_dim = in_dim
        for i, h_dim in enumerate(hidden_dims):
            is_last_layer = (i == len(hidden_dims) - 1)
            layers.append(layer_cls(current_dim, h_dim))
            
            if is_last_layer:
                if add_final_norm:
                    layers.append(nn.LayerNorm(h_dim, eps=1e-5))
                if final_activation is not None:
                    layers.append(final_activation)
            else:
                layers.append(nn.LayerNorm(h_dim, eps=1e-5))
                layers.append(activation)
            current_dim = h_dim
        return nn.Sequential(*layers)

    @override(TorchRNN)
    def get_initial_state(self) -> List[TensorType]:
        return [torch.zeros(self.state_size) for _ in range(self.num_mamba_layers)]
        
    def reset_noise(self):
        if self.use_noisy:
            for module in self.policy_head.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()
            for module in self.value_head.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:

        obs_flat = input_dict.get("obs_flat", input_dict.get("obs"))
        if obs_flat is None: raise ValueError("Missing obs")
        BT, obs_dim = obs_flat.shape; device = obs_flat.device
        
        if "prev_actions" in input_dict:
            prev_actions_input = input_dict["prev_actions"]
        else:
            prev_actions_input = torch.full((BT,), self.action_space.n, dtype=torch.long, device=device)

        if seq_lens is None:
            seq_lens = torch.full((BT,), 1, dtype=torch.int64, device=device); T_max = 1; B = BT
        else:
            seq_lens = seq_lens.to(dtype=torch.int64); T_max = seq_lens.max().item(); B = seq_lens.shape[0]

        seq_mask = None
        if B * T_max != BT:
            obs_unpadded = torch.split(obs_flat, seq_lens.tolist())
            obs_padded = nn.utils.rnn.pad_sequence(obs_unpadded, batch_first=True, padding_value=0.0)
            B_pad, T_max_pad, _ = obs_padded.shape
            obs_flat = obs_padded.reshape(B_pad * T_max_pad, obs_dim)

            prev_actions_unpadded = torch.split(prev_actions_input.long(), seq_lens.tolist())
            prev_actions_padded = nn.utils.rnn.pad_sequence(prev_actions_unpadded, batch_first=True, padding_value=self.action_space.n)
            prev_actions = prev_actions_padded.view(B_pad, T_max_pad)

            seq_mask = torch.arange(T_max_pad, device=device).unsqueeze(0) < seq_lens.unsqueeze(1)
            BT = B_pad * T_max_pad
        else:
            prev_actions = prev_actions_input.long().view(B, T_max)
        
        with autocast(enabled=AMP_AVAILABLE and device.type == 'cuda', dtype=torch.bfloat16):
            
            mlp_features = self.mlp_encoder(obs_flat)
            pa_emb = self.prev_action_embed(prev_actions.view(BT))
            
            x = torch.cat([mlp_features, pa_emb], dim=-1)
            x = self.input_proj(x)
            x = x.view(B, T_max, self.d_model)

            h_states = [s.to(device) for s in state]
            new_states = []
            for i, mamba_block in enumerate(self.mamba_blocks):
                if self.use_checkpoint and self.training:
                    x, h_last = checkpoint(mamba_block, x, h_states[i], use_reentrant=False)
                else:
                    x, h_last = mamba_block(x, h_states[i])
                new_states.append(h_last.detach())
            x_mamba = x
            
            x_core = self.final_norm(x_mamba)
            x_core_flat = x_core.reshape(BT, self.d_model)

            if seq_mask is not None:
                mask_flat = seq_mask.reshape(BT, 1).to(x_core_flat.dtype)
                x_core_flat = x_core_flat * mask_flat
            
            if self.use_checkpoint and self.training:
                policy_logits_raw = checkpoint(self.policy_head, x_core_flat, use_reentrant=False)
                value_raw = checkpoint(self.value_head, x_core_flat, use_reentrant=False)
            else:
                policy_logits_raw = self.policy_head(x_core_flat)
                value_raw = self.value_head(x_core_flat)

            logits = policy_logits_raw.float() 
            
            if self.use_distributional:
                self._value_out = value_raw.float()
            else:
                self._value_out = value_raw.float().squeeze(-1)
                
        if "action_mask" in input_dict:
            action_mask = input_dict["action_mask"].float()
            if action_mask.shape[0] != BT:
                action_mask_unpadded = torch.split(action_mask, seq_lens.tolist())
                action_mask_padded = nn.utils.rnn.pad_sequence(action_mask_unpadded, batch_first=True, padding_value=0.0)
                action_mask = action_mask_padded.view(BT, -1)
            
            all_zero = (action_mask.sum(dim=1) == 0)
            if all_zero.any():
                action_mask[all_zero] = 1.0

            inf_mask = torch.log(action_mask.clamp(min=1e-6))
            logits = logits + inf_mask
        
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            logits = torch.nan_to_num(logits, nan=0.0, posinf=1e9, neginf=-1e9)

        return logits, new_states

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Gibt den Critic-Output zurück (Verteilungs-Logits oder Skalar)."""
        assert self._value_out is not None
        return self._value_out