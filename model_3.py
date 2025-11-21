import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, List, Dict, Optional, Any
from gymnasium import spaces

from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TensorType

try:
    from torch.amp import autocast 
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import autocast
        AMP_AVAILABLE = True
    except ImportError:
        class autocast:
            def __init__(self, *args, **kwargs): pass
            def __enter__(self): pass
            def __exit__(self, *args): pass
        AMP_AVAILABLE = False

AMP_AVAILABLE = False

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
        expected_h_size = B * self.d_model * self.d_state
        
        if h_prev.numel() != expected_h_size:
            if h_prev.numel() > expected_h_size:
                 h_prev = h_prev[:B]
        
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
        x_out = x + out
        h_flat = h.reshape(B, -1)
        
        return x_out, h_flat


class GFootballMambaRLModule(TorchRLModule, ValueFunctionAPI):
    @override(RLModule)
    def setup(self):
        cfg = self.config.model_config_dict
        
        self.d_model = cfg.get("d_model", 128)
        self.mamba_state = cfg.get("mamba_state", 8)
        self.num_mamba_layers = cfg.get("num_mamba_layers", 3)
        self.num_stages = cfg.get("num_stages", 8)
        
        obs_space = self.config.observation_space
        action_space = self.config.action_space
        
        if isinstance(obs_space, spaces.Dict) and "obs" in obs_space.spaces:
            actual_obs_space = obs_space.spaces["obs"]
        else:
            actual_obs_space = obs_space
        
        self.total_obs_dim = 460
        self.num_actions = action_space.n
        
        self.mlp_hidden_dims = cfg.get("mlp_hidden_dims", [256, 128])
        self.mlp_activation = self._get_activation(cfg.get("mlp_activation", "silu"))
        
        self.mlp_encoder = self._build_mlp(
            self.total_obs_dim,
            self.mlp_hidden_dims,
            self.mlp_activation,
            add_final_norm=True
        )
        self.mlp_output_dim = self.mlp_hidden_dims[-1]
        
        self.prev_action_emb_dim = cfg.get("prev_action_emb", 16)
        self.prev_action_embed = nn.Embedding(self.num_actions + 1, self.prev_action_emb_dim)
        
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
        
        self.use_checkpoint = cfg.get("gradient_checkpointing", False)
        self.final_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        
        self.head_hidden_dims = cfg.get("head_hidden_dims", [128])
        self.head_activation = self._get_activation(cfg.get("head_activation", "silu"))
        
        self.policy_head = self._build_mlp(
            self.d_model,
            self.head_hidden_dims + [self.num_actions],
            self.head_activation,
            final_activation=None
        )
        
        self.use_distributional = cfg.get("use_distributional", True)
        if self.use_distributional:
            self.v_min = cfg.get("v_min", -10.0)
            self.v_max = cfg.get("v_max", 10.0)
            self.num_atoms = cfg.get("num_atoms", 51)
            self.value_output_dim = self.num_atoms
            self.register_buffer("atoms", torch.linspace(self.v_min, self.v_max, self.num_atoms))
        else:
            self.value_output_dim = 1
            
        self.value_heads = nn.ModuleList([
            self._build_mlp(
                self.d_model,
                self.head_hidden_dims + [self.value_output_dim],
                self.head_activation,
                final_activation=None
            )
            for _ in range(self.num_stages)
        ])
        
        self.state_size = self.d_model * self.mamba_state
        
        weights_path = cfg.get("pretrained_weights_path")
        if weights_path and os.path.exists(weights_path):
            self.load_state_dict(torch.load(weights_path, map_location="cpu"), strict=False)

    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'tanh': nn.Tanh()
        }
        return activations.get(name, nn.SiLU())
    
    def _build_mlp(self, in_dim: int, hidden_dims: List[int],
                      activation: nn.Module,
                      final_activation: Optional[nn.Module] = None,
                      add_final_norm: bool = False) -> nn.Module:
        layers = []
        current_dim = in_dim
        for i, h_dim in enumerate(hidden_dims):
            is_last = (i == len(hidden_dims) - 1)
            layers.append(nn.Linear(current_dim, h_dim))
            
            if is_last:
                if add_final_norm:
                    layers.append(nn.LayerNorm(h_dim, eps=1e-5))
                if final_activation is not None:
                    layers.append(final_activation)
            else:
                layers.append(nn.LayerNorm(h_dim, eps=1e-5))
                layers.append(activation)
            current_dim = h_dim
        return nn.Sequential(*layers)
    
    @override(RLModule)
    def get_initial_state(self) -> Dict[str, TensorType]:
        return {
            f"h_{i}": torch.zeros(self.state_size)
            for i in range(self.num_mamba_layers)
        }
    
    def _process_observation(self, obs_flat: torch.Tensor) -> Tuple[torch.Tensor, int, int, bool]:
        has_time_dim = False

        if obs_flat.dim() == 4:
            T, B, stack, feats = obs_flat.shape
            has_time_dim = True
            obs_flat = obs_flat.permute(1, 0, 2, 3)
            obs_flat = obs_flat.reshape(B * T, stack * feats)
        elif obs_flat.dim() == 3:
            B, stack, feats = obs_flat.shape
            T = 1
            obs_flat = obs_flat.reshape(B * T, stack * feats)
        elif obs_flat.dim() == 2:
            B = obs_flat.shape[0]
            T = 1
        elif obs_flat.dim() == 1:
            B = 1
            T = 1
            obs_flat = obs_flat.unsqueeze(0)
        else:
            raise ValueError(f"Unexpected obs shape: {obs_flat.shape}")

        return obs_flat, B, T, has_time_dim

    @override(RLModule)
    def _forward(self, batch: Dict[str, TensorType], 
                 explore: bool = True,
                 inference: bool = False,
                 **kwargs) -> Dict[str, TensorType]:
        
        if isinstance(batch["obs"], dict):
            obs_flat = batch["obs"]["obs"]
            stage_indices = batch["obs"]["stage_index"]
        else:
            obs_flat = batch["obs"]
            if obs_flat.dim() >= 2:
                batch_size = obs_flat.shape[0]
            else:
                batch_size = 1
            stage_indices = torch.zeros(batch_size, 1, dtype=torch.long, device=obs_flat.device)
        
        obs_flat, B, T_max, has_time_dim = self._process_observation(obs_flat)
        
        device = obs_flat.device
        BT = B * T_max
        
        obs_flat = obs_flat.reshape(BT, 460)
        
        if stage_indices.dim() == 1:
            stage_indices = stage_indices.unsqueeze(-1)
        if stage_indices.dim() == 2:
            if T_max > 1:
                stage_indices = stage_indices.unsqueeze(1).expand(-1, T_max, -1)
            else:
                stage_indices = stage_indices.view(B, 1, 1)
        elif stage_indices.dim() == 3:
            pass
        
        stage_indices_flat = stage_indices.reshape(BT, -1).squeeze(-1).long()
        
        if "prev_actions" in batch:
            prev_actions = batch["prev_actions"]
            if prev_actions.dim() == 1:
                prev_actions = prev_actions.unsqueeze(0)
            if prev_actions.numel() != BT:
                if prev_actions.shape[0] == B:
                    prev_actions = prev_actions.unsqueeze(1).expand(B, T_max)
        else:
            prev_actions = torch.full((BT,), self.num_actions, dtype=torch.long, device=device)
        
        prev_actions_flat = prev_actions.reshape(BT).long()
        
        state_in = {}
        for i in range(self.num_mamba_layers):
            key = f"h_{i}"
            if key in batch:
                state_tensor = batch[key]
                
                if state_tensor.dim() == 3: 
                    state_tensor = state_tensor[:, 0, :] 
                elif state_tensor.shape[0] == B * T_max and T_max > 1:
                    state_tensor = state_tensor.view(B, T_max, -1)[:, 0, :]
                
                if state_tensor.shape[0] != B:
                    if state_tensor.shape[0] == 1 and B > 1:
                        state_tensor = state_tensor.expand(B, -1)
                    elif B == 1:
                        state_tensor = state_tensor[:1]
                state_in[i] = state_tensor
            else:
                state_in[i] = torch.zeros((B, self.state_size), device=device)
        
        with autocast(device_type='cuda' if device.type == 'cuda' else 'cpu', 
                      dtype=torch.bfloat16, 
                      enabled=AMP_AVAILABLE and device.type == 'cuda'):
            
            mlp_features = self.mlp_encoder(obs_flat)
            pa_emb = self.prev_action_embed(prev_actions_flat)
            x = torch.cat([mlp_features, pa_emb], dim=-1)
            x = self.input_proj(x)
            
            x = x.view(B, T_max, self.d_model)
            
            state_out = {}
            for i, mamba_block in enumerate(self.mamba_blocks):
                h_prev = state_in[i]
                if self.use_checkpoint and self.training:
                    x, h_last = checkpoint(mamba_block, x, h_prev, use_reentrant=False)
                else:
                    x, h_last = mamba_block(x, h_prev)
                state_out[f"h_{i}"] = h_last.detach()
            
            x_core = self.final_norm(x)
            x_core_flat = x_core.reshape(BT, self.d_model)
            
            if self.use_checkpoint and self.training:
                policy_logits_flat = checkpoint(self.policy_head, x_core_flat, use_reentrant=False)
            else:
                policy_logits_flat = self.policy_head(x_core_flat)
            
            value_outputs = torch.zeros((BT, self.value_output_dim), 
                                        device=device, dtype=x_core_flat.dtype)
            
            for stage_idx in range(self.num_stages):
                stage_mask = (stage_indices_flat == stage_idx)
                if stage_mask.any():
                    stage_features = x_core_flat[stage_mask]
                    if self.use_checkpoint and self.training:
                        stage_values = checkpoint(self.value_heads[stage_idx], stage_features, use_reentrant=False)
                    else:
                        stage_values = self.value_heads[stage_idx](stage_features)
                    
                    value_outputs[stage_mask] = stage_values.to(value_outputs.dtype)
            
            if "action_mask" in batch:
                action_mask = batch["action_mask"]
                if action_mask.dim() == 2:
                    action_mask = action_mask.unsqueeze(1).expand(-1, T_max, -1)
                action_mask_flat = action_mask.reshape(BT, -1).float()
                
                mask_sum = action_mask_flat.sum(dim=1)
                action_mask_flat[mask_sum == 0] = 1.0
                
                inf_mask = torch.log(action_mask_flat.clamp(min=1e-10))
                policy_logits_flat = policy_logits_flat + inf_mask
            
            policy_logits_flat = torch.nan_to_num(policy_logits_flat, nan=0.0, posinf=1e9, neginf=-1e9)
            
        if self.use_distributional:
            probs = F.softmax(value_outputs.float(), dim=-1)
            atoms = self.atoms.to(probs.device).view(1, 1, -1)
            vf_preds_flat = (probs * atoms).sum(dim=-1)
        else:
            vf_preds_flat = value_outputs.squeeze(-1)

        if has_time_dim:
            policy_logits = policy_logits_flat.view(B, T_max, self.num_actions).permute(1, 0, 2).contiguous()
            vf_preds = vf_preds_flat.view(B, T_max).permute(1, 0).contiguous()
        else:
            policy_logits = policy_logits_flat
            vf_preds = vf_preds_flat

        # WICHTIG: Cast auf float32 für den RLlib Metrics Logger
        policy_logits = policy_logits.float()
        vf_preds = vf_preds.float()

        state_out_time_major = {}
        for key, tensor in state_out.items():
            if has_time_dim:
                state_out_time_major[key] = tensor.unsqueeze(0)
            else:
                state_out_time_major[key] = tensor
        
        outputs = {
            "action_dist_inputs": policy_logits, 
            "vf_preds": vf_preds,
            **state_out_time_major
        }

        return outputs

    @override(TorchRLModule)
    def forward_inference(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        return self._forward(batch, explore=False, inference=True, **kwargs)
    
    @override(TorchRLModule)
    def forward_exploration(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        return self._forward(batch, explore=True, inference=False, **kwargs)
    
    @override(TorchRLModule)
    def forward_train(self, batch: Dict[str, TensorType], **kwargs) -> Dict[str, TensorType]:
        return self._forward(batch, explore=False, inference=False, **kwargs)

    @override(ValueFunctionAPI)
    def compute_values(self, batch: Dict[str, TensorType], **kwargs) -> TensorType:
        out = self._forward(batch, explore=False, inference=False)
        return out["vf_preds"].float()