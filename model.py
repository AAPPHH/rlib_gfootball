import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.jit as jit
from gymnasium.spaces import Space
from typing import Tuple, List, Dict

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.view_requirement import ViewRequirement


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        
        self.in_norm = nn.LayerNorm(in_dim)
        
        self.spline_weight = nn.Parameter(torch.randn(out_dim, in_dim, grid_size) * 0.1)
        self.scale_base = nn.Parameter(torch.ones(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        
        self.rbf_beta = nn.Parameter(torch.ones(1) * 2.0)
        
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        
        B = x.shape[0]
        x_expanded = x.unsqueeze(1)
        
        grid = self.grid.to(x.dtype).to(x.device)
        grid_expanded = grid.view(1, 1, 1, -1)
        x_grid = x_expanded.unsqueeze(-1)
        
        beta = self.rbf_beta.clamp(0.5, 6.0)
        basis = torch.exp(-((x_grid - grid_expanded) ** 2) * beta)
        
        spline_out = torch.einsum('bdig,oid->bod', basis, self.spline_weight)
        
        base_out = torch.einsum('bi,oi->bo', x, self.scale_base)
        spline_sum = spline_out.sum(dim=-1)
        
        return base_out + spline_sum + self.bias


class GNNEncoder(nn.Module):
    def __init__(self, node_dim: int, hidden_dim: int):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim, bias=False)
        self.edge_weight = nn.Parameter(torch.ones(1) * 0.5)
    
    def forward(self, node_features: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        x = self.node_proj(node_features) 
        x_agg = torch.bmm(adj_matrix, x) 
        x = x * (1 - self.edge_weight) + x_agg * self.edge_weight
        graph_emb = x.mean(dim=1) 
        return graph_emb


class CompactS6Layer(nn.Module):
    def __init__(self, d_model: int, d_state: int = 8):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.norm = nn.LayerNorm(d_model)
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
        
        self.d_model = cfg.get("d_model", 96)
        self.mamba_state = cfg.get("mamba_state", 8)
        self.num_mamba_layers = cfg.get("num_mamba_layers", 2)
        
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.view_requirements["prev_actions"] = ViewRequirement(
            data_col="actions", shift=-1, space=self.action_space
        )

        self.prev_action_emb_dim = cfg.get("prev_action_emb", 16)
        
        self.use_kan = cfg.get("use_kan", True)
        self.kan_grid = cfg.get("kan_grid", 5)
        
        self.use_gnn = cfg.get("use_gnn", True)
        self.gnn_hidden = cfg.get("gnn_hidden", 32)
        self.gnn_k = cfg.get("gnn_k", 6)
        
        self.use_checkpoint = cfg.get("gradient_checkpointing", False)
        
        self.num_frames = 4
        total_obs_dim = int(np.prod(obs_space.shape))
        
        if total_obs_dim % self.num_frames != 0:
            raise ValueError(f"Obs dim {total_obs_dim} not divisible by {self.num_frames}")
        
        self.frame_dim = total_obs_dim // self.num_frames
        
        if self.frame_dim != 115:
            print(f"WARNING: frame_dim={self.frame_dim}, expected 115")
        
        self.num_players = 22
        self.register_buffer("adj_eye", torch.eye(self.num_players, dtype=torch.float32))
        
        if self.use_kan:
            self.frame_encoder = KANLayer(self.frame_dim, self.d_model, grid_size=self.kan_grid)
        else:
            self.frame_encoder = nn.Linear(self.frame_dim, self.d_model)
        
        self.frame_norm = nn.LayerNorm(self.d_model)
        
        if self.use_gnn:
            self.node_feature_dim = 8
            self.gnn_encoder = GNNEncoder(self.node_feature_dim, self.gnn_hidden)

        self.prev_action_embed = nn.Embedding(action_space.n, self.prev_action_emb_dim)
        
        self.input_dim = self.d_model + self.gnn_hidden + self.prev_action_emb_dim
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        
        self.mamba_blocks = nn.ModuleList([
            CompactS6Layer(
                d_model=self.d_model,
                d_state=self.mamba_state
            )
            for _ in range(self.num_mamba_layers)
        ])
        
        self.final_norm = nn.LayerNorm(self.d_model)
        
        self.policy_head = nn.Sequential(
            nn.Linear(self.d_model, max(64, self.d_model // 2)),
            nn.ReLU(),
            nn.Linear(max(64, self.d_model // 2), num_outputs)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.d_model, max(32, self.d_model // 4)),
            nn.ReLU(),
            nn.Linear(max(32, self.d_model // 4), 1)
        )
        
        self._value_out = None
        
        self.state_size = self.d_model * self.mamba_state
        
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"[GFootballMamba] Simple115v2 compatible (Memory Optimized):")
        print(f"  - d_model: {self.d_model}")
        print(f"  - State size per layer: {self.state_size}")
        print(f"  - Mamba layers: {self.num_mamba_layers}")
        print(f"  - Gradient Checkpointing: {self.use_checkpoint}")
        print(f"  - Total params: {total_params/1e6:.2f}M")

    @override(TorchRNN)
    def get_initial_state(self) -> List[TensorType]:
        return [
            torch.zeros(self.state_size)
            for _ in range(self.num_mamba_layers)
        ]
    
    @staticmethod
    @jit.script
    def _parse_frame_to_graph_impl(frame_curr: torch.Tensor, frame_prev: torch.Tensor,
                                     adj_eye: torch.Tensor, gnn_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B = frame_curr.size(0)
        device = frame_curr.device
        
        ball_pos = frame_curr[:, :2]
        left_pos = frame_curr[:, 3:25].view(B, 11, 2)
        right_pos = frame_curr[:, 25:47].view(B, 11, 2)
        
        ball_pos_prev = frame_prev[:, :2]
        left_pos_prev = frame_prev[:, 3:25].view(B, 11, 2)
        right_pos_prev = frame_prev[:, 25:47].view(B, 11, 2)
        
        all_pos = torch.cat([left_pos, right_pos], dim=1)
        all_pos_prev = torch.cat([left_pos_prev, right_pos_prev], dim=1)
        
        all_vel = all_pos - all_pos_prev

        team_id = torch.cat([
            torch.zeros(B, 11, 1, device=device),
            torch.ones(B, 11, 1, device=device)
        ], dim=1) 
        
        ball_pos_exp = ball_pos.unsqueeze(1) 
        ball_dist = torch.norm(all_pos - ball_pos_exp, dim=-1, keepdim=True) 
        ball_prox = torch.exp(-ball_dist * 2)
        
        left_center = left_pos.mean(dim=1, keepdim=True)
        right_center = right_pos.mean(dim=1, keepdim=True)
        team_center = torch.cat([
            left_center.expand(-1, 11, -1), 
            right_center.expand(-1, 11, -1)
        ], dim=1)
        delta_team = all_pos - team_center
        
        node_features = torch.cat([all_pos, all_vel, team_id, ball_prox, delta_team], dim=-1)
        
        pos_i = all_pos.unsqueeze(2) 
        pos_j = all_pos.unsqueeze(1) 
        
        eye = adj_eye.to(device).unsqueeze(0)
        
        with torch.no_grad():
            dists = torch.norm(pos_i - pos_j, dim=-1)
            dists = dists + eye * 1e6
            _, idx = torch.topk(-dists, gnn_k, dim=-1)
        
        adj = torch.zeros_like(dists).scatter_(-1, idx, 1.0)
        adj = (adj + eye).clamp_(max=1.0)
        adj = adj / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        
        return node_features, adj

    def _parse_frame_to_graph(self, frame_curr: torch.Tensor, frame_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._parse_frame_to_graph_impl(frame_curr, frame_prev, self.adj_eye, self.gnn_k)
    
    def _encode_frames(self, obs_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = obs_frames.size(0)
        device = obs_frames.device
        
        obs_flat = obs_frames.reshape(B * self.num_frames, self.frame_dim)
        frame_emb = self.frame_encoder(obs_flat)
        frame_emb = frame_emb.reshape(B, self.num_frames, self.d_model)
        frame_emb = F.relu(self.frame_norm(frame_emb))
        
        frame_features = frame_emb.mean(dim=1)
        
        del obs_flat, frame_emb
        
        if self.use_gnn:
            frame_curr = obs_frames[:, -1, :]
            frame_prev = obs_frames[:, -2, :] if self.num_frames > 1 else frame_curr
            
            with torch.no_grad():
                nodes, adj = self._parse_frame_to_graph(frame_curr, frame_prev)
            
            gnn_features = self.gnn_encoder(nodes, adj)
            
            del nodes, adj, frame_curr, frame_prev
        else:
            gnn_features = torch.zeros(B, self.gnn_hidden, device=device)
        
        return frame_features, gnn_features

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:

        obs_flat = input_dict.get("obs_flat", input_dict.get("obs"))
        if obs_flat is None:
            raise ValueError("Missing obs")

        BT, obs_dim = obs_flat.shape
        device = obs_flat.device
        
        if seq_lens is None:
            seq_lens = torch.full((BT,), 1, dtype=torch.int64, device=device)
            T_max = 1
            B = BT
        else:
            seq_lens = seq_lens.to(dtype=torch.int64)
            T_max = int(seq_lens.max().item())
            B = seq_lens.shape[0]

        if B * T_max != BT:
            obs_unpadded = torch.split(obs_flat, seq_lens.int().tolist())
            obs_padded = nn.utils.rnn.pad_sequence(obs_unpadded, batch_first=True, padding_value=0.0)
            B, T_max, _ = obs_padded.shape
            obs_flat = obs_padded.reshape(B * T_max, obs_dim)
            BT = B * T_max
            
            prev_actions_unpadded = torch.split(input_dict["prev_actions"].long(), seq_lens.int().tolist())
            prev_actions_padded = nn.utils.rnn.pad_sequence(prev_actions_unpadded, batch_first=True, padding_value=0)
            prev_actions = prev_actions_padded.view(B, T_max)
        else:
            prev_actions = input_dict["prev_actions"].long().view(B, T_max)

        
        obs_frames = obs_flat.view(BT, self.num_frames, self.frame_dim)
        
        frame_emb, gnn_emb = self._encode_frames(obs_frames)

        pa_emb = self.prev_action_embed(prev_actions)
        pa_emb_flat = pa_emb.view(BT, self.prev_action_emb_dim)

        x = torch.cat([frame_emb, gnn_emb, pa_emb_flat], dim=-1)
        
        del frame_emb, gnn_emb, pa_emb_flat, pa_emb, obs_frames, obs_flat
        
        x = self.input_proj(x)
        
        x = x.view(B, T_max, self.d_model)
        
        h_states = []
        for s in state:
            s_batched = s.to(device)
            h_states.append(s_batched)
        
        new_states = []
        
        for i, mamba_block in enumerate(self.mamba_blocks):
            if self.use_checkpoint and self.training:
                x, h_last = checkpoint(mamba_block, x, h_states[i], use_reentrant=False)
            else:
                x, h_last = mamba_block(x, h_states[i])
                
            new_states.append(h_last.detach())
        
        x = self.final_norm(x)
        
        x_flat = x.reshape(BT, self.d_model)
        
        del x
        
        logits = self.policy_head(x_flat)
        self._value_out = self.value_head(x_flat).squeeze(-1)

        return logits, new_states

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._value_out is not None
        return self._value_out