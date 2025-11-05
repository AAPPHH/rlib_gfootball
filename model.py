import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import torch.jit as jit
from gymnasium.spaces import Space
from typing import Tuple, List, Dict, Optional

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.view_requirement import ViewRequirement

from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_add_pool
from torch_geometric.nn.pool import knn_graph

try:
    from torch.cuda.amp import autocast
    AMP_AVAILABLE = True
except ImportError:
    class autocast:
        def __init__(self, *args, **kwargs): pass
        def __enter__(self): pass
        def __exit__(self, *args): pass
    AMP_AVAILABLE = False
    print("Warning: torch.cuda.amp not available. Automatic Mixed Precision (AMP) disabled.")


class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5,
                 use_layernorm: bool = True, activation: str = 'silu'):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.activation = activation

        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.in_norm = nn.LayerNorm(in_dim)

        self.spline_weight = nn.Parameter(torch.randn(out_dim, in_dim, grid_size) * 0.1)
        self.scale_base = nn.Parameter(torch.ones(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.rbf_beta = nn.Parameter(torch.ones(1) * 2.0)

        self.register_buffer('grid', torch.linspace(-1, 1, grid_size))

        nn.init.kaiming_uniform_(self.scale_base, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_layernorm:
            x = self.in_norm(x)

        B = x.shape[0]
        grid = self.grid.to(x.dtype).to(x.device)
        beta = self.rbf_beta.clamp(0.5, 10.0)
        basis = torch.exp(-((x.unsqueeze(-1) - grid.view(1, 1, -1)) ** 2) * beta)
        spline = torch.einsum('big,oig->bo', basis, self.spline_weight)
        base = torch.einsum('bi,oi->bo', x, self.scale_base)
        output = base + spline + self.bias

        if self.activation == 'silu':
            output = F.silu(output)
        elif self.activation == 'gelu':
            output = F.gelu(output)
        elif self.activation == 'relu':
            output = F.relu(output)
        elif self.activation == 'none':
            pass
        return output

class KANBlock(nn.Module):
    def __init__(self, dims: List[int], grid_size: int = 5,
                 use_residual: bool = True, dropout: float = 0.0):
        super().__init__()
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(KANLayer(
                dims[i], dims[i+1],
                grid_size=grid_size,
                activation='silu' if i < len(dims) - 2 else 'none'
            ))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        if use_residual and dims[0] != dims[-1]:
            self.residual_proj = nn.Linear(dims[0], dims[-1])
        else:
            self.residual_proj = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        for layer in self.layers:
            x = layer(x)
            if self.dropout is not None:
                x = self.dropout(x)
        if self.use_residual:
            if self.residual_proj is not None:
                identity = self.residual_proj(identity)
            x = x + identity
        return x

class PyGGNNEncoder(nn.Module):
    def __init__(self, node_feature_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 3, gnn_type: str = 'gat', dropout: float = 0.1):
        super().__init__()
        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        if gnn_type == 'gat':
            self.num_heads = 4
            assert hidden_dim % self.num_heads == 0, "hidden_dim must be divisible by num_heads for GAT"
        
        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim, eps=1e-5),
            nn.SiLU()
        )
        self.gnn_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = hidden_dim
            if gnn_type == 'gcn':
                layer = GCNConv(in_dim, out_dim)
            elif gnn_type == 'gat':
                layer = GATConv(in_dim, out_dim // self.num_heads, heads=self.num_heads, dropout=dropout)
            elif gnn_type == 'sage':
                layer = SAGEConv(in_dim, out_dim)
            else:
                raise ValueError(f"Unknown GNN type: {gnn_type}")
            self.gnn_layers.append(layer)
            self.layer_norms.append(nn.LayerNorm(out_dim, eps=1e-5))
        self.global_pool_type = 'mean'
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim, eps=1e-5),
            nn.SiLU()
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor,
                batch: Optional[torch.LongTensor] = None) -> torch.Tensor:
        x = self.node_encoder(x)
        for i, (gnn_layer, norm) in enumerate(zip(self.gnn_layers, self.layer_norms)):
            identity = x
            x = gnn_layer(x, edge_index)
            x = norm(x)
            x = F.silu(x)
            x = self.dropout(x)
            x = x + identity
        if batch is None:
            if self.global_pool_type == 'mean':
                x = x.mean(dim=0, keepdim=True)
            elif self.global_pool_type == 'add':
                x = x.sum(dim=0, keepdim=True)
            elif self.global_pool_type == 'max':
                x = x.max(dim=0, keepdim=True)[0]
        else:
            if self.global_pool_type == 'mean':
                x = global_mean_pool(x, batch)
            elif self.global_pool_type == 'add':
                x = global_add_pool(x, batch)
        x = self.output_proj(x)
        return x

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
        self.kan_grid = cfg.get("kan_grid", 8)
        self.kan_hidden_dim = cfg.get("kan_hidden_dim", 64)
        self.gnn_hidden = cfg.get("gnn_hidden", 64)
        self.gnn_output = cfg.get("gnn_output", 128)
        self.gnn_layers = cfg.get("gnn_layers", 3)
        self.gnn_type = cfg.get("gnn_type", 'gat')
        self.gnn_k_neighbors = cfg.get("gnn_k_neighbors", 6)
        self.include_ball_node = cfg.get("include_ball_node", True)
        self.include_global_node = cfg.get("include_global_node", True)
        self.include_team_nodes = cfg.get("include_team_nodes", True)
        self.include_possession_node = cfg.get("include_possession_node", True)
        self.include_action_node = cfg.get("include_action_node", True)
        self.use_checkpoint = cfg.get("gradient_checkpointing", False)

        self.num_frames = 4
        total_obs_dim = int(np.prod(obs_space.shape))
        if total_obs_dim % self.num_frames != 0:
            raise ValueError(f"Obs dim {total_obs_dim} not divisible by {self.num_frames}")
        self.frame_dim = total_obs_dim // self.num_frames
        if self.frame_dim != 115:
            print(f"WARNING: frame_dim={self.frame_dim}, expected 115 for 'simple115v2' spec")

        self.num_players = 22
        self.num_left_players = 11
        self.num_nodes = self.num_players
        if self.include_ball_node: self.num_nodes += 1
        if self.include_global_node: self.num_nodes += 1
        if self.include_team_nodes: self.num_nodes += 2
        if self.include_possession_node: self.num_nodes += 1
        if self.include_action_node: self.num_nodes += 1
        self.node_feature_dim = 12

        self.node_feature_norm = nn.LayerNorm(self.node_feature_dim, eps=1e-5)

        self.gnn_encoder = PyGGNNEncoder(
            node_feature_dim=self.node_feature_dim, hidden_dim=self.gnn_hidden,
            output_dim=self.gnn_output, num_layers=self.gnn_layers,
            gnn_type=self.gnn_type, dropout=cfg.get("gnn_dropout", 0.1)
        )
        print(f"Using PyTorch Geometric GNN Encoder (type={self.gnn_type})")
        
        self.prev_action_embed = nn.Embedding(action_space.n + 1, self.prev_action_emb_dim)

        self.input_dim = self.gnn_output + self.prev_action_emb_dim
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_dim, self.d_model), nn.LayerNorm(self.d_model, eps=1e-5), nn.SiLU()
        )
        self.mamba_blocks = nn.ModuleList([
            CompactS6Layer(d_model=self.d_model, d_state=self.mamba_state)
            for _ in range(self.num_mamba_layers)
        ])
        
        self.final_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.policy_head = KANBlock(
            dims=[self.d_model, self.kan_hidden_dim, self.kan_hidden_dim, num_outputs],
            grid_size=self.kan_grid, use_residual=False, dropout=cfg.get("kan_dropout", 0.0)
        )
        
        self.value_norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.value_kan = KANBlock(
            dims=[self.d_model, self.kan_hidden_dim // 2, self.kan_hidden_dim // 4, 1],
            grid_size=self.kan_grid, use_residual=False, dropout=cfg.get("kan_dropout", 0.0)
        )
        self.value_output = nn.Tanh()
        self.value_scale = float(cfg.get("value_scale", 100.0))
        
        self._value_out = None
        self.state_size = self.d_model * self.mamba_state
        total_params = sum(p.numel() for p in self.parameters())
        print(f"[GFootballMamba] Config:")
        print(f"  - d_model: {self.d_model}, mamba_layers: {self.num_mamba_layers}")
        print(f"  - GNN: {self.gnn_type} ({self.gnn_layers}L), hidden={self.gnn_hidden}, out={self.gnn_output}")
        print(f"  - Graph nodes: {self.num_nodes} (Ball: {self.include_ball_node}, Global: {self.include_global_node}, Teams: {self.include_team_nodes})")
        print(f"  - Total params: {total_params/1e6:.2f}M")

    @override(TorchRNN)
    def get_initial_state(self) -> List[TensorType]:
        return [torch.zeros(self.state_size) for _ in range(self.num_mamba_layers)]

    def _parse_obs_frames(self, obs_frames):
        B = obs_frames.shape[0]; device = obs_frames.device
        frame_curr = obs_frames[:, -1, :]; frame_prev = obs_frames[:, -2, :] if self.num_frames > 1 else frame_curr
        frame_prev2 = obs_frames[:, -3, :] if self.num_frames > 2 else frame_prev
        left_pos = frame_curr[:, 0:22].view(B, 11, 2); left_dir = frame_curr[:, 22:44].view(B, 11, 2)
        right_pos = frame_curr[:, 44:66].view(B, 11, 2); right_dir = frame_curr[:, 66:88].view(B, 11, 2)
        ball_pos = frame_curr[:, 88:91]; ball_dir = frame_curr[:, 91:94]
        ball_own = frame_curr[:, 94:97]; active_player = frame_curr[:, 97:108]; game_mode = frame_curr[:, 108:115]
        all_pos = torch.cat([left_pos, right_pos], dim=1); all_dir = torch.cat([left_dir, right_dir], dim=1)
        left_pos_prev = frame_prev[:, 0:22].view(B, 11, 2); right_pos_prev = frame_prev[:, 44:66].view(B, 11, 2)
        all_pos_prev = torch.cat([left_pos_prev, right_pos_prev], dim=1); ball_pos_prev = frame_prev[:, 88:91]
        left_pos_prev2 = frame_prev2[:, 0:22].view(B, 11, 2); right_pos_prev2 = frame_prev2[:, 44:66].view(B, 11, 2)
        all_pos_prev2 = torch.cat([left_pos_prev2, right_pos_prev2], dim=1); ball_pos_prev2 = frame_prev2[:, 88:91]
        all_vel = all_pos - all_pos_prev; ball_vel = ball_pos - ball_pos_prev
        all_vel_prev = all_pos_prev - all_pos_prev2; all_accel = all_vel - all_vel_prev
        ball_vel_prev = ball_pos_prev - ball_pos_prev2; ball_accel = ball_vel - ball_vel_prev
        return (B, device, left_pos, right_pos, all_pos, all_dir, all_vel, all_accel,
                ball_pos, ball_dir, ball_vel, ball_accel, ball_own, active_player, game_mode)

    def _extract_graph_features(self, obs_frames: torch.Tensor) -> torch.Tensor:
        (B, device, left_pos, right_pos, all_pos, all_dir, all_vel, all_accel,
         ball_pos, ball_dir, ball_vel, ball_accel, ball_own, active_player,
         game_mode) = self._parse_obs_frames(obs_frames)

        team_id = torch.cat([torch.zeros(B, self.num_left_players, 1, device=device),
                             torch.ones(B, self.num_players - self.num_left_players, 1, device=device)], dim=1)
        ball_pos_2d = ball_pos[:, :2].unsqueeze(1)
        ball_dist = torch.norm(all_pos - ball_pos_2d, dim=-1, keepdim=True); ball_prox = torch.exp(-ball_dist * 2)
        active_feat = torch.cat([active_player, torch.zeros(B, self.num_players - self.num_left_players, device=device)], dim=1).unsqueeze(-1)
        pad_player = torch.zeros(B, self.num_players, 1, device=device)
        player_features = torch.cat([all_pos, all_dir, all_vel, all_accel, team_id, ball_prox, active_feat, pad_player], dim=-1)

        node_features_list = [player_features]; node_indices = {}; current_idx = self.num_players
        if self.include_ball_node:
            ball_features = torch.cat([ball_pos, ball_dir, ball_vel, ball_accel], dim=-1).unsqueeze(1)
            node_features_list.append(ball_features); node_indices['ball_idx'] = current_idx; current_idx += 1
        if self.include_global_node:
            pad = torch.zeros(B, self.node_feature_dim - game_mode.shape[1], device=device)
            global_features = torch.cat([game_mode, pad], dim=-1).unsqueeze(1)
            node_features_list.append(global_features); node_indices['global_idx'] = current_idx; current_idx += 1
        if self.include_team_nodes:
            left_centroid = left_pos.mean(dim=1); left_spread = left_pos.std(dim=1)
            team_id_left = torch.full((B, 1), 0.0, device=device); pad_team = torch.zeros(B, self.node_feature_dim - 5, device=device)
            left_team_feat = torch.cat([left_centroid, left_spread, team_id_left, pad_team], dim=-1).unsqueeze(1)
            right_centroid = right_pos.mean(dim=1); right_spread = right_pos.std(dim=1)
            team_id_right = torch.full((B, 1), 1.0, device=device)
            right_team_feat = torch.cat([right_centroid, right_spread, team_id_right, pad_team], dim=-1).unsqueeze(1)
            node_features_list.extend([left_team_feat, right_team_feat]); node_indices['left_team_idx'] = current_idx; node_indices['right_team_idx'] = current_idx + 1; current_idx += 2
        if self.include_possession_node:
            pad = torch.zeros(B, self.node_feature_dim - ball_own.shape[1], device=device)
            poss_features = torch.cat([ball_own, pad], dim=-1).unsqueeze(1)
            node_features_list.append(poss_features); node_indices['poss_idx'] = current_idx; current_idx += 1
        if self.include_action_node:
            pad = torch.zeros(B, self.node_feature_dim - active_player.shape[1], device=device)
            action_features = torch.cat([active_player, pad], dim=-1).unsqueeze(1)
            node_features_list.append(action_features); node_indices['action_idx'] = current_idx; current_idx += 1

        node_features = torch.cat(node_features_list, dim=1)
        N = node_features.size(1); assert N == self.num_nodes, f"Node count mismatch: {N} != {self.num_nodes}"
        x_flat_nodes = node_features.view(B * N, self.node_feature_dim)
        batch_pyg = torch.arange(B, device=device).repeat_interleave(N)
        
        player_pos_flat = all_pos.view(B * self.num_players, 2)
        batch_players = torch.arange(B, device=device).repeat_interleave(self.num_players)
        edge_index_players = knn_graph(player_pos_flat, k=self.gnn_k_neighbors, batch=batch_players, loop=True)
        
        all_edges_list = [edge_index_players]
        
        batch_offsets = torch.arange(B, device=device, dtype=torch.long) * N
        
        def add_batched_edges(src_node_idx_in_graph, target_node_indices_in_graph, self_loop=False):
            if isinstance(src_node_idx_in_graph, int):
                src_node_idx_in_graph = torch.full((B,), src_node_idx_in_graph, device=device, dtype=torch.long)

            if isinstance(target_node_indices_in_graph, int):
                target_idx = torch.full((B, 1), target_node_indices_in_graph, device=device, dtype=torch.long)
            elif target_node_indices_in_graph.dim() == 1:
                target_idx = target_node_indices_in_graph.unsqueeze(0).expand(B, -1)
            else:
                target_idx = target_node_indices_in_graph

            src_abs = batch_offsets + src_node_idx_in_graph
            tgt_abs = batch_offsets.view(-1, 1) + target_idx

            num_targets_per_graph = tgt_abs.shape[1]

            src_expanded = src_abs.unsqueeze(1).expand(-1, num_targets_per_graph).reshape(-1)
            tgt_flat = tgt_abs.reshape(-1)
            
            assert src_expanded.numel() == tgt_flat.numel()

            edges_fwd = torch.stack([src_expanded, tgt_flat], dim=0)
            edges_bwd = torch.stack([tgt_flat, src_expanded], dim=0)
            new_edges = [edges_fwd, edges_bwd]
            
            if self_loop:
                self_loops = torch.stack([src_abs, src_abs], dim=0)
                new_edges.append(self_loops)

            return new_edges
            
        player_indices_rel = torch.arange(self.num_players, device=device, dtype=torch.long)
        left_player_indices_rel = torch.arange(self.num_left_players, device=device, dtype=torch.long)
        right_player_indices_rel = torch.arange(self.num_left_players, self.num_players, device=device, dtype=torch.long)
        all_node_indices_rel = torch.arange(N, device=device, dtype=torch.long)

        if self.include_ball_node:
            idx = node_indices['ball_idx']
            all_edges_list.extend(add_batched_edges(idx, player_indices_rel, self_loop=True))
        if self.include_team_nodes:
            idx_l = node_indices['left_team_idx']; idx_r = node_indices['right_team_idx']
            all_edges_list.extend(add_batched_edges(idx_l, left_player_indices_rel, self_loop=True))
            all_edges_list.extend(add_batched_edges(idx_r, right_player_indices_rel, self_loop=True))
            if self.include_ball_node:
                all_edges_list.extend(add_batched_edges(idx_l, node_indices['ball_idx']))
                all_edges_list.extend(add_batched_edges(idx_r, node_indices['ball_idx']))
        if self.include_possession_node:
            idx = node_indices['poss_idx']
            if self.include_ball_node:
                all_edges_list.extend(add_batched_edges(idx, node_indices['ball_idx'], self_loop=True))
            else:
                all_edges_list.extend(add_batched_edges(idx, idx, self_loop=True))
        if self.include_action_node:
            idx = node_indices['action_idx']
            all_edges_list.extend(add_batched_edges(idx, left_player_indices_rel, self_loop=True))
            if self.include_ball_node:
                all_edges_list.extend(add_batched_edges(idx, node_indices['ball_idx']))
        if self.include_global_node:
            idx = node_indices['global_idx']
            all_other_nodes_rel = all_node_indices_rel[all_node_indices_rel != idx]
            all_edges_list.extend(add_batched_edges(idx, all_other_nodes_rel, self_loop=True))
            if self.include_team_nodes:
                all_edges_list.extend(add_batched_edges(idx, node_indices['left_team_idx']))
                all_edges_list.extend(add_batched_edges(idx, node_indices['right_team_idx']))

        edge_index = torch.cat(all_edges_list, dim=1)
        x_norm = self.node_feature_norm(x_flat_nodes)
        
        # --- START: MODIFIZIERTER GNN-AUFRUF ---
        if self.use_checkpoint and self.training:
            # `use_reentrant=False` ist oft effizienter und für die meisten GNNs sicher
            graph_embeddings = checkpoint(self.gnn_encoder, x_norm, edge_index, batch_pyg, use_reentrant=False)
        else:
            graph_embeddings = self.gnn_encoder(x_norm, edge_index, batch_pyg)
        # --- ENDE: MODIFIZIERTER GNN-AUFRUF ---
            
        return graph_embeddings

    @override(TorchModelV2)
    def forward(self, input_dict: Dict[str, TensorType], state: List[TensorType],
                seq_lens: TensorType) -> Tuple[TensorType, List[TensorType]]:

        obs_flat = input_dict.get("obs_flat", input_dict.get("obs"))
        if obs_flat is None: raise ValueError("Missing obs")
        BT, obs_dim = obs_flat.shape; device = obs_flat.device

        is_training = state is not None

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

        obs_frames = obs_flat.view(BT, self.num_frames, self.frame_dim)
        
        # --- START: MODIFIZIERTER AUTOCAST-BLOCK ---
        # Der autocast-Block umschließt jetzt ALLES, inklusive der KAN-Köpfe
        with autocast(enabled=AMP_AVAILABLE and device.type == 'cuda', dtype=torch.bfloat16):
            gnn_features = self._extract_graph_features(obs_frames)
            pa_emb = self.prev_action_embed(prev_actions.view(BT))
            
            x = torch.cat([gnn_features, pa_emb], dim=-1)
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

            x_mamba = x
            
            x_policy = self.final_norm(x_mamba)
            x_flat_policy = x_policy.reshape(BT, self.d_model)
            if seq_mask is not None:
                mask_flat = seq_mask.reshape(BT, 1).to(x_flat_policy.dtype)
                x_flat_policy = x_flat_policy * mask_flat
            
            x_value = self.value_norm(x_mamba)
            x_flat_value = x_value.reshape(BT, self.d_model)
            if seq_mask is not None:
                mask_flat_val = seq_mask.reshape(BT, 1).to(x_flat_value.dtype)
                x_flat_value = x_flat_value * mask_flat_val
        
            x_flat_policy_c = torch.clamp(x_flat_policy, -10.0, 10.0)
            x_flat_value_c = torch.clamp(x_flat_value, -10.0, 10.0)

            if self.use_checkpoint and self.training:
                logits_raw = checkpoint(self.policy_head, x_flat_policy_c, use_reentrant=False)
                v_raw = checkpoint(self.value_kan, x_flat_value_c, use_reentrant=False)
            else:
                logits_raw = self.policy_head(x_flat_policy_c)
                v_raw = self.value_kan(x_flat_value_c)

            logits = logits_raw.float() 
            value = self.value_output(v_raw.float()) * self.value_scale
            
        self._value_out = value.squeeze(-1)

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
        assert self._value_out is not None
        return self._value_out