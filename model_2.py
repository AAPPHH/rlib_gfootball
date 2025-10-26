import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast 
import torch.jit as jit
from gymnasium.spaces import Space
from typing import Tuple, Optional, List, Dict

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.view_requirement import ViewRequirement

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

class FiLMLayer(nn.Module):
    def __init__(self, base_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, base_dim, bias=False)
        self.beta = nn.Linear(cond_dim, base_dim, bias=False)
        
        # Init: gamma ≈ 1, beta ≈ 0 (neutraler Start)
        nn.init.constant_(self.gamma.weight, 0.0)
        nn.init.constant_(self.beta.weight, 0.0)
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = 1.0 + self.gamma(cond)  # [B, base_dim]
        beta = self.beta(cond)           # [B, base_dim]
        return gamma * x + beta


class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1):
        super().__init__()
        padding = (kernel - 1) * dilation // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=kernel, padding=padding,
                                     dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


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
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        expand = 2
        inner_dim = dim * expand
        self.norm = EfficientGroupNorm(dim, eps=1e-5)
        self.conv = DepthwiseSeparableConv1d(dim, inner_dim * 2, kernel, dilation)
        self.proj = nn.Conv1d(inner_dim, dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_scale = nn.Parameter(torch.ones(1, 1, dim) * 0.1)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x.transpose(1, 2))
        conv_out = self.conv(x_norm)
        gate, value = conv_out.chunk(2, dim=1)
        out = value * torch.sigmoid(gate)
        out = self.proj(out).transpose(1, 2)
        out = self.dropout(out)
        return out * self.layer_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            out = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            out = self._forward_impl(x)
        return x + out


class GFootballGNN(TorchRNN, nn.Module):
    
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
            
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})
        self.gru_hidden = cfg.get("gru_hidden", 320)
        
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.view_requirements["prev_actions"] = ViewRequirement(
            data_col="actions", shift=-1, space=self.action_space
        )

        self.d_model = cfg.get("d_model", 96)
        self.prev_action_emb_dim = cfg.get("prev_action_emb", 16)
        
        self.tcn_kernel = cfg.get("tcn_kernel", 3)
        self.tcn_dilations = cfg.get("tcn_dilations", [1, 2])
        self.use_checkpoint = cfg.get("gradient_checkpointing", True)
        self.dropout = cfg.get("dropout", 0.05)
        
        self.use_gnn = cfg.get("use_gnn", True)
        self.gnn_hidden = cfg.get("gnn_hidden", 32) 
        self.gnn_last_only = cfg.get("gnn_last_only", False)
        self.gnn_k = cfg.get("gnn_k", 6)
        
        self.num_frames = 4
        total_obs_dim = int(np.prod(obs_space.shape))
        
        if total_obs_dim % self.num_frames != 0:
            raise ValueError(f"Obs dim {total_obs_dim} not divisible by {self.num_frames}")
        
        self.frame_dim = total_obs_dim // self.num_frames
        
        if self.frame_dim != 115:
            print(f"WARNING: frame_dim={self.frame_dim}, expected 115")
        
        self.num_players = 22
        self.register_buffer("adj_eye", torch.eye(self.num_players, dtype=torch.float32))
            
        if self.use_gnn:
            # Node-Features: [x, y, vx, vy, team_id, ball_prox, Δx_team, Δy_team] = 8 Dimensionen
            self.node_feature_dim = 8
            self.gnn_encoder = GNNEncoder(self.node_feature_dim, self.gnn_hidden)
            
            # FiLM statt Concat: Base-Encoder bleibt klein
            self.frame_encoder = nn.Linear(self.frame_dim, self.d_model)
            self.film_layer = FiLMLayer(self.d_model, self.gnn_hidden)
            
            if self.gnn_last_only:
                self.null_gnn = nn.Parameter(torch.zeros(1, self.gnn_hidden))
        else:
            self.frame_encoder = nn.Linear(self.frame_dim, self.d_model)
        
        self.frame_norm = nn.LayerNorm(self.d_model)
        
        self.tcn_blocks = nn.ModuleList([
            EfficientTCNBlock(self.d_model, self.tcn_kernel, dil, self.dropout, self.use_checkpoint)
            for dil in self.tcn_dilations
        ])

        self.prev_action_embed = nn.Embedding(action_space.n, self.prev_action_emb_dim)
        
        self.gru_input_size = self.d_model + self.prev_action_emb_dim
        self.gru = nn.GRU(self.gru_input_size, self.gru_hidden, batch_first=True)

        self.policy_head = nn.Sequential(
            nn.Linear(self.gru_hidden, max(64, self.gru_hidden // 2)),
            nn.ReLU(),
            nn.Linear(max(64, self.gru_hidden // 2), num_outputs)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.gru_hidden, max(32, self.gru_hidden // 4)),
            nn.ReLU(),
            nn.Linear(max(32, self.gru_hidden // 4), 1)
        )
        
        with torch.no_grad():
            for name, param in self.gru.named_parameters():
                if "bias_ih" in name or "bias_hh" in name:
                    nn.init.constant_(param, 0.0)
        
        self.fuse_dim = self.gru_hidden
        self.adapt_gnn = nn.Linear(self.d_model, self.fuse_dim, bias=False)
        self.adapt_tcn = nn.Linear(self.d_model, self.fuse_dim, bias=False)
        self.adapt_gru = nn.Identity()
        self.fuse_logits = nn.Parameter(torch.tensor([0.15, 0.35, 0.50]))
        self.fuse_ln = nn.LayerNorm(self.fuse_dim)
        
        self.fuse_temp = nn.Parameter(torch.tensor(1.0))
        self.head_gate = nn.Sequential(nn.Linear(self.fuse_dim, self.fuse_dim), 
                                       nn.SiLU())
        self.pi_ln = nn.LayerNorm(self.fuse_dim)
        self.v_ln  = nn.LayerNorm(self.fuse_dim)

        self._value_out = None
    
    @override(TorchRNN)
    def get_initial_state(self) -> List[torch.Tensor]:
        return [torch.zeros(self.gru_hidden, dtype=torch.float32)]
    
    @staticmethod
    @jit.script
    def _parse_frame_to_graph_impl(
        frame_curr: torch.Tensor,
        frame_prev: torch.Tensor,
        adj_eye: torch.Tensor, 
        gnn_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B = frame_curr.size(0)
        device = frame_curr.device
        
        ball_pos = frame_curr[:, 0:2] 
        
        # Aktuelle Positionen
        left_pos = frame_curr[:, 3:25].reshape(B, 11, 2)
        right_pos = frame_curr[:, 25:47].reshape(B, 11, 2)
        all_pos = torch.cat([left_pos, right_pos], dim=1)  # [B, 22, 2]
        
        # Vorherige Positionen (für Geschwindigkeit)
        left_pos_prev = frame_prev[:, 3:25].reshape(B, 11, 2)
        right_pos_prev = frame_prev[:, 25:47].reshape(B, 11, 2)
        all_pos_prev = torch.cat([left_pos_prev, right_pos_prev], dim=1)  # [B, 22, 2]
        
        # Geschwindigkeit: v = pos_t - pos_t-1
        all_vel = all_pos - all_pos_prev  # [B, 22, 2]
        
        # Team-ID (0 = links, 1 = rechts)
        team_id = torch.cat([
            torch.zeros(B, 11, 1, device=device),
            torch.ones(B, 11, 1, device=device)
        ], dim=1) 
        
        # Distanz zum Ball
        ball_pos_exp = ball_pos.unsqueeze(1) 
        ball_dist = torch.norm(all_pos - ball_pos_exp, dim=-1, keepdim=True) 
        ball_prox = torch.exp(-ball_dist * 2)
        
        # Team-Struktur: Δp_team = pi - mean(p_team)
        left_center = left_pos.mean(dim=1, keepdim=True)  # [B, 1, 2]
        right_center = right_pos.mean(dim=1, keepdim=True)
        team_center = torch.cat([
            left_center.expand(-1, 11, -1), 
            right_center.expand(-1, 11, -1)
        ], dim=1)  # [B, 22, 2]
        delta_team = all_pos - team_center  # [B, 22, 2]
        
        # Node-Features: [x, y, vx, vy, team_id, ball_prox, Δx_team, Δy_team] = 8 Features
        node_features = torch.cat([all_pos, all_vel, team_id, ball_prox, delta_team], dim=-1)
        
        # Optimierte K-NN Adjacency (binär, no_grad für Distanzen)
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
    
    def _encode_frames(self, obs_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        B = obs_frames.size(0)
        device = obs_frames.device
        
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            if self.use_gnn:
                if self.gnn_last_only:
                    # Preallocate - spart temporäre List-Allokationen
                    gnn_stacked = torch.empty(B, self.num_frames, self.gnn_hidden, 
                                              device=device, dtype=torch.bfloat16)
                    
                    frame_curr = obs_frames[:, -1, :]
                    frame_prev = obs_frames[:, -2, :] if self.num_frames > 1 else frame_curr
                    nodes, adj = self._parse_frame_to_graph(frame_curr, frame_prev)
                    gnn_emb_last = self.gnn_encoder(nodes, adj)
                    
                    # In-place assignment statt List+Stack
                    null_emb = self.null_gnn.expand(B, -1).to(torch.bfloat16)
                    for t in range(self.num_frames - 1):
                        gnn_stacked[:, t] = null_emb
                    gnn_stacked[:, -1] = gnn_emb_last
                
                else:
                    # Alle Frames: List+Stack OK bei nur 4 Frames
                    gnn_embs = []
                    for f in range(self.num_frames):
                        frame_curr = obs_frames[:, f, :]
                        frame_prev = obs_frames[:, f - 1, :] if f > 0 else frame_curr
                        nodes, adj = self._parse_frame_to_graph(frame_curr, frame_prev)
                        gnn_emb = self.gnn_encoder(nodes, adj) 
                        gnn_embs.append(gnn_emb)
                    
                    gnn_stacked = torch.stack(gnn_embs, dim=1)
                
                # FiLM-Conditioning statt Concat
                x = self.frame_encoder(obs_frames)  # [B, T, d_model]
                
                # Pro Frame: FiLM-Modulation
                x_modulated = []
                for t in range(self.num_frames):
                    x_t = x[:, t, :]                # [B, d_model]
                    gnn_t = gnn_stacked[:, t, :]    # [B, gnn_hidden]
                    x_t_mod = self.film_layer(x_t, gnn_t)
                    x_modulated.append(x_t_mod)
                
                x = torch.stack(x_modulated, dim=1)  # [B, T, d_model]
            else:
                x = self.frame_encoder(obs_frames) 
            
            x = F.relu(self.frame_norm(x))
            
            for block in self.tcn_blocks:
                x = block(x)
            
            z_tcn_last = x[:, -1, :]
            
            z_t = 0.6 * x.mean(dim=1) + 0.4 * x.max(dim=1)[0]
            
            if self.use_gnn:
                # Für z_gnn_last: letzten Frame mit GNN nochmal encodieren
                x_last = self.frame_encoder(obs_frames[:, -1, :])
                z_gnn_last = self.film_layer(x_last, gnn_stacked[:, -1, :])
            else:
                z_gnn_last = z_tcn_last
            
        return z_t, z_tcn_last, z_gnn_last

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

        seq_lens_cpu_int64 = seq_lens.cpu()

        obs_frames = obs_flat.view(BT, self.num_frames, self.frame_dim)

        z_t, z_tcn_last, z_gnn_last = self._encode_frames(obs_frames)

        pa_emb = self.prev_action_embed(prev_actions) 
        pa_emb_flat = pa_emb.view(BT, self.prev_action_emb_dim)

        gru_in_flat = torch.cat([z_t, pa_emb_flat], dim=-1)
        gru_in = gru_in_flat.view(B, T_max, self.gru_input_size)

        h_in = state[0].to(device).unsqueeze(0)
        packed_gru_in = nn.utils.rnn.pack_padded_sequence(
            gru_in, seq_lens_cpu_int64, batch_first=True, enforce_sorted=False
        )
        
        packed_gru_out, h_out = self.gru(packed_gru_in, h_in)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_gru_out, batch_first=True, total_length=T_max
        )

        gru_out_flat = gru_out.reshape(BT, self.gru_hidden)

        logits_bf16 = None
        value_bf16 = None
        
        with autocast(device_type=device.type, dtype=torch.bfloat16):
            h_gnn = self.adapt_gnn(z_gnn_last)
            h_tcn = self.adapt_tcn(z_tcn_last)
            h_gru = self.adapt_gru(gru_out_flat)
            
            temp = self.fuse_temp.clamp(min=0.25, max=4.0)
            w = torch.softmax(self.fuse_logits / temp, dim=0)
            
            h_fused = self.fuse_ln(w[0]*h_gnn + w[1]*h_tcn + w[2]*h_gru)
            
            h_fused = self.head_gate(h_fused)
            
            logits_bf16 = self.policy_head(self.pi_ln(h_fused))
            value_bf16 = self.value_head(self.v_ln(h_fused.detach())).squeeze(-1)
        
        logits = logits_bf16.float()
        self._value_out = value_bf16.float()

        new_state = [h_out.squeeze(0).detach()]
        return logits, new_state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._value_out is not None
        return self._value_out