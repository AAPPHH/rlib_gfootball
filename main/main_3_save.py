import math
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import ray
import gfootball.env as football_env
FEATURE_NAMES = [
    'ball_x', 'ball_y', 'ball_z', 'ball_speed', 'ball_dir_x', 'ball_dir_y', 'ball_owned_team',
    'rel_ball_x', 'rel_ball_y', 'dist_to_ball', 'angle_to_ball',
    'dist_to_goal', 'goal_angle', 'shooting_angle', 'in_shooting_range', 'dist_to_own_goal',
    'keeper_dist', 'keeper_angle',
    'tm1_rel_x', 'tm1_rel_y', 'tm1_dir_x', 'tm1_dir_y',
    'tm2_rel_x', 'tm2_rel_y', 'tm2_dir_x', 'tm2_dir_y',
    'tm3_rel_x', 'tm3_rel_y', 'tm3_dir_x', 'tm3_dir_y',
    'tm4_rel_x', 'tm4_rel_y', 'tm4_dir_x', 'tm4_dir_y',
    'tm5_rel_x', 'tm5_rel_y', 'tm5_dir_x', 'tm5_dir_y',
    'op1_rel_x', 'op1_rel_y', 'op1_dir_x', 'op1_dir_y',
    'op2_rel_x', 'op2_rel_y', 'op2_dir_x', 'op2_dir_y',
    'op3_rel_x', 'op3_rel_y', 'op3_dir_x', 'op3_dir_y',
    'op4_rel_x', 'op4_rel_y', 'op4_dir_x', 'op4_dir_y',
    'op5_rel_x', 'op5_rel_y', 'op5_dir_x', 'op5_dir_y',
    'teammates_ahead', 'defenders_ahead', 'numerical_advantage', 'free_teammates', 'team_spread',
    'nearest_opponent_dist', 'space_ahead', 'pressure_index', 'passing_lanes_open',
    'offside_line_x', 'is_offside',
    'in_attack_third', 'in_middle_third', 'in_defense_third', 'on_left_wing', 'on_right_wing',
    'sticky_sprint', 'sticky_dribble', 'sticky_dir_x', 'sticky_dir_y',
    'game_mode_normal', 'game_mode_kickoff', 'game_mode_goal_kick', 'game_mode_free_kick',
    'game_mode_corner', 'game_mode_throw_in', 'game_mode_penalty', 'steps_remaining',
    'score_diff', 'winning', 'losing', 'drawing',
    'shooting_opportunity', 'attack_momentum', 'counter_attack_potential',
]
FEATURE_GROUPS = {
    'ball_state': list(range(0, 7)),
    'relative_ball': list(range(7, 11)),
    'goal_geometry': list(range(11, 16)),
    'keeper': list(range(16, 18)),
    'closest_teammates': list(range(18, 38)),
    'closest_opponents': list(range(38, 58)),
    'team_structure': list(range(58, 63)),
    'pressure_space': list(range(63, 67)),
    'offside': list(range(67, 69)),
    'zones': list(range(69, 74)),
    'sticky': list(range(74, 78)),
    'game_state': list(range(78, 86)),
    'score_context': list(range(86, 90)),
    'composite': list(range(90, 93)),
}
FEATURE_DIM = 93
OBS_DIM = 460
class FeatureEngineer:
    FEATURE_DIM = 93
    GOAL_POS = np.array([1.0, 0.0], dtype=np.float32)
    OWN_GOAL_POS = np.array([-1.0, 0.0], dtype=np.float32)
    GOAL_TOP = np.array([1.0, 0.044], dtype=np.float32)
    GOAL_BOTTOM = np.array([1.0, -0.044], dtype=np.float32)
    @staticmethod
    def extract_features(obs: np.ndarray, active_player_override: int = None) -> np.ndarray:
        squeeze_output = False
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
            squeeze_output = True
        B = obs.shape[0]
        obs115 = obs[:, :115] if obs.shape[1] >= 115 else np.pad(obs, ((0, 0), (0, 115 - obs.shape[1])), mode='constant')
        feat = np.zeros((B, FEATURE_DIM), dtype=np.float32)
        left_pos = obs115[:, 0:22].reshape(B, 11, 2)
        left_dir = obs115[:, 22:44].reshape(B, 11, 2)
        right_pos = obs115[:, 44:66].reshape(B, 11, 2)
        right_dir = obs115[:, 66:88].reshape(B, 11, 2)
        ball_pos = obs115[:, 88:90]
        ball_z = obs115[:, 90:91]
        ball_dir = obs115[:, 91:94]
        ball_owned = obs115[:, 94:97]
        ball_owned_team = np.argmax(ball_owned, axis=1) - 1
        if active_player_override is not None:
            active_idx = np.full(B, active_player_override, dtype=np.int32)
        else:
            active_idx = obs115[:, 97].astype(np.int32)
        active_idx = np.clip(active_idx, 0, 10)
        game_mode = obs115[:, 98:105]
        sticky = obs115[:, 105:115]
        batch_idx = np.arange(B)
        active_pos = left_pos[batch_idx, active_idx]
        ball_speed = np.linalg.norm(ball_dir[:, :2], axis=1)
        feat[:, 0] = ball_pos[:, 0]
        feat[:, 1] = ball_pos[:, 1]
        feat[:, 2] = np.clip(ball_z[:, 0], 0, 1)
        feat[:, 3] = np.clip(ball_speed, 0, 2)
        feat[:, 4] = ball_dir[:, 0]
        feat[:, 5] = ball_dir[:, 1]
        feat[:, 6] = ball_owned_team / 2.0
        rel_ball = ball_pos - active_pos
        dist_to_ball = np.linalg.norm(rel_ball, axis=1)
        angle_to_ball = np.arctan2(rel_ball[:, 1], rel_ball[:, 0])
        feat[:, 7] = rel_ball[:, 0]
        feat[:, 8] = rel_ball[:, 1]
        feat[:, 9] = np.clip(dist_to_ball, 0, 2)
        feat[:, 10] = angle_to_ball / np.pi
        goal_vec = FeatureEngineer.GOAL_POS - ball_pos
        dist_to_goal = np.linalg.norm(goal_vec, axis=1)
        goal_angle = np.abs(np.arctan2(goal_vec[:, 1], goal_vec[:, 0]))
        vec_top = FeatureEngineer.GOAL_TOP - ball_pos
        vec_bottom = FeatureEngineer.GOAL_BOTTOM - ball_pos
        shooting_angle = np.abs(np.arctan2(vec_top[:, 1], vec_top[:, 0]) - np.arctan2(vec_bottom[:, 1], vec_bottom[:, 0]))
        dist_to_own_goal = np.linalg.norm(FeatureEngineer.OWN_GOAL_POS - ball_pos, axis=1)
        feat[:, 11] = np.clip(dist_to_goal, 0, 2)
        feat[:, 12] = goal_angle / np.pi
        feat[:, 13] = shooting_angle / np.pi
        feat[:, 14] = (dist_to_goal < 0.35).astype(np.float32)
        feat[:, 15] = np.clip(dist_to_own_goal, 0, 2)
        right_x = right_pos[:, :, 0]
        keeper_idx = np.argmax(right_x, axis=1)
        keeper_pos = right_pos[batch_idx, keeper_idx]
        keeper_dist = np.linalg.norm(ball_pos - keeper_pos, axis=1)
        keeper_vec = ball_pos - keeper_pos
        keeper_angle = np.arctan2(keeper_vec[:, 1], keeper_vec[:, 0])
        keeper_valid = keeper_pos[:, 0] > 0.7
        feat[:, 16] = np.where(keeper_valid, np.clip(keeper_dist, 0, 2), 1.0)
        feat[:, 17] = np.where(keeper_valid, keeper_angle / np.pi, 0.0)
        left_active = np.abs(left_pos[:, :, 0]) > 0.01
        right_active = np.abs(right_pos[:, :, 0]) > 0.01
        tm_rel = left_pos - active_pos[:, None, :]
        tm_dist = np.linalg.norm(tm_rel, axis=2)
        tm_dist[batch_idx, active_idx] = 999.0
        tm_dist = np.where(left_active, tm_dist, 999.0)
        tm_sorted_idx = np.argsort(tm_dist, axis=1)
        for i in range(5):
            idx = tm_sorted_idx[:, i]
            rel = left_pos[batch_idx, idx] - active_pos
            dirs = left_dir[batch_idx, idx]
            valid = tm_dist[batch_idx, idx] < 100
            feat[:, 18+i*4] = np.where(valid, rel[:, 0], 0)
            feat[:, 19+i*4] = np.where(valid, rel[:, 1], 0)
            feat[:, 20+i*4] = np.where(valid, dirs[:, 0], 0)
            feat[:, 21+i*4] = np.where(valid, dirs[:, 1], 0)
        op_rel = right_pos - active_pos[:, None, :]
        op_dist = np.linalg.norm(op_rel, axis=2)
        op_dist = np.where(right_active, op_dist, 999.0)
        op_sorted_idx = np.argsort(op_dist, axis=1)
        for i in range(5):
            idx = op_sorted_idx[:, i]
            rel = right_pos[batch_idx, idx] - active_pos
            dirs = right_dir[batch_idx, idx]
            valid = op_dist[batch_idx, idx] < 100
            feat[:, 38+i*4] = np.where(valid, rel[:, 0], 0)
            feat[:, 39+i*4] = np.where(valid, rel[:, 1], 0)
            feat[:, 40+i*4] = np.where(valid, dirs[:, 0], 0)
            feat[:, 41+i*4] = np.where(valid, dirs[:, 1], 0)
        ball_x = ball_pos[:, 0]
        left_x = left_pos[:, :, 0]
        teammates_ahead = np.sum((left_x > ball_x[:, None]) & left_active, axis=1) / 11.0
        defenders_ahead = np.sum((right_x > ball_x[:, None]) & right_active, axis=1) / 11.0
        numerical_adv = teammates_ahead * 11 - defenders_ahead * 11
        pairwise_dist = np.linalg.norm(left_pos[:, :, None, :] - right_pos[:, None, :, :], axis=3)
        min_opp_dist = np.min(np.where(right_active[:, None, :], pairwise_dist, 10.0), axis=2)
        free_mask = (min_opp_dist > 0.15) & left_active
        free_teammates = np.sum(free_mask, axis=1) / 11.0
        left_y_valid = np.where(left_active, left_pos[:, :, 1], np.nan)
        team_spread = np.nanstd(left_y_valid, axis=1)
        team_spread = np.nan_to_num(team_spread, nan=0.0)
        feat[:, 58] = teammates_ahead
        feat[:, 59] = defenders_ahead
        feat[:, 60] = np.clip(numerical_adv / 5.0, -1, 1)
        feat[:, 61] = free_teammates
        feat[:, 62] = np.clip(team_spread, 0, 0.5)
        opp_dist_to_active = np.linalg.norm(right_pos - active_pos[:, None, :], axis=2)
        opp_dist_to_active = np.where(right_active, opp_dist_to_active, 10.0)
        feat[:, 63] = np.clip(np.min(opp_dist_to_active, axis=1), 0, 1)
        opp_ahead_mask = (right_x > ball_x[:, None]) & right_active
        opp_dist_ahead = np.where(opp_ahead_mask, opp_dist_to_active, 10.0)
        space_ahead = np.min(opp_dist_ahead, axis=1)
        feat[:, 64] = np.clip(np.where(np.any(opp_ahead_mask, axis=1), space_ahead, 1.0), 0, 1)
        feat[:, 65] = np.clip(np.sum(opp_dist_to_active < 0.2, axis=1) / 3.0, 0, 1)
        tm_in_range = (tm_dist < 0.3) & (tm_dist > 0.05)
        feat[:, 66] = np.clip(np.sum(tm_in_range & free_mask, axis=1) / 5.0, 0, 1)
        sorted_right_x = np.sort(right_x, axis=1)
        offside_line = np.maximum(ball_x, sorted_right_x[:, 1])
        active_x = left_pos[batch_idx, active_idx, 0]
        feat[:, 67] = offside_line
        feat[:, 68] = ((active_x > offside_line) & (ball_owned_team == 0)).astype(np.float32)
        feat[:, 69] = (ball_x > 0.33).astype(np.float32)
        feat[:, 70] = ((ball_x >= -0.33) & (ball_x <= 0.33)).astype(np.float32)
        feat[:, 71] = (ball_x < -0.33).astype(np.float32)
        ball_y = ball_pos[:, 1]
        feat[:, 72] = (ball_y > 0.2).astype(np.float32)
        feat[:, 73] = (ball_y < -0.2).astype(np.float32)
        sticky_dirs = sticky[:, :8]
        sticky_dir_idx = np.argmax(sticky_dirs, axis=1)
        sticky_dir_active = np.any(sticky_dirs > 0, axis=1)
        sticky_angle = sticky_dir_idx * (2 * np.pi / 8)
        feat[:, 74] = sticky[:, 8]
        feat[:, 75] = sticky[:, 9]
        feat[:, 76] = np.where(sticky_dir_active, np.cos(sticky_angle), 0)
        feat[:, 77] = np.where(sticky_dir_active, np.sin(sticky_angle), 0)
        feat[:, 78:85] = game_mode
        feat[:, 85] = 0.5
        feat[:, 86] = 0.0
        feat[:, 87] = 0.0
        feat[:, 88] = 0.0
        feat[:, 89] = 1.0
        feat[:, 90] = shooting_angle * np.where(keeper_valid, np.clip(keeper_dist, 0, 1), 1.0)
        feat[:, 91] = np.clip(ball_dir[:, 0] * ball_speed, -1, 1)
        feat[:, 92] = np.clip(feat[:, 64], 0, 1) * 0.4 + np.clip(feat[:, 91], 0, 1) * 0.3 + np.clip(numerical_adv / 5.0 + 0.5, 0, 1) * 0.3
        if squeeze_output:
            return feat[0]
        return feat
@dataclass
class ModelConfig:
    obs_dim: int = OBS_DIM
    feature_dim: int = FEATURE_DIM
    d_model: int = 256
    mamba_d_state: int = 64
    mamba_layers: int = 4
    num_actions: int = 19
    action_emb_dim: int = 16
    encoder_hidden: List[int] = None
    policy_hidden: List[int] = None
    value_hidden: List[int] = None
    use_distributional: bool = True
    v_min: float = -10.0
    v_max: float = 10.0
    num_atoms: int = 51
    num_stages: int = 10
    stage_emb_dim: int = 8
    dropout: float = 0.0
    segment_size: int = 16
    def __post_init__(self):
        if self.encoder_hidden is None:
            self.encoder_hidden = [256, 256]
        if self.policy_hidden is None:
            self.policy_hidden = [256]
        if self.value_hidden is None:
            self.value_hidden = [256]
class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, segment_size=16):
        super().__init__()
        self.d_model, self.d_state = d_model, d_state
        self.segment_size = segment_size
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        self.A_log_diag = nn.Parameter(torch.log(torch.linspace(1.0, d_state, d_state)))
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.B_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.C_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        nn.init.zeros_(self.dt_proj.bias)
        nn.init.normal_(self.dt_proj.weight, std=0.01)
    def _process_segment(self, x_in_seg, z_seg, dt_seg, B_t_seg, C_t_seg, h):
        outputs = []
        A_diag = -torch.exp(self.A_log_diag)
        seg_len = x_in_seg.shape[1]
        for t in range(seg_len):
            dA = torch.exp(dt_seg[:, t, :].unsqueeze(-1) * A_diag.unsqueeze(0).unsqueeze(0))
            dB = dt_seg[:, t, :].unsqueeze(-1) * B_t_seg[:, t, :].unsqueeze(1)
            h = h * dA + x_in_seg[:, t, :].unsqueeze(-1) * dB
            outputs.append((h * C_t_seg[:, t, :].unsqueeze(1)).sum(dim=-1))
        return torch.stack(outputs, dim=1), h
    def forward(self, x, h):
        B, L, D = x.shape
        h = h.view(B, self.d_model, self.d_state)
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_in, z = xz.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(x_in))
        B_t, C_t = self.B_proj(x_in), self.C_proj(x_in)
        if self.training and L > self.segment_size:
            all_outputs = []
            for seg_start in range(0, L, self.segment_size):
                seg_end = min(seg_start + self.segment_size, L)
                x_in_seg = x_in[:, seg_start:seg_end].contiguous()
                z_seg = z[:, seg_start:seg_end].contiguous()
                dt_seg = dt[:, seg_start:seg_end].contiguous()
                B_t_seg = B_t[:, seg_start:seg_end].contiguous()
                C_t_seg = C_t[:, seg_start:seg_end].contiguous()
                seg_out, h = checkpoint.checkpoint(self._process_segment, x_in_seg, z_seg, dt_seg, B_t_seg, C_t_seg, h, use_reentrant=False)
                all_outputs.append(seg_out)
            y = torch.cat(all_outputs, dim=1)
        else:
            outputs = []
            A_diag = -torch.exp(self.A_log_diag)
            for t in range(L):
                dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A_diag.unsqueeze(0).unsqueeze(0))
                dB = dt[:, t, :].unsqueeze(-1) * B_t[:, t, :].unsqueeze(1)
                h = h * dA + x_in[:, t, :].unsqueeze(-1) * dB
                outputs.append((h * C_t[:, t, :].unsqueeze(1)).sum(dim=-1))
            y = torch.stack(outputs, dim=1)
        out = y * F.silu(z) + x_in * self.D
        return x + self.dropout(self.out_proj(out)), h.view(B, -1)
class MambaEncoder(nn.Module):
    def __init__(self, input_dim, d_model=256, d_state=64, num_layers=4, dropout=0.0, segment_size=16):
        super().__init__()
        self.d_model, self.d_state, self.num_layers = d_model, d_state, num_layers
        self.output_dim = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([MambaBlock(d_model, d_state, dropout, segment_size) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.state_size = d_model * d_state
    def forward(self, x, hidden_state=None):
        B, L, D = x.shape
        x = self.input_proj(x)
        if hidden_state is None:
            hidden_state = [torch.zeros(B, self.state_size, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        new_states = []
        for i, layer in enumerate(self.layers):
            x, h_new = layer(x, hidden_state[i])
            new_states.append(h_new)
        return self.final_norm(x), new_states
    def get_initial_hidden_state(self, batch_size, device):
        return [torch.zeros(batch_size, self.state_size, device=device) for _ in range(self.num_layers)]
class GFootballPolicyValueNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.stage_embedding = nn.Embedding(config.num_stages, config.stage_emb_dim)
        self.action_embedding = nn.Embedding(config.num_actions, config.action_emb_dim)
        obs_input_dim = config.obs_dim + config.feature_dim + config.stage_emb_dim + config.action_emb_dim
        encoder_layers = []
        in_dim = obs_input_dim
        for hidden_dim in config.encoder_hidden:
            encoder_layers.extend([nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        self.obs_encoder = nn.Sequential(*encoder_layers)
        self.mamba = MambaEncoder(input_dim=in_dim, d_model=config.d_model, d_state=config.mamba_d_state, num_layers=config.mamba_layers, dropout=config.dropout, segment_size=config.segment_size)
        policy_layers = []
        in_dim = self.mamba.output_dim
        for hidden_dim in config.policy_hidden:
            policy_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        policy_layers.append(nn.Linear(in_dim, config.num_actions))
        self.policy_head = nn.Sequential(*policy_layers)
        self.num_value_heads = 5
        self.num_reward_types = 2
        value_hidden = config.value_hidden[0] if config.value_hidden else 256
        self.value_fc1_dense = nn.Linear(self.mamba.output_dim, value_hidden * self.num_value_heads)
        self.value_fc1_sparse = nn.Linear(self.mamba.output_dim, value_hidden * self.num_value_heads)
        if config.use_distributional:
            self.value_fc2_dense = nn.Linear(value_hidden * self.num_value_heads, config.num_atoms * self.num_value_heads)
            self.value_fc2_sparse = nn.Linear(value_hidden * self.num_value_heads, config.num_atoms * self.num_value_heads)
            self.register_buffer('value_support', torch.linspace(config.v_min, config.v_max, config.num_atoms))
            self.value_out_dim = config.num_atoms
        else:
            self.value_fc2_dense = nn.Linear(value_hidden * self.num_value_heads, self.num_value_heads)
            self.value_fc2_sparse = nn.Linear(value_hidden * self.num_value_heads, self.num_value_heads)
            self.value_support = None
            self.value_out_dim = 1
        self._init_weights()
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module not in [m for layer in self.mamba.layers for m in [layer.in_proj, layer.B_proj, layer.C_proj, layer.out_proj, layer.dt_proj]]:
                    nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_fc1_dense.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.value_fc2_dense.weight, gain=1.0)
        nn.init.orthogonal_(self.value_fc1_sparse.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.value_fc2_sparse.weight, gain=1.0)
    def _normalize_obs(self, obs):
        if obs.dim() == 1:
            return obs.unsqueeze(0).unsqueeze(1), True
        elif obs.dim() == 2:
            return obs.unsqueeze(1), True
        elif obs.dim() == 3:
            return obs, False
        else:
            raise ValueError("obs must be 1D, 2D, or 3D")
    def _normalize_index(self, idx, B, L, device, default_val=0):
        if idx is None:
            return torch.full((B, L), default_val, dtype=torch.long, device=device)
        if idx.dim() == 0:
            return idx.long().expand(B, L).contiguous()
        elif idx.dim() == 1:
            return idx.long().unsqueeze(1).expand(B, L).contiguous()
        elif idx.dim() == 2:
            if idx.shape[1] == 1 and L > 1:
                return idx.long().expand(B, L).contiguous()
            return idx.long()
        else:
            raise ValueError("Index must be 0D, 1D, or 2D")
    def forward(self, obs, features, stage_idx, prev_action=None, hidden_state=None, return_hidden=False, reward_type=None):
        obs, squeeze = self._normalize_obs(obs)
        B, L, _ = obs.shape
        device = obs.device
        use_amp = device.type == 'cuda' and self.training
        if features.dim() == 1:
            features = features.unsqueeze(0).unsqueeze(1)
        elif features.dim() == 2:
            features = features.unsqueeze(1)
        stage_idx = self._normalize_index(stage_idx, B, L, device, 0)
        prev_action = self._normalize_index(prev_action, B, L, device, 0)
        if reward_type is None:
            reward_type = torch.zeros(B, dtype=torch.long, device=device)
        elif isinstance(reward_type, int):
            reward_type = torch.full((B,), reward_type, dtype=torch.long, device=device)
        elif isinstance(reward_type, torch.Tensor):
            if reward_type.dim() == 0:
                reward_type = reward_type.expand(B)
            reward_type = reward_type.long().to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            stage_emb = self.stage_embedding(stage_idx)
            action_emb = self.action_embedding(prev_action)
            x = torch.cat([obs, features, stage_emb, action_emb], dim=-1)
            x = self.obs_encoder(x)
        x = x.float()
        x, new_hidden = self.mamba(x, hidden_state)
        x = x.float()
        logits = self.policy_head(x)
        v_dense = F.relu(self.value_fc1_dense(x))
        v_dense = self.value_fc2_dense(v_dense)
        v_sparse = F.relu(self.value_fc1_sparse(x))
        v_sparse = self.value_fc2_sparse(v_sparse)
        Bs, Ls = v_dense.shape[:2]
        v_dense = v_dense.view(Bs, Ls, self.num_value_heads, self.value_out_dim)
        v_sparse = v_sparse.view(Bs, Ls, self.num_value_heads, self.value_out_dim)
        if self.config.use_distributional:
            value_logits_dense = v_dense.mean(dim=2)
            value_probs_dense = F.softmax(value_logits_dense, dim=-1)
            value_dense = (value_probs_dense * self.value_support).sum(-1, keepdim=True)
            value_logits_sparse = v_sparse.mean(dim=2)
            value_probs_sparse = F.softmax(value_logits_sparse, dim=-1)
            value_sparse = (value_probs_sparse * self.value_support).sum(-1, keepdim=True)
        else:
            value_dense = v_dense.mean(dim=2)
            value_sparse = v_sparse.mean(dim=2)
            value_logits_dense = None
            value_logits_sparse = None
        rt_expanded = reward_type.view(B, 1, 1).expand(B, Ls, 1)
        value = torch.where(rt_expanded == 0, value_dense, value_sparse)
        if self.config.use_distributional:
            rt_logits = reward_type.view(B, 1, 1).expand(B, Ls, self.config.num_atoms)
            value_logits = torch.where(rt_logits == 0, value_logits_dense, value_logits_sparse)
        else:
            value_logits = None
        log_probs = F.log_softmax(logits, dim=-1)
        if squeeze:
            logits, value, log_probs = logits.squeeze(1), value.squeeze(1), log_probs.squeeze(1)
            if value_logits is not None:
                value_logits = value_logits.squeeze(1)
        result = {'logits': logits, 'value': value.squeeze(-1) if value.dim() > 1 else value, 'log_probs': log_probs}
        if value_logits is not None:
            result['value_logits'] = value_logits
        if return_hidden:
            result['hidden_state'] = new_hidden
        return result
    def get_action(self, obs, features, stage_idx, prev_action=None, hidden_state=None, deterministic=False, reward_type=None):
        output = self.forward(obs, features, stage_idx, prev_action, hidden_state, return_hidden=True, reward_type=reward_type)
        logits = output['logits']
        if torch.isnan(logits).any():
            logits = torch.zeros_like(logits)
        logits = logits.clamp(min=-20.0, max=20.0)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), output['value'], output.get('hidden_state')
    def evaluate_actions(self, obs, features, stage_idx, actions, prev_action=None, hidden_state=None, reward_type=None):
        output = self.forward(obs, features, stage_idx, prev_action, hidden_state, reward_type=reward_type)
        logits = output['logits'].clamp(min=-20.0, max=20.0)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), output['value']
    def get_initial_hidden_state(self, batch_size, device):
        return self.mamba.get_initial_hidden_state(batch_size, device)
def create_model(config_dict=None):
    return GFootballPolicyValueNet(ModelConfig(**(config_dict or {})))
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
@dataclass
class StageConfig:
    stage_id: int
    env_name: str
    representation: str = "simple115v2"
    left_agents: int = 1
    right_agents: int = 0
    max_steps: int = 3000
    rewards: str = "scoring,checkpoints"
    reward_type: int = -1
    def __post_init__(self):
        if self.reward_type == -1:
            self.reward_type = 1 if self.rewards == "scoring" else 0
def get_default_stages():
    return [
        StageConfig(0, "academy_empty_goal_close", "simple115v2", 1, 0, 400),
        StageConfig(1, "academy_empty_goal", "simple115v2", 1, 0, 400),
        StageConfig(2, "academy_run_to_score", "simple115v2", 1, 0, 400),
        StageConfig(3, "academy_run_to_score_with_keeper", "simple115v2", 1, 0, 400),
        StageConfig(4, "academy_pass_and_shoot_with_keeper", "simple115v2", 2, 0, 400),
        StageConfig(5, "academy_3_vs_1_with_keeper", "simple115v2", 3, 0, 400),
        StageConfig(6, "academy_counterattack_easy", "simple115v2", 4, 0, 600),
        StageConfig(7, "academy_counterattack_hard", "simple115v2", 4, 0, 600),
        StageConfig(8, "academy_single_goal_versus_lazy", "simple115v2", 4, 0, 1000),
        StageConfig(9, "academy_single_goal_versus_lazy", "simple115v2", 11, 0, 1000),
    ]
@dataclass
class TrainingConfig:
    stages: List[StageConfig] = field(default_factory=list)
    final_stage_target_win_rate: float = 0.95
    max_steps_without_progress: int = 500_000
    num_workers: int = 20
    envs_per_worker: int = 1
    trajectory_length: int = 128
    queue_size: int = 64
    batch_size: int = 2048
    minibatch_size: int = 512
    num_epochs: int = 4
    learning_rate: float = 3e-4
    lr_schedule: str = "reduce_on_plateau"
    lr_warmup_steps: int = 5000
    lr_min: float = 1e-6
    lr_plateau_window: int = 200
    lr_plateau_threshold: float = 0.01
    lr_plateau_factor: float = 0.5
    lr_plateau_cooldown: int = 100
    lr_plateau_min_episodes: int = 500
    lr_plateau_stage_protection: int = 300
    lr_plateau_max_reductions: int = 5
    max_grad_norm: float = 0.5
    gamma: float = 1.0
    rho_bar: float = 1.0
    c_bar: float = 1.0
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    si_lambda: float = 0.5
    total_steps: int = 100_000_000
    log_interval: int = 10
    checkpoint_interval: int = 100
    feature_importance_interval: int = 50
    weight_sync_interval: int = 5
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    def __post_init__(self):
        if not self.stages:
            self.stages = get_default_stages()
@dataclass
class StageBaseline:
    stage_id: int
    episode_return_mean: float = 0.0
    episode_return_std: float = 1.0
    step_reward_mean: float = 0.0
    step_reward_std: float = 0.01
    win_rate: float = 0.0
    calibrated: bool = False
    def normalize_return(self, raw_return):
        if not self.calibrated or self.episode_return_std < 1e-6:
            return raw_return
        return (raw_return - self.episode_return_mean) / self.episode_return_std
    def to_dict(self):
        return asdict(self)
    @classmethod
    def from_dict(cls, d):
        return cls(**d)
@ray.remote
def _calibrate_stage_batch(stage_dict, num_episodes, worker_id):
    stage = StageConfig(**stage_dict)
    env = football_env.create_environment(env_name=stage.env_name, representation=stage.representation, number_of_left_players_agent_controls=stage.left_agents, number_of_right_players_agent_controls=stage.right_agents, rewards=stage.rewards, write_goal_dumps=False, write_full_episode_dumps=False, render=False, write_video=False)
    returns, wins, step_rewards, lengths = [], [], [], []
    for ep in range(num_episodes):
        obs, done, ep_return, ep_steps = env.reset(), False, 0.0, 0
        while not done and ep_steps < stage.max_steps:
            actions = [s.sample() for s in env.action_space] if isinstance(env.action_space, list) else env.action_space.sample()
            obs, reward, done, info = env.step(actions)
            step_reward = float(sum(reward)) if isinstance(reward, (list, np.ndarray)) else float(reward)
            step_rewards.append(step_reward)
            ep_return += step_reward
            ep_steps += 1
        returns.append(ep_return)
        lengths.append(ep_steps)
        won = info["score"][0] > info["score"][1] if isinstance(info, dict) and "score" in info else ep_return > 0
        wins.append(1.0 if won else 0.0)
    env.close()
    return {'stage_id': stage.stage_id, 'worker_id': worker_id, 'returns': returns, 'wins': wins, 'step_rewards': step_rewards, 'lengths': lengths}
class BaselineCalibrator:
    def __init__(self, stages, save_path):
        self.stages, self.save_path = stages, save_path
        self.baselines = {s.stage_id: StageBaseline(s.stage_id) for s in stages}
    def load(self):
        if not self.save_path.exists():
            return False
        try:
            with open(self.save_path) as f:
                data = json.load(f)
            for k, v in data.items():
                sid = int(k)
                if sid in self.baselines:
                    self.baselines[sid] = StageBaseline.from_dict(v)
            return all(b.calibrated for b in self.baselines.values())
        except Exception as e:
            print(f"Failed to load baselines: {e}")
            return False
    def save(self):
        with open(self.save_path, 'w') as f:
            json.dump({str(k): v.to_dict() for k, v in self.baselines.items()}, f, indent=2)
    def calibrate(self, num_episodes=100, num_workers=8):
        print(f"Calibrating baselines ({num_episodes} eps/stage, {num_workers} workers)...")
        episodes_per_worker = max(1, num_episodes // num_workers)
        futures = [_calibrate_stage_batch.remote(asdict(stage), episodes_per_worker, worker_id) for stage in self.stages for worker_id in range(num_workers)]
        stage_data = {s.stage_id: {'returns': [], 'wins': [], 'step_rewards': [], 'lengths': []} for s in self.stages}
        total, completed = len(futures), 0
        while futures:
            done, futures = ray.wait(futures, num_returns=1, timeout=None)
            for ref in done:
                result = ray.get(ref)
                sid = result['stage_id']
                stage_data[sid]['returns'].extend(result['returns'])
                stage_data[sid]['wins'].extend(result['wins'])
                stage_data[sid]['step_rewards'].extend(result['step_rewards'])
                stage_data[sid]['lengths'].extend(result['lengths'])
                completed += 1
                print(f"\r  Progress: {completed}/{total}", end="", flush=True)
        print()
        for stage in self.stages:
            data = stage_data[stage.stage_id]
            bl = self.baselines[stage.stage_id]
            bl.episode_return_mean = float(np.mean(data['returns']))
            bl.episode_return_std = max(float(np.std(data['returns'])), 0.01)
            bl.step_reward_mean = float(np.mean(data['step_rewards'])) if data['step_rewards'] else 0.0
            bl.step_reward_std = max(float(np.std(data['step_rewards'])), 0.001) if data['step_rewards'] else 0.01
            bl.win_rate = float(np.mean(data['wins']))
            bl.calibrated = True
            print(f"  Stage {stage.stage_id}: return={bl.episode_return_mean:.3f}±{bl.episode_return_std:.3f}, win={bl.win_rate:.1%}")
        self.save()
@ray.remote
class CurriculumController:
    T_LEARN, T_UNLOCK, T_MASTERY = 0.65, 0.85, 0.95
    MIN_EPS_UNLOCK, LP_WINDOW, STALENESS_COEFF, MIN_WEIGHT, SUSTAINED_WINDOW = 100, 200, 0.2, 0.05, 50
    def __init__(self, stages, baselines, final_target_win_rate=0.95, initial_state=None):
        self.stages = [StageConfig(**s) if isinstance(s, dict) else s for s in stages]
        self.baselines = {int(k): StageBaseline.from_dict(v) if isinstance(v, dict) else v for k, v in baselines.items()}
        self.num_stages = len(self.stages)
        self.final_target_win_rate = final_target_win_rate
        self.perfect_achieved = {s.stage_id: False for s in self.stages}
        self._restore_state(initial_state) if initial_state else self._init_fresh()
    def _init_fresh(self):
        self.episode_count, self.unlocked_stages, self.learned_stages, self.mastered_stages = 0, {0}, set(), set()
        self.stage_stats = {s.stage_id: {'episodes': 0, 'ema_return': 0.0, 'ema_win': 0.0, 'ema_win_slow': 0.0, 'sustained_peak': 0.0, 'max_reward': -float('inf'), 'recent_wins': [], 'recent_returns': [], 'last_sampled': 0, 'lp_history': []} for s in self.stages}
    def _restore_state(self, state):
        self.episode_count = state.get('episode_count', 0)
        self.unlocked_stages = set(state.get('unlocked_stages', [0]))
        self.learned_stages = set(state.get('learned_stages', []))
        self.mastered_stages = set(state.get('mastered_stages', []))
        self.perfect_achieved = state.get('perfect_achieved', {s.stage_id: False for s in self.stages})
        saved_stats = state.get('stage_stats', {})
        self.stage_stats = {}
        for s in self.stages:
            sid_str = str(s.stage_id)
            if sid_str in saved_stats:
                ss = saved_stats[sid_str]
                self.stage_stats[s.stage_id] = {'episodes': ss.get('episodes', 0), 'ema_return': ss.get('ema_return', 0.0), 'ema_win': ss.get('ema_win', 0.0), 'ema_win_slow': ss.get('ema_win_slow', ss.get('ema_win', 0.0)), 'sustained_peak': ss.get('sustained_peak', ss.get('peak_win', 0.0)), 'max_reward': ss.get('max_reward', -float('inf')), 'recent_wins': [], 'recent_returns': [], 'last_sampled': ss.get('last_sampled', 0), 'lp_history': []}
            else:
                self.stage_stats[s.stage_id] = {'episodes': 0, 'ema_return': 0.0, 'ema_win': 0.0, 'ema_win_slow': 0.0, 'sustained_peak': 0.0, 'max_reward': -float('inf'), 'recent_wins': [], 'recent_returns': [], 'last_sampled': 0, 'lp_history': []}
        print(f"  Restored: ep={self.episode_count}, learned={sorted(self.learned_stages)}, mastered={sorted(self.mastered_stages)}")
    def _compute_learning_progress(self, sid):
        return self.stage_stats[sid]['ema_win'] - self.stage_stats[sid]['ema_win_slow']
    def _compute_staleness(self, sid):
        stats = self.stage_stats[sid]
        return 0.0 if stats['episodes'] == 0 else min((self.episode_count - stats['last_sampled']) / 500.0, 2.0)
    def _compute_forgetting(self, sid):
        return max(0.0, self.stage_stats[sid]['sustained_peak'] - self.stage_stats[sid]['ema_win'])
    def _compute_weight(self, sid):
        stats = self.stage_stats[sid]
        if stats['episodes'] < 50:
            base = 1.0
        else:
            lp, forgetting = max(0.0, self._compute_learning_progress(sid)) * 10.0, self._compute_forgetting(sid) * 10.0
            if sid in self.mastered_stages:
                base = 0.1 + forgetting
            elif sid in self.learned_stages:
                base = 0.3 + lp + forgetting
            elif stats['ema_win'] < 0.1:
                base = 0.5 + lp
            elif stats['ema_win'] < 0.3:
                base = 0.8 + lp
            else:
                base = 1.0 + lp
        return max(self.MIN_WEIGHT, base + self.STALENESS_COEFF * self._compute_staleness(sid))
    def get_stage(self):
        available = sorted(self.unlocked_stages)
        if len(available) == 1:
            self.stage_stats[available[0]]['last_sampled'] = self.episode_count
            return asdict(self.stages[available[0]])
        weights = {sid: self._compute_weight(sid) for sid in available}
        total = sum(weights.values())
        probs = {k: v / total for k, v in weights.items()}
        chosen = np.random.choice(list(probs.keys()), p=list(probs.values()))
        self.stage_stats[chosen]['last_sampled'] = self.episode_count
        return asdict(self.stages[chosen])
    def report_episode(self, stage_id, episode_return, won):
        self.episode_count += 1
        stats = self.stage_stats[stage_id]
        stats['episodes'] += 1
        stats['recent_wins'].append(1.0 if won else 0.0)
        if len(stats['recent_wins']) > 100:
            stats['recent_wins'].pop(0)
        stats['recent_returns'].append(episode_return)
        if len(stats['recent_returns']) > 100:
            stats['recent_returns'].pop(0)
        if episode_return > stats['max_reward']:
            stats['max_reward'] = episode_return
        alpha_fast, alpha_slow = 0.02, 0.005
        if stats['episodes'] == 1:
            stats['ema_return'], stats['ema_win'], stats['ema_win_slow'] = episode_return, float(won), float(won)
        else:
            stats['ema_return'] = (1 - alpha_fast) * stats['ema_return'] + alpha_fast * episode_return
            stats['ema_win'] = (1 - alpha_fast) * stats['ema_win'] + alpha_fast * float(won)
            stats['ema_win_slow'] = (1 - alpha_slow) * stats['ema_win_slow'] + alpha_slow * float(won)
        if len(stats['recent_wins']) >= self.SUSTAINED_WINDOW:
            recent_mean = np.mean(stats['recent_wins'][-self.SUSTAINED_WINDOW:])
            if recent_mean > stats['sustained_peak']:
                stats['sustained_peak'] = recent_mean
        stats['lp_history'].append(self._compute_learning_progress(stage_id))
        if len(stats['lp_history']) > self.LP_WINDOW:
            stats['lp_history'].pop(0)
        self._check_learned(stage_id)
        self._check_unlock(stage_id)
        self._check_mastery(stage_id)
        return self._check_special_saves(stage_id)
    def _check_special_saves(self, stage_id):
        result = {'save_95_all': False, 'save_100': False, 'stage_100': None}
        stats = self.stage_stats[stage_id]
        if len(stats['recent_wins']) >= 50:
            recent_wr = np.mean(stats['recent_wins'][-50:])
            if recent_wr >= 1.0 and not self.perfect_achieved.get(stage_id, False):
                self.perfect_achieved[stage_id] = True
                result['save_100'] = True
                result['stage_100'] = stage_id
                print(f"\n🏆 PERFECT 100% on Stage {stage_id}!\n")
        all_95 = True
        for sid in range(self.num_stages):
            s = self.stage_stats[sid]
            if s['episodes'] < 50:
                all_95 = False
                break
            wr = np.mean(s['recent_wins'][-50:]) if len(s['recent_wins']) >= 50 else 0
            if wr < 0.95:
                all_95 = False
                break
        if all_95:
            result['save_95_all'] = True
        return result
    def _check_learned(self, stage_id):
        if stage_id in self.learned_stages:
            return
        stats = self.stage_stats[stage_id]
        if stats['episodes'] < 100 or stats['ema_win'] < self.T_LEARN:
            return
        if np.mean(stats['recent_wins']) if len(stats['recent_wins']) >= 50 else 0 < 0.5:
            return
        self.learned_stages.add(stage_id)
        print(f"\n📚 STAGE {stage_id} LEARNED! (ema={stats['ema_win']:.1%})\n")
    def _check_unlock(self, stage_id):
        next_stage = stage_id + 1
        if next_stage >= self.num_stages or next_stage in self.unlocked_stages:
            return
        if stage_id > 0 and (stage_id - 1) not in self.mastered_stages:
            return
        stats = self.stage_stats[stage_id]
        if stats['episodes'] < self.MIN_EPS_UNLOCK:
            return
        recent_wr = np.mean(stats['recent_wins']) if stats['recent_wins'] else 0
        if recent_wr >= self.T_UNLOCK:
            self.unlocked_stages.add(next_stage)
            print(f"\n🔓 STAGE {next_stage} UNLOCKED! (Stage {stage_id}: {recent_wr:.1%})\n")
    def _check_mastery(self, stage_id):
        if stage_id in self.mastered_stages:
            return
        stats = self.stage_stats[stage_id]
        if stats['episodes'] < 200:
            return
        recent_wr = np.mean(stats['recent_wins']) if stats['recent_wins'] else 0
        if recent_wr >= self.T_MASTERY:
            self.mastered_stages.add(stage_id)
            print(f"\n⭐ STAGE {stage_id} MASTERED! ({recent_wr:.1%})\n")
    def is_training_complete(self):
        final_stage = self.num_stages - 1
        if final_stage not in self.learned_stages:
            return False
        return np.mean(self.stage_stats[final_stage]['recent_wins']) if self.stage_stats[final_stage]['recent_wins'] else 0 >= self.final_target_win_rate
    def get_progress_summary(self):
        lines = []
        for sid in sorted(self.unlocked_stages):
            stats = self.stage_stats[sid]
            status = "⭐" if sid in self.mastered_stages else ("📚" if sid in self.learned_stages else "🔓")
            if stats['episodes'] > 0:
                recent_wr = np.mean(stats['recent_wins']) if stats['recent_wins'] else 0
                lines.append(f"{status}S{sid}:{recent_wr:.0%}")
            else:
                lines.append(f"{status}S{sid}:--")
        return " | ".join(lines)
    def get_stats(self):
        weights = {sid: self._compute_weight(sid) for sid in self.unlocked_stages}
        total_w = sum(weights.values())
        return {'episode_count': self.episode_count, 'unlocked_stages': list(self.unlocked_stages), 'learned_stages': list(self.learned_stages), 'mastered_stages': list(self.mastered_stages), 'highest_unlocked': max(self.unlocked_stages) if self.unlocked_stages else 0, 'training_complete': self.is_training_complete(), 'perfect_achieved': self.perfect_achieved, 'sample_probs': {sid: w / total_w for sid, w in weights.items()}, 'stage_stats': {str(sid): {'episodes': s['episodes'], 'ema_return': s['ema_return'], 'ema_win': s['ema_win'], 'ema_win_slow': s['ema_win_slow'], 'sustained_peak': s['sustained_peak'], 'max_reward': s['max_reward'] if s['max_reward'] > -float('inf') else 0, 'learning_progress': self._compute_learning_progress(sid), 'forgetting': self._compute_forgetting(sid), 'recent_win_rate': np.mean(s['recent_wins']) if s['recent_wins'] else 0} for sid, s in self.stage_stats.items()}}
@ray.remote
class SamplerWorker:
    MAX_AGENTS = 11
    def __init__(self, worker_id, model_config, stages, baselines):
        self.worker_id = worker_id
        self.stages = {s['stage_id']: StageConfig(**s) for s in stages}
        self.baselines = {int(k): StageBaseline.from_dict(v) for k, v in baselines.items()}
        self.device = torch.device('cpu')
        self.model = create_model(model_config)
        self.model.to(self.device)
        self.model.eval()
        self.feature_engineer = FeatureEngineer()
        self.env, self.current_stage, self.current_obs, self.current_features = None, None, None, None
        self.hidden_state, self.prev_action, self.episode_return, self.episode_steps = None, None, 0.0, 0
    def set_weights(self, weights):
        self.model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})
    def collect_trajectory(self, trajectory_length, curriculum_controller):
        obs_list, feature_list, action_list, reward_list, done_list = [], [], [], [], []
        value_list, log_prob_list, stage_list, mask_list, reward_type_list = [], [], [], [], []
        episode_returns, episode_wins, episode_lengths, episode_stages, episode_max_rewards = [], [], [], [], []
        special_saves = []
        steps, ep_max_reward = 0, -float('inf')
        while steps < trajectory_length:
            if self.env is None or self._should_switch_stage():
                self._setup_env(StageConfig(**ray.get(curriculum_controller.get_stage.remote())))
                ep_max_reward = -float('inf')
            num_agents = self.current_stage.left_agents
            current_reward_type = self.current_stage.reward_type
            stage_tensor = torch.tensor(self.current_stage.stage_id, device=self.device)
            reward_type_tensor = torch.tensor(current_reward_type, device=self.device)
            if num_agents == 1:
                obs_tensor = torch.from_numpy(self.current_obs[0]).float().to(self.device)
                feature_tensor = torch.from_numpy(self.current_features[0]).float().to(self.device)
                with torch.no_grad():
                    action, log_prob, value, self.hidden_state = self.model.get_action(obs_tensor, feature_tensor, stage_tensor, prev_action=self.prev_action, hidden_state=self.hidden_state, reward_type=reward_type_tensor)
                action_int, log_prob_float, value_float = action.item(), log_prob.item(), value.item()
                action_full = np.zeros(self.MAX_AGENTS, dtype=np.int64)
                log_prob_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                value_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                mask_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                action_full[0], log_prob_full[0], value_full[0], mask_full[0] = action_int, log_prob_float, value_float, 1.0
                env_action = action_int
            else:
                obs_batch = torch.from_numpy(self.current_obs[:num_agents]).float().to(self.device)
                feat_batch = torch.from_numpy(self.current_features[:num_agents]).float().to(self.device)
                reward_type_batch = torch.full((num_agents,), current_reward_type, device=self.device)
                with torch.no_grad():
                    actions, log_probs, values, _ = self.model.get_action(obs_batch, feat_batch, stage_tensor, prev_action=None, hidden_state=None, reward_type=reward_type_batch)
                action_full = np.zeros(self.MAX_AGENTS, dtype=np.int64)
                log_prob_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                value_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                mask_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                action_full[:num_agents] = actions.cpu().numpy()
                log_prob_full[:num_agents] = log_probs.cpu().numpy()
                value_full[:num_agents] = values.cpu().numpy()
                mask_full[:num_agents] = 1.0
                env_action = actions.cpu().numpy().tolist()
            raw_obs, reward, done, info = self.env.step(env_action)
            self._update_obs(raw_obs)
            step_reward = float(sum(reward)) if isinstance(reward, (list, np.ndarray)) else float(reward)
            self.episode_return += step_reward
            self.episode_steps += 1
            if step_reward > ep_max_reward:
                ep_max_reward = step_reward
            episode_done = bool(done) or self.episode_steps >= self.current_stage.max_steps
            reward_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            done_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            if num_agents > 0:
                per_agent_reward = step_reward / num_agents
                for i in range(num_agents):
                    reward_full[i], done_full[i] = per_agent_reward, float(episode_done)
            obs_padded = np.zeros((self.MAX_AGENTS, OBS_DIM), dtype=np.float32)
            feature_padded = np.zeros((self.MAX_AGENTS, FEATURE_DIM), dtype=np.float32)
            obs_padded[:num_agents] = self.current_obs[:num_agents]
            feature_padded[:num_agents] = self.current_features[:num_agents]
            obs_list.append(obs_padded)
            feature_list.append(feature_padded)
            action_list.append(action_full)
            reward_list.append(reward_full)
            done_list.append(done_full)
            value_list.append(value_full)
            log_prob_list.append(log_prob_full)
            stage_list.append(self.current_stage.stage_id)
            mask_list.append(mask_full)
            reward_type_list.append(current_reward_type)
            self.prev_action = torch.tensor(action_full[0], device=self.device)
            steps += 1
            if episode_done:
                won = info["score"][0] > info["score"][1] if isinstance(info, dict) and "score" in info else self.episode_return > 0
                episode_returns.append(self.episode_return)
                episode_wins.append(won)
                episode_lengths.append(self.episode_steps)
                episode_stages.append(self.current_stage.stage_id)
                episode_max_rewards.append(ep_max_reward)
                save_result = ray.get(curriculum_controller.report_episode.remote(self.current_stage.stage_id, self.episode_return, won))
                if save_result:
                    special_saves.append(save_result)
                self._reset_episode()
                ep_max_reward = -float('inf')
        return {'obs': np.array(obs_list, dtype=np.float32), 'features': np.array(feature_list, dtype=np.float32), 'actions': np.array(action_list, dtype=np.int64), 'rewards': np.array(reward_list, dtype=np.float32), 'dones': np.array(done_list, dtype=np.float32), 'values': np.array(value_list, dtype=np.float32), 'log_probs': np.array(log_prob_list, dtype=np.float32), 'stage_ids': np.array(stage_list, dtype=np.int64), 'agent_masks': np.array(mask_list, dtype=np.float32), 'reward_types': np.array(reward_type_list, dtype=np.int64), 'worker_id': self.worker_id, 'episode_returns': episode_returns, 'episode_wins': episode_wins, 'episode_lengths': episode_lengths, 'episode_stages': episode_stages, 'episode_max_rewards': episode_max_rewards, 'special_saves': special_saves}
    def _setup_env(self, stage):
        if self.env is not None:
            self.env.close()
        self.current_stage = stage
        self.env = football_env.create_environment(env_name=stage.env_name, representation=stage.representation, number_of_left_players_agent_controls=stage.left_agents, number_of_right_players_agent_controls=stage.right_agents, stacked=True, rewards=stage.rewards, write_goal_dumps=False, write_full_episode_dumps=False, render=False, write_video=False)
        self._reset_episode()
    def _reset_episode(self):
        self._update_obs(self.env.reset())
        self.hidden_state = self.model.get_initial_hidden_state(1, self.device)
        self.prev_action, self.episode_return, self.episode_steps = None, 0.0, 0
    def _update_obs(self, raw_obs):
        if not isinstance(raw_obs, np.ndarray):
            raw_obs = np.array(raw_obs)
        if raw_obs.ndim == 1:
            raw_obs = raw_obs.reshape(1, -1)
        elif raw_obs.ndim == 3:
            num_agents = raw_obs.shape[0]
            raw_obs = raw_obs.reshape(num_agents, -1)
        self.current_obs = raw_obs.astype(np.float32)
        self.current_features = self.feature_engineer.extract_features(self.current_obs)
    def _should_switch_stage(self):
        return self.episode_steps == 0 and np.random.random() < 0.1
    def close(self):
        if self.env is not None:
            self.env.close()
def parallel_vtrace_scan(values, rewards, dones, rho, c, gamma, bootstrap_value):
    N = len(values)
    not_done = 1 - dones
    v_next = torch.cat([values[1:], bootstrap_value.unsqueeze(0)])
    delta = rho * (rewards + gamma * not_done * v_next - values)
    b = gamma * not_done * c
    a = values + delta - b * v_next
    a_flip, b_flip = a.flip(0), b.flip(0)
    n_steps = int(math.ceil(math.log2(N))) if N > 1 else 1
    for d in range(n_steps):
        stride = 2 ** d
        if stride >= N:
            break
        a_shifted = torch.cat([torch.zeros(stride, device=a.device, dtype=a.dtype), a_flip[:-stride]])
        b_shifted = torch.cat([torch.zeros(stride, device=b.device, dtype=b.dtype), b_flip[:-stride]])
        a_flip = a_flip + b_flip * a_shifted
        b_flip = b_flip * b_shifted
    return a_flip.flip(0)
class Learner:
    def __init__(self, config, model_config, writer=None):
        self.config, self.device, self.writer = config, torch.device(config.device), writer
        self.model = create_model(model_config)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, eps=1e-5, weight_decay=1e-5)
        self.scaler = torch.amp.GradScaler('cuda')
        self.update_count, self.nan_count = 0, 0
        self.ema_enabled = True
        self.ema_decay = 0.999
        self.ema_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.ema_start_step = 1000
        self.current_highest_stage = 0
        self.steps_since_stage_change = 0
        self.lr_reductions_this_stage = 0
        self.cooldown_counter = 0
        self.lp_history = []
        self.current_lr = config.learning_rate
        self.si_omega, self.si_prev_params, self.si_running_sum = {}, {}, {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.si_omega[name] = torch.zeros_like(param, device=self.device)
                self.si_prev_params[name] = param.data.clone()
                self.si_running_sum[name] = torch.zeros_like(param, device=self.device)
        self.si_lambda, self.si_epsilon = config.si_lambda, 1e-3
        self.feature_importance_history = defaultdict(list)
        print(f"Learner on {self.device}, params: {count_parameters(self.model):,}")
    def _update_ema(self):
        if not self.ema_enabled or self.update_count < self.ema_start_step:
            return
        with torch.no_grad():
            for k, v in self.model.state_dict().items():
                if k in self.ema_weights:
                    self.ema_weights[k].mul_(self.ema_decay).add_(v, alpha=1 - self.ema_decay)
    def get_weights(self, use_ema=True):
        if use_ema and self.ema_enabled and self.update_count >= self.ema_start_step:
            return {k: v.cpu().numpy() for k, v in self.ema_weights.items()}
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
    def _compute_feature_importance(self, batch):
        self.model.eval()
        n_samples = min(256, len(batch['obs']))
        idx = np.random.choice(len(batch['obs']), n_samples, replace=False)
        features = batch['features'][idx].clone().requires_grad_(True)
        obs = batch['obs'][idx]
        stage_ids = batch['stage_ids'][idx]
        actions = batch['actions'][idx]
        reward_types = batch['reward_types'][idx]
        log_probs, _, values = self.model.evaluate_actions(obs, features, stage_ids, actions, reward_type=reward_types)
        loss = -log_probs.mean() + values.mean()
        loss.backward()
        grad = features.grad.abs().mean(dim=0).cpu().numpy()
        feat_var = features.detach().var(dim=0).cpu().numpy()
        VAR_THRESHOLD = 1e-4
        active_mask = feat_var > VAR_THRESHOLD
        importance = np.where(active_mask, grad, 0.0)
        if importance.max() > 0:
            importance = importance / importance.max()
        self.model.train()
        return importance, {}
    def _log_feature_importance(self, importance, stage_importance=None):
        if self.writer is None:
            return
        for i, imp in enumerate(importance):
            if i < len(FEATURE_NAMES):
                self.writer.add_scalar(f'feature_importance/global/{FEATURE_NAMES[i]}', imp, self.update_count)
        for group_name, indices in FEATURE_GROUPS.items():
            group_imp = np.mean([importance[i] for i in indices if i < len(importance)])
            self.writer.add_scalar(f'feature_importance/groups/{group_name}', group_imp, self.update_count)
            self.feature_importance_history[group_name].append(group_imp)
    def _compute_vtrace_batch(self, trajectories):
        gamma, rho_bar, c_bar = self.config.gamma, self.config.rho_bar, self.config.c_bar
        all_obs, all_features, all_actions = [], [], []
        all_rewards, all_dones, all_behavior_log_probs, all_stage_ids, all_reward_types = [], [], [], [], []
        traj_boundaries = [0]
        for traj in trajectories:
            masks = traj['agent_masks']
            T, A = masks.shape
            mask_flat = masks.reshape(-1) > 0
            if mask_flat.sum() == 0:
                continue
            all_obs.append(traj['obs'].reshape(-1, traj['obs'].shape[-1])[mask_flat])
            all_features.append(traj['features'].reshape(-1, traj['features'].shape[-1])[mask_flat])
            all_actions.append(traj['actions'].reshape(-1)[mask_flat])
            all_rewards.append(traj['rewards'].reshape(-1)[mask_flat])
            all_dones.append(traj['dones'].reshape(-1)[mask_flat])
            all_behavior_log_probs.append(traj['log_probs'].reshape(-1)[mask_flat])
            all_stage_ids.append(np.repeat(traj['stage_ids'], A)[mask_flat])
            all_reward_types.append(np.repeat(traj['reward_types'], A)[mask_flat])
            traj_boundaries.append(traj_boundaries[-1] + mask_flat.sum())
        if not all_obs:
            return None
        obs = torch.from_numpy(np.concatenate(all_obs)).float().to(self.device, non_blocking=True)
        features = torch.from_numpy(np.concatenate(all_features)).float().to(self.device, non_blocking=True)
        actions = torch.from_numpy(np.concatenate(all_actions)).long().to(self.device, non_blocking=True)
        rewards = torch.from_numpy(np.concatenate(all_rewards)).float().to(self.device, non_blocking=True)
        dones = torch.from_numpy(np.concatenate(all_dones)).float().to(self.device, non_blocking=True)
        behavior_log_probs = torch.from_numpy(np.concatenate(all_behavior_log_probs)).float().to(self.device, non_blocking=True)
        stage_ids = torch.from_numpy(np.concatenate(all_stage_ids)).long().to(self.device, non_blocking=True)
        reward_types = torch.from_numpy(np.concatenate(all_reward_types)).long().to(self.device, non_blocking=True)
        with torch.no_grad():
            target_log_probs, _, values = self.model.evaluate_actions(obs, features, stage_ids, actions, reward_type=reward_types)
        target_log_probs, values = target_log_probs.float(), values.float()
        log_ratios = (target_log_probs - behavior_log_probs).clamp(-20, 20)
        ratios = torch.exp(log_ratios)
        rho = torch.clamp(ratios, max=rho_bar)
        c = torch.clamp(ratios, max=c_bar)
        vtrace_targets = torch.zeros_like(values)
        pg_advantages = torch.zeros_like(values)
        num_trajs = len(traj_boundaries) - 1
        for i in range(num_trajs):
            start, end = traj_boundaries[i], traj_boundaries[i + 1]
            if end <= start:
                continue
            traj_values = values[start:end]
            traj_rewards = rewards[start:end]
            traj_dones = dones[start:end]
            traj_rho = rho[start:end]
            traj_c = c[start:end]
            bootstrap = traj_values[-1] * (1 - traj_dones[-1])
            vtrace_targets[start:end] = parallel_vtrace_scan(traj_values, traj_rewards, traj_dones, traj_rho, traj_c, gamma, bootstrap)
            vs_plus_one = torch.cat([vtrace_targets[start+1:end], bootstrap.unsqueeze(0)])
            pg_advantages[start:end] = traj_rho * (traj_rewards + gamma * (1 - traj_dones) * vs_plus_one - traj_values)
        return {'obs': obs, 'features': features, 'actions': actions, 'stage_ids': stage_ids, 'advantages': pg_advantages, 'vtrace_targets': vtrace_targets, 'rho': rho, 'reward_types': reward_types}
    def update(self, trajectories, global_step=0):
        self.model.train()
        batch = self._compute_vtrace_batch(trajectories)
        if batch is None:
            return {}
        if torch.isnan(batch['obs']).any() or torch.isinf(batch['obs']).any():
            self.nan_count += 1
            return {'nan_skipped': 1.0}
        if self.update_count % self.config.feature_importance_interval == 0:
            importance, stage_importance = self._compute_feature_importance(batch)
            self._log_feature_importance(importance, stage_importance)
        advantages = batch['advantages']
        vtrace_targets = batch['vtrace_targets']
        rho = batch['rho']
        reward_types = batch['reward_types']
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std
        batch_size = len(advantages)
        all_indices = []
        for epoch in range(self.config.num_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.config.minibatch_size):
                all_indices.append(perm[start:start + self.config.minibatch_size])
        total_loss, policy_loss_sum, value_loss_sum, entropy_sum = 0.0, 0.0, 0.0, 0.0
        dense_value_loss_sum, sparse_value_loss_sum = 0.0, 0.0
        dense_count, sparse_count = 0, 0
        rho_sum, num_updates, skipped = 0.0, 0, 0
        grad_norm = torch.tensor(0.0)
        for mb_idx in all_indices:
            mb_obs = batch['obs'][mb_idx]
            mb_features = batch['features'][mb_idx]
            mb_actions = batch['actions'][mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_vtrace_targets = vtrace_targets[mb_idx]
            mb_stage_ids = batch['stage_ids'][mb_idx]
            mb_rho = rho[mb_idx]
            mb_reward_types = reward_types[mb_idx]
            try:
                log_probs, entropy, values = self.model.evaluate_actions(mb_obs, mb_features, mb_stage_ids, mb_actions, reward_type=mb_reward_types)
                if torch.isnan(log_probs).any() or torch.isnan(values).any():
                    self.nan_count += 1
                    skipped += 1
                    continue
                policy_loss = -(log_probs * mb_advantages.detach()).mean()
                value_loss = F.mse_loss(values, mb_vtrace_targets.detach())
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.config.value_coeff * value_loss + self.config.entropy_coeff * entropy_loss
                dense_mask = mb_reward_types == 0
                sparse_mask = mb_reward_types == 1
                if dense_mask.any():
                    dense_vl = F.mse_loss(values[dense_mask], mb_vtrace_targets[dense_mask]).item()
                    dense_value_loss_sum += dense_vl
                    dense_count += 1
                if sparse_mask.any():
                    sparse_vl = F.mse_loss(values[sparse_mask], mb_vtrace_targets[sparse_mask]).item()
                    sparse_value_loss_sum += sparse_vl
                    sparse_count += 1
                if self.si_lambda > 0:
                    si_loss = sum((self.si_omega[n] * (p - self.si_prev_params[n]).pow(2)).sum() for n, p in self.model.named_parameters() if p.requires_grad and n in self.si_omega)
                    loss = loss + self.si_lambda * si_loss
            except ValueError:
                self.nan_count += 1
                skipped += 1
                continue
            if torch.isnan(loss) or torch.isinf(loss):
                self.nan_count += 1
                skipped += 1
                continue
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                self.nan_count += 1
                skipped += 1
                self.scaler.update()
                continue
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._update_ema()
            if num_updates % 4 == 0:
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.si_running_sum and param.grad is not None:
                        self.si_running_sum[name] += -param.grad.data * (param.data - self.si_prev_params[name])
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += -entropy_loss.item()
            rho_sum += mb_rho.mean().item()
            num_updates += 1
        self.update_count += 1
        if self.update_count % 10 == 0:
            self._consolidate_si()
        if num_updates == 0:
            return {'nan_skipped': float(skipped)}
        explained_var = 0.0
        if self.update_count % 10 == 0:
            with torch.no_grad():
                var_y = vtrace_targets.var()
                sample_idx = torch.randperm(len(vtrace_targets))[:min(2048, len(vtrace_targets))]
                _, _, sample_values = self.model.evaluate_actions(batch['obs'][sample_idx], batch['features'][sample_idx], batch['stage_ids'][sample_idx], batch['actions'][sample_idx], reward_type=batch['reward_types'][sample_idx])
                explained_var = float(1 - (vtrace_targets[sample_idx] - sample_values.float()).var() / var_y if var_y > 1e-6 else 0.0)
        total_dense_samples = (reward_types == 0).sum().item()
        total_sparse_samples = (reward_types == 1).sum().item()
        return {'loss/total': total_loss / num_updates, 'loss/policy': policy_loss_sum / num_updates, 'loss/value': value_loss_sum / num_updates, 'loss/entropy': entropy_sum / num_updates, 'vtrace/rho_mean': rho_sum / num_updates, 'vtrace/rho_max': rho.max().item(), 'vtrace/explained_variance': explained_var, 'train/lr': self.optimizer.param_groups[0]['lr'], 'train/nan_count': self.nan_count, 'train/grad_norm': float(grad_norm), 'value_heads/dense_samples': total_dense_samples, 'value_heads/sparse_samples': total_sparse_samples, 'value_heads/dense_loss': dense_value_loss_sum / dense_count if dense_count > 0 else 0, 'value_heads/sparse_loss': sparse_value_loss_sum / sparse_count if sparse_count > 0 else 0}
    def _consolidate_si(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.si_omega:
                delta = param.data - self.si_prev_params[name]
                self.si_omega[name] = torch.clamp(self.si_omega[name] + self.si_running_sum[name] / (delta.pow(2) + self.si_epsilon), min=0.0)
                self.si_prev_params[name] = param.data.clone()
                self.si_running_sum[name].zero_()
    def update_curriculum_state(self, highest_unlocked_stage: int, frontier_lp: float, frontier_episodes: int):
        self.steps_since_stage_change += 1
        if highest_unlocked_stage > self.current_highest_stage:
            self.current_highest_stage = highest_unlocked_stage
            self.steps_since_stage_change = 0
            self.lr_reductions_this_stage = 0
            self.lp_history.clear()
        self.lp_history.append(frontier_lp)
        if len(self.lp_history) > self.config.lr_plateau_window:
            self.lp_history.pop(0)
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
        self._update_lr(frontier_episodes)
    def _check_plateau(self, frontier_episodes: int) -> bool:
        cfg = self.config
        if self.update_count < cfg.lr_warmup_steps:
            return False
        if len(self.lp_history) < cfg.lr_plateau_window:
            return False
        if frontier_episodes < cfg.lr_plateau_min_episodes:
            return False
        if self.steps_since_stage_change < cfg.lr_plateau_stage_protection:
            return False
        if self.cooldown_counter > 0:
            return False
        if self.lr_reductions_this_stage >= cfg.lr_plateau_max_reductions:
            return False
        if self.current_lr <= cfg.lr_min:
            return False
        avg_lp = np.mean(self.lp_history)
        return avg_lp < cfg.lr_plateau_threshold
    def _update_lr(self, frontier_episodes: int = 0):
        cfg = self.config
        if self.update_count < cfg.lr_warmup_steps:
            warmup_factor = self.update_count / cfg.lr_warmup_steps
            lr = cfg.learning_rate * warmup_factor
            for pg in self.optimizer.param_groups:
                pg['lr'] = max(lr, 1e-7)
            return
        if self._check_plateau(frontier_episodes):
            self.current_lr = max(self.current_lr * cfg.lr_plateau_factor, cfg.lr_min)
            self.lr_reductions_this_stage += 1
            self.cooldown_counter = cfg.lr_plateau_cooldown
            self.lp_history.clear()
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.current_lr
    def save_checkpoint(self, path, extra=None):
        ckpt = {'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'update_count': self.update_count, 'nan_count': self.nan_count, 'si_omega': {k: v.cpu() for k, v in self.si_omega.items()}, 'si_prev_params': {k: v.cpu() for k, v in self.si_prev_params.items()}, 'feature_importance_history': dict(self.feature_importance_history), 'current_lr': self.current_lr, 'current_highest_stage': self.current_highest_stage, 'lr_reductions_this_stage': self.lr_reductions_this_stage, 'lp_history': list(self.lp_history), 'ema_weights': {k: v.cpu() for k, v in self.ema_weights.items()} if self.ema_enabled else None}
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.update_count = ckpt.get('update_count', 0)
        self.nan_count = ckpt.get('nan_count', 0)
        if 'si_omega' in ckpt:
            for name in self.si_omega:
                if name in ckpt['si_omega']:
                    self.si_omega[name] = ckpt['si_omega'][name].to(self.device)
        if 'si_prev_params' in ckpt:
            for name in self.si_prev_params:
                if name in ckpt['si_prev_params']:
                    self.si_prev_params[name] = ckpt['si_prev_params'][name].to(self.device)
        if 'feature_importance_history' in ckpt:
            self.feature_importance_history = defaultdict(list, ckpt['feature_importance_history'])
        self.current_lr = ckpt.get('current_lr', self.config.learning_rate)
        self.current_highest_stage = ckpt.get('current_highest_stage', 0)
        self.lr_reductions_this_stage = ckpt.get('lr_reductions_this_stage', 0)
        self.lp_history = list(ckpt.get('lp_history', []))
        if self.ema_enabled and 'ema_weights' in ckpt and ckpt['ema_weights'] is not None:
            for k, v in ckpt['ema_weights'].items():
                if k in self.ema_weights:
                    self.ema_weights[k] = v.to(self.device)
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.current_lr
        return ckpt
class IMPALATrainer:
    def __init__(self, config, model_config, resume_from=None):
        self.config, self.model_config, self.resume_from = config, model_config, resume_from
        self.log_dir, self.checkpoint_dir = Path(config.log_dir), Path(config.checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.baselines_path = self.checkpoint_dir / "baselines.json"
        self.calibrator = BaselineCalibrator(config.stages, self.baselines_path)
        self.learner, self.curriculum, self.workers, self.writer = None, None, [], None
        self.total_episodes, self.total_steps, self.start_time = 0, 0, None
        self.last_highest_unlocked, self.last_progress_step = 0, 0
        self.episode_returns_buffer = defaultdict(list)
        self.episode_wins_buffer = defaultdict(list)
        self.episode_lengths_buffer = defaultdict(list)
        self.episode_max_rewards_buffer = defaultdict(list)
        self.checkpoint_data = None
        self.saved_95_all = False
        self.saved_100_stages = set()
    def setup(self):
        print("=" * 60)
        print(f"IMPALA TRAINER - STAGES 0-9 (LAZY)")
        if self.resume_from:
            print(f"RESUMING FROM: {self.resume_from}")
        print("=" * 60)
        run_name = f"gfootball_stage9_{time.strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=self.log_dir / run_name)
        if not self.calibrator.load():
            self.calibrator.calibrate(num_episodes=100, num_workers=min(self.config.num_workers, 8))
        stages_dict = [asdict(s) for s in self.config.stages]
        baselines_dict = {k: v.to_dict() for k, v in self.calibrator.baselines.items()}
        self.learner = Learner(self.config, self.model_config, self.writer)
        curriculum_initial_state = None
        if self.resume_from:
            checkpoint_path = Path(self.resume_from)
            if checkpoint_path.exists():
                print(f"Loading: {checkpoint_path}")
                self.checkpoint_data = self.learner.load_checkpoint(checkpoint_path)
                self.total_steps = self.checkpoint_data.get('total_steps', 0)
                self.total_episodes = self.checkpoint_data.get('total_episodes', 0)
                curriculum_initial_state = self.checkpoint_data.get('curriculum_stats', None)
                self.saved_95_all = self.checkpoint_data.get('saved_95_all', False)
                self.saved_100_stages = set(self.checkpoint_data.get('saved_100_stages', []))
        self.curriculum = CurriculumController.remote(stages_dict, baselines_dict, final_target_win_rate=self.config.final_stage_target_win_rate, initial_state=curriculum_initial_state)
        self.workers = [SamplerWorker.remote(worker_id=i, model_config=self.model_config, stages=stages_dict, baselines=baselines_dict) for i in range(self.config.num_workers)]
        self._sync_weights()
        if curriculum_initial_state:
            self.last_highest_unlocked = len(curriculum_initial_state.get('learned_stages', []))
            self.last_progress_step = self.total_steps
    def _sync_weights(self):
        ray.get([w.set_weights.remote(self.learner.get_weights()) for w in self.workers])
    def _aggregate_episode_stats(self, trajectories):
        for traj in trajectories:
            for ret, won, length, stage_id in zip(traj.get('episode_returns', []), traj.get('episode_wins', []), traj.get('episode_lengths', []), traj.get('episode_stages', [])):
                self.episode_returns_buffer[stage_id].append(ret)
                self.episode_wins_buffer[stage_id].append(1.0 if won else 0.0)
                self.episode_lengths_buffer[stage_id].append(length)
            for stage_id, max_r in zip(traj.get('episode_stages', []), traj.get('episode_max_rewards', [])):
                self.episode_max_rewards_buffer[stage_id].append(max_r)
    def _check_special_saves(self, trajectories):
        for traj in trajectories:
            for save_info in traj.get('special_saves', []):
                if save_info.get('save_100') and save_info.get('stage_100') not in self.saved_100_stages:
                    stage = save_info['stage_100']
                    self.saved_100_stages.add(stage)
                    path = self.checkpoint_dir / f"checkpoint_100pct_stage{stage}.pt"
                    self.learner.save_checkpoint(path, extra={'total_steps': self.total_steps, 'total_episodes': self.total_episodes, 'curriculum_stats': ray.get(self.curriculum.get_stats.remote()), 'saved_95_all': self.saved_95_all, 'saved_100_stages': list(self.saved_100_stages), 'milestone': f'100% on stage {stage}'})
                    print(f"\n💾 SAVED: 100% checkpoint for Stage {stage} -> {path}\n")
                if save_info.get('save_95_all') and not self.saved_95_all:
                    self.saved_95_all = True
                    path = self.checkpoint_dir / f"checkpoint_95pct_all_stages.pt"
                    self.learner.save_checkpoint(path, extra={'total_steps': self.total_steps, 'total_episodes': self.total_episodes, 'curriculum_stats': ray.get(self.curriculum.get_stats.remote()), 'saved_95_all': self.saved_95_all, 'saved_100_stages': list(self.saved_100_stages), 'milestone': '95% on all stages'})
                    print(f"\n💾 SAVED: 95% ALL STAGES checkpoint -> {path}\n")
    def _log_episode_stats(self):
        if self.writer is None:
            return
        for stage_id in sorted(self.episode_returns_buffer.keys()):
            returns = self.episode_returns_buffer[stage_id]
            wins = self.episode_wins_buffer[stage_id]
            lengths = self.episode_lengths_buffer[stage_id]
            max_rewards = self.episode_max_rewards_buffer.get(stage_id, [])
            if returns:
                self.writer.add_scalar(f'episode/return_stage_{stage_id}', np.mean(returns), self.total_steps)
                self.writer.add_scalar(f'episode/win_rate_stage_{stage_id}', np.mean(wins), self.total_steps)
                self.writer.add_scalar(f'episode/length_stage_{stage_id}', np.mean(lengths), self.total_steps)
                if max_rewards:
                    self.writer.add_scalar(f'episode/max_reward_stage_{stage_id}', max(max_rewards), self.total_steps)
        self.episode_returns_buffer.clear()
        self.episode_wins_buffer.clear()
        self.episode_lengths_buffer.clear()
        self.episode_max_rewards_buffer.clear()
    def train(self):
        print("Starting training...")
        self.start_time = time.time()
        if self.last_progress_step == 0:
            self.last_progress_step = self.total_steps
        pending = {w.collect_trajectory.remote(self.config.trajectory_length, self.curriculum): w for w in self.workers}
        trajectories_buffer = []
        update_count = self.learner.update_count
        while True:
            curriculum_stats = ray.get(self.curriculum.get_stats.remote())
            if curriculum_stats.get('training_complete', False):
                print("\n🎉 TRAINING COMPLETE - 95% on Stage 9!")
                break
            if self.total_steps >= self.config.total_steps:
                print("\n⚠️ Max steps reached")
                break
            num_learned = len(curriculum_stats.get('learned_stages', []))
            if num_learned > self.last_highest_unlocked:
                self.last_highest_unlocked = num_learned
                self.last_progress_step = self.total_steps
            elif self.total_steps - self.last_progress_step > self.config.max_steps_without_progress:
                print(f"\n⚠️ No progress for {self.config.max_steps_without_progress:,} steps")
                break
            done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
            for ref in done_refs:
                worker = pending.pop(ref)
                try:
                    trajectory = ray.get(ref)
                    trajectories_buffer.append(trajectory)
                    self.total_steps += int(len(trajectory['obs']) * (trajectory['agent_masks'].sum() / len(trajectory['obs'])))
                    self.total_episodes += len(trajectory['episode_returns'])
                    self._aggregate_episode_stats([trajectory])
                    self._check_special_saves([trajectory])
                except Exception as e:
                    print(f"Worker error: {e}")
                pending[worker.collect_trajectory.remote(self.config.trajectory_length, self.curriculum)] = worker
            if sum(len(t['obs']) * t['agent_masks'].sum() / len(t['obs']) for t in trajectories_buffer) >= self.config.batch_size:
                curriculum_stats = ray.get(self.curriculum.get_stats.remote())
                highest_unlocked = curriculum_stats.get('highest_unlocked', 0)
                stage_stats = curriculum_stats.get('stage_stats', {})
                frontier_stats = stage_stats.get(str(highest_unlocked), {})
                frontier_lp = frontier_stats.get('learning_progress', 0.0)
                frontier_episodes = frontier_stats.get('episodes', 0)
                self.learner.update_curriculum_state(highest_unlocked, frontier_lp, frontier_episodes)
                stats = self.learner.update(trajectories_buffer, global_step=self.total_steps)
                trajectories_buffer = []
                update_count += 1
                if update_count % self.config.weight_sync_interval == 0:
                    self._sync_weights()
                if update_count % self.config.log_interval == 0:
                    self._log_progress(update_count, stats)
                    self._log_episode_stats()
                if update_count % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(update_count)
        self._save_checkpoint(update_count, final=True)
        print("\n=== FINAL STATUS ===")
        print(ray.get(self.curriculum.get_progress_summary.remote()))
        if self.writer:
            self.writer.close()
    def _log_progress(self, update_count, stats):
        elapsed = time.time() - self.start_time
        sps = self.total_steps / elapsed if elapsed > 0 else 0
        curriculum_stats = ray.get(self.curriculum.get_stats.remote())
        learned = curriculum_stats.get('learned_stages', [])
        mastered = curriculum_stats.get('mastered_stages', [])
        stage_stats = curriculum_stats.get('stage_stats', {})
        sample_probs = curriculum_stats.get('sample_probs', {})
        lr = self.learner.current_lr
        print(f"[{update_count}] {self.total_steps / 1e6:.1f}M | {sps / 1e3:.0f}k sps | LR:{lr:.1e} | Loss:{stats.get('loss/total', 0):.3f}")
        print(f"Mastered: {sorted(mastered)} | Learned: {sorted(learned)}")
        if stage_stats:
            frontier = []
            for sid in sorted(curriculum_stats.get('unlocked_stages', [0])):
                if str(sid) in stage_stats and stage_stats[str(sid)]['episodes'] > 0:
                    s = stage_stats[str(sid)]
                    wr = s.get('recent_win_rate', s['ema_win'])
                    marker = "⭐" if sid in mastered else ("📚" if sid in learned else "")
                    frontier.append(f"S{sid}:{wr:.0%}{marker}")
            print(f"{' | '.join(frontier)}")
        if self.writer:
            self.writer.add_scalar('train/lr', lr, self.total_steps)
    def _save_checkpoint(self, update_count, final=False):
        path = self.checkpoint_dir / f"checkpoint_{'final' if final else f'update_{update_count}'}.pt"
        self.learner.save_checkpoint(path, extra={'total_steps': self.total_steps, 'total_episodes': self.total_episodes, 'curriculum_stats': ray.get(self.curriculum.get_stats.remote()), 'saved_95_all': self.saved_95_all, 'saved_100_stages': list(self.saved_100_stages)})
        print(f"Saved: {path}")
    def close(self):
        if self.writer:
            self.writer.close()
        for w in self.workers:
            try:
                ray.get(w.close.remote())
            except:
                pass
        ray.shutdown()
def main():
    RESUME_FROM = None
    CHECKPOINT_DIR, LOG_DIR, NUM_WORKERS = "./checkpoints_stage9", "./logs_stage9", 24
    model_config = {
        'obs_dim': OBS_DIM,
        'feature_dim': FEATURE_DIM,
        'd_model': 128,
        'mamba_d_state': 64,
        'mamba_layers': 4,
        'encoder_hidden': [128],
        'policy_hidden': [128],
        'value_hidden': [128],
        'use_distributional': True,
        'dropout': 0.0,
        'num_stages': 10,
        'segment_size': 4,
    }
    config = TrainingConfig(
        stages=get_default_stages(),
        final_stage_target_win_rate=0.95,
        max_steps_without_progress=100_000_000,
        num_workers=NUM_WORKERS,
        trajectory_length=128,
        batch_size=1028,
        minibatch_size=128,
        num_epochs=2,
        learning_rate=3e-4,
        lr_schedule="constant",
        gamma=1.0,
        rho_bar=1.0,
        c_bar=1.0,
        entropy_coeff=0.01,
        value_coeff=0.5,
        si_lambda=1.0,
        max_grad_norm=0.5,
        total_steps=1_000_000_000,
        log_interval=10,
        checkpoint_interval=100,
        feature_importance_interval=50,
        weight_sync_interval=25,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR
    )
    trainer = IMPALATrainer(config, model_config, resume_from=RESUME_FROM)
    try:
        trainer.setup()
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        trainer.close()
if __name__ == "__main__":
    main()