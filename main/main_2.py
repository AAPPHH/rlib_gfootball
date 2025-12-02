import math
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
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
    'ball_x', 'ball_y', 'ball_z', 'ball_owned_team',
    'ball_speed', 'ball_dir_x', 'ball_dir_y', 'moving_to_goal',
    'dist_to_goal', 'goal_angle', 'shooting_angle', 'in_shooting_range',
    'keeper_dist', 'keeper_angle_to_ball',
    'teammates_ahead', 'defenders_ahead', 'numerical_advantage',
    'in_attack_third', 'in_defense_third', 'on_wing',
    'shooting_opportunity', 'attack_momentum',
    'sticky_sprint', 'sticky_dribble', 'sticky_dir_x', 'sticky_dir_y',
    'dist_to_own_goal',
    'nearest_opponent_dist', 'space_ahead', 'free_teammates',
    'offside_line_x', 'is_offside',
    'tiredness',
]

FEATURE_GROUPS = {
    'ball_position': [0, 1, 2, 3], 'ball_movement': [4, 5, 6, 7],
    'goal_threat': [8, 9, 10, 11], 'keeper': [12, 13],
    'team_structure': [14, 15, 16], 'zones': [17, 18, 19],
    'composite': [20, 21], 'sticky': [22, 23, 24, 25],
    'defense': [26], 'pressure_space': [27, 28, 29],
    'offside': [30, 31], 'player_state': [32],
}

FEATURE_DIM = 33
OBS_DIM = 460

class FeatureEngineer:
    FEATURE_DIM = 33
    GOAL_POS = np.array([1.0, 0.0], dtype=np.float32)
    OWN_GOAL_POS = np.array([-1.0, 0.0], dtype=np.float32)
    
    @staticmethod
    def extract_features(obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        B = obs.shape[0]
        obs115 = obs[:, :115] if obs.shape[1] >= 115 else np.pad(obs, ((0,0), (0, 115-obs.shape[1])))
        feat = np.zeros((B, FEATURE_DIM), dtype=np.float32)
        ball_pos = obs115[:, 88:90]
        bx, by = ball_pos[:, 0], ball_pos[:, 1]
        bz = obs115[:, 90] if obs115.shape[1] > 90 else np.zeros(B)
        bot = obs115[:, 108] if obs115.shape[1] > 108 else np.zeros(B)
        feat[:, 0], feat[:, 1], feat[:, 2], feat[:, 3] = bx, by, np.clip(bz, 0, 1), bot
        bdir = obs115[:, 91:93] if obs115.shape[1] > 92 else np.zeros((B, 2))
        bspd = np.linalg.norm(bdir, axis=1)
        feat[:, 4] = np.clip(bspd, 0, 2)
        feat[:, 5], feat[:, 6] = bdir[:, 0], bdir[:, 1]
        feat[:, 7] = np.clip(bdir[:, 0], -1, 1)
        dg = np.linalg.norm(ball_pos - FeatureEngineer.GOAL_POS, axis=1)
        gv = FeatureEngineer.GOAL_POS - ball_pos
        ga = np.abs(np.arctan2(gv[:, 1], gv[:, 0]))
        pt, pb = np.array([1.0, 0.044]), np.array([1.0, -0.044])
        vt, vb = pt - ball_pos, pb - ball_pos
        sa = np.abs(np.arctan2(vt[:, 1], vt[:, 0]) - np.arctan2(vb[:, 1], vb[:, 0]))
        feat[:, 8] = np.clip(dg, 0, 2)
        feat[:, 9] = ga / np.pi
        feat[:, 10] = sa / np.pi
        feat[:, 11] = (dg < 0.35).astype(np.float32)
        rx, ry = obs115[:, 44:66:2], obs115[:, 45:66:2]
        rte = np.any(np.abs(rx) > 0.01, axis=1)
        ki = np.argmax(rx, axis=1)
        kx, ky = rx[np.arange(B), ki], ry[np.arange(B), ki]
        kpos = np.stack([kx, ky], axis=1)
        kd = np.linalg.norm(ball_pos - kpos, axis=1)
        ktb = ball_pos - kpos
        ka = np.arctan2(ktb[:, 1], ktb[:, 0])
        kv = rte & (kx > 0.7)
        feat[:, 12] = np.clip(np.where(kv, kd, 1.0), 0, 2)
        feat[:, 13] = np.where(kv, ka, 0.0) / np.pi
        lx, ly = obs115[:, 0:22:2], obs115[:, 1:22:2]
        feat[:, 14] = np.sum(lx > bx[:, None], axis=1) / 11.0
        feat[:, 15] = np.sum(rx > bx[:, None], axis=1) / 11.0
        na = np.sum(lx > bx[:, None], axis=1) - np.sum(rx > bx[:, None], axis=1)
        feat[:, 16] = np.clip(na / 5.0, -1, 1)
        feat[:, 17] = (bx > 0.33).astype(np.float32)
        feat[:, 18] = (bx < -0.33).astype(np.float32)
        feat[:, 19] = (np.abs(by) > 0.25).astype(np.float32)
        feat[:, 20] = sa * np.clip(np.where(kv, kd, 1.0), 0, 1)
        feat[:, 21] = np.clip(bdir[:, 0] * bspd, -1, 1)
        stk = obs115[:, 96:106] if obs115.shape[1] > 105 else np.zeros((B, 10))
        feat[:, 22] = stk[:, 8]
        feat[:, 23] = stk[:, 9]
        ds = stk[:, :8]
        di = np.argmax(ds, axis=1)
        da = np.any(ds > 0, axis=1)
        ang = di * (2 * np.pi / 8)
        feat[:, 24] = np.where(da, np.cos(ang), 0)
        feat[:, 25] = np.where(da, np.sin(ang), 0)
        feat[:, 26] = np.clip(np.linalg.norm(ball_pos - FeatureEngineer.OWN_GOAL_POS, axis=1), 0, 2)
        rpos = np.stack([rx, ry], axis=2)
        bpe = ball_pos[:, None, :]
        od = np.linalg.norm(rpos - bpe, axis=2)
        oa = np.abs(rx) > 0.01
        od = np.where(oa, od, 10.0)
        feat[:, 27] = np.clip(np.min(od, axis=1), 0, 1)
        oam = (rx > bx[:, None]) & oa
        oad = np.where(oam, od, 10.0)
        sah = np.min(oad, axis=1)
        feat[:, 28] = np.clip(np.where(np.any(oam, axis=1), sah, 1.0), 0, 1)
        lpos = np.stack([lx, ly], axis=2)
        le, re = lpos[:, :, None, :], rpos[:, None, :, :]
        pwd = np.linalg.norm(le - re, axis=3)
        la = np.abs(lx) > 0.01
        mod = np.min(pwd, axis=2)
        fm = (mod > 0.15) & la
        feat[:, 29] = np.sum(fm, axis=1) / 11.0
        srx = np.sort(rx, axis=1)
        sld = srx[:, 1]
        ofl = np.maximum(bx, sld)
        feat[:, 30] = ofl
        api = obs115[:, 107].astype(int) if obs115.shape[1] > 107 else np.zeros(B, dtype=int)
        api = np.clip(api, 0, 10)
        apx = lx[np.arange(B), api]
        ios = (apx > ofl) & (bot == 1)
        feat[:, 31] = ios.astype(np.float32)
        feat[:, 32] = stk[:, 8] * 0.1
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
    num_stages: int = 14
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
                if seg_start > 0:
                    seg_out, h = checkpoint.checkpoint(
                        self._process_segment, x_in_seg, z_seg, dt_seg, B_t_seg, C_t_seg, h,
                        use_reentrant=False
                    )
                else:
                    seg_out, h = self._process_segment(x_in_seg, z_seg, dt_seg, B_t_seg, C_t_seg, h)
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
        self.mamba = MambaEncoder(
            input_dim=in_dim, d_model=config.d_model, d_state=config.mamba_d_state,
            num_layers=config.mamba_layers, dropout=config.dropout, segment_size=config.segment_size
        )
        policy_layers = []
        in_dim = self.mamba.output_dim
        for hidden_dim in config.policy_hidden:
            policy_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        policy_layers.append(nn.Linear(in_dim, config.num_actions))
        self.policy_head = nn.Sequential(*policy_layers)
        self.num_value_heads = 5
        value_hidden = config.value_hidden[0] if config.value_hidden else 256
        self.value_fc1 = nn.Linear(self.mamba.output_dim, value_hidden * self.num_value_heads)
        if config.use_distributional:
            self.value_fc2 = nn.Linear(value_hidden * self.num_value_heads, config.num_atoms * self.num_value_heads)
            self.register_buffer('value_support', torch.linspace(config.v_min, config.v_max, config.num_atoms))
            self.value_out_dim = config.num_atoms
        else:
            self.value_fc2 = nn.Linear(value_hidden * self.num_value_heads, self.num_value_heads)
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
        nn.init.orthogonal_(self.value_fc1.weight, gain=math.sqrt(2))
        nn.init.orthogonal_(self.value_fc2.weight, gain=1.0)

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

    def forward(self, obs, features, stage_idx, prev_action=None, hidden_state=None, return_hidden=False):
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
        with torch.amp.autocast('cuda', enabled=use_amp):
            stage_emb = self.stage_embedding(stage_idx)
            action_emb = self.action_embedding(prev_action)
            x = torch.cat([obs, features, stage_emb, action_emb], dim=-1)
            x = self.obs_encoder(x)
        x = x.float()
        x, new_hidden = self.mamba(x, hidden_state)
        with torch.amp.autocast('cuda', enabled=use_amp):
            x_amp = x
            logits = self.policy_head(x_amp)
            v = F.relu(self.value_fc1(x_amp))
            v = self.value_fc2(v)
            Bs, Ls = v.shape[:2]
            v = v.view(Bs, Ls, self.num_value_heads, self.value_out_dim)
            if self.config.use_distributional:
                value_logits = v.mean(dim=2)
                value_probs = F.softmax(value_logits, dim=-1)
                value = (value_probs * self.value_support).sum(-1, keepdim=True)
            else:
                value = v.mean(dim=2)
                value_logits = None
            log_probs = F.log_softmax(logits, dim=-1)
        logits, value, log_probs = logits.float(), value.float(), log_probs.float()
        if squeeze:
            logits, value, log_probs = logits.squeeze(1), value.squeeze(1), log_probs.squeeze(1)
            if value_logits is not None:
                value_logits = value_logits.squeeze(1).float()
        result = {'logits': logits, 'value': value.squeeze(-1) if value.dim() > 1 else value, 'log_probs': log_probs}
        if value_logits is not None:
            result['value_logits'] = value_logits
        if return_hidden:
            result['hidden_state'] = new_hidden
        return result

    def get_action(self, obs, features, stage_idx, prev_action=None, hidden_state=None, deterministic=False):
        output = self.forward(obs, features, stage_idx, prev_action, hidden_state, return_hidden=True)
        logits = output['logits']
        if torch.isnan(logits).any():
            logits = torch.zeros_like(logits)
        logits = logits.clamp(min=-20.0, max=20.0)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), output['value'], output.get('hidden_state')

    def evaluate_actions(self, obs, features, stage_idx, actions, prev_action=None, hidden_state=None):
        output = self.forward(obs, features, stage_idx, prev_action, hidden_state)
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
        StageConfig(10, "academy_single_goal_versus_lazy", "simple115v2", 11, 0, 1000, "scoring"),
        StageConfig(11, "11_vs_11_easy_stochastic", "simple115v2", 11, 0, 3000, "scoring"),
        StageConfig(12, "11_vs_11_stochastic", "simple115v2", 11, 0, 3000, "scoring"),
        StageConfig(13, "11_vs_11_hard_stochastic", "simple115v2", 11, 0, 3000, "scoring"),
    ]

@dataclass
class TrainingConfig:
    stages: List[StageConfig] = field(default_factory=list)
    final_stage_target_win_rate: float = 0.5
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
    env = football_env.create_environment(
        env_name=stage.env_name, representation=stage.representation,
        number_of_left_players_agent_controls=stage.left_agents,
        number_of_right_players_agent_controls=stage.right_agents,
        rewards=stage.rewards, write_goal_dumps=False, write_full_episode_dumps=False,
        render=False, write_video=False
    )
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
        futures = [_calibrate_stage_batch.remote(asdict(stage), episodes_per_worker, worker_id)
                   for stage in self.stages for worker_id in range(num_workers)]
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

    def __init__(self, stages, baselines, final_target_win_rate=0.5, initial_state=None):
        self.stages = [StageConfig(**s) if isinstance(s, dict) else s for s in stages]
        self.baselines = {int(k): StageBaseline.from_dict(v) if isinstance(v, dict) else v for k, v in baselines.items()}
        self.num_stages = len(self.stages)
        self.final_target_win_rate = final_target_win_rate
        self._restore_state(initial_state) if initial_state else self._init_fresh()

    def _init_fresh(self):
        self.episode_count, self.unlocked_stages, self.learned_stages, self.mastered_stages = 0, {0}, set(), set()
        self.stage_stats = {s.stage_id: {'episodes': 0, 'ema_return': 0.0, 'ema_win': 0.0, 'ema_win_slow': 0.0,
                                         'sustained_peak': 0.0, 'max_reward': -float('inf'), 'recent_wins': [],
                                         'recent_returns': [], 'last_sampled': 0, 'lp_history': []} for s in self.stages}

    def _restore_state(self, state):
        self.episode_count = state.get('episode_count', 0)
        self.unlocked_stages = set(state.get('unlocked_stages', [0]))
        self.learned_stages = set(state.get('learned_stages', []))
        self.mastered_stages = set(state.get('mastered_stages', []))
        saved_stats = state.get('stage_stats', {})
        self.stage_stats = {}
        for s in self.stages:
            sid_str = str(s.stage_id)
            if sid_str in saved_stats:
                ss = saved_stats[sid_str]
                self.stage_stats[s.stage_id] = {
                    'episodes': ss.get('episodes', 0), 'ema_return': ss.get('ema_return', 0.0),
                    'ema_win': ss.get('ema_win', 0.0), 'ema_win_slow': ss.get('ema_win_slow', ss.get('ema_win', 0.0)),
                    'sustained_peak': ss.get('sustained_peak', ss.get('peak_win', 0.0)),
                    'max_reward': ss.get('max_reward', -float('inf')), 'recent_wins': [], 'recent_returns': [],
                    'last_sampled': ss.get('last_sampled', 0), 'lp_history': []
                }
            else:
                self.stage_stats[s.stage_id] = {'episodes': 0, 'ema_return': 0.0, 'ema_win': 0.0, 'ema_win_slow': 0.0,
                                                'sustained_peak': 0.0, 'max_reward': -float('inf'), 'recent_wins': [],
                                                'recent_returns': [], 'last_sampled': 0, 'lp_history': []}
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
        return {
            'episode_count': self.episode_count, 'unlocked_stages': list(self.unlocked_stages),
            'learned_stages': list(self.learned_stages), 'mastered_stages': list(self.mastered_stages),
            'highest_unlocked': max(self.unlocked_stages) if self.unlocked_stages else 0,
            'training_complete': self.is_training_complete(),
            'sample_probs': {sid: w / total_w for sid, w in weights.items()},
            'stage_stats': {str(sid): {
                'episodes': s['episodes'], 'ema_return': s['ema_return'], 'ema_win': s['ema_win'],
                'ema_win_slow': s['ema_win_slow'], 'sustained_peak': s['sustained_peak'],
                'max_reward': s['max_reward'] if s['max_reward'] > -float('inf') else 0,
                'learning_progress': self._compute_learning_progress(sid),
                'forgetting': self._compute_forgetting(sid),
                'recent_win_rate': np.mean(s['recent_wins']) if s['recent_wins'] else 0
            } for sid, s in self.stage_stats.items()}
        }

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
        value_list, log_prob_list, stage_list, mask_list = [], [], [], []
        episode_returns, episode_wins, episode_lengths, episode_stages, episode_max_rewards = [], [], [], [], []
        steps, ep_max_reward = 0, -float('inf')
        while steps < trajectory_length:
            if self.env is None or self._should_switch_stage():
                self._setup_env(StageConfig(**ray.get(curriculum_controller.get_stage.remote())))
                ep_max_reward = -float('inf')
            num_agents = self.current_stage.left_agents
            stage_tensor = torch.tensor(self.current_stage.stage_id, device=self.device)
            if num_agents == 1:
                obs_tensor = torch.from_numpy(self.current_obs[0]).float().to(self.device)
                feature_tensor = torch.from_numpy(self.current_features[0]).float().to(self.device)
                with torch.no_grad():
                    action, log_prob, value, self.hidden_state = self.model.get_action(
                        obs_tensor, feature_tensor, stage_tensor, prev_action=self.prev_action, hidden_state=self.hidden_state
                    )
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
                with torch.no_grad():
                    actions, log_probs, values, _ = self.model.get_action(
                        obs_batch, feat_batch, stage_tensor, prev_action=None, hidden_state=None
                    )
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
            self.prev_action = torch.tensor(action_full[0], device=self.device)
            steps += 1
            if episode_done:
                won = info["score"][0] > info["score"][1] if isinstance(info, dict) and "score" in info else self.episode_return > 0
                episode_returns.append(self.episode_return)
                episode_wins.append(won)
                episode_lengths.append(self.episode_steps)
                episode_stages.append(self.current_stage.stage_id)
                episode_max_rewards.append(ep_max_reward)
                ray.get(curriculum_controller.report_episode.remote(self.current_stage.stage_id, self.episode_return, won))
                self._reset_episode()
                ep_max_reward = -float('inf')
        return {
            'obs': np.array(obs_list, dtype=np.float32), 'features': np.array(feature_list, dtype=np.float32),
            'actions': np.array(action_list, dtype=np.int64), 'rewards': np.array(reward_list, dtype=np.float32),
            'dones': np.array(done_list, dtype=np.float32), 'values': np.array(value_list, dtype=np.float32),
            'log_probs': np.array(log_prob_list, dtype=np.float32), 'stage_ids': np.array(stage_list, dtype=np.int64),
            'agent_masks': np.array(mask_list, dtype=np.float32), 'worker_id': self.worker_id,
            'episode_returns': episode_returns, 'episode_wins': episode_wins, 'episode_lengths': episode_lengths,
            'episode_stages': episode_stages, 'episode_max_rewards': episode_max_rewards
        }

    def _setup_env(self, stage):
        if self.env is not None:
            self.env.close()
        self.current_stage = stage
        self.env = football_env.create_environment(
            env_name=stage.env_name, representation=stage.representation,
            number_of_left_players_agent_controls=stage.left_agents,
            number_of_right_players_agent_controls=stage.right_agents,
            stacked=True, rewards=stage.rewards, write_goal_dumps=False,
            write_full_episode_dumps=False, render=False, write_video=False
        )
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
        print(f"  V-Trace: rho_bar={config.rho_bar}, c_bar={config.c_bar}")
        print(f"  EMA: decay={self.ema_decay}")
        print(f"  Segment Checkpointing: {model_config.get('segment_size', 16)} steps")
        print(f"  Mixed Precision: encoder/heads=fp16, mamba=fp32")
    
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
        log_probs, _, values = self.model.evaluate_actions(obs, features, stage_ids, actions)
        loss = -log_probs.mean() + values.mean()
        loss.backward()
        grad = features.grad.abs().mean(dim=0).cpu().numpy()
        feat_var = features.detach().var(dim=0).cpu().numpy()
        VAR_THRESHOLD = 1e-4
        active_mask = feat_var > VAR_THRESHOLD
        importance = np.where(active_mask, grad, 0.0)
        if importance.max() > 0:
            importance = importance / importance.max()
        unique_stages = stage_ids.unique().cpu().numpy()
        stage_importance = {}
        for stage in unique_stages:
            stage_mask = stage_ids == stage
            if stage_mask.sum() < 10:
                continue
            stage_features = features[stage_mask].detach()
            stage_grad = features.grad[stage_mask].abs().mean(dim=0).cpu().numpy()
            stage_var = stage_features.var(dim=0).cpu().numpy()
            stage_active = stage_var > VAR_THRESHOLD
            stage_imp = np.where(stage_active, stage_grad, 0.0)
            if stage_imp.max() > 0:
                stage_imp = stage_imp / stage_imp.max()
            stage_importance[int(stage)] = stage_imp
        self.model.train()
        n_active = active_mask.sum()
        if self.update_count % 100 == 0:
            active_names = [FEATURE_NAMES[i] for i in range(len(active_mask)) if active_mask[i]]
            inactive_names = [FEATURE_NAMES[i] for i in range(len(active_mask)) if not active_mask[i]]
            print(f"Active features ({n_active}/{FEATURE_DIM}): {', '.join(active_names[:10])}...")
            if inactive_names:
                print(f"Inactive (var<{VAR_THRESHOLD}): {', '.join(inactive_names)}")
        return importance, stage_importance

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
        if stage_importance:
            for stage, imp in stage_importance.items():
                stage_str = f'stage_{stage:02d}'
                for i, val in enumerate(imp):
                    if i < len(FEATURE_NAMES):
                        self.writer.add_scalar(f'feature_importance/{stage_str}/{FEATURE_NAMES[i]}', val, self.update_count)
                for group_name, indices in FEATURE_GROUPS.items():
                    group_imp = np.mean([imp[i] for i in indices if i < len(imp)])
                    self.writer.add_scalar(f'feature_importance/{stage_str}_groups/{group_name}', group_imp, self.update_count)

    def _compute_vtrace_batch(self, trajectories):
        gamma, rho_bar, c_bar = self.config.gamma, self.config.rho_bar, self.config.c_bar
        all_obs, all_features, all_actions = [], [], []
        all_rewards, all_dones, all_behavior_log_probs, all_stage_ids = [], [], [], []
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
        with torch.no_grad():
            target_log_probs, _, values = self.model.evaluate_actions(obs, features, stage_ids, actions)
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
            vtrace_targets[start:end] = parallel_vtrace_scan(
                traj_values, traj_rewards, traj_dones, traj_rho, traj_c, gamma, bootstrap
            )
            vs_plus_one = torch.cat([vtrace_targets[start+1:end], bootstrap.unsqueeze(0)])
            pg_advantages[start:end] = traj_rho * (traj_rewards + gamma * (1 - traj_dones) * vs_plus_one - traj_values)
        return {
            'obs': obs, 'features': features, 'actions': actions, 'stage_ids': stage_ids,
            'advantages': pg_advantages, 'vtrace_targets': vtrace_targets, 'rho': rho,
        }

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
        
        # Pre-compute normalized advantages once
        advantages = batch['advantages']
        vtrace_targets = batch['vtrace_targets']
        rho = batch['rho']
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std
        
        # Pre-generate all minibatch indices for all epochs
        batch_size = len(advantages)
        all_indices = []
        for epoch in range(self.config.num_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.config.minibatch_size):
                all_indices.append(perm[start:start + self.config.minibatch_size])
        
        total_loss, policy_loss_sum, value_loss_sum, entropy_sum = 0.0, 0.0, 0.0, 0.0
        rho_sum, num_updates, skipped = 0.0, 0, 0
        grad_norm = torch.tensor(0.0)
        
        # Single loop over all minibatches
        for mb_idx in all_indices:
            mb_obs = batch['obs'][mb_idx]
            mb_features = batch['features'][mb_idx]
            mb_actions = batch['actions'][mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_vtrace_targets = vtrace_targets[mb_idx]
            mb_stage_ids = batch['stage_ids'][mb_idx]
            mb_rho = rho[mb_idx]
            
            try:
                log_probs, entropy, values = self.model.evaluate_actions(mb_obs, mb_features, mb_stage_ids, mb_actions)
                if torch.isnan(log_probs).any() or torch.isnan(values).any():
                    self.nan_count += 1
                    skipped += 1
                    continue
                policy_loss = -(log_probs * mb_advantages.detach()).mean()
                value_loss = F.mse_loss(values, mb_vtrace_targets.detach())
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.config.value_coeff * value_loss + self.config.entropy_coeff * entropy_loss
                if self.si_lambda > 0:
                    si_loss = sum((self.si_omega[n] * (p - self.si_prev_params[n]).pow(2)).sum()
                                  for n, p in self.model.named_parameters() if p.requires_grad and n in self.si_omega)
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
            
            # SI update (less frequent)
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
        
        # Skip expensive explained variance computation most of the time
        if self.update_count % 10 == 0:
            with torch.no_grad():
                var_y = vtrace_targets.var()
                sample_idx = torch.randperm(len(vtrace_targets))[:min(2048, len(vtrace_targets))]
                _, _, sample_values = self.model.evaluate_actions(
                    batch['obs'][sample_idx], batch['features'][sample_idx], 
                    batch['stage_ids'][sample_idx], batch['actions'][sample_idx]
                )
                explained_var = float(1 - (vtrace_targets[sample_idx] - sample_values.float()).var() / var_y if var_y > 1e-6 else 0.0)
        else:
            explained_var = 0.0
        
        return {
            'loss/total': total_loss / num_updates, 'loss/policy': policy_loss_sum / num_updates,
            'loss/value': value_loss_sum / num_updates, 'loss/entropy': entropy_sum / num_updates,
            'vtrace/rho_mean': rho_sum / num_updates, 'vtrace/rho_max': rho.max().item(),
            'vtrace/explained_variance': explained_var,
            'train/lr': self.optimizer.param_groups[0]['lr'], 'train/nan_count': self.nan_count,
            'train/grad_norm': float(grad_norm)
        }

    def _consolidate_si(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.si_omega:
                delta = param.data - self.si_prev_params[name]
                self.si_omega[name] = torch.clamp(
                    self.si_omega[name] + self.si_running_sum[name] / (delta.pow(2) + self.si_epsilon), min=0.0
                )
                self.si_prev_params[name] = param.data.clone()
                self.si_running_sum[name].zero_()

    def update_curriculum_state(self, highest_unlocked_stage: int, frontier_lp: float, frontier_episodes: int):
        self.steps_since_stage_change += 1
        if highest_unlocked_stage > self.current_highest_stage:
            self.current_highest_stage = highest_unlocked_stage
            self.steps_since_stage_change = 0
            self.lr_reductions_this_stage = 0
            self.lp_history.clear()
            print(f"\n🔓 Stage {highest_unlocked_stage} unlocked - LR bleibt bei {self.current_lr:.2e}, Plateau-Counter reset\n")
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
            old_lr = self.current_lr
            self.current_lr = max(self.current_lr * cfg.lr_plateau_factor, cfg.lr_min)
            self.lr_reductions_this_stage += 1
            self.cooldown_counter = cfg.lr_plateau_cooldown
            avg_lp = np.mean(self.lp_history) if self.lp_history else 0
            print(f"\n📉 LR PLATEAU DETECTED!")
            print(f"   LP avg: {avg_lp:.4f} < {cfg.lr_plateau_threshold}")
            print(f"   LR: {old_lr:.2e} → {self.current_lr:.2e}")
            print(f"   Reductions this stage: {self.lr_reductions_this_stage}/{cfg.lr_plateau_max_reductions}")
            print(f"   Cooldown: {cfg.lr_plateau_cooldown} updates\n")
            self.lp_history.clear()
            if self.writer:
                self.writer.add_scalar('train/lr_reduction', 1.0, self.update_count)
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.current_lr

    def save_checkpoint(self, path, extra=None):
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'nan_count': self.nan_count,
            'si_omega': {k: v.cpu() for k, v in self.si_omega.items()},
            'si_prev_params': {k: v.cpu() for k, v in self.si_prev_params.items()},
            'feature_importance_history': dict(self.feature_importance_history),
            'current_lr': self.current_lr,
            'current_highest_stage': self.current_highest_stage,
            'lr_reductions_this_stage': self.lr_reductions_this_stage,
            'lp_history': list(self.lp_history),
            'ema_weights': {k: v.cpu() for k, v in self.ema_weights.items()} if self.ema_enabled else None,
        }
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
        print(f"  Loaded LR state: lr={self.current_lr:.2e}, stage={self.current_highest_stage}, reductions={self.lr_reductions_this_stage}")
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

    def setup(self):
        print("=" * 60)
        print("IMPALA TRAINER - V-TRACE + 33 FEATURES")
        if self.resume_from:
            print(f"RESUMING FROM: {self.resume_from}")
        print("=" * 60)
        run_name = f"gfootball_vtrace_{time.strftime('%Y%m%d_%H%M%S')}"
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
        self.curriculum = CurriculumController.remote(
            stages_dict, baselines_dict, final_target_win_rate=self.config.final_stage_target_win_rate,
            initial_state=curriculum_initial_state
        )
        self.workers = [
            SamplerWorker.remote(worker_id=i, model_config=self.model_config, stages=stages_dict, baselines=baselines_dict)
            for i in range(self.config.num_workers)
        ]
        self._sync_weights()
        if curriculum_initial_state:
            self.last_highest_unlocked = len(curriculum_initial_state.get('learned_stages', []))
            self.last_progress_step = self.total_steps

    def _sync_weights(self):
        ray.get([w.set_weights.remote(self.learner.get_weights()) for w in self.workers])

    def _aggregate_episode_stats(self, trajectories):
        for traj in trajectories:
            for ret, won, length, stage_id in zip(
                traj.get('episode_returns', []), traj.get('episode_wins', []),
                traj.get('episode_lengths', []), traj.get('episode_stages', [])
            ):
                self.episode_returns_buffer[stage_id].append(ret)
                self.episode_wins_buffer[stage_id].append(1.0 if won else 0.0)
                self.episode_lengths_buffer[stage_id].append(length)
            for stage_id, max_r in zip(traj.get('episode_stages', []), traj.get('episode_max_rewards', [])):
                self.episode_max_rewards_buffer[stage_id].append(max_r)

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
                print("\n🎉 TRAINING COMPLETE!")
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
        def fmt_range(stages):
            if not stages:
                return "∅"
            stages = sorted(stages)
            return f"{stages[0]}-{stages[-1]}" if stages == list(range(stages[0], stages[-1] + 1)) else ",".join(map(str, stages))
        lr = self.learner.current_lr
        lr_info = f"LR:{lr:.1e}"
        if self.learner.lr_reductions_this_stage > 0:
            lr_info += f"(↓{self.learner.lr_reductions_this_stage})"
        if self.learner.cooldown_counter > 0:
            lr_info += f"[cd:{self.learner.cooldown_counter}]"
        rho_info = f"ρ:{stats.get('vtrace/rho_mean', 1.0):.2f}/{stats.get('vtrace/rho_max', 1.0):.1f}"
        print(f"[{update_count}] {self.total_steps / 1e6:.1f}M | {sps / 1e3:.0f}k sps | {lr_info} | {rho_info} | Loss:{stats.get('loss/total', 0):.3f}")
        print(f"Mastered: {fmt_range(mastered)} | Learned: {fmt_range(learned)}")
        if sample_probs:
            top_probs = sorted(sample_probs.items(), key=lambda x: -x[1])[:5]
            print(f"Sample: {' '.join([f'S{sid}:{p:.0%}' for sid, p in top_probs if p > 0.01])}")
        if stage_stats:
            frontier = []
            for sid in sorted(curriculum_stats.get('unlocked_stages', [0])):
                if str(sid) in stage_stats and stage_stats[str(sid)]['episodes'] > 0:
                    s = stage_stats[str(sid)]
                    wr = s.get('recent_win_rate', s['ema_win'])
                    lp = s.get('learning_progress', 0)
                    peak = s.get('sustained_peak', 0)
                    marker = "⭐" if sid in mastered else ("📚" if sid in learned else "")
                    delta = int((peak - wr) * 100)
                    lp_str = f"lp={lp:.3f}" if sid == self.learner.current_highest_stage else ""
                    r_val = s.get('max_reward', 0)
                    frontier.append(f"S{sid}:{wr:.0%}{marker}{'(↓' + str(delta) + ')' if delta > 5 else ''} r={r_val:.0f} {lp_str}".strip())
            print(f"{' | '.join(frontier)}")
        if self.writer:
            self.writer.add_scalar('train/lr', lr, self.total_steps)
            self.writer.add_scalar('train/lr_reductions_this_stage', self.learner.lr_reductions_this_stage, self.total_steps)
            self.writer.add_scalar('vtrace/rho_mean', stats.get('vtrace/rho_mean', 1.0), self.total_steps)
            self.writer.add_scalar('vtrace/rho_max', stats.get('vtrace/rho_max', 1.0), self.total_steps)
            if self.learner.lp_history:
                self.writer.add_scalar('train/lp_avg', np.mean(self.learner.lp_history), self.total_steps)
        if self.learner.feature_importance_history:
            top_groups = sorted(
                [(g, np.mean(v[-10:]) if v else 0) for g, v in self.learner.feature_importance_history.items()],
                key=lambda x: -x[1]
            )[:5]
            print(f"Features: {' '.join([f'{g}:{imp:.4f}' for g, imp in top_groups])}")

    def _save_checkpoint(self, update_count, final=False):
        path = self.checkpoint_dir / f"checkpoint_{'final' if final else f'update_{update_count}'}.pt"
        self.learner.save_checkpoint(path, extra={
            'total_steps': self.total_steps, 'total_episodes': self.total_episodes,
            'curriculum_stats': ray.get(self.curriculum.get_stats.remote())
        })
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
    CHECKPOINT_DIR, LOG_DIR, NUM_WORKERS = "./checkpoints_vtrace", "./logs_vtrace", 24
    model_config = {
        'obs_dim': 460,
        'feature_dim': FEATURE_DIM,
        'd_model': 128,
        'mamba_d_state':  64,
        'mamba_layers': 2,
        'encoder_hidden': [128],
        'policy_hidden': [128],
        'value_hidden': [128],
        'use_distributional': True,
        'dropout': 0.0,
        'num_stages': 14,
        'segment_size': 16,
    }
    config = TrainingConfig(
        stages=get_default_stages(),
        final_stage_target_win_rate=0.5,
        max_steps_without_progress=100_000_000,
        num_workers=NUM_WORKERS,
        trajectory_length=128,
        batch_size=4096,
        minibatch_size=512,
        num_epochs=2,
        learning_rate=3e-4,
        lr_schedule="constant",
        gamma=1.0,
        rho_bar=1.0,
        c_bar=1.0,
        entropy_coeff=0.0001,
        value_coeff=0.5,
        si_lambda=0.5,
        max_grad_norm=0.5,
        total_steps=1_000_000_000,
        log_interval=10,
        checkpoint_interval=100,
        feature_importance_interval=50,
        weight_sync_interval=5,
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