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
import trueskill
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


# TrueSkill environment singleton
TS_ENV = trueskill.TrueSkill(draw_probability=0.0)


def win_probability(r1: trueskill.Rating, r2: trueskill.Rating) -> float:
    """Calculate win probability of r1 vs r2"""
    delta_mu = r1.mu - r2.mu
    denom = math.sqrt(2 * TS_ENV.beta**2 + r1.sigma**2 + r2.sigma**2)
    return TS_ENV.cdf(delta_mu / denom)


@dataclass 
class Opponent:
    name: str
    opponent_type: str
    rating: trueskill.Rating
    difficulty: float = 0.5
    weights_path: str = None
    matches_played: int = 0


@ray.remote
class OpponentPool:
    DENSE_TO_SPARSE_THRESHOLD = 0.7
    SPARSE_LOCK_THRESHOLD = 0.8
    
    def __init__(self, snapshot_dir: str, anchor_ratio: float = 0.5,
                 snapshot_interval: int = 200, max_snapshots: int = 30):
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.anchor_ratio = anchor_ratio
        self.snapshot_interval = snapshot_interval
        self.max_snapshots = max_snapshots
        
        self.agent_rating = TS_ENV.create_rating(mu=15.0, sigma=8.333)
        
        self.anchors = {
            'bot_easy': Opponent('bot_easy', 'bot', TS_ENV.create_rating(mu=18.0, sigma=2.0), 0.05),
            'bot_medium': Opponent('bot_medium', 'bot', TS_ENV.create_rating(mu=28.0, sigma=2.0), 0.6),
            'bot_hard': Opponent('bot_hard', 'bot', TS_ENV.create_rating(mu=38.0, sigma=2.0), 0.95),
        }
        
        self.snapshots: List[Opponent] = []
        self.update_count = 0
        self.updates_since_snapshot = 0
        
        self.dense_ratio = 1.0
        self.dense_ratio_locked = False
        
        self.match_history = defaultdict(list)
        self.dense_results = defaultdict(list)
        self.sparse_results = defaultdict(list)
        
        self.total_matches = 0
        self.dense_matches = 0
        self.sparse_matches = 0
        
        self.episode_rewards = []
        self.max_reward_seen = -float('inf')
    
    def get_dense_ratio(self) -> float:
        return self.dense_ratio
    
    def select_opponent(self) -> Tuple[str, str, float, int]:
        self.update_count += 1
        
        use_dense = np.random.random() < self.dense_ratio
        reward_type = 0 if use_dense else 1
        
        if np.random.random() < self.anchor_ratio or not self.snapshots:
            opponent = self._select_anchor()
        else:
            opponent = self._select_snapshot()
        
        return (opponent.name, opponent.opponent_type, opponent.difficulty, reward_type)
    
    def _select_anchor(self) -> Opponent:
        my_mu = self.agent_rating.mu
        
        if my_mu < 20:
            weights = {'bot_easy': 0.8, 'bot_medium': 0.2, 'bot_hard': 0.0}
        elif my_mu < 28:
            weights = {'bot_easy': 0.3, 'bot_medium': 0.5, 'bot_hard': 0.2}
        elif my_mu < 35:
            weights = {'bot_easy': 0.1, 'bot_medium': 0.4, 'bot_hard': 0.5}
        else:
            weights = {'bot_easy': 0.05, 'bot_medium': 0.25, 'bot_hard': 0.7}
        
        names = list(weights.keys())
        probs = np.array([weights[n] for n in names])
        probs = probs / probs.sum()
        chosen = np.random.choice(names, p=probs)
        return self.anchors[chosen]
    
    def _select_snapshot(self) -> Opponent:
        if not self.snapshots:
            return self._select_anchor()
        
        candidates = []
        for snap in self.snapshots:
            quality = trueskill.quality_1vs1(self.agent_rating, snap.rating)
            candidates.append((snap, quality))
        
        snaps, scores = zip(*candidates)
        scores = np.array(scores)
        scores = scores + 0.1
        probs = scores / scores.sum()
        
        idx = np.random.choice(len(snaps), p=probs)
        return snaps[idx]
    
    def report_match(self, opponent_name: str, won: bool, reward_type: int, episode_reward: float = 0.0):
        self.total_matches += 1
        if reward_type == 0:
            self.dense_matches += 1
        else:
            self.sparse_matches += 1
        
        self.episode_rewards.append(episode_reward)
        if len(self.episode_rewards) > 500:
            self.episode_rewards.pop(0)
        if episode_reward > self.max_reward_seen:
            self.max_reward_seen = episode_reward
        
        if opponent_name.startswith('snapshot_'):
            opp = None
            for s in self.snapshots:
                if s.name == opponent_name:
                    opp = s
                    break
            if opp is None:
                return
        elif opponent_name in self.anchors:
            opp = self.anchors[opponent_name]
        else:
            return
        
        if won:
            new_agent, new_opp = trueskill.rate_1vs1(self.agent_rating, opp.rating)
        else:
            new_opp, new_agent = trueskill.rate_1vs1(opp.rating, self.agent_rating)
        
        self.agent_rating = new_agent
        opp.rating = new_opp
        opp.matches_played += 1
        
        self.match_history[opponent_name].append(won)
        if len(self.match_history[opponent_name]) > 200:
            self.match_history[opponent_name].pop(0)
        
        if reward_type == 0:
            self.dense_results[opponent_name].append(won)
            if len(self.dense_results[opponent_name]) > 200:
                self.dense_results[opponent_name].pop(0)
        else:
            self.sparse_results[opponent_name].append(won)
            if len(self.sparse_results[opponent_name]) > 200:
                self.sparse_results[opponent_name].pop(0)
        
        self._update_dense_ratio()
    
    def _update_dense_ratio(self):
        if self.dense_ratio_locked:
            return
        
        easy_dense = self.dense_results.get('bot_easy', [])
        if len(easy_dense) < 50:
            return
        
        easy_wr_dense = np.mean(easy_dense[-50:])
        
        if easy_wr_dense >= self.DENSE_TO_SPARSE_THRESHOLD:
            old_ratio = self.dense_ratio
            
            easy_sparse = self.sparse_results.get('bot_easy', [])
            if len(easy_sparse) >= 30:
                easy_wr_sparse = np.mean(easy_sparse[-30:])
                if easy_wr_sparse >= self.SPARSE_LOCK_THRESHOLD:
                    target = 0.0
                else:
                    progress = easy_wr_sparse / self.SPARSE_LOCK_THRESHOLD
                    target = max(0.0, 1.0 - progress)
            else:
                target = max(0.3, 1.0 - (easy_wr_dense - 0.5) * 2)
            
            self.dense_ratio = 0.95 * self.dense_ratio + 0.05 * target
            self.dense_ratio = max(0.0, min(1.0, self.dense_ratio))
            
            if self.dense_ratio < 0.05:
                self.dense_ratio = 0.0
                self.dense_ratio_locked = True
                print(f"\n🎯 PHASE COMPLETE: Now 100% sparse rewards\n")
            elif abs(old_ratio - self.dense_ratio) > 0.05:
                print(f"  Dense ratio: {old_ratio:.0%} → {self.dense_ratio:.0%}")
    
    def maybe_save_snapshot(self, weights: dict, update_count: int) -> bool:
        self.updates_since_snapshot += 1
        
        if self.updates_since_snapshot < self.snapshot_interval:
            return False
        
        if self.agent_rating.sigma > 6.0:
            return False
        
        self.updates_since_snapshot = 0
        
        snapshot_path = self.snapshot_dir / f"snapshot_{update_count}.pt"
        torch.save(weights, snapshot_path)
        
        snap_name = f"snapshot_{len(self.snapshots)}"
        new_snap = Opponent(
            name=snap_name,
            opponent_type='snapshot',
            rating=TS_ENV.create_rating(mu=self.agent_rating.mu, sigma=max(self.agent_rating.sigma, 4.0)),
            weights_path=str(snapshot_path)
        )
        self.snapshots.append(new_snap)
        
        if len(self.snapshots) > self.max_snapshots:
            self._prune_snapshots()
        
        print(f"📸 Snapshot {snap_name}: μ={self.agent_rating.mu:.1f}")
        return True
    
    def _prune_snapshots(self):
        if len(self.snapshots) <= self.max_snapshots:
            return
        
        sorted_snaps = sorted(self.snapshots, key=lambda s: s.rating.mu)
        keep_indices = set([0, len(sorted_snaps) - 1])
        
        n_middle = self.max_snapshots - 2
        if n_middle > 0 and len(sorted_snaps) > 2:
            step = (len(sorted_snaps) - 1) / (n_middle + 1)
            for i in range(1, n_middle + 1):
                keep_indices.add(min(int(i * step), len(sorted_snaps) - 1))
        
        for i, snap in enumerate(sorted_snaps):
            if i not in keep_indices and snap.weights_path:
                try:
                    Path(snap.weights_path).unlink()
                except:
                    pass
        
        self.snapshots = [sorted_snaps[i] for i in sorted(keep_indices)]
        for i, snap in enumerate(self.snapshots):
            snap.name = f'snapshot_{i}'
    
    def get_snapshot_weights(self, snapshot_name: str) -> Optional[dict]:
        for snap in self.snapshots:
            if snap.name == snapshot_name:
                if snap.weights_path and Path(snap.weights_path).exists():
                    return torch.load(snap.weights_path, map_location='cpu', weights_only=False)
        return None
    
    def get_stats(self) -> dict:
        anchor_wr = {}
        for name in self.anchors:
            hist = self.match_history.get(name, [])
            anchor_wr[name] = np.mean(hist[-100:]) if hist else 0.0
        
        dense_wr = {}
        for name in self.anchors:
            hist = self.dense_results.get(name, [])
            dense_wr[name] = np.mean(hist[-50:]) if hist else 0.0
        
        sparse_wr = {}
        for name in self.anchors:
            hist = self.sparse_results.get(name, [])
            sparse_wr[name] = np.mean(hist[-50:]) if hist else 0.0
        
        expected_wr = {
            'vs_easy': win_probability(self.agent_rating, self.anchors['bot_easy'].rating),
            'vs_medium': win_probability(self.agent_rating, self.anchors['bot_medium'].rating),
            'vs_hard': win_probability(self.agent_rating, self.anchors['bot_hard'].rating),
        }
        
        return {
            'agent_mu': self.agent_rating.mu,
            'agent_sigma': self.agent_rating.sigma,
            'dense_ratio': self.dense_ratio,
            'dense_locked': self.dense_ratio_locked,
            'num_snapshots': len(self.snapshots),
            'total_matches': self.total_matches,
            'dense_matches': self.dense_matches,
            'sparse_matches': self.sparse_matches,
            'anchor_wr': anchor_wr,
            'dense_wr': dense_wr,
            'sparse_wr': sparse_wr,
            'expected_wr': expected_wr,
            'max_reward': self.max_reward_seen if self.max_reward_seen > -float('inf') else 0.0,
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
        }
    
    def is_training_complete(self) -> bool:
        if not self.dense_ratio_locked:
            return False
        sparse_hard = self.sparse_results.get('bot_hard', [])
        if len(sparse_hard) < 50:
            return False
        return np.mean(sparse_hard[-50:]) >= 0.6
    
    def get_state(self) -> dict:
        return {
            'agent_rating': {'mu': self.agent_rating.mu, 'sigma': self.agent_rating.sigma},
            'dense_ratio': self.dense_ratio,
            'dense_ratio_locked': self.dense_ratio_locked,
            'snapshots': [
                {'name': s.name, 'rating': {'mu': s.rating.mu, 'sigma': s.rating.sigma}, 
                 'weights_path': s.weights_path, 'matches_played': s.matches_played}
                for s in self.snapshots
            ],
            'match_history': {k: list(v) for k, v in self.match_history.items()},
            'dense_results': {k: list(v) for k, v in self.dense_results.items()},
            'sparse_results': {k: list(v) for k, v in self.sparse_results.items()},
            'update_count': self.update_count,
            'total_matches': self.total_matches,
            'dense_matches': self.dense_matches,
            'sparse_matches': self.sparse_matches,
        }
    
    def restore_state(self, state: dict):
        r = state['agent_rating']
        self.agent_rating = TS_ENV.create_rating(mu=r['mu'], sigma=r['sigma'])
        self.dense_ratio = state.get('dense_ratio', 1.0)
        self.dense_ratio_locked = state.get('dense_ratio_locked', False)
        
        self.snapshots = []
        for s in state.get('snapshots', []):
            r = s['rating']
            snap = Opponent(
                name=s['name'],
                opponent_type='snapshot',
                rating=TS_ENV.create_rating(mu=r['mu'], sigma=r['sigma']),
                weights_path=s.get('weights_path')
            )
            snap.matches_played = s.get('matches_played', 0)
            self.snapshots.append(snap)
        
        self.match_history = defaultdict(list, {k: list(v) for k, v in state.get('match_history', {}).items()})
        self.dense_results = defaultdict(list, {k: list(v) for k, v in state.get('dense_results', {}).items()})
        self.sparse_results = defaultdict(list, {k: list(v) for k, v in state.get('sparse_results', {}).items()})
        self.update_count = state.get('update_count', 0)
        self.total_matches = state.get('total_matches', 0)
        self.dense_matches = state.get('dense_matches', 0)
        self.sparse_matches = state.get('sparse_matches', 0)
        
        print(f"  Restored: μ={self.agent_rating.mu:.1f}, dense_ratio={self.dense_ratio:.0%}, snaps={len(self.snapshots)}")


@dataclass
class ModelConfig:
    obs_dim: int = OBS_DIM
    feature_dim: int = FEATURE_DIM
    d_model: int = 128
    mamba_d_state: int = 64
    mamba_layers: int = 2
    num_actions: int = 19
    action_emb_dim: int = 16
    encoder_hidden: List[int] = None
    policy_hidden: List[int] = None
    value_hidden: List[int] = None
    use_distributional: bool = True
    v_min: float = -10.0
    v_max: float = 10.0
    num_atoms: int = 51
    dropout: float = 0.0
    segment_size: int = 16

    def __post_init__(self):
        if self.encoder_hidden is None:
            self.encoder_hidden = [128]
        if self.policy_hidden is None:
            self.policy_hidden = [128]
        if self.value_hidden is None:
            self.value_hidden = [128]


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
        for t in range(x_in_seg.shape[1]):
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
    def __init__(self, input_dim, d_model=128, d_state=64, num_layers=2, dropout=0.0, segment_size=16):
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
        self.action_embedding = nn.Embedding(config.num_actions, config.action_emb_dim)
        obs_input_dim = config.obs_dim + config.feature_dim + config.action_emb_dim
        
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
        value_hidden = config.value_hidden[0] if config.value_hidden else 128
        
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

    def forward(self, obs, features, prev_action=None, hidden_state=None, return_hidden=False, reward_type=None):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(1)
            squeeze = True
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        
        B, L, _ = obs.shape
        device = obs.device
        
        if features.dim() == 1:
            features = features.unsqueeze(0).unsqueeze(1)
        elif features.dim() == 2:
            features = features.unsqueeze(1)
        
        if prev_action is None:
            prev_action = torch.zeros(B, L, dtype=torch.long, device=device)
        elif prev_action.dim() == 0:
            prev_action = prev_action.expand(B, L)
        elif prev_action.dim() == 1:
            prev_action = prev_action.unsqueeze(1).expand(B, L)
        
        if reward_type is None:
            reward_type = torch.zeros(B, dtype=torch.long, device=device)
        elif isinstance(reward_type, int):
            reward_type = torch.full((B,), reward_type, dtype=torch.long, device=device)
        elif isinstance(reward_type, torch.Tensor):
            if reward_type.dim() == 0:
                reward_type = reward_type.expand(B)
            reward_type = reward_type.long().to(device)
        
        action_emb = self.action_embedding(prev_action)
        x = torch.cat([obs, features, action_emb], dim=-1)
        x = self.obs_encoder(x)
        x, new_hidden = self.mamba(x, hidden_state)
        
        logits = self.policy_head(x)
        
        v_dense = F.relu(self.value_fc1_dense(x))
        v_dense = self.value_fc2_dense(v_dense)
        v_dense = v_dense.view(B, L, self.num_value_heads, self.value_out_dim)
        
        v_sparse = F.relu(self.value_fc1_sparse(x))
        v_sparse = self.value_fc2_sparse(v_sparse)
        v_sparse = v_sparse.view(B, L, self.num_value_heads, self.value_out_dim)
        
        if self.config.use_distributional:
            vl_dense = v_dense.mean(dim=2)
            vp_dense = F.softmax(vl_dense, dim=-1)
            val_dense = (vp_dense * self.value_support).sum(-1, keepdim=True)
            
            vl_sparse = v_sparse.mean(dim=2)
            vp_sparse = F.softmax(vl_sparse, dim=-1)
            val_sparse = (vp_sparse * self.value_support).sum(-1, keepdim=True)
        else:
            val_dense = v_dense.mean(dim=2)
            val_sparse = v_sparse.mean(dim=2)
        
        rt_exp = reward_type.view(B, 1, 1).expand(B, L, 1)
        value = torch.where(rt_exp == 0, val_dense, val_sparse)
        
        log_probs = F.log_softmax(logits, dim=-1)
        
        if squeeze:
            logits = logits.squeeze(1)
            value = value.squeeze(1)
            log_probs = log_probs.squeeze(1)
        
        result = {'logits': logits, 'value': value.squeeze(-1), 'log_probs': log_probs}
        if return_hidden:
            result['hidden_state'] = new_hidden
        return result

    def get_action(self, obs, features, prev_action=None, hidden_state=None, deterministic=False, reward_type=None):
        output = self.forward(obs, features, prev_action, hidden_state, return_hidden=True, reward_type=reward_type)
        logits = output['logits'].clamp(min=-20.0, max=20.0)
        if torch.isnan(logits).any():
            logits = torch.zeros_like(logits)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), output['value'], output.get('hidden_state')

    def evaluate_actions(self, obs, features, actions, prev_action=None, hidden_state=None, reward_type=None):
        output = self.forward(obs, features, prev_action, hidden_state, reward_type=reward_type)
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
class TrainingConfig:
    left_agents: int = 11
    max_episode_steps: int = 500  # Shorter episodes for faster feedback
    
    anchor_ratio: float = 0.5
    snapshot_interval: int = 200
    max_snapshots: int = 30
    
    num_workers: int = 24
    trajectory_length: int = 512  # Longer trajectories for better credit assignment
    batch_size: int = 4096
    minibatch_size: int = 512
    num_epochs: int = 2
    
    learning_rate: float = 3e-4
    lr_min: float = 1e-6
    lr_warmup_steps: int = 5000
    max_grad_norm: float = 0.3  # Reduced for stability
    
    gamma: float = 0.999
    rho_bar: float = 1.0
    c_bar: float = 1.0
    entropy_coeff: float = 0.01  # Increased for more exploration
    value_coeff: float = 0.5
    entropy_floor: float = 1.0  # Skip update if entropy below this
    ppo_clip: float = 0.2  # PPO clipping range
    
    total_steps: int = 1_000_000_000
    log_interval: int = 10
    checkpoint_interval: int = 100
    weight_sync_interval: int = 5
    
    log_dir: str = "./logs_selfplay"
    checkpoint_dir: str = "./checkpoints_selfplay"
    snapshot_dir: str = "./snapshots"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@ray.remote
class SamplerWorker:
    MAX_AGENTS = 11

    def __init__(self, worker_id, model_config, config):
        self.worker_id = worker_id
        self.config = config
        self.device = torch.device('cpu')
        
        self.model = create_model(model_config)
        self.model.to(self.device)
        self.model.eval()
        
        self.opponent_model = create_model(model_config)
        self.opponent_model.to(self.device)
        self.opponent_model.eval()
        
        self.feature_engineer = FeatureEngineer()
        self.env = None
        self.current_obs = None
        self.current_features = None
        self.hidden_state = None
        self.prev_action = None
        self.episode_return = 0.0
        self.episode_steps = 0
        
        self.current_opponent_name = None
        self.current_reward_type = 0
        self.using_selfplay = False

    def set_weights(self, weights):
        self.model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})

    def set_opponent_weights(self, weights):
        if weights:
            self.opponent_model.load_state_dict({k: torch.from_numpy(v.copy()) if isinstance(v, np.ndarray) else v for k, v in weights.items()})

    def collect_trajectory(self, trajectory_length, opponent_pool):
        obs_list, feature_list, action_list, reward_list, done_list = [], [], [], [], []
        value_list, log_prob_list, mask_list, reward_type_list = [], [], [], []
        episode_returns, episode_wins, episode_lengths, episode_opponents, episode_reward_types = [], [], [], [], []
        
        steps = 0
        while steps < trajectory_length:
            if self.env is None or self.episode_steps == 0:
                opp_name, opp_type, opp_difficulty, reward_type = ray.get(opponent_pool.select_opponent.remote())
                self._setup_env(opp_name, opp_type, opp_difficulty, reward_type, opponent_pool)
            
            num_agents = self.config.left_agents
            rt_tensor = torch.tensor(self.current_reward_type, device=self.device)
            
            if num_agents == 1:
                obs_tensor = torch.from_numpy(self.current_obs[0]).float().to(self.device)
                feature_tensor = torch.from_numpy(self.current_features[0]).float().to(self.device)
                with torch.no_grad():
                    action, log_prob, value, self.hidden_state = self.model.get_action(
                        obs_tensor, feature_tensor, prev_action=self.prev_action, 
                        hidden_state=self.hidden_state, reward_type=rt_tensor
                    )
                action_full = np.zeros(self.MAX_AGENTS, dtype=np.int64)
                log_prob_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                value_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                mask_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                action_full[0] = action.item()
                log_prob_full[0] = log_prob.item()
                value_full[0] = value.item()
                mask_full[0] = 1.0
                left_actions = [action.item()]
            else:
                obs_batch = torch.from_numpy(self.current_obs[:num_agents]).float().to(self.device)
                feat_batch = torch.from_numpy(self.current_features[:num_agents]).float().to(self.device)
                rt_batch = torch.full((num_agents,), self.current_reward_type, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    actions, log_probs, values, _ = self.model.get_action(
                        obs_batch, feat_batch, prev_action=None, hidden_state=None, reward_type=rt_batch
                    )
                action_full = np.zeros(self.MAX_AGENTS, dtype=np.int64)
                log_prob_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                value_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                mask_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
                action_full[:num_agents] = actions.cpu().numpy()
                log_prob_full[:num_agents] = log_probs.cpu().numpy()
                value_full[:num_agents] = values.cpu().numpy()
                mask_full[:num_agents] = 1.0
                left_actions = actions.cpu().numpy().tolist()
            
            if self.using_selfplay:
                right_obs = self.current_obs[num_agents:num_agents*2] if len(self.current_obs) >= num_agents*2 else self.current_obs[:num_agents]
                right_feat = self.current_features[num_agents:num_agents*2] if len(self.current_features) >= num_agents*2 else self.current_features[:num_agents]
                right_obs_batch = torch.from_numpy(right_obs).float().to(self.device)
                right_feat_batch = torch.from_numpy(right_feat).float().to(self.device)
                rt_opp = torch.full((num_agents,), self.current_reward_type, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    opp_actions, _, _, _ = self.opponent_model.get_action(
                        right_obs_batch, right_feat_batch, prev_action=None, hidden_state=None, reward_type=rt_opp
                    )
                env_action = left_actions + opp_actions.cpu().numpy().tolist()
            else:
                env_action = left_actions
            
            raw_obs, reward, done, info = self.env.step(env_action)
            self._update_obs(raw_obs)
            
            step_reward = float(sum(reward)) if isinstance(reward, (list, np.ndarray)) else float(reward)
            self.episode_return += step_reward
            self.episode_steps += 1
            
            episode_done = bool(done) or self.episode_steps >= self.config.max_episode_steps
            
            reward_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            done_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            per_agent_reward = step_reward / num_agents
            for i in range(num_agents):
                reward_full[i] = per_agent_reward
                done_full[i] = float(episode_done)
            
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
            mask_list.append(mask_full)
            reward_type_list.append(self.current_reward_type)
            
            if num_agents == 1:
                self.prev_action = torch.tensor(action_full[0], device=self.device)
            
            steps += 1
            
            if episode_done:
                score = info.get("score", [0, 0]) if isinstance(info, dict) else [0, 0]
                won = score[0] > score[1]
                
                episode_returns.append(self.episode_return)
                episode_wins.append(won)
                episode_lengths.append(self.episode_steps)
                episode_opponents.append(self.current_opponent_name)
                episode_reward_types.append(self.current_reward_type)
                
                ray.get(opponent_pool.report_match.remote(self.current_opponent_name, won, self.current_reward_type, self.episode_return))
                self._reset_episode()
        
        return {
            'obs': np.array(obs_list, dtype=np.float32),
            'features': np.array(feature_list, dtype=np.float32),
            'actions': np.array(action_list, dtype=np.int64),
            'rewards': np.array(reward_list, dtype=np.float32),
            'dones': np.array(done_list, dtype=np.float32),
            'values': np.array(value_list, dtype=np.float32),
            'log_probs': np.array(log_prob_list, dtype=np.float32),
            'agent_masks': np.array(mask_list, dtype=np.float32),
            'reward_types': np.array(reward_type_list, dtype=np.int64),
            'worker_id': self.worker_id,
            'episode_returns': episode_returns,
            'episode_wins': episode_wins,
            'episode_lengths': episode_lengths,
            'episode_opponents': episode_opponents,
            'episode_reward_types': episode_reward_types,
        }

    def _setup_env(self, opp_name, opp_type, opp_difficulty, reward_type, opponent_pool):
        if self.env is not None:
            self.env.close()
        
        self.current_opponent_name = opp_name
        self.current_reward_type = reward_type
        rewards = "scoring,checkpoints" if reward_type == 0 else "scoring"
        
        if opp_type == 'bot':
            self.using_selfplay = False
            self.env = football_env.create_environment(
                env_name="11_vs_11_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=self.config.left_agents,
                number_of_right_players_agent_controls=0,
                other_config_options={'right_team_difficulty': opp_difficulty, 'left_team_difficulty': 1.0},
                stacked=True,
                rewards=rewards,
                write_goal_dumps=False,
                write_full_episode_dumps=False,
                render=False,
                write_video=False
            )
        else:
            weights = ray.get(opponent_pool.get_snapshot_weights.remote(opp_name))
            if weights:
                self.set_opponent_weights(weights)
            self.using_selfplay = True
            self.env = football_env.create_environment(
                env_name="11_vs_11_stochastic",
                representation="simple115v2",
                number_of_left_players_agent_controls=self.config.left_agents,
                number_of_right_players_agent_controls=self.config.left_agents,
                stacked=True,
                rewards=rewards,
                write_goal_dumps=False,
                write_full_episode_dumps=False,
                render=False,
                write_video=False
            )
        
        self._reset_episode()

    def _reset_episode(self):
        raw_obs = self.env.reset()
        self._update_obs(raw_obs)
        self.hidden_state = self.model.get_initial_hidden_state(1, self.device)
        self.prev_action = None
        self.episode_return = 0.0
        self.episode_steps = 0

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
        self.config = config
        self.device = torch.device(config.device)
        self.writer = writer
        
        self.model = create_model(model_config)
        self.model.to(self.device)
        
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate, eps=1e-5, weight_decay=1e-5)
        self.scaler = torch.amp.GradScaler('cuda')
        
        self.update_count = 0
        self.current_lr = config.learning_rate
        
        self.ema_enabled = True
        self.ema_decay = 0.999
        self.ema_weights = {k: v.clone() for k, v in self.model.state_dict().items()}
        self.ema_start_step = 1000
        
        print(f"Learner: {count_parameters(self.model):,} params, dual value heads")

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

    def _compute_vtrace_batch(self, trajectories):
        gamma, rho_bar, c_bar = self.config.gamma, self.config.rho_bar, self.config.c_bar
        all_obs, all_features, all_actions = [], [], []
        all_rewards, all_dones, all_behavior_log_probs, all_reward_types = [], [], [], []
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
            all_reward_types.append(np.repeat(traj['reward_types'], A)[mask_flat])
            traj_boundaries.append(traj_boundaries[-1] + mask_flat.sum())
        
        if not all_obs:
            return None
        
        obs = torch.from_numpy(np.concatenate(all_obs)).float().to(self.device)
        features = torch.from_numpy(np.concatenate(all_features)).float().to(self.device)
        actions = torch.from_numpy(np.concatenate(all_actions)).long().to(self.device)
        rewards = torch.from_numpy(np.concatenate(all_rewards)).float().to(self.device)
        dones = torch.from_numpy(np.concatenate(all_dones)).float().to(self.device)
        behavior_log_probs = torch.from_numpy(np.concatenate(all_behavior_log_probs)).float().to(self.device)
        reward_types = torch.from_numpy(np.concatenate(all_reward_types)).long().to(self.device)
        
        with torch.no_grad():
            target_log_probs, _, values = self.model.evaluate_actions(obs, features, actions, reward_type=reward_types)
        
        log_ratios = (target_log_probs - behavior_log_probs).clamp(-20, 20)
        ratios = torch.exp(log_ratios)
        rho = torch.clamp(ratios, max=rho_bar)
        c = torch.clamp(ratios, max=c_bar)
        
        vtrace_targets = torch.zeros_like(values)
        pg_advantages = torch.zeros_like(values)
        
        for i in range(len(traj_boundaries) - 1):
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
            'obs': obs, 'features': features, 'actions': actions,
            'advantages': pg_advantages, 'vtrace_targets': vtrace_targets, 
            'rho': rho, 'reward_types': reward_types,
            'old_log_probs': target_log_probs,  # For PPO clipping
        }

    def update(self, trajectories):
        self.model.train()
        batch = self._compute_vtrace_batch(trajectories)
        if batch is None:
            return {}
        
        advantages = batch['advantages']
        vtrace_targets = batch['vtrace_targets']
        rho = batch['rho']
        reward_types = batch['reward_types']
        old_log_probs = batch['old_log_probs']
        
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - advantages.mean()) / adv_std
        
        batch_size = len(advantages)
        all_indices = []
        for _ in range(self.config.num_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.config.minibatch_size):
                all_indices.append(perm[start:start + self.config.minibatch_size])
        
        total_loss, policy_loss_sum, value_loss_sum, entropy_sum = 0.0, 0.0, 0.0, 0.0
        num_updates, skipped = 0, 0
        grad_norm = torch.tensor(0.0)
        
        for mb_idx in all_indices:
            mb_obs = batch['obs'][mb_idx]
            mb_features = batch['features'][mb_idx]
            mb_actions = batch['actions'][mb_idx]
            mb_advantages = advantages[mb_idx]
            mb_vtrace_targets = vtrace_targets[mb_idx]
            mb_reward_types = reward_types[mb_idx]
            mb_old_log_probs = old_log_probs[mb_idx]
            
            try:
                log_probs, entropy, values = self.model.evaluate_actions(
                    mb_obs, mb_features, mb_actions, reward_type=mb_reward_types
                )
                if torch.isnan(log_probs).any() or torch.isnan(values).any():
                    skipped += 1
                    continue
                
                # Entropy floor - skip if policy is too deterministic
                mean_entropy = entropy.mean().item()
                if mean_entropy < self.config.entropy_floor:
                    skipped += 1
                    continue
                
                # PPO clipped policy loss
                ratio = torch.exp(log_probs - mb_old_log_probs.detach())
                clipped_ratio = torch.clamp(ratio, 1.0 - self.config.ppo_clip, 1.0 + self.config.ppo_clip)
                policy_loss1 = -ratio * mb_advantages.detach()
                policy_loss2 = -clipped_ratio * mb_advantages.detach()
                policy_loss = torch.max(policy_loss1, policy_loss2).mean()
                
                value_loss = F.mse_loss(values, mb_vtrace_targets.detach())
                entropy_loss = -entropy.mean()
                loss = policy_loss + self.config.value_coeff * value_loss + self.config.entropy_coeff * entropy_loss
                
                # Loss clipping - skip extreme losses
                if abs(loss.item()) > 5.0:
                    skipped += 1
                    continue
            except:
                skipped += 1
                continue
            
            if torch.isnan(loss) or torch.isinf(loss):
                skipped += 1
                continue
            
            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                skipped += 1
                self.scaler.update()
                continue
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self._update_ema()
            
            total_loss += loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            entropy_sum += -entropy_loss.item()
            num_updates += 1
        
        self.update_count += 1
        self._update_lr()
        
        if num_updates == 0:
            return {'skipped': float(skipped)}
        
        return {
            'loss/total': total_loss / num_updates,
            'loss/policy': policy_loss_sum / num_updates,
            'loss/value': value_loss_sum / num_updates,
            'loss/entropy': entropy_sum / num_updates,
            'train/grad_norm': float(grad_norm),
            'train/lr': self.current_lr,
            'vtrace/rho_mean': float(rho.mean()),
            'vtrace/rho_max': float(rho.max()),
            'vtrace/rho_min': float(rho.min()),
        }

    def _update_lr(self):
        if self.update_count < self.config.lr_warmup_steps:
            self.current_lr = self.config.learning_rate * (self.update_count / self.config.lr_warmup_steps)
        for pg in self.optimizer.param_groups:
            pg['lr'] = max(self.current_lr, self.config.lr_min)

    def save_checkpoint(self, path, extra=None):
        ckpt = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'current_lr': self.current_lr,
            'ema_weights': {k: v.cpu() for k, v in self.ema_weights.items()},
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.update_count = ckpt.get('update_count', 0)
        self.current_lr = ckpt.get('current_lr', self.config.learning_rate)
        if 'ema_weights' in ckpt and ckpt['ema_weights']:
            for k, v in ckpt['ema_weights'].items():
                if k in self.ema_weights:
                    self.ema_weights[k] = v.to(self.device)
        for pg in self.optimizer.param_groups:
            pg['lr'] = self.current_lr
        return ckpt


class SelfPlayTrainer:
    def __init__(self, config, model_config, resume_from=None):
        self.config = config
        self.model_config = model_config
        self.resume_from = resume_from
        
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.snapshot_dir = Path(config.snapshot_dir)
        
        for d in [self.log_dir, self.checkpoint_dir, self.snapshot_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        self.learner = None
        self.opponent_pool = None
        self.workers = []
        self.writer = None
        
        self.total_episodes = 0
        self.total_steps = 0
        self.start_time = None

    def setup(self):
        print("=" * 60)
        print("IMPALA + TRUESKILL SELF-PLAY + DUAL VALUE HEADS")
        print("=" * 60)
        
        run_name = f"selfplay_{time.strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=self.log_dir / run_name)
        
        self.learner = Learner(self.config, self.model_config, self.writer)
        
        pool_state = None
        if self.resume_from:
            p = Path(self.resume_from)
            if p.exists():
                print(f"Loading: {p}")
                ckpt = self.learner.load_checkpoint(p)
                self.total_steps = ckpt.get('total_steps', 0)
                self.total_episodes = ckpt.get('total_episodes', 0)
                pool_state = ckpt.get('pool_state')
        
        self.opponent_pool = OpponentPool.remote(
            snapshot_dir=str(self.snapshot_dir),
            anchor_ratio=self.config.anchor_ratio,
            snapshot_interval=self.config.snapshot_interval,
            max_snapshots=self.config.max_snapshots,
        )
        
        if pool_state:
            ray.get(self.opponent_pool.restore_state.remote(pool_state))
        
        self.workers = [
            SamplerWorker.remote(worker_id=i, model_config=self.model_config, config=self.config)
            for i in range(self.config.num_workers)
        ]
        
        self._sync_weights()
        print(f"Workers: {len(self.workers)}")

    def _sync_weights(self):
        weights = self.learner.get_weights()
        ray.get([w.set_weights.remote(weights) for w in self.workers])

    def train(self):
        print("Training...")
        self.start_time = time.time()
        
        pending = {w.collect_trajectory.remote(self.config.trajectory_length, self.opponent_pool): w for w in self.workers}
        trajectories_buffer = []
        update_count = self.learner.update_count
        
        while True:
            if ray.get(self.opponent_pool.is_training_complete.remote()):
                print("\n🎉 TRAINING COMPLETE!")
                break
            
            if self.total_steps >= self.config.total_steps:
                print("\n⚠️ Max steps")
                break
            
            done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
            
            for ref in done_refs:
                worker = pending.pop(ref)
                try:
                    traj = ray.get(ref)
                    trajectories_buffer.append(traj)
                    self.total_steps += int(len(traj['obs']) * traj['agent_masks'].sum() / len(traj['obs']))
                    self.total_episodes += len(traj['episode_returns'])
                except Exception as e:
                    print(f"Worker error: {e}")
                pending[worker.collect_trajectory.remote(self.config.trajectory_length, self.opponent_pool)] = worker
            
            if sum(len(t['obs']) * t['agent_masks'].sum() / len(t['obs']) for t in trajectories_buffer) >= self.config.batch_size:
                stats = self.learner.update(trajectories_buffer)
                trajectories_buffer = []
                update_count += 1
                
                ray.get(self.opponent_pool.maybe_save_snapshot.remote(
                    self.learner.get_weights(use_ema=False), update_count
                ))
                
                if update_count % self.config.weight_sync_interval == 0:
                    self._sync_weights()
                
                if update_count % self.config.log_interval == 0:
                    self._log_progress(update_count, stats)
                
                if update_count % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(update_count)
        
        self._save_checkpoint(update_count, final=True)
        if self.writer:
            self.writer.close()

    def _log_progress(self, update_count, stats):
        elapsed = time.time() - self.start_time
        sps = self.total_steps / elapsed if elapsed > 0 else 0
        
        pool_stats = ray.get(self.opponent_pool.get_stats.remote())
        
        mu = pool_stats['agent_mu']
        sigma = pool_stats['agent_sigma']
        dr = pool_stats['dense_ratio']
        
        exp = pool_stats['expected_wr']
        dense_wr = pool_stats['dense_wr']
        sparse_wr = pool_stats['sparse_wr']
        actual_wr = pool_stats['anchor_wr']
        
        total_m = pool_stats['total_matches']
        
        phase = "DENSE" if dr > 0.5 else ("TRANS" if dr > 0 else "SPARSE")
        
        print(f"[{update_count}] {self.total_steps/1e6:.1f}M | {sps/1e3:.0f}k sps | "
              f"μ={mu:.1f}±{sigma:.1f} | {phase} {dr:.0%} | "
              f"E:{exp['vs_easy']:.0%} M:{exp['vs_medium']:.0%} H:{exp['vs_hard']:.0%} | "
              f"matches={total_m} eps={self.total_episodes}")
        
        max_r = pool_stats.get('max_reward', 0)
        mean_r = pool_stats.get('mean_reward', 0)
        
        loss_total = stats.get('loss/total', 0)
        entropy = stats.get('loss/entropy', 0)
        rho_mean = stats.get('vtrace/rho_mean', 1.0)
        rho_max = stats.get('vtrace/rho_max', 1.0)
        
        if total_m > 0:
            print(f"  Actual: E:{actual_wr['bot_easy']:.0%} M:{actual_wr['bot_medium']:.0%} H:{actual_wr['bot_hard']:.0%} | "
                  f"R: max={max_r:.2f} mean={mean_r:.2f} | "
                  f"Loss:{loss_total:.3f} H:{entropy:.3f} ρ:{rho_mean:.2f}/{rho_max:.1f}")
        
        if self.writer:
            self.writer.add_scalar('skill/mu', mu, self.total_steps)
            self.writer.add_scalar('skill/sigma', sigma, self.total_steps)
            self.writer.add_scalar('phase/dense_ratio', dr, self.total_steps)
            self.writer.add_scalar('episodes/total', self.total_episodes, self.total_steps)
            self.writer.add_scalar('matches/total', total_m, self.total_steps)
            for k, v in exp.items():
                self.writer.add_scalar(f'expected/{k}', v, self.total_steps)
            for k, v in actual_wr.items():
                self.writer.add_scalar(f'actual/{k}', v, self.total_steps)
            for k, v in stats.items():
                self.writer.add_scalar(k, v, self.total_steps)

    def _save_checkpoint(self, update_count, final=False):
        path = self.checkpoint_dir / f"checkpoint_{'final' if final else f'{update_count}'}.pt"
        pool_state = ray.get(self.opponent_pool.get_state.remote())
        self.learner.save_checkpoint(path, extra={
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'pool_state': pool_state,
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
    
    model_config = {
        'obs_dim': OBS_DIM,
        'feature_dim': FEATURE_DIM,
        'd_model': 128,
        'mamba_d_state': 64,
        'mamba_layers': 2,
        'encoder_hidden': [128],
        'policy_hidden': [128],
        'value_hidden': [128],
        'use_distributional': True,
        'dropout': 0.0,
        'segment_size': 16,
    }
    
    config = TrainingConfig(
        left_agents=11,
        max_episode_steps=500,  # Shorter for faster feedback
        anchor_ratio=0.5,
        snapshot_interval=1000,
        max_snapshots=30,
        num_workers=24,
        trajectory_length=512,  # Longer for better credit assignment
        batch_size=4096,
        minibatch_size=512,
        num_epochs=2,
        learning_rate=3e-4,
        gamma=1,
        rho_bar=1.0,
        c_bar=1.0,
        entropy_coeff=0.01,  # Higher for more exploration
        value_coeff=0.5,
        total_steps=1_000_000_000,
        log_interval=10,
        checkpoint_interval=100,
        weight_sync_interval=50,  # For off-policy ~0.9 rho target
        log_dir="./logs_selfplay",
        checkpoint_dir="./checkpoints_selfplay",
        snapshot_dir="./snapshots",
    )
    
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    trainer = SelfPlayTrainer(config, model_config, resume_from=RESUME_FROM)
    
    try:
        trainer.setup()
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()