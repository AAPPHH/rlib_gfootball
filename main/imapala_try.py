"""
GFootball Mamba IMPALA - Full Async with V-trace
- Asynchrone Workers mit kontinuierlichem Rollout
- V-trace Off-Policy Korrektur
- Queue-basierter Learner
- Cluster-ready (SLURM kompatibel)
"""

import time
import os
from pathlib import Path
from collections import deque
from typing import List, Optional, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import ray
import gfootball.env as football_env


FEATURE_DIM = 93
OBS_DIM = 115


# =============================================================================
# GOLDEN MEMORY REPLAY BUFFER
# =============================================================================

class GoldenMemory:
    """
    Replay Buffer der die besten Rollouts behält.
    - Priorität nach Episode Return oder Win
    - Mixe frische + golden Rollouts für Training
    """
    
    def __init__(
        self, 
        capacity: int = 256,
        selection_mode: str = "top_return",  # "top_return", "wins_only", "mixed"
        golden_ratio: float = 0.25,  # Anteil golden samples im Batch
    ):
        self.capacity = capacity
        self.selection_mode = selection_mode
        self.golden_ratio = golden_ratio
        
        self.buffer: List[dict] = []
        self.returns: List[float] = []  # Für Sortierung
        
    def add(self, rollout: dict):
        """Füge Rollout hinzu wenn er gut genug ist."""
        # Berechne Return für diesen Rollout
        episodes = rollout.get('completed_episodes', [])
        if not episodes:
            # Kein vollständiges Episode - nutze sum rewards
            rollout_return = float(np.sum(rollout['rewards']))
            won = False
        else:
            # Durchschnitt über completed episodes
            rollout_return = np.mean([ep['return'] for ep in episodes])
            won = any(ep['won'] for ep in episodes)
        
        # Selection logic
        should_add = False
        
        if self.selection_mode == "wins_only":
            should_add = won
        elif self.selection_mode == "top_return":
            if len(self.buffer) < self.capacity:
                should_add = True
            else:
                # Nur hinzufügen wenn besser als schlechtester
                min_return = min(self.returns) if self.returns else float('-inf')
                should_add = rollout_return > min_return
        elif self.selection_mode == "mixed":
            # 50% Chance oder wenn Win
            should_add = won or (np.random.random() < 0.5 and len(self.buffer) < self.capacity)
        
        if should_add:
            if len(self.buffer) >= self.capacity:
                # Entferne schlechtesten
                min_idx = np.argmin(self.returns)
                self.buffer.pop(min_idx)
                self.returns.pop(min_idx)
            
            self.buffer.append(rollout.copy())
            self.returns.append(rollout_return)
    
    def sample(self, n: int, prioritized: bool = True) -> List[dict]:
        """Sample n Rollouts aus dem Buffer."""
        if len(self.buffer) == 0:
            return []
        
        n = min(n, len(self.buffer))
        
        if prioritized:
            # Höhere Returns = höhere Wahrscheinlichkeit
            returns_arr = np.array(self.returns)
            # Shift to positive
            shifted = returns_arr - returns_arr.min() + 1.0
            probs = shifted / shifted.sum()
            indices = np.random.choice(len(self.buffer), size=n, replace=False, p=probs)
        else:
            indices = np.random.choice(len(self.buffer), size=n, replace=False)
        
        return [self.buffer[i] for i in indices]
    
    def get_golden_batch(self, fresh_rollouts: List[dict], batch_size: int) -> List[dict]:
        """
        Mixe frische Rollouts mit Golden Memory.
        Returns batch_size Rollouts.
        """
        n_golden = int(batch_size * self.golden_ratio)
        n_fresh = batch_size - n_golden
        
        # Fresh rollouts
        if len(fresh_rollouts) >= n_fresh:
            fresh_batch = fresh_rollouts[:n_fresh]
        else:
            fresh_batch = fresh_rollouts
            n_golden = batch_size - len(fresh_batch)
        
        # Golden samples
        golden_batch = self.sample(n_golden) if n_golden > 0 else []
        
        return fresh_batch + golden_batch
    
    def stats(self) -> dict:
        if not self.returns:
            return {'size': 0, 'mean_return': 0, 'max_return': 0, 'min_return': 0}
        return {
            'size': len(self.buffer),
            'mean_return': np.mean(self.returns),
            'max_return': np.max(self.returns),
            'min_return': np.min(self.returns),
        }
    
    def __len__(self):
        return len(self.buffer)


# =============================================================================
# FEATURE ENGINEER
# =============================================================================

class FeatureEngineer:
    GOAL_POS = np.array([1.0, 0.0], dtype=np.float32)
    OWN_GOAL_POS = np.array([-1.0, 0.0], dtype=np.float32)
    GOAL_TOP = np.array([1.0, 0.044], dtype=np.float32)
    GOAL_BOTTOM = np.array([1.0, -0.044], dtype=np.float32)

    @staticmethod
    def extract_features(obs: np.ndarray) -> np.ndarray:
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
        feat[:, 86:90] = [0.0, 0.0, 0.0, 1.0]
        
        feat[:, 90] = shooting_angle * np.where(keeper_valid, np.clip(keeper_dist, 0, 1), 1.0)
        feat[:, 91] = np.clip(ball_dir[:, 0] * ball_speed, -1, 1)
        feat[:, 92] = np.clip(feat[:, 64], 0, 1) * 0.4 + np.clip(feat[:, 91], 0, 1) * 0.3 + np.clip(numerical_adv / 5.0 + 0.5, 0, 1) * 0.3
        
        if squeeze_output:
            return feat[0]
        return feat


# =============================================================================
# MAMBA BLOCK
# =============================================================================

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
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
    
    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_in, z = xz.chunk(2, dim=-1)
        
        dt = F.softplus(self.dt_proj(x_in))
        B_t = self.B_proj(x_in)
        C_t = self.C_proj(x_in)
        
        A_diag = -torch.exp(self.A_log_diag)
        outputs = []
        
        for t in range(L):
            dA = torch.exp(dt[:, t, :].unsqueeze(-1) * A_diag.unsqueeze(0).unsqueeze(0))
            dB = dt[:, t, :].unsqueeze(-1) * B_t[:, t, :].unsqueeze(1)
            h = h * dA + x_in[:, t, :].unsqueeze(-1) * dB
            y_t = (h * C_t[:, t, :].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        out = y * F.silu(z) + x_in * self.D
        out = x + self.dropout(self.out_proj(out))
        
        return out, h


# =============================================================================
# MAMBA IMPALA NETWORK
# =============================================================================

class MambaIMPALANet(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        feature_dim: int,
        num_actions: int = 19,
        d_model: int = 128,
        d_state: int = 16,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.num_actions = num_actions
        
        input_dim = obs_dim + feature_dim
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout) 
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.policy_head = nn.Linear(d_model, num_actions)
        self.value_head = nn.Linear(d_model, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.input_proj.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.input_proj.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
    
    def get_initial_hidden(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        return [
            torch.zeros(batch_size, self.d_model, self.d_state, device=device)
            for _ in range(self.num_layers)
        ]
    
    def forward_sequence(
        self,
        obs: torch.Tensor,
        features: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward über Sequenz.
        obs: (B, L, obs_dim)
        features: (B, L, feature_dim)
        Returns: logits (B, L, A), values (B, L), new_hidden
        """
        B, L, _ = obs.shape
        device = obs.device
        
        if hidden is None:
            hidden = self.get_initial_hidden(B, device)
        
        x = torch.cat([obs, features], dim=-1)
        x = self.input_proj(x)
        
        new_hidden = []
        for i, layer in enumerate(self.mamba_layers):
            x, h_new = layer(x, hidden[i])
            new_hidden.append(h_new)
        
        x = self.final_norm(x)
        logits = self.policy_head(x)
        values = self.value_head(x).squeeze(-1)
        
        return logits, values, new_hidden
    
    def forward(
        self,
        obs: torch.Tensor,
        features: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Single step forward."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
            features = features.unsqueeze(0).unsqueeze(0)
            squeeze = True
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)
            features = features.unsqueeze(1)
            squeeze = True
        else:
            squeeze = False
        
        logits, values, new_hidden = self.forward_sequence(obs, features, hidden)
        
        if squeeze:
            logits = logits.squeeze(1)
            values = values.squeeze(1)
        
        return logits, values, new_hidden
    
    def get_action(
        self,
        obs: torch.Tensor,
        features: torch.Tensor,
        hidden: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        logits, value, new_hidden = self.forward(obs, features, hidden)
        dist = Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value, new_hidden


# =============================================================================
# V-TRACE
# =============================================================================

def compute_vtrace(
    behavior_log_probs: torch.Tensor,  # (B, T)
    target_log_probs: torch.Tensor,    # (B, T)
    actions: torch.Tensor,             # (B, T)
    rewards: torch.Tensor,             # (B, T)
    values: torch.Tensor,              # (B, T)
    bootstrap_values: torch.Tensor,    # (B,)
    dones: torch.Tensor,               # (B, T)
    gamma: float = 0.99,
    rho_bar: float = 1.0,              # Importance sampling truncation
    c_bar: float = 1.0,                # Trace cutting coefficient
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute V-trace targets and advantages.
    
    Returns:
        vs: V-trace value targets (B, T)
        advantages: Policy gradient advantages (B, T)  
        rhos: Clipped importance weights (B, T)
    """
    B, T = rewards.shape
    device = rewards.device
    
    # Importance sampling ratios
    log_rhos = target_log_probs - behavior_log_probs
    rhos = torch.exp(log_rhos)
    
    # Clip importance weights
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)
    
    # Temporal difference errors
    # δ_t = ρ_t * (r_t + γ * V(s_{t+1}) - V(s_t))
    values_t_plus_1 = torch.cat([values[:, 1:], bootstrap_values.unsqueeze(1)], dim=1)
    
    # Mask für Episode-Ende
    not_done = 1.0 - dones
    
    deltas = clipped_rhos * (rewards + gamma * values_t_plus_1 * not_done - values)
    
    # V-trace targets (backward pass)
    # v_s = V(s) + Σ_{t=s}^{T-1} γ^{t-s} (Π_{i=s}^{t-1} c_i) δ_t
    vs_minus_v = torch.zeros(B, T, device=device)
    
    # Accumulate from the end
    acc = torch.zeros(B, device=device)
    for t in reversed(range(T)):
        acc = deltas[:, t] + gamma * cs[:, t] * acc * not_done[:, t]
        vs_minus_v[:, t] = acc
    
    vs = values + vs_minus_v
    
    # Policy gradient advantages
    # A_t = ρ_t * (r_t + γ * v_{t+1} - V(s_t))
    vs_t_plus_1 = torch.cat([vs[:, 1:], bootstrap_values.unsqueeze(1)], dim=1)
    advantages = clipped_rhos * (rewards + gamma * vs_t_plus_1 * not_done - values)
    
    return vs, advantages, clipped_rhos


# =============================================================================
# ASYNC WORKER
# =============================================================================

@ray.remote
class AsyncWorker:
    """Asynchroner Worker der kontinuierlich Rollouts sammelt."""
    
    def __init__(
        self, 
        worker_id: int, 
        num_agents: int = 1, 
        d_model: int = 128,
        d_state: int = 16, 
        num_layers: int = 2,
        rollout_length: int = 256,
    ):
        self.worker_id = worker_id
        self.num_agents = num_agents
        self.rollout_length = rollout_length
        self.feature_engineer = FeatureEngineer()
        
        self.env = football_env.create_environment(
            env_name="11_vs_11_easy_stochastic",
            representation="simple115v2",
            number_of_left_players_agent_controls=num_agents,
            rewards="scoring,checkpoints",
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            write_video=False,
        )
        
        self.device = torch.device('cpu')
        self.model = MambaIMPALANet(
            OBS_DIM, FEATURE_DIM,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
        ).to(self.device)
        self.model.eval()
        
        self.episode_return = 0.0
        self.episode_steps = 0
        self.current_obs = None
        self.current_features = None
        self.hidden_state = None
        self._reset()
    
    def set_weights(self, weights: dict):
        self.model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})
    
    def _reset(self):
        raw_obs = self.env.reset()
        self._update_obs(raw_obs)
        self.episode_return = 0.0
        self.episode_steps = 0
        self.hidden_state = self.model.get_initial_hidden(1, self.device)
    
    def _update_obs(self, raw_obs):
        if isinstance(raw_obs, list):
            raw_obs = np.array(raw_obs)
        if raw_obs.ndim == 1:
            raw_obs = raw_obs.reshape(1, -1)
        self.current_obs = raw_obs[0][:OBS_DIM].astype(np.float32)
        self.current_features = self.feature_engineer.extract_features(self.current_obs)
    
    def collect_rollout(self) -> dict:
        """Sammle einen Rollout mit behavior policy log probs."""
        # Reset hidden state am Anfang jedes Rollouts für Konsistenz
        self.hidden_state = self.model.get_initial_hidden(1, self.device)
        
        obs_list = []
        features_list = []
        actions_list = []
        rewards_list = []
        dones_list = []
        behavior_log_probs_list = []  # Log probs von behavior policy (Worker)
        
        step_rewards = []
        completed_episodes = []
        
        for _ in range(self.rollout_length):
            obs_t = torch.from_numpy(self.current_obs).float().unsqueeze(0)
            feat_t = torch.from_numpy(self.current_features).float().unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, _, new_hidden = self.model.get_action(
                    obs_t, feat_t, self.hidden_state
                )
            
            self.hidden_state = new_hidden
            action_int = action.item()
            
            obs_list.append(self.current_obs.copy())
            features_list.append(self.current_features.copy())
            actions_list.append(action_int)
            behavior_log_probs_list.append(log_prob.item())
            
            env_action = action_int if self.num_agents == 1 else [action_int]
            raw_obs, reward, done, info = self.env.step(env_action)
            
            step_reward = float(reward) if np.isscalar(reward) else float(reward[0]) if len(reward) > 0 else 0.0
            step_rewards.append(step_reward)
            rewards_list.append(step_reward)
            
            self.episode_return += step_reward
            self.episode_steps += 1
            
            episode_done = bool(done) or self.episode_steps >= 3000
            dones_list.append(1.0 if episode_done else 0.0)
            
            if episode_done:
                if isinstance(info, dict) and "score" in info:
                    won = info["score"][0] > info["score"][1]
                else:
                    won = self.episode_return > 0
                
                completed_episodes.append({
                    'return': self.episode_return,
                    'won': won,
                    'length': self.episode_steps,
                })
                self._reset()
            else:
                self._update_obs(raw_obs)
        
        # Bootstrap value für letzten State
        obs_t = torch.from_numpy(self.current_obs).float().unsqueeze(0)
        feat_t = torch.from_numpy(self.current_features).float().unsqueeze(0)
        with torch.no_grad():
            _, bootstrap_value, _ = self.model.forward(obs_t, feat_t, self.hidden_state)
        
        return {
            'obs': np.array(obs_list, dtype=np.float32),
            'features': np.array(features_list, dtype=np.float32),
            'actions': np.array(actions_list, dtype=np.int64),
            'rewards': np.array(rewards_list, dtype=np.float32),
            'dones': np.array(dones_list, dtype=np.float32),
            'behavior_log_probs': np.array(behavior_log_probs_list, dtype=np.float32),
            'bootstrap_value': bootstrap_value.item(),
            'completed_episodes': completed_episodes,
            'step_rewards': step_rewards,
            'worker_id': self.worker_id,
        }
    
    def close(self):
        self.env.close()


# =============================================================================
# IMPALA LEARNER
# =============================================================================

class IMPALALearner:
    """
    IMPALA Learner mit V-trace.
    - Asynchrone Weight Updates
    - Off-Policy Korrektur via V-trace
    - Batch Processing von mehreren Rollouts
    """
    
    def __init__(
        self,
        num_workers: int = 24,
        num_agents: int = 1,
        rollout_length: int = 256,
        batch_size: int = 8,  # Rollouts pro Update
        lr: float = 3e-4,
        gamma: float = 0.99,
        rho_bar: float = 1.0,
        c_bar: float = 1.0,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 40.0,
        optimizer: str = "adam",  # "adam", "adamw", "rmsprop"
        weight_decay: float = 0.0,
        # Golden Memory
        use_golden_memory: bool = True,
        golden_capacity: int = 256,
        golden_ratio: float = 0.25,  # 25% des Batches aus Golden Memory
        golden_mode: str = "top_return",  # "top_return", "wins_only", "mixed"
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints_impala",
        d_model: int = 128,
        d_state: int = 16,
        num_layers: int = 2,
    ):
        self.num_workers = num_workers
        self.rollout_length = rollout_length
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.rho_bar = rho_bar
        self.c_bar = c_bar
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Learner Model
        self.model = MambaIMPALANet(
            OBS_DIM, FEATURE_DIM,
            d_model=d_model,
            d_state=d_state,
            num_layers=num_layers,
        ).to(self.device)
        
        self.optimizer_name = optimizer
        
        self.optimizer = self._create_optimizer(optimizer, lr, weight_decay)
        
        # Ray init
        ray.init(ignore_reinit_error=True)
        
        # Async Workers
        self.workers = [
            AsyncWorker.remote(
                worker_id=i,
                num_agents=num_agents,
                d_model=d_model,
                d_state=d_state,
                num_layers=num_layers,
                rollout_length=rollout_length,
            )
            for i in range(num_workers)
        ]
        
        # Stats
        self.total_steps = 0
        self.update_count = 0
        self.start_time = None
        
        self.episode_returns = deque(maxlen=100)
        self.episode_wins = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=1000)
        
        # Async tracking
        self.pending_rollouts: Dict[ray.ObjectRef, AsyncWorker] = {}
        self.rollout_queue: List[dict] = []
        
        # Golden Memory
        self.use_golden_memory = use_golden_memory
        self.golden_memory = GoldenMemory(
            capacity=golden_capacity,
            selection_mode=golden_mode,
            golden_ratio=golden_ratio,
        ) if use_golden_memory else None
        
        print(f"Mamba IMPALA Learner")
        print(f"  Device: {self.device}")
        print(f"  Workers: {num_workers}")
        print(f"  Rollout: {rollout_length} steps")
        print(f"  Batch size: {batch_size} rollouts")
        print(f"  Optimizer: {optimizer} (lr={lr}, wd={weight_decay})")
        print(f"  V-trace: ρ̄={rho_bar}, c̄={c_bar}")
        print(f"  Mamba: d_model={d_model}, d_state={d_state}, layers={num_layers}")
        if use_golden_memory:
            print(f"  Golden Memory: capacity={golden_capacity}, ratio={golden_ratio}, mode={golden_mode}")
        print(f"  Params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _create_optimizer(self, optimizer: str, lr: float, weight_decay: float):
        """Create optimizer based on config."""
        params = self.model.parameters()
        
        if optimizer == "adam":
            return torch.optim.Adam(params, lr=lr, eps=1e-5, weight_decay=weight_decay)
        elif optimizer == "adamw":
            return torch.optim.AdamW(params, lr=lr, eps=1e-5, weight_decay=weight_decay or 0.01)
        elif optimizer == "rmsprop":
            return torch.optim.RMSprop(params, lr=lr, eps=1e-5, alpha=0.99, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")
    
    def _get_weights(self) -> dict:
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
    
    def _update_worker_weights(self, worker):
        """Update einzelnen Worker async."""
        weights = self._get_weights()
        worker.set_weights.remote(weights)
    
    def _broadcast_weights(self):
        """Broadcast weights zu allen Workern."""
        weights = self._get_weights()
        ray.get([w.set_weights.remote(weights) for w in self.workers])
    
    def _compute_target_log_probs(
        self,
        obs: torch.Tensor,
        features: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Berechne log probs und values mit target policy (Learner).
        """
        logits, values, _ = self.model.forward_sequence(obs, features, hidden=None)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy
    
    def _update(self, rollouts: List[dict]) -> dict:
        """
        IMPALA Update mit V-trace.
        """
        B = len(rollouts)
        L = self.rollout_length
        
        # Stack alle Rollouts
        obs_batch = np.stack([r['obs'] for r in rollouts])
        feat_batch = np.stack([r['features'] for r in rollouts])
        actions_batch = np.stack([r['actions'] for r in rollouts])
        rewards_batch = np.stack([r['rewards'] for r in rollouts])
        dones_batch = np.stack([r['dones'] for r in rollouts])
        behavior_log_probs_batch = np.stack([r['behavior_log_probs'] for r in rollouts])
        bootstrap_values = np.array([r['bootstrap_value'] for r in rollouts])
        
        # To tensors
        obs = torch.from_numpy(obs_batch).float().to(self.device)
        features = torch.from_numpy(feat_batch).float().to(self.device)
        actions = torch.from_numpy(actions_batch).long().to(self.device)
        rewards = torch.from_numpy(rewards_batch).float().to(self.device)
        dones = torch.from_numpy(dones_batch).float().to(self.device)
        behavior_log_probs = torch.from_numpy(behavior_log_probs_batch).float().to(self.device)
        bootstrap_vals = torch.from_numpy(bootstrap_values).float().to(self.device)
        
        # Forward pass mit target policy
        target_log_probs, values, entropy = self._compute_target_log_probs(obs, features, actions)
        
        # V-trace targets
        with torch.no_grad():
            vs, advantages, rhos = compute_vtrace(
                behavior_log_probs=behavior_log_probs,
                target_log_probs=target_log_probs.detach(),
                actions=actions,
                rewards=rewards,
                values=values.detach(),
                bootstrap_values=bootstrap_vals,
                dones=dones,
                gamma=self.gamma,
                rho_bar=self.rho_bar,
                c_bar=self.c_bar,
            )
        
        # Policy loss (policy gradient mit V-trace advantages)
        policy_loss = -(target_log_probs * advantages.detach()).mean()
        
        # Value loss (MSE zu V-trace targets)
        value_loss = F.mse_loss(values, vs.detach())
        
        # Entropy bonus
        entropy_loss = -entropy.mean()
        
        # Total loss
        loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.update_count += 1
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'mean_rho': rhos.mean().item(),
            'max_rho': rhos.max().item(),
        }
    
    def train(
        self, 
        total_steps: int = 50_000_000, 
        log_interval: int = 10, 
        checkpoint_interval: int = 100,
        weight_sync_interval: int = 1,  # Wie oft weights broadcasten
    ):
        print(f"\nStarting IMPALA training for {total_steps:,} steps...")
        self.start_time = time.time()
        
        # Initial weight sync
        self._broadcast_weights()
        
        # Starte alle Worker
        for worker in self.workers:
            ref = worker.collect_rollout.remote()
            self.pending_rollouts[ref] = worker
        
        while self.total_steps < total_steps:
            # Warte auf mindestens batch_size Rollouts
            while len(self.rollout_queue) < self.batch_size:
                # Warte auf einen fertigen Rollout
                done_refs, _ = ray.wait(list(self.pending_rollouts.keys()), num_returns=1)
                
                for ref in done_refs:
                    worker = self.pending_rollouts.pop(ref)
                    rollout = ray.get(ref)
                    self.rollout_queue.append(rollout)
                    
                    # Stats tracken
                    self.total_steps += len(rollout['obs'])
                    for ep in rollout['completed_episodes']:
                        self.episode_returns.append(ep['return'])
                        self.episode_wins.append(1.0 if ep['won'] else 0.0)
                        self.episode_lengths.append(ep['length'])
                    self.recent_rewards.extend(rollout['step_rewards'])
                    
                    # Add to Golden Memory
                    if self.use_golden_memory:
                        self.golden_memory.add(rollout)
                    
                    # Worker sofort wieder starten (async!)
                    # Update weights periodisch
                    if self.update_count % weight_sync_interval == 0:
                        self._update_worker_weights(worker)
                    
                    new_ref = worker.collect_rollout.remote()
                    self.pending_rollouts[new_ref] = worker
            
            # Batch für Update - mit Golden Memory Mix
            fresh_batch = self.rollout_queue[:self.batch_size]
            self.rollout_queue = self.rollout_queue[self.batch_size:]
            
            if self.use_golden_memory and len(self.golden_memory) > 0:
                batch = self.golden_memory.get_golden_batch(fresh_batch, self.batch_size)
            else:
                batch = fresh_batch
            
            # Update
            stats = self._update(batch)
            
            # Logging
            if self.update_count % log_interval == 0:
                elapsed = time.time() - self.start_time
                sps = self.total_steps / elapsed if elapsed > 0 else 0
                
                win_rate = np.mean(self.episode_wins) * 100 if self.episode_wins else 0
                mean_return = np.mean(self.episode_returns) if self.episode_returns else 0
                mean_step_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
                
                print(f"[{self.update_count:4d}] {self.total_steps/1e6:.2f}M | "
                      f"{sps/1e3:.1f}k sps | "
                      f"Win: {win_rate:.1f}% | "
                      f"Ret: {mean_return:.2f} | "
                      f"Loss: {stats['loss']:.3f} | "
                      f"Ent: {stats['entropy']:.3f} | "
                      f"ρ: {stats['mean_rho']:.2f}" + 
                      (f" | GM: {len(self.golden_memory)}" if self.use_golden_memory else ""))
            
            # Checkpoint
            if self.update_count % checkpoint_interval == 0:
                self._save_checkpoint()
        
        self._save_checkpoint(final=True)
        print(f"\nTraining complete! Final win rate: {np.mean(self.episode_wins)*100:.1f}%")
    
    def _save_checkpoint(self, final: bool = False):
        name = "final" if final else f"update_{self.update_count}"
        path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'win_rate': np.mean(self.episode_wins) if self.episode_wins else 0,
            'config': {
                'd_model': self.d_model,
                'd_state': self.d_state,
                'num_layers': self.num_layers,
                'rho_bar': self.rho_bar,
                'c_bar': self.c_bar,
                'optimizer': self.optimizer_name,
            }
        }
        
        # Save Golden Memory
        if self.use_golden_memory:
            checkpoint['golden_memory'] = {
                'buffer': self.golden_memory.buffer,
                'returns': self.golden_memory.returns,
                'stats': self.golden_memory.stats(),
            }
        
        torch.save(checkpoint, path)
        print(f"  Saved: {path}")
    
    def close(self):
        # Cancel pending
        for ref in self.pending_rollouts.keys():
            ray.cancel(ref, force=True)
        
        # Close workers
        for w in self.workers:
            try:
                ray.get(w.close.remote(), timeout=5)
            except:
                pass
        
        ray.shutdown()


# =============================================================================
# MAIN
# =============================================================================

def main():
    learner = IMPALALearner(
        num_workers=24,
        num_agents=1,
        rollout_length=256,
        batch_size=8,  # 8 Rollouts pro Update
        lr=3e-4,
        gamma=0.99,
        rho_bar=1.0,
        c_bar=1.0,
        value_coeff=0.5,
        entropy_coeff=0.001,
        max_grad_norm=40.0,
        optimizer="adam",  # "adam", "adamw", "rmsprop"
        weight_decay=0.0,
        # Golden Memory
        use_golden_memory=True,
        golden_capacity=256,
        golden_ratio=0.25,  # 25% des Batches aus Golden Memory
        golden_mode="top_return",  # "top_return", "wins_only", "mixed"
        device="cuda",
        checkpoint_dir="./checkpoints_impala",
        d_model=128,
        d_state=16,
        num_layers=2,
    )
    
    try:
        learner.train(
            total_steps=50_000_000,
            log_interval=10,
            checkpoint_interval=100,
            weight_sync_interval=1,
        )
    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        learner.close()


if __name__ == "__main__":
    main()