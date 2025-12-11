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
from torch.cuda.amp import autocast, GradScaler
import ray
import gfootball.env as football_env

FEATURE_DIM = 93
OBS_DIM = 115
NUM_AGENTS = 11
NUM_ACTIONS = 19

ROLE_ENCODING = np.array([
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
    [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
    [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
], dtype=np.float32)


class GoldenMemory:
    def __init__(self, capacity: int = 256, selection_mode: str = "top_return", golden_ratio: float = 0.25,
                 max_sample_count: int = 10, max_age_updates: int = 500, min_win_margin: float = 0.5,
                 diversity_bonus: float = 0.3):
        self.capacity = capacity
        self.selection_mode = selection_mode
        self.golden_ratio = golden_ratio
        self.max_sample_count = max_sample_count
        self.max_age_updates = max_age_updates
        self.min_win_margin = min_win_margin
        self.diversity_bonus = diversity_bonus
        self.buffer: List[dict] = []
        self.returns: List[float] = []
        self.wins: List[bool] = []
        self.sample_counts: List[int] = []
        self.added_at_update: List[int] = []
        self.current_update = 0

    def add(self, rollout: dict):
        episodes = rollout.get('completed_episodes', [])
        if not episodes:
            rollout_return = float(np.sum(rollout['rewards']))
            won = False
        else:
            rollout_return = np.mean([ep['return'] for ep in episodes])
            won = any(ep['won'] for ep in episodes)
        if won and rollout_return < self.min_win_margin:
            won = False
        should_add = False
        if self.selection_mode == "wins_only":
            should_add = won
        elif self.selection_mode == "top_return":
            if len(self.buffer) < self.capacity:
                should_add = True
            else:
                min_return = min(self.returns) if self.returns else float('-inf')
                should_add = rollout_return > min_return
        elif self.selection_mode == "mixed":
            if won:
                should_add = True
            elif len(self.buffer) < self.capacity:
                should_add = True
            else:
                non_win_returns = [r for r, w in zip(self.returns, self.wins) if not w]
                if non_win_returns:
                    should_add = rollout_return > min(non_win_returns)
        if should_add:
            if len(self.buffer) >= self.capacity:
                idx_to_remove = self._get_worst_index(won)
                self._remove_at(idx_to_remove)
            self.buffer.append(rollout.copy())
            self.returns.append(rollout_return)
            self.wins.append(won)
            self.sample_counts.append(0)
            self.added_at_update.append(self.current_update)

    def _remove_at(self, idx: int):
        self.buffer.pop(idx)
        self.returns.pop(idx)
        self.wins.pop(idx)
        self.sample_counts.pop(idx)
        self.added_at_update.pop(idx)

    def _get_worst_index(self, new_is_win: bool) -> int:
        oversampled = [i for i, c in enumerate(self.sample_counts) if c >= self.max_sample_count]
        if oversampled:
            return oversampled[0]
        too_old = [i for i, t in enumerate(self.added_at_update) if self.current_update - t > self.max_age_updates]
        if too_old:
            return min(too_old, key=lambda i: self.added_at_update[i])
        non_win_indices = [i for i, w in enumerate(self.wins) if not w]
        if non_win_indices:
            return min(non_win_indices, key=lambda i: self.returns[i])
        return int(np.argmin(self.returns))

    def sample(self, n: int, prioritized: bool = True) -> List[dict]:
        if len(self.buffer) == 0:
            return []
        self._cleanup_stale()
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        if prioritized:
            weights = self._compute_sample_weights()
            probs = weights / weights.sum()
            indices = np.random.choice(len(self.buffer), size=n, replace=False, p=probs)
        else:
            indices = np.random.choice(len(self.buffer), size=n, replace=False)
        for i in indices:
            self.sample_counts[i] += 1
        return [self.buffer[i] for i in indices]

    def _compute_sample_weights(self) -> np.ndarray:
        weights = np.zeros(len(self.buffer))
        returns_arr = np.array(self.returns)
        shifted = returns_arr - returns_arr.min() + 1.0
        for i in range(len(self.buffer)):
            base_weight = shifted[i]
            if self.wins[i]:
                base_weight *= 2.0
            sample_penalty = 1.0 / (1.0 + self.sample_counts[i] * 0.2)
            age = self.current_update - self.added_at_update[i]
            age_factor = np.exp(-age / self.max_age_updates)
            weights[i] = base_weight * sample_penalty * age_factor
        return weights

    def _cleanup_stale(self):
        indices_to_remove = []
        for i in range(len(self.buffer)):
            age = self.current_update - self.added_at_update[i]
            if age > self.max_age_updates:
                indices_to_remove.append(i)
            elif self.sample_counts[i] >= self.max_sample_count:
                indices_to_remove.append(i)
        for i in sorted(indices_to_remove, reverse=True):
            self._remove_at(i)

    def tick_update(self):
        self.current_update += 1

    def get_golden_batch(self, fresh_rollouts: List[dict], batch_size: int) -> List[dict]:
        effective_ratio = self.golden_ratio
        if len(self.buffer) < 10:
            effective_ratio *= 0.5
        win_ratio = sum(self.wins) / max(len(self.wins), 1)
        if win_ratio < 0.1:
            effective_ratio *= 0.5
        n_golden = int(batch_size * effective_ratio)
        n_fresh = batch_size - n_golden
        if len(fresh_rollouts) >= n_fresh:
            fresh_batch = fresh_rollouts[:n_fresh]
        else:
            fresh_batch = fresh_rollouts
            n_golden = batch_size - len(fresh_batch)
        golden_batch = self.sample(n_golden) if n_golden > 0 else []
        return fresh_batch + golden_batch

    def stats(self) -> dict:
        if not self.returns:
            return {'size': 0, 'mean_return': 0, 'max_return': 0, 'min_return': 0,
                    'num_wins': 0, 'win_ratio': 0, 'mean_samples': 0, 'mean_age': 0}
        num_wins = sum(self.wins)
        return {
            'size': len(self.buffer), 'mean_return': np.mean(self.returns),
            'max_return': np.max(self.returns), 'min_return': np.min(self.returns),
            'num_wins': num_wins, 'win_ratio': num_wins / len(self.buffer),
            'mean_samples': np.mean(self.sample_counts),
            'mean_age': np.mean([self.current_update - t for t in self.added_at_update])}

    def __len__(self):
        return len(self.buffer)


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

    @staticmethod
    def extract_features_multi(obs_multi: np.ndarray) -> np.ndarray:
        return FeatureEngineer.extract_features(obs_multi)


class ResidualMLPBlock(nn.Module):
    """Residual MLP block with pre-norm and SiLU activation"""
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = d_model * expansion
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.5)
        nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = F.silu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return residual + x


class PopArtValueHead(nn.Module):
    """PopArt: Preserving Outputs Precisely, while Adaptively Rescaling Targets (DeepMind)"""
    def __init__(self, input_dim: int, beta: float = 3e-4):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(input_dim, 1)
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sigma', torch.ones(1))
        self.register_buffer('nu', torch.ones(1))
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns denormalized value predictions"""
        normalized = self.linear(x).squeeze(-1)
        return normalized * self.sigma + self.mu

    def forward_normalized(self, x: torch.Tensor) -> torch.Tensor:
        """Returns normalized value predictions (for training)"""
        return self.linear(x).squeeze(-1)

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize V-trace targets for training"""
        return (targets - self.mu) / self.sigma

    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor):
        """Update running statistics and adjust weights to preserve outputs"""
        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()
        batch_mean = targets.mean()
        batch_sq_mean = (targets ** 2).mean()
        self.mu = (1 - self.beta) * self.mu + self.beta * batch_mean
        self.nu = (1 - self.beta) * self.nu + self.beta * batch_sq_mean
        self.sigma = torch.sqrt(torch.clamp(self.nu - self.mu ** 2, min=1e-4))
        self.linear.weight.data = self.linear.weight.data * old_sigma / self.sigma
        self.linear.bias.data = (self.linear.bias.data * old_sigma + old_mu - self.mu) / self.sigma


class MultiAgentMLPNet(nn.Module):
    """
    Pure MLP architecture with parameter count equivalent to the Mamba model (~660k params).
    Uses residual blocks with expansion factor for expressivity.
    Config: d_model=224, layers=3, expansion=2 → 663,916 params (vs Mamba 658,732)
    """
    def __init__(self, obs_dim: int = OBS_DIM, feature_dim: int = FEATURE_DIM, num_actions: int = NUM_ACTIONS,
                 num_agents: int = NUM_AGENTS, agent_embed_dim: int = 8, action_embed_dim: int = 16,
                 d_model: int = 224, num_layers: int = 3, expansion: int = 2, dropout: float = 0.0,
                 popart_beta: float = 3e-4):
        super().__init__()
        self.obs_dim = obs_dim
        self.feature_dim = feature_dim
        self.num_actions = num_actions
        self.num_agents = num_agents
        self.agent_embed_dim = agent_embed_dim
        self.action_embed_dim = action_embed_dim
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Role encoding
        role_matrix = torch.tensor(ROLE_ENCODING, dtype=torch.float32)
        self.register_buffer('role_encoding', role_matrix)
        self.role_dim = 4
        
        # Embeddings
        self.agent_embedding = nn.Embedding(num_agents, agent_embed_dim)
        nn.init.normal_(self.agent_embedding.weight, std=0.1)
        self.action_embedding = nn.Embedding(num_actions + 1, action_embed_dim)
        nn.init.normal_(self.action_embedding.weight, std=0.02)
        
        # Input projection
        input_dim = obs_dim + feature_dim + self.role_dim + agent_embed_dim + action_embed_dim
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Residual MLP blocks
        self.mlp_layers = nn.ModuleList([
            ResidualMLPBlock(d_model, expansion=expansion, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Output normalization
        self.final_norm = nn.LayerNorm(d_model)
        
        # Policy and value heads
        self.policy_head = nn.Linear(d_model, num_actions)
        self.value_head = PopArtValueHead(d_model, beta=popart_beta)
        
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"MultiAgentMLPNet (PopArt + AMP): {total_params:,} params ({trainable_params:,} trainable)")

    def _init_weights(self):
        nn.init.orthogonal_(self.input_proj.weight, gain=np.sqrt(2))
        nn.init.zeros_(self.input_proj.bias)
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)

    def forward_sequence(self, obs: torch.Tensor, features: torch.Tensor, agent_ids: torch.Tensor,
                         prev_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for sequences. MLP processes each timestep independently.
        Returns denormalized values (for V-trace bootstrap).
        """
        B, L, _ = obs.shape
        
        # Expand agent_ids and prev_actions if needed
        if agent_ids.dim() == 1:
            agent_ids_expanded = agent_ids.unsqueeze(1).expand(-1, L)
        else:
            agent_ids_expanded = agent_ids
        if prev_actions.dim() == 1:
            prev_actions_expanded = prev_actions.unsqueeze(1).expand(-1, L)
        else:
            prev_actions_expanded = prev_actions
        
        # Get embeddings
        role = self.role_encoding[agent_ids_expanded]
        agent_embed = self.agent_embedding(agent_ids_expanded)
        action_embed = self.action_embedding(prev_actions_expanded)
        
        # Concatenate inputs
        x = torch.cat([obs, features, role, agent_embed, action_embed], dim=-1)
        
        # Input projection
        x = self.input_proj(x)
        
        # MLP layers (process all timesteps in parallel - no recurrence needed)
        for layer in self.mlp_layers:
            x = layer(x)
        
        # Final norm and heads
        x = self.final_norm(x)
        logits = self.policy_head(x)
        values = self.value_head(x)
        
        return logits, values

    def forward(self, obs: torch.Tensor, features: torch.Tensor, agent_ids: torch.Tensor,
                prev_actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Flexible forward pass handling various input shapes"""
        squeeze_batch = False
        squeeze_seq = False
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).unsqueeze(0)
            features = features.unsqueeze(0).unsqueeze(0)
            if agent_ids.dim() == 0:
                agent_ids = agent_ids.unsqueeze(0)
            if prev_actions.dim() == 0:
                prev_actions = prev_actions.unsqueeze(0)
            squeeze_batch = True
            squeeze_seq = True
        elif obs.dim() == 2:
            obs = obs.unsqueeze(1)
            features = features.unsqueeze(1)
            squeeze_seq = True
        
        logits, values = self.forward_sequence(obs, features, agent_ids, prev_actions)
        
        if squeeze_seq:
            logits = logits.squeeze(1)
            values = values.squeeze(1)
        if squeeze_batch:
            logits = logits.squeeze(0)
            values = values.squeeze(0)
        
        return logits, values

    def get_action(self, obs: torch.Tensor, features: torch.Tensor, agent_ids: torch.Tensor,
                   prev_actions: torch.Tensor, deterministic: bool = False
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action for inference"""
        logits, values = self.forward(obs, features, agent_ids, prev_actions)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions, log_probs, values


def compute_vtrace(behavior_log_probs: torch.Tensor, target_log_probs: torch.Tensor, rewards: torch.Tensor,
                   values: torch.Tensor, bootstrap_values: torch.Tensor, dones: torch.Tensor,
                   gamma: float = 0.99, rho_bar: float = 1.0, c_bar: float = 1.0
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T = rewards.shape
    device = rewards.device
    log_rhos = (target_log_probs - behavior_log_probs).clamp(-20, 20)
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)
    not_done = 1.0 - dones
    values_t_plus_1 = torch.cat([values[:, 1:], bootstrap_values.unsqueeze(1)], dim=1)
    deltas = clipped_rhos * (rewards + gamma * values_t_plus_1 * not_done - values)
    vs_minus_v = torch.zeros(B, T, device=device)
    acc = torch.zeros(B, device=device)
    for t in reversed(range(T)):
        acc = deltas[:, t] + gamma * cs[:, t] * acc * not_done[:, t]
        vs_minus_v[:, t] = acc
    vs = values + vs_minus_v
    vs_t_plus_1 = torch.cat([vs[:, 1:], bootstrap_values.unsqueeze(1)], dim=1)
    advantages = clipped_rhos * (rewards + gamma * vs_t_plus_1 * not_done - values)
    return vs, advantages, clipped_rhos


@ray.remote
class MultiAgentWorker:
    def __init__(self, worker_id: int, num_agents: int = NUM_AGENTS, d_model: int = 224,
                 num_layers: int = 3, expansion: int = 2, agent_embed_dim: int = 8, action_embed_dim: int = 16,
                 popart_beta: float = 3e-4, rollout_length: int = 128, env_name: str = "11_vs_11_easy_stochastic"):
        self.worker_id = worker_id
        self.num_agents = num_agents
        self.num_actions = NUM_ACTIONS
        self.rollout_length = rollout_length
        self.feature_engineer = FeatureEngineer()
        self.env = football_env.create_environment(
            env_name=env_name, representation="simple115v2", number_of_left_players_agent_controls=num_agents,
            rewards="scoring,checkpoints", write_goal_dumps=False, write_full_episode_dumps=False,
            render=False, write_video=False)
        self.device = torch.device('cpu')
        self.model = MultiAgentMLPNet(
            obs_dim=OBS_DIM, feature_dim=FEATURE_DIM, num_agents=num_agents, agent_embed_dim=agent_embed_dim,
            action_embed_dim=action_embed_dim, d_model=d_model, num_layers=num_layers, expansion=expansion,
            popart_beta=popart_beta
        ).to(self.device)
        self.model.eval()
        self.agent_ids = torch.arange(num_agents, dtype=torch.long, device=self.device)
        self.episode_return = 0.0
        self.episode_steps = 0
        self.current_obs = None
        self.current_features = None
        self.prev_actions = None
        self._reset()

    def set_weights(self, weights: dict):
        self.model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})

    def _reset(self):
        raw_obs = self.env.reset()
        self._update_obs(raw_obs)
        self.episode_return = 0.0
        self.episode_steps = 0
        self.prev_actions = torch.full((self.num_agents,), self.num_actions, dtype=torch.long, device=self.device)

    def _update_obs(self, raw_obs):
        raw_obs = np.array(raw_obs) if isinstance(raw_obs, list) else raw_obs
        self.current_obs = raw_obs[:, :OBS_DIM].astype(np.float32)
        self.current_features = self.feature_engineer.extract_features_multi(self.current_obs)

    def collect_rollout(self) -> dict:
        obs_list, features_list, actions_list, behavior_log_probs_list = [], [], [], []
        prev_actions_list = []
        rewards_list, dones_list, step_rewards, completed_episodes = [], [], [], []
        
        for _ in range(self.rollout_length):
            obs_t = torch.from_numpy(self.current_obs).float().to(self.device)
            feat_t = torch.from_numpy(self.current_features).float().to(self.device)
            
            with torch.no_grad():
                actions, log_probs, values = self.model.get_action(
                    obs_t, feat_t, self.agent_ids, self.prev_actions
                )
            
            obs_list.append(self.current_obs.copy())
            features_list.append(self.current_features.copy())
            prev_actions_list.append(self.prev_actions.cpu().numpy().copy())
            actions_list.append(actions.cpu().numpy())
            behavior_log_probs_list.append(log_probs.cpu().numpy())
            self.prev_actions = actions.clone()
            
            raw_obs, reward, done, info = self.env.step(actions.cpu().tolist())
            if hasattr(reward, '__len__'):
                step_reward = float(np.sum(reward))
            else:
                step_reward = float(reward)
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
                completed_episodes.append({'return': self.episode_return, 'won': won, 'length': self.episode_steps})
                self._reset()
            else:
                self._update_obs(raw_obs)
        
        obs_t = torch.from_numpy(self.current_obs).float().to(self.device)
        feat_t = torch.from_numpy(self.current_features).float().to(self.device)
        with torch.no_grad():
            _, bootstrap_values = self.model.forward(
                obs_t, feat_t, self.agent_ids, self.prev_actions
            )
        
        return {
            'obs': np.array(obs_list, dtype=np.float32),
            'features': np.array(features_list, dtype=np.float32),
            'prev_actions': np.array(prev_actions_list, dtype=np.int64),
            'actions': np.array(actions_list, dtype=np.int64),
            'rewards': np.array(rewards_list, dtype=np.float32),
            'dones': np.array(dones_list, dtype=np.float32),
            'behavior_log_probs': np.array(behavior_log_probs_list, dtype=np.float32),
            'bootstrap_values': bootstrap_values.cpu().numpy(),
            'completed_episodes': completed_episodes,
            'step_rewards': step_rewards,
            'worker_id': self.worker_id
        }

    def close(self):
        self.env.close()


class MultiAgentIMPALALearner:
    def __init__(self, num_workers: int = 24, num_agents: int = NUM_AGENTS, rollout_length: int = 128,
                 batch_size: int = 8, lr: float = 3e-4, gamma: float = 0.99, rho_bar: float = 1.0,
                 c_bar: float = 1.0, value_coeff: float = 0.5, entropy_coeff: float = 0.01,
                 max_grad_norm: float = 40.0, optimizer: str = "adam", weight_decay: float = 0.0,
                 d_model: int = 224, num_layers: int = 3, expansion: int = 2,
                 agent_embed_dim: int = 8, action_embed_dim: int = 16, popart_beta: float = 3e-4,
                 use_golden_memory: bool = True, golden_capacity: int = 256,
                 golden_ratio: float = 0.25, golden_mode: str = "top_return", golden_max_samples: int = 10,
                 golden_max_age: int = 500, golden_min_margin: float = 0.5,
                 env_name: str = "11_vs_11_easy_stochastic", device: str = "cuda",
                 checkpoint_dir: str = "./checkpoints_mlp_amp", resume_from: str = None,
                 use_amp: bool = True):
        self.num_workers = num_workers
        self.num_agents = num_agents
        self.num_actions = NUM_ACTIONS
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
        self.num_layers = num_layers
        self.expansion = expansion
        self.agent_embed_dim = agent_embed_dim
        self.action_embed_dim = action_embed_dim
        self.popart_beta = popart_beta
        self.env_name = env_name
        self.use_amp = use_amp and torch.cuda.is_available()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        self.model = MultiAgentMLPNet(
            obs_dim=OBS_DIM, feature_dim=FEATURE_DIM, num_agents=num_agents, agent_embed_dim=agent_embed_dim,
            action_embed_dim=action_embed_dim, d_model=d_model, num_layers=num_layers, expansion=expansion,
            popart_beta=popart_beta
        ).to(self.device)
        
        self.agent_ids = torch.arange(num_agents, dtype=torch.long, device=self.device)
        self.optimizer_name = optimizer
        self.optimizer = self._create_optimizer(optimizer, lr, weight_decay)
        
        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.use_amp)
        
        ray.init(ignore_reinit_error=True)
        self.workers = [
            MultiAgentWorker.remote(worker_id=i, num_agents=num_agents, d_model=d_model,
                                    num_layers=num_layers, expansion=expansion, agent_embed_dim=agent_embed_dim,
                                    action_embed_dim=action_embed_dim, popart_beta=popart_beta,
                                    rollout_length=rollout_length, env_name=env_name)
            for i in range(num_workers)]
        
        self.total_steps = 0
        self.update_count = 0
        self.start_time = None
        self.episode_returns = deque(maxlen=100)
        self.episode_wins = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=1000)
        self.pending_rollouts: Dict[ray.ObjectRef, any] = {}
        self.rollout_queue: List[dict] = []
        self.use_golden_memory = use_golden_memory
        self.golden_memory = GoldenMemory(
            capacity=golden_capacity, selection_mode=golden_mode, golden_ratio=golden_ratio,
            max_sample_count=golden_max_samples, max_age_updates=golden_max_age,
            min_win_margin=golden_min_margin) if use_golden_memory else None
        
        print(f"\n{'='*60}")
        print(f"Multi-Agent MLP IMPALA + PopArt + AMP")
        print(f"{'='*60}")
        print(f"  Device: {self.device}")
        print(f"  Mixed Precision (AMP): {self.use_amp}")
        print(f"  Workers: {num_workers}")
        print(f"  Agents: {num_agents} (parameter sharing)")
        print(f"  Rollout: {rollout_length} steps")
        print(f"  Batch size: {batch_size} rollouts")
        print(f"  Optimizer: {optimizer} (lr={lr}, wd={weight_decay})")
        print(f"  V-trace: ρ̄={rho_bar}, c̄={c_bar}, γ={gamma}")
        print(f"  PopArt: β={popart_beta} (adaptive value normalization)")
        print(f"  Model: d_model={d_model}, layers={num_layers}, expansion={expansion}")
        print(f"  Agent Embedding: {agent_embed_dim} dim + 4 dim role")
        print(f"  Action Embedding: {action_embed_dim} dim")
        print(f"  Environment: {env_name}")
        if use_golden_memory:
            print(f"  Golden Memory: capacity={golden_capacity}, ratio={golden_ratio}, mode={golden_mode}")
        print(f"{'='*60}\n")
        
        if resume_from is not None:
            self.load_checkpoint(resume_from)

    def _create_optimizer(self, optimizer: str, lr: float, weight_decay: float):
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
        weights = self._get_weights()
        worker.set_weights.remote(weights)

    def _broadcast_weights(self):
        weights = self._get_weights()
        ray.get([w.set_weights.remote(weights) for w in self.workers])

    def _compute_target_log_probs(self, obs: torch.Tensor, features: torch.Tensor, agent_ids: torch.Tensor,
                                  prev_actions: torch.Tensor, actions: torch.Tensor
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns log_probs, denormalized values (for V-trace), entropy"""
        logits, values_denorm = self.model.forward_sequence(obs, features, agent_ids, prev_actions)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values_denorm, entropy

    def _update(self, rollouts: List[dict]) -> dict:
        B = len(rollouts)
        T = self.rollout_length
        N = self.num_agents
        
        obs_batch = np.stack([r['obs'] for r in rollouts])
        feat_batch = np.stack([r['features'] for r in rollouts])
        prev_actions_batch = np.stack([r['prev_actions'] for r in rollouts])
        actions_batch = np.stack([r['actions'] for r in rollouts])
        behavior_lp_batch = np.stack([r['behavior_log_probs'] for r in rollouts])
        rewards_batch = np.stack([r['rewards'] for r in rollouts])
        dones_batch = np.stack([r['dones'] for r in rollouts])
        bootstrap_batch = np.stack([r['bootstrap_values'] for r in rollouts])
        
        rewards_expanded = np.tile(rewards_batch[:, :, None], (1, 1, N))
        dones_expanded = np.tile(dones_batch[:, :, None], (1, 1, N))
        
        obs = torch.from_numpy(obs_batch).float().to(self.device)
        obs = obs.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        features = torch.from_numpy(feat_batch).float().to(self.device)
        features = features.permute(0, 2, 1, 3).reshape(B * N, T, -1)
        prev_actions = torch.from_numpy(prev_actions_batch).long().to(self.device)
        prev_actions = prev_actions.permute(0, 2, 1).reshape(B * N, T)
        actions = torch.from_numpy(actions_batch).long().to(self.device)
        actions = actions.permute(0, 2, 1).reshape(B * N, T)
        behavior_log_probs = torch.from_numpy(behavior_lp_batch).float().to(self.device)
        behavior_log_probs = behavior_log_probs.permute(0, 2, 1).reshape(B * N, T)
        rewards = torch.from_numpy(rewards_expanded).float().to(self.device)
        rewards = rewards.permute(0, 2, 1).reshape(B * N, T)
        dones = torch.from_numpy(dones_expanded).float().to(self.device)
        dones = dones.permute(0, 2, 1).reshape(B * N, T)
        bootstrap_vals = torch.from_numpy(bootstrap_batch).float().to(self.device)
        bootstrap_vals = bootstrap_vals.reshape(B * N)
        agent_ids = self.agent_ids.repeat(B)
        
        # Forward pass with AMP
        with autocast(enabled=self.use_amp):
            target_log_probs, values_denorm, entropy = self._compute_target_log_probs(
                obs, features, agent_ids, prev_actions, actions
            )
        
        # V-trace computation (in float32 for numerical stability)
        with torch.no_grad():
            vs, advantages, rhos = compute_vtrace(
                behavior_log_probs=behavior_log_probs,
                target_log_probs=target_log_probs.float().detach(),
                rewards=rewards, values=values_denorm.float().detach(),
                bootstrap_values=bootstrap_vals.float(),
                dones=dones, gamma=self.gamma, rho_bar=self.rho_bar, c_bar=self.c_bar)
            
            # Update PopArt statistics
            self.model.value_head.update_stats(vs)
            
            # Normalize targets and values with updated statistics
            vs_normalized = self.model.value_head.normalize_targets(vs)
            values_normalized = self.model.value_head.normalize_targets(values_denorm.float())
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Loss computation with AMP
        with autocast(enabled=self.use_amp):
            policy_loss = -(target_log_probs * advantages.detach()).mean()
            value_loss = F.mse_loss(values_normalized, vs_normalized.detach())
            entropy_loss = -entropy.mean()
            loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
        
        # Backward with gradient scaling
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        self.update_count += 1
        if self.use_golden_memory:
            self.golden_memory.tick_update()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.mean().item(),
            'grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
            'mean_rho': rhos.mean().item(),
            'max_rho': rhos.max().item(),
            'popart_mu': self.model.value_head.mu.item(),
            'popart_sigma': self.model.value_head.sigma.item(),
            'amp_scale': self.scaler.get_scale() if self.use_amp else 1.0
        }

    def train(self, total_steps: int = 100_000_000, log_interval: int = 10, checkpoint_interval: int = 100,
              weight_sync_interval: int = 1):
        print(f"Starting Multi-Agent MLP IMPALA training for {total_steps:,} steps...")
        print(f"  Effective samples per update: {self.batch_size} * {self.rollout_length} * {self.num_agents} = {self.batch_size * self.rollout_length * self.num_agents:,}")
        self.start_time = time.time()
        self._broadcast_weights()
        
        for worker in self.workers:
            ref = worker.collect_rollout.remote()
            self.pending_rollouts[ref] = worker
        
        while self.total_steps < total_steps:
            while len(self.rollout_queue) < self.batch_size:
                done_refs, _ = ray.wait(list(self.pending_rollouts.keys()), num_returns=1)
                for ref in done_refs:
                    worker = self.pending_rollouts.pop(ref)
                    rollout = ray.get(ref)
                    self.rollout_queue.append(rollout)
                    self.total_steps += len(rollout['obs']) * self.num_agents
                    for ep in rollout['completed_episodes']:
                        self.episode_returns.append(ep['return'])
                        self.episode_wins.append(1.0 if ep['won'] else 0.0)
                        self.episode_lengths.append(ep['length'])
                    self.recent_rewards.extend(rollout['step_rewards'])
                    if self.use_golden_memory:
                        self.golden_memory.add(rollout)
                    if self.update_count % weight_sync_interval == 0:
                        self._update_worker_weights(worker)
                    new_ref = worker.collect_rollout.remote()
                    self.pending_rollouts[new_ref] = worker
            
            fresh_batch = self.rollout_queue[:self.batch_size]
            self.rollout_queue = self.rollout_queue[self.batch_size:]
            
            if self.use_golden_memory and len(self.golden_memory) > 0:
                batch = self.golden_memory.get_golden_batch(fresh_batch, self.batch_size)
            else:
                batch = fresh_batch
            
            stats = self._update(batch)
            win_rate = np.mean(self.episode_wins) * 100 if self.episode_wins else 0
            mean_return = np.mean(self.episode_returns) if self.episode_returns else 0
            max_return = np.max(self.episode_returns) if self.episode_returns else 0
            
            if self.update_count % log_interval == 0:
                elapsed = time.time() - self.start_time
                sps = self.total_steps / elapsed if elapsed > 0 else 0
                gm_stats = self.golden_memory.stats() if self.use_golden_memory else {}
                gm_info = f" | GM: {gm_stats.get('size', 0)}({gm_stats.get('num_wins', 0)}W)" if self.use_golden_memory else ""
                popart_info = f" | μ:{stats['popart_mu']:.1f} σ:{stats['popart_sigma']:.1f}"
                amp_info = f" | scale:{stats['amp_scale']:.0f}" if self.use_amp else ""
                print(f"[{self.update_count:5d}] {self.total_steps/1e6:.2f}M | "
                      f"{sps/1e3:.1f}k sps | Win: {win_rate:5.1f}% | Ret: {mean_return:6.2f} | "
                      f"Max: {max_return:6.2f} | Loss: {stats['loss']:.3f} | Ent: {stats['entropy']:.3f} | "
                      f"ρ: {stats['mean_rho']:.2f}{popart_info}{amp_info}{gm_info}")
            
            if self.update_count % checkpoint_interval == 0:
                self._save_checkpoint()
            
            if win_rate == 100.0 and len(self.episode_wins) >= 20:
                print(f"\n🎉 100% Win Rate erreicht nach {self.total_steps/1e6:.2f}M steps!")
                self._save_checkpoint(name=f"perfect_{self.update_count}")
                break
        
        self._save_checkpoint(final=True)
        print(f"\nTraining complete! Final win rate: {np.mean(self.episode_wins)*100:.1f}%")

    def _save_checkpoint(self, final: bool = False, name: str = None):
        if name is None:
            name = "final" if final else f"update_{self.update_count}"
        path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'total_steps': self.total_steps, 'update_count': self.update_count,
            'win_rate': np.mean(self.episode_wins) if self.episode_wins else 0,
            'episode_returns': list(self.episode_returns), 'episode_wins': list(self.episode_wins),
            'episode_lengths': list(self.episode_lengths),
            'config': {
                'num_agents': self.num_agents, 'd_model': self.d_model, 'num_layers': self.num_layers,
                'expansion': self.expansion, 'agent_embed_dim': self.agent_embed_dim,
                'action_embed_dim': self.action_embed_dim, 'rho_bar': self.rho_bar, 'c_bar': self.c_bar,
                'optimizer': self.optimizer_name, 'lr': self.lr, 'gamma': self.gamma,
                'entropy_coeff': self.entropy_coeff, 'value_coeff': self.value_coeff,
                'env_name': self.env_name, 'use_amp': self.use_amp
            }
        }
        if self.use_golden_memory and self.golden_memory is not None:
            checkpoint['golden_memory'] = {
                'buffer': self.golden_memory.buffer, 'returns': self.golden_memory.returns,
                'wins': self.golden_memory.wins, 'sample_counts': self.golden_memory.sample_counts,
                'added_at_update': self.golden_memory.added_at_update,
                'current_update': self.golden_memory.current_update,
                'stats': self.golden_memory.stats()
            }
        torch.save(checkpoint, path)
        print(f"  💾 Saved: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        print(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            opt_state = checkpoint['optimizer_state_dict']
            for state in opt_state['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)
            self.optimizer.load_state_dict(opt_state)
        if self.use_amp and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict'] is not None:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        self.total_steps = checkpoint.get('total_steps', 0)
        self.update_count = checkpoint.get('update_count', 0)
        if 'episode_returns' in checkpoint:
            self.episode_returns = deque(checkpoint['episode_returns'], maxlen=100)
        if 'episode_wins' in checkpoint:
            self.episode_wins = deque(checkpoint['episode_wins'], maxlen=100)
        if 'episode_lengths' in checkpoint:
            self.episode_lengths = deque(checkpoint['episode_lengths'], maxlen=100)
        if self.use_golden_memory:
            self.golden_memory.buffer = []
            self.golden_memory.returns = []
            self.golden_memory.wins = []
            self.golden_memory.sample_counts = []
            self.golden_memory.added_at_update = []
            self.golden_memory.current_update = self.update_count
        win_rate = checkpoint.get('win_rate', 0) * 100
        print(f"  Restored: {self.total_steps/1e6:.2f}M steps, {self.update_count} updates, {win_rate:.1f}% win rate")
        return checkpoint

    def close(self):
        for ref in list(self.pending_rollouts.keys()):
            try:
                ray.cancel(ref)
            except Exception:
                pass
        for w in self.workers:
            try:
                ray.get(w.close.remote(), timeout=5)
            except Exception:
                pass
        ray.shutdown()


def main():
    RESUME_FROM = None
    learner = MultiAgentIMPALALearner(
        num_workers=24, num_agents=2, env_name="11_vs_11_easy_stochastic",
        rollout_length=256, batch_size=8, lr=1e-4, gamma=0.999, rho_bar=1.0, c_bar=1.0,
        value_coeff=0.5, entropy_coeff=0.0005, max_grad_norm=40.0, optimizer="adam", weight_decay=0.0,
        # MLP architecture (~660k params, equivalent to Mamba model)
        d_model=224, num_layers=3, expansion=2,
        agent_embed_dim=8, action_embed_dim=16, popart_beta=3e-4,
        use_golden_memory=True, golden_capacity=256, golden_ratio=0.25, golden_mode="mixed",
        golden_max_samples=10, golden_max_age=500, golden_min_margin=0.5,
        device="cuda", checkpoint_dir="./checkpoints_mlp_amp", resume_from=RESUME_FROM,
        use_amp=True  # Mixed precision training
    )
    try:
        learner.train(total_steps=100_000_000_000, log_interval=10, checkpoint_interval=100, weight_sync_interval=1)
    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        learner.close()


if __name__ == "__main__":
    main()