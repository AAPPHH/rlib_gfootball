"""
GFootball Simple IPPO mit Feature Engineer
- Korrektes Episode-Tracking (auch während laufender Episodes)
- Dein 93-Feature Feature Engineer
- Einfache MLP-Architektur (bewährt stabil)
- Sauberes Monitoring
"""

import math
import time
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import ray
import gfootball.env as football_env


# =============================================================================
# FEATURE ENGINEER (deine 93 Features)
# =============================================================================

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

FEATURE_DIM = 93
OBS_DIM = 115  # simple115v2


class FeatureEngineer:
    """Dein 93-Feature Feature Engineer."""
    
    GOAL_POS = np.array([1.0, 0.0], dtype=np.float32)
    OWN_GOAL_POS = np.array([-1.0, 0.0], dtype=np.float32)
    GOAL_TOP = np.array([1.0, 0.044], dtype=np.float32)
    GOAL_BOTTOM = np.array([1.0, -0.044], dtype=np.float32)

    @staticmethod
    def extract_features(obs: np.ndarray) -> np.ndarray:
        """Extract 93 engineered features from simple115v2 observation."""
        squeeze_output = False
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
            squeeze_output = True
        
        B = obs.shape[0]
        obs115 = obs[:, :115] if obs.shape[1] >= 115 else np.pad(obs, ((0, 0), (0, 115 - obs.shape[1])), mode='constant')
        feat = np.zeros((B, FEATURE_DIM), dtype=np.float32)
        
        # Parse observation
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
        
        # Ball state (0-6)
        feat[:, 0] = ball_pos[:, 0]
        feat[:, 1] = ball_pos[:, 1]
        feat[:, 2] = np.clip(ball_z[:, 0], 0, 1)
        feat[:, 3] = np.clip(ball_speed, 0, 2)
        feat[:, 4] = ball_dir[:, 0]
        feat[:, 5] = ball_dir[:, 1]
        feat[:, 6] = ball_owned_team / 2.0
        
        # Relative ball (7-10)
        rel_ball = ball_pos - active_pos
        dist_to_ball = np.linalg.norm(rel_ball, axis=1)
        angle_to_ball = np.arctan2(rel_ball[:, 1], rel_ball[:, 0])
        feat[:, 7] = rel_ball[:, 0]
        feat[:, 8] = rel_ball[:, 1]
        feat[:, 9] = np.clip(dist_to_ball, 0, 2)
        feat[:, 10] = angle_to_ball / np.pi
        
        # Goal geometry (11-15)
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
        
        # Keeper (16-17)
        right_x = right_pos[:, :, 0]
        keeper_idx = np.argmax(right_x, axis=1)
        keeper_pos = right_pos[batch_idx, keeper_idx]
        keeper_dist = np.linalg.norm(ball_pos - keeper_pos, axis=1)
        keeper_vec = ball_pos - keeper_pos
        keeper_angle = np.arctan2(keeper_vec[:, 1], keeper_vec[:, 0])
        keeper_valid = keeper_pos[:, 0] > 0.7
        feat[:, 16] = np.where(keeper_valid, np.clip(keeper_dist, 0, 2), 1.0)
        feat[:, 17] = np.where(keeper_valid, keeper_angle / np.pi, 0.0)
        
        # Closest teammates (18-37)
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
        
        # Closest opponents (38-57)
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
        
        # Team structure (58-62)
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
        
        # Pressure/space (63-66)
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
        
        # Offside (67-68)
        sorted_right_x = np.sort(right_x, axis=1)
        offside_line = np.maximum(ball_x, sorted_right_x[:, 1])
        active_x = left_pos[batch_idx, active_idx, 0]
        feat[:, 67] = offside_line
        feat[:, 68] = ((active_x > offside_line) & (ball_owned_team == 0)).astype(np.float32)
        
        # Zones (69-73)
        feat[:, 69] = (ball_x > 0.33).astype(np.float32)
        feat[:, 70] = ((ball_x >= -0.33) & (ball_x <= 0.33)).astype(np.float32)
        feat[:, 71] = (ball_x < -0.33).astype(np.float32)
        ball_y = ball_pos[:, 1]
        feat[:, 72] = (ball_y > 0.2).astype(np.float32)
        feat[:, 73] = (ball_y < -0.2).astype(np.float32)
        
        # Sticky actions (74-77)
        sticky_dirs = sticky[:, :8]
        sticky_dir_idx = np.argmax(sticky_dirs, axis=1)
        sticky_dir_active = np.any(sticky_dirs > 0, axis=1)
        sticky_angle = sticky_dir_idx * (2 * np.pi / 8)
        feat[:, 74] = sticky[:, 8]  # sprint
        feat[:, 75] = sticky[:, 9]  # dribble
        feat[:, 76] = np.where(sticky_dir_active, np.cos(sticky_angle), 0)
        feat[:, 77] = np.where(sticky_dir_active, np.sin(sticky_angle), 0)
        
        # Game state (78-85)
        feat[:, 78:85] = game_mode
        feat[:, 85] = 0.5  # steps_remaining placeholder
        
        # Score context (86-89)
        feat[:, 86] = 0.0  # score_diff
        feat[:, 87] = 0.0  # winning
        feat[:, 88] = 0.0  # losing
        feat[:, 89] = 1.0  # drawing
        
        # Composite features (90-92)
        feat[:, 90] = shooting_angle * np.where(keeper_valid, np.clip(keeper_dist, 0, 1), 1.0)
        feat[:, 91] = np.clip(ball_dir[:, 0] * ball_speed, -1, 1)
        feat[:, 92] = np.clip(feat[:, 64], 0, 1) * 0.4 + np.clip(feat[:, 91], 0, 1) * 0.3 + np.clip(numerical_adv / 5.0 + 0.5, 0, 1) * 0.3
        
        if squeeze_output:
            return feat[0]
        return feat


# =============================================================================
# NETWORK
# =============================================================================

def orthogonal_init(layer, gain=1.0):
    """Orthogonal initialization - critical for RL stability."""
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=gain)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)


class SimplePPONet(nn.Module):
    """Simple MLP for PPO - proven architecture."""
    
    def __init__(self, obs_dim: int, feature_dim: int, num_actions: int = 19, hidden: int = 256):
        super().__init__()
        input_dim = obs_dim + feature_dim  # 115 + 93 = 208
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )
        
        # Value head
        self.value = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        
        # Apply orthogonal init
        self.apply(lambda m: orthogonal_init(m, gain=np.sqrt(2)))
        orthogonal_init(self.policy[-1], gain=0.01)  # Small init for policy
        orthogonal_init(self.value[-1], gain=1.0)
        
    def forward(self, obs: torch.Tensor, features: torch.Tensor):
        x = torch.cat([obs, features], dim=-1)
        h = self.encoder(x)
        logits = self.policy(h)
        value = self.value(h).squeeze(-1)
        return logits, value
    
    def get_action(self, obs: torch.Tensor, features: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs, features)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
    
    def evaluate(self, obs: torch.Tensor, features: torch.Tensor, actions: torch.Tensor):
        logits, value = self.forward(obs, features)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, value


# =============================================================================
# ROLLOUT STORAGE
# =============================================================================

@dataclass
class RolloutBuffer:
    """Storage for rollout data."""
    obs: List[np.ndarray]
    features: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]
    log_probs: List[float]
    values: List[float]
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.obs = []
        self.features = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.log_probs = []
        self.values = []
    
    def add(self, obs, features, action, reward, done, log_prob, value):
        self.obs.append(obs)
        self.features.append(features)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def __len__(self):
        return len(self.obs)


# =============================================================================
# WORKER
# =============================================================================

@ray.remote
class RolloutWorker:
    """Collects rollouts from environment."""
    
    def __init__(self, worker_id: int, num_agents: int = 1):
        self.worker_id = worker_id
        self.num_agents = num_agents
        self.feature_engineer = FeatureEngineer()
        
        # Create environment
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
        
        # Model (will be set via set_weights)
        self.device = torch.device('cpu')
        self.model = SimplePPONet(OBS_DIM, FEATURE_DIM).to(self.device)
        self.model.eval()
        
        # Episode tracking
        self.episode_return = 0.0
        self.episode_steps = 0
        self.current_obs = None
        self.current_features = None
        self._reset()
    
    def set_weights(self, weights: dict):
        """Update model weights."""
        self.model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
    
    def _reset(self):
        """Reset environment and episode tracking."""
        raw_obs = self.env.reset()
        self._update_obs(raw_obs)
        self.episode_return = 0.0
        self.episode_steps = 0
    
    def _update_obs(self, raw_obs):
        """Process raw observation."""
        if isinstance(raw_obs, list):
            raw_obs = np.array(raw_obs)
        if raw_obs.ndim == 1:
            raw_obs = raw_obs.reshape(1, -1)
        self.current_obs = raw_obs[0][:OBS_DIM].astype(np.float32)
        self.current_features = self.feature_engineer.extract_features(self.current_obs)
    
    def collect_rollout(self, num_steps: int = 256) -> dict:
        """Collect rollout data."""
        buffer = RolloutBuffer()
        
        # Tracking für diese Rollout
        step_rewards = []  # Alle step rewards
        completed_episodes = []  # (return, won, length)
        
        for _ in range(num_steps):
            # Get action
            obs_t = torch.from_numpy(self.current_obs).float().unsqueeze(0)
            feat_t = torch.from_numpy(self.current_features).float().unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value = self.model.get_action(obs_t, feat_t)
            
            action_int = action.item()
            
            # Step environment
            env_action = action_int if self.num_agents == 1 else [action_int]
            raw_obs, reward, done, info = self.env.step(env_action)
            
            # Process reward
            step_reward = float(reward) if np.isscalar(reward) else float(reward[0]) if len(reward) > 0 else 0.0
            step_rewards.append(step_reward)
            
            self.episode_return += step_reward
            self.episode_steps += 1
            
            # Check episode end
            episode_done = bool(done) or self.episode_steps >= 3000
            
            # Store transition
            buffer.add(
                obs=self.current_obs.copy(),
                features=self.current_features.copy(),
                action=action_int,
                reward=step_reward,
                done=episode_done,
                log_prob=log_prob.item(),
                value=value.item(),
            )
            
            # Handle episode end
            if episode_done:
                # Determine win
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
        
        # Get bootstrap value for GAE
        obs_t = torch.from_numpy(self.current_obs).float().unsqueeze(0)
        feat_t = torch.from_numpy(self.current_features).float().unsqueeze(0)
        with torch.no_grad():
            _, bootstrap_value = self.model.forward(obs_t, feat_t)
        
        return {
            'obs': np.array(buffer.obs, dtype=np.float32),
            'features': np.array(buffer.features, dtype=np.float32),
            'actions': np.array(buffer.actions, dtype=np.int64),
            'rewards': np.array(buffer.rewards, dtype=np.float32),
            'dones': np.array(buffer.dones, dtype=np.float32),
            'log_probs': np.array(buffer.log_probs, dtype=np.float32),
            'values': np.array(buffer.values, dtype=np.float32),
            'bootstrap_value': bootstrap_value.item(),
            # Episode stats
            'completed_episodes': completed_episodes,
            'step_rewards': step_rewards,
            'running_episode_return': self.episode_return,  # Laufende Episode
            'running_episode_steps': self.episode_steps,
        }
    
    def close(self):
        self.env.close()


# =============================================================================
# TRAINER
# =============================================================================

class PPOTrainer:
    """Simple PPO Trainer with proper monitoring."""
    
    def __init__(
        self,
        num_workers: int = 16,
        num_agents: int = 1,
        rollout_length: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.01,
        max_grad_norm: float = 0.5,
        num_epochs: int = 4,
        minibatch_size: int = 512,
        device: str = "cuda",
        checkpoint_dir: str = "./checkpoints_ippo_v2",
    ):
        self.num_workers = num_workers
        self.num_agents = num_agents
        self.rollout_length = rollout_length
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.minibatch_size = minibatch_size
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Model
        self.model = SimplePPONet(OBS_DIM, FEATURE_DIM).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        
        # Workers
        ray.init(ignore_reinit_error=True)
        self.workers = [
            RolloutWorker.remote(worker_id=i, num_agents=num_agents)
            for i in range(num_workers)
        ]
        
        # Stats
        self.total_steps = 0
        self.update_count = 0
        self.start_time = None
        
        # Episode tracking mit Ringbuffer
        self.episode_returns = deque(maxlen=100)
        self.episode_wins = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.recent_rewards = deque(maxlen=1000)  # Step rewards
        
        print(f"PPO Trainer initialized")
        print(f"  Device: {self.device}")
        print(f"  Workers: {num_workers}")
        print(f"  Agents: {num_agents}")
        print(f"  Params: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def _get_weights(self) -> dict:
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
    
    def _sync_weights(self):
        weights = self._get_weights()
        ray.get([w.set_weights.remote(weights) for w in self.workers])
    
    def _compute_gae(self, rewards, values, dones, bootstrap_value):
        """Compute GAE advantages."""
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)
        
        last_gae = 0.0
        last_value = bootstrap_value
        
        for t in reversed(range(T)):
            if dones[t]:
                delta = rewards[t] - values[t]
                last_gae = delta
            else:
                delta = rewards[t] + self.gamma * last_value - values[t]
                last_gae = delta + self.gamma * self.gae_lambda * last_gae
            
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
            last_value = values[t]
        
        return advantages, returns
    
    def _update(self, rollouts: list) -> dict:
        """PPO update step."""
        # Aggregate rollouts
        all_obs, all_features, all_actions = [], [], []
        all_advantages, all_returns, all_old_log_probs = [], [], []
        
        for rollout in rollouts:
            advantages, returns = self._compute_gae(
                rollout['rewards'],
                rollout['values'],
                rollout['dones'],
                rollout['bootstrap_value'],
            )
            all_obs.append(rollout['obs'])
            all_features.append(rollout['features'])
            all_actions.append(rollout['actions'])
            all_advantages.append(advantages)
            all_returns.append(returns)
            all_old_log_probs.append(rollout['log_probs'])
        
        # Concatenate
        obs = torch.from_numpy(np.concatenate(all_obs)).float().to(self.device)
        features = torch.from_numpy(np.concatenate(all_features)).float().to(self.device)
        actions = torch.from_numpy(np.concatenate(all_actions)).long().to(self.device)
        advantages = torch.from_numpy(np.concatenate(all_advantages)).float().to(self.device)
        returns = torch.from_numpy(np.concatenate(all_returns)).float().to(self.device)
        old_log_probs = torch.from_numpy(np.concatenate(all_old_log_probs)).float().to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        batch_size = len(obs)
        total_loss, policy_loss_sum, value_loss_sum, entropy_sum = 0, 0, 0, 0
        num_updates = 0
        
        for _ in range(self.num_epochs):
            indices = torch.randperm(batch_size, device=self.device)
            
            for start in range(0, batch_size, self.minibatch_size):
                end = min(start + self.minibatch_size, batch_size)
                mb_idx = indices[start:end]
                
                # Evaluate
                log_probs, entropy, values = self.model.evaluate(
                    obs[mb_idx], features[mb_idx], actions[mb_idx]
                )
                
                # Policy loss (clipped)
                ratio = torch.exp(log_probs - old_log_probs[mb_idx])
                surr1 = ratio * advantages[mb_idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages[mb_idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = F.mse_loss(values, returns[mb_idx])
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += entropy.mean().item()
                num_updates += 1
        
        self.update_count += 1
        
        return {
            'loss': total_loss / num_updates,
            'policy_loss': policy_loss_sum / num_updates,
            'value_loss': value_loss_sum / num_updates,
            'entropy': entropy_sum / num_updates,
        }
    
    def train(self, total_steps: int = 50_000_000, log_interval: int = 10, checkpoint_interval: int = 100):
        """Main training loop."""
        print(f"\nStarting training for {total_steps:,} steps...")
        self.start_time = time.time()
        
        # Initial weight sync
        self._sync_weights()
        
        # Start initial rollouts
        pending = {w.collect_rollout.remote(self.rollout_length): w for w in self.workers}
        
        while self.total_steps < total_steps:
            # Wait for rollouts
            done_refs, _ = ray.wait(list(pending.keys()), num_returns=len(pending))
            
            rollouts = []
            for ref in done_refs:
                worker = pending.pop(ref)
                rollout = ray.get(ref)
                rollouts.append(rollout)
                
                # Track steps
                self.total_steps += len(rollout['obs'])
                
                # Track completed episodes
                for ep in rollout['completed_episodes']:
                    self.episode_returns.append(ep['return'])
                    self.episode_wins.append(1.0 if ep['won'] else 0.0)
                    self.episode_lengths.append(ep['length'])
                
                # Track step rewards
                self.recent_rewards.extend(rollout['step_rewards'])
                
                # Queue next rollout
                pending[worker.collect_rollout.remote(self.rollout_length)] = worker
            
            # Update
            stats = self._update(rollouts)
            
            # Sync weights
            self._sync_weights()
            
            # Logging
            if self.update_count % log_interval == 0:
                elapsed = time.time() - self.start_time
                sps = self.total_steps / elapsed if elapsed > 0 else 0
                
                # Episode stats
                win_rate = np.mean(self.episode_wins) * 100 if self.episode_wins else 0
                mean_return = np.mean(self.episode_returns) if self.episode_returns else 0
                mean_length = np.mean(self.episode_lengths) if self.episode_lengths else 0
                
                # Step reward stats (zeigt auch bei langen Episodes was)
                mean_step_reward = np.mean(self.recent_rewards) if self.recent_rewards else 0
                
                print(f"[{self.update_count:4d}] {self.total_steps/1e6:.2f}M steps | "
                      f"{sps/1e3:.1f}k sps | "
                      f"Win: {win_rate:.1f}% | "
                      f"Return: {mean_return:.2f} | "
                      f"StepR: {mean_step_reward:.4f} | "
                      f"Loss: {stats['loss']:.3f} | "
                      f"Ent: {stats['entropy']:.3f}")
            
            # Checkpoint
            if self.update_count % checkpoint_interval == 0:
                self._save_checkpoint()
        
        # Final checkpoint
        self._save_checkpoint(final=True)
        print(f"\nTraining complete! Final win rate: {np.mean(self.episode_wins)*100:.1f}%")
    
    def _save_checkpoint(self, final: bool = False):
        name = "final" if final else f"update_{self.update_count}"
        path = self.checkpoint_dir / f"checkpoint_{name}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'update_count': self.update_count,
            'win_rate': np.mean(self.episode_wins) if self.episode_wins else 0,
        }, path)
        print(f"  Saved: {path}")
    
    def close(self):
        for w in self.workers:
            try:
                ray.get(w.close.remote())
            except:
                pass
        ray.shutdown()


# =============================================================================
# MAIN
# =============================================================================

def main():
    trainer = PPOTrainer(
        num_workers=24,
        num_agents=1,  # Start mit 1 Agent
        rollout_length=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        value_coeff=0.5,
        entropy_coeff=0.01,
        max_grad_norm=0.5,
        num_epochs=4,
        minibatch_size=512,
        device="cuda",
        checkpoint_dir="./checkpoints_ippo_v2",
    )
    
    try:
        trainer.train(
            total_steps=50_000_000,
            log_interval=10,
            checkpoint_interval=100,
        )
    except KeyboardInterrupt:
        print("\nInterrupted!")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()