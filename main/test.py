"""
Direct 11v11 training without curriculum.
Single stage: academy_single_goal_versus_lazy
"""
import math
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import ray
import gfootball.env as football_env


class FeatureEngineer:
    GOAL_POS = np.array([1.0, 0.0], dtype=np.float32)
    OWN_GOAL_POS = np.array([-1.0, 0.0], dtype=np.float32)
    POST_TOP = np.array([1.0, 0.044], dtype=np.float32)
    POST_BOTTOM = np.array([1.0, -0.044], dtype=np.float32)
    
    @staticmethod
    def extract_features(obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        B = obs.shape[0]
        obs115 = obs[:, :115] if obs.shape[1] >= 115 else np.pad(obs, ((0, 0), (0, 115 - obs.shape[1])))
        feat = np.zeros((B, 32), dtype=np.float32)
        ball_x = obs115[:, 0]
        ball_y = obs115[:, 1]
        ball_z = obs115[:, 2]
        ball_dir_x = obs115[:, 3]
        ball_dir_y = obs115[:, 4]
        ball_pos = np.stack([ball_x, ball_y], axis=-1)
        dist_to_goal = np.linalg.norm(ball_pos - FeatureEngineer.GOAL_POS, axis=-1)
        dist_to_own_goal = np.linalg.norm(ball_pos - FeatureEngineer.OWN_GOAL_POS, axis=-1)
        goal_vec = FeatureEngineer.GOAL_POS - ball_pos
        goal_angle = np.arctan2(goal_vec[:, 1], goal_vec[:, 0])
        in_shooting_range = (dist_to_goal < 0.3).astype(np.float32)
        in_penalty_area = ((ball_x > 0.8) & (np.abs(ball_y) < 0.2)).astype(np.float32)
        ball_speed = np.sqrt(ball_dir_x**2 + ball_dir_y**2)
        ball_dir = np.stack([ball_dir_x, ball_dir_y], axis=-1)
        ball_dir_norm = ball_dir / (ball_speed[:, None] + 1e-8)
        goal_vec_norm = goal_vec / (np.linalg.norm(goal_vec, axis=-1, keepdims=True) + 1e-8)
        moving_to_goal = np.sum(ball_dir_norm * goal_vec_norm, axis=-1)
        moving_to_goal = np.where(ball_speed > 0.01, moving_to_goal, 0.0)
        keeper_x = obs115[:, 88]
        keeper_y = obs115[:, 89]
        keeper_pos = np.stack([keeper_x, keeper_y], axis=-1)
        keeper_dist = np.linalg.norm(ball_pos - keeper_pos, axis=-1)
        keeper_to_ball = ball_pos - keeper_pos
        keeper_angle_to_ball = np.arctan2(keeper_to_ball[:, 1], keeper_to_ball[:, 0])
        vec_top = FeatureEngineer.POST_TOP - ball_pos
        vec_bottom = FeatureEngineer.POST_BOTTOM - ball_pos
        angle_top = np.arctan2(vec_top[:, 1], vec_top[:, 0])
        angle_bottom = np.arctan2(vec_bottom[:, 1], vec_bottom[:, 0])
        shooting_angle = np.abs(angle_top - angle_bottom)
        shooting_angle = np.where(dist_to_goal > 0.01, shooting_angle, 0.0)
        ball_progress = (ball_x + 1.0) / 2.0
        left_x = obs115[:, 22:44:2]
        right_x = obs115[:, 44:66:2]
        teammates_ahead = (left_x > ball_x[:, None]).sum(axis=1)
        defenders_ahead = (right_x > ball_x[:, None]).sum(axis=1)
        numerical_advantage = teammates_ahead - defenders_ahead
        feat[:, 0] = dist_to_goal
        feat[:, 1] = dist_to_own_goal
        feat[:, 2] = goal_angle / np.pi
        feat[:, 3] = in_shooting_range
        feat[:, 4] = in_penalty_area
        feat[:, 5] = ball_speed
        feat[:, 6] = moving_to_goal
        feat[:, 7] = keeper_dist
        feat[:, 8] = keeper_angle_to_ball / np.pi
        feat[:, 9] = shooting_angle / np.pi
        feat[:, 10] = ball_progress
        feat[:, 11] = ball_z
        feat[:, 12] = teammates_ahead / 11.0
        feat[:, 13] = defenders_ahead / 11.0
        feat[:, 14] = numerical_advantage / 11.0
        feat[:, 15] = (keeper_dist < 0.15).astype(np.float32)
        feat[:, 16] = (shooting_angle > 0.1).astype(np.float32)
        feat[:, 17] = ball_x
        feat[:, 18] = ball_y
        feat[:, 19] = ball_dir_x
        feat[:, 20] = ball_dir_y
        if obs.shape[1] >= 100:
            feat[:, 21] = obs115[:, 96]
            feat[:, 22] = obs115[:, 97]
            feat[:, 23] = obs115[:, 98]
            feat[:, 24] = obs115[:, 99]
        feat[:, 25] = (ball_x > 0.5).astype(np.float32)
        feat[:, 26] = (ball_x < -0.5).astype(np.float32)
        feat[:, 27] = (np.abs(ball_y) > 0.3).astype(np.float32)
        feat[:, 28] = np.maximum(0, 1.0 - dist_to_goal)
        feat[:, 29] = np.maximum(0, shooting_angle) * np.maximum(0, 1.0 - keeper_dist)
        feat[:, 30] = ball_progress * np.where(moving_to_goal > 0, 1.0, 0.5)
        feat[:, 31] = in_penalty_area * in_shooting_range
        return feat


def hippo_legs_matrix(N: int) -> torch.Tensor:
    P = torch.zeros(N, N)
    for n in range(N):
        for k in range(n + 1):
            P[n, k] = math.sqrt(2 * n + 1) * math.sqrt(2 * k + 1)
            if n > k:
                P[n, k] *= 1.0
            elif n == k:
                P[n, k] = n + 1
    return -P


@dataclass
class ModelConfig:
    obs_dim: int = 460
    feature_dim: int = 32
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
    num_stages: int = 1  # Only 1 stage now
    stage_emb_dim: int = 8
    dropout: float = 0.0
    
    def __post_init__(self):
        if self.encoder_hidden is None:
            self.encoder_hidden = [256, 256]
        if self.policy_hidden is None:
            self.policy_hidden = [256]
        if self.value_hidden is None:
            self.value_hidden = [256]


class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        hippo_A = hippo_legs_matrix(d_state)
        self.register_buffer('A', hippo_A)
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
        h = h.view(B, self.d_model, self.d_state)
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_in, z = xz.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(x_in))
        A_diag = -torch.exp(self.A_log_diag)
        B_t = self.B_proj(x_in)
        C_t = self.C_proj(x_in)
        outputs = []
        for t in range(L):
            dt_t = dt[:, t, :]
            B_t_t = B_t[:, t, :]
            C_t_t = C_t[:, t, :]
            x_t = x_in[:, t, :]
            dA = torch.exp(dt_t.unsqueeze(-1) * A_diag.unsqueeze(0).unsqueeze(0))
            dB = dt_t.unsqueeze(-1) * B_t_t.unsqueeze(1)
            h = h * dA + x_t.unsqueeze(-1) * dB
            y_t = (h * C_t_t.unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        y = torch.stack(outputs, dim=1)
        out = y * F.silu(z) + x_in * self.D
        out = self.out_proj(out)
        out = self.dropout(out)
        h_out = h.view(B, -1)
        return x + out, h_out


class MambaEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 256, d_state: int = 64, num_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        self.output_dim = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        self.state_size = d_model * d_state
        print(f"Mamba: {num_layers} layers, d_model={d_model}, d_state={d_state}")
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        B, L, D = x.shape
        x = self.input_proj(x)
        if hidden_state is None:
            hidden_state = [torch.zeros(B, self.state_size, device=x.device, dtype=x.dtype) for _ in range(self.num_layers)]
        new_states = []
        for i, layer in enumerate(self.layers):
            x, h_new = layer(x, hidden_state[i])
            new_states.append(h_new)
        x = self.final_norm(x)
        return x, new_states
    
    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        return [torch.zeros(batch_size, self.state_size, device=device) for _ in range(self.num_layers)]


class GFootballPolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig):
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
            input_dim=in_dim,
            d_model=config.d_model,
            d_state=config.mamba_d_state,
            num_layers=config.mamba_layers,
            dropout=config.dropout
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
    
    def _normalize_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if obs.dim() == 1:
            return obs.unsqueeze(0).unsqueeze(1), True
        elif obs.dim() == 2:
            return obs.unsqueeze(1), True
        elif obs.dim() == 3:
            return obs, False
        else:
            raise ValueError(f"obs must be 1D, 2D, or 3D, got {obs.dim()}D")
    
    def _normalize_index(self, idx: Optional[torch.Tensor], B: int, L: int, device: torch.device, default_val: int = 0) -> torch.Tensor:
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
            raise ValueError(f"Index must be 0D, 1D, or 2D, got {idx.dim()}D")
        
    def forward(self, obs: torch.Tensor, features: torch.Tensor, stage_idx: torch.Tensor, prev_action: Optional[torch.Tensor] = None, hidden_state: Optional[List[torch.Tensor]] = None, return_hidden: bool = False) -> Dict[str, Any]:
        obs, squeeze_output = self._normalize_obs(obs)
        B, L, _ = obs.shape
        device = obs.device
        if features.dim() == 1:
            features = features.unsqueeze(0).unsqueeze(1)
        elif features.dim() == 2:
            features = features.unsqueeze(1)
        stage_idx = self._normalize_index(stage_idx, B, L, device, default_val=0)
        prev_action = self._normalize_index(prev_action, B, L, device, default_val=0)
        stage_emb = self.stage_embedding(stage_idx)
        action_emb = self.action_embedding(prev_action)
        x = torch.cat([obs, features, stage_emb, action_emb], dim=-1)
        x = self.obs_encoder(x)
        x, new_hidden = self.mamba(x, hidden_state)
        logits = self.policy_head(x)
        v = F.relu(self.value_fc1(x))
        v = self.value_fc2(v)
        B_size, L_size = v.shape[:2]
        v = v.view(B_size, L_size, self.num_value_heads, self.value_out_dim)
        if self.config.use_distributional:
            value_logits = v.mean(dim=2)
            value_probs = F.softmax(value_logits, dim=-1)
            value = (value_probs * self.value_support).sum(-1, keepdim=True)
        else:
            value = v.mean(dim=2)
            value_logits = None
        log_probs = F.log_softmax(logits, dim=-1)
        if squeeze_output:
            logits = logits.squeeze(1)
            value = value.squeeze(1)
            log_probs = log_probs.squeeze(1)
            if value_logits is not None:
                value_logits = value_logits.squeeze(1)
        result = {'logits': logits, 'value': value.squeeze(-1) if value.dim() > 1 else value, 'log_probs': log_probs}
        if value_logits is not None:
            result['value_logits'] = value_logits
        if return_hidden:
            result['hidden_state'] = new_hidden
        return result
    
    def get_action(self, obs: torch.Tensor, features: torch.Tensor, stage_idx: torch.Tensor, prev_action: Optional[torch.Tensor] = None, hidden_state: Optional[List[torch.Tensor]] = None, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        output = self.forward(obs, features, stage_idx, prev_action, hidden_state, return_hidden=True)
        logits = output['logits']
        if torch.isnan(logits).any():
            logits = torch.zeros_like(logits)
        logits = logits.clamp(min=-20.0, max=20.0)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, output['value'], output.get('hidden_state')
    
    def evaluate_actions(self, obs: torch.Tensor, features: torch.Tensor, stage_idx: torch.Tensor, actions: torch.Tensor, prev_action: Optional[torch.Tensor] = None, hidden_state: Optional[List[torch.Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        output = self.forward(obs, features, stage_idx, prev_action, hidden_state)
        logits = output['logits']
        logits = logits.clamp(min=-20.0, max=20.0)
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, output['value']
    
    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        return self.mamba.get_initial_hidden_state(batch_size, device)


def create_model(config_dict: Optional[Dict] = None) -> GFootballPolicyValueNet:
    if config_dict is None:
        config_dict = {}
    config = ModelConfig(**config_dict)
    return GFootballPolicyValueNet(config)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@dataclass 
class TrainingConfig:
    env_name: str = "academy_single_goal_versus_lazy"
    num_agents: int = 11
    max_episode_steps: int = 1000
    num_workers: int = 20
    trajectory_length: int = 128
    batch_size: int = 4096
    minibatch_size: int = 1024
    num_epochs: int = 5  # Paper: 5
    learning_rate: float = 5e-4  # Paper: 5e-4
    lr_schedule: str = "cosine_restarts"
    max_grad_norm: float = 0.5
    gamma: float = 1.0  # Paper: 1.0 is critical for 11v11 stability!
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.0001  # Paper: 0.0001 (very small!)
    value_coeff: float = 0.5
    total_steps: int = 100_000_000
    log_interval: int = 10
    checkpoint_interval: int = 100
    weight_sync_interval: int = 5
    log_dir: str = "./logs_11v11"
    checkpoint_dir: str = "./checkpoints_11v11"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@ray.remote
class SamplerWorker:
    MAX_AGENTS = 11
    OBS_DIM = 460
    FEATURE_DIM = 32
    
    def __init__(self, worker_id: int, model_config: Dict, env_name: str, num_agents: int, max_steps: int):
        self.worker_id = worker_id
        self.env_name = env_name
        self.num_agents = num_agents
        self.max_steps = max_steps
        self.device = torch.device('cpu')
        self.model = create_model(model_config)
        self.model.to(self.device)
        self.model.eval()
        self.feature_engineer = FeatureEngineer()
        self.env = None
        self.current_obs = None
        self.current_features = None
        self.hidden_state = None
        self.prev_action = None
        self.episode_return = 0.0
        self.episode_steps = 0
        self._setup_env()
        
    def set_weights(self, weights: Dict[str, np.ndarray]):
        state_dict = {k: torch.from_numpy(v.copy()) for k, v in weights.items()}
        self.model.load_state_dict(state_dict)
        
    def _setup_env(self):
        if self.env is not None:
            self.env.close()
        self.env = football_env.create_environment(
            env_name=self.env_name,
            representation="simple115v2",
            number_of_left_players_agent_controls=self.num_agents,
            number_of_right_players_agent_controls=0,
            stacked=True,
            rewards='scoring,checkpoints',
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
                
    def _get_obs_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.current_obs[0]).float().to(self.device)
    
    def _get_feature_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.current_features[0]).float().to(self.device)
        
    def collect_trajectory(self, trajectory_length: int) -> Dict:
        obs_list, feature_list, action_list, reward_list, done_list = [], [], [], [], []
        value_list, log_prob_list, mask_list = [], [], []
        episode_returns, episode_wins, episode_lengths = [], [], []
        steps = 0
        
        while steps < trajectory_length:
            # =========================================================
            # IPPO KEY: Each agent gets its OWN action from OWN observation!
            # =========================================================
            action_full = np.zeros(self.MAX_AGENTS, dtype=np.int64)
            log_prob_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            value_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            mask_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            
            env_action = []
            stage_tensor = torch.tensor(0, device=self.device)
            
            with torch.no_grad():
                for i in range(self.num_agents):
                    # Get THIS agent's observation and features
                    obs_i = torch.from_numpy(self.current_obs[i]).float().to(self.device)
                    feat_i = torch.from_numpy(self.current_features[i]).float().to(self.device)
                    
                    # Get THIS agent's action (each agent decides independently!)
                    action_i, log_prob_i, value_i, _ = self.model.get_action(
                        obs_i, feat_i, stage_tensor,
                        prev_action=self.prev_action, hidden_state=self.hidden_state
                    )
                    
                    # Store per-agent data
                    action_full[i] = action_i.item()
                    log_prob_full[i] = log_prob_i.item()
                    value_full[i] = value_i.item()
                    mask_full[i] = 1.0
                    
                    env_action.append(int(action_i.item()))
            
            # Step environment with 11 DIFFERENT actions
            raw_obs, reward, done, info = self.env.step(env_action)
            self._update_obs(raw_obs)
            
            if isinstance(reward, (list, np.ndarray)):
                step_reward = float(sum(reward))
            else:
                step_reward = float(reward)
            
            self.episode_return += step_reward
            self.episode_steps += 1
            
            terminated = bool(done)
            truncated = self.episode_steps >= self.max_steps
            episode_done = terminated or truncated
            
            # Distribute reward across agents
            reward_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            done_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            per_agent_reward = step_reward / self.num_agents
            for i in range(self.num_agents):
                reward_full[i] = per_agent_reward
                done_full[i] = float(episode_done)
            
            # Pad observations
            obs_padded = np.zeros((self.MAX_AGENTS, self.OBS_DIM), dtype=np.float32)
            obs_padded[:self.num_agents] = self.current_obs[:self.num_agents]
            feature_padded = np.zeros((self.MAX_AGENTS, self.FEATURE_DIM), dtype=np.float32)
            feature_padded[:self.num_agents] = self.current_features[:self.num_agents]
            
            obs_list.append(obs_padded)
            feature_list.append(feature_padded)
            action_list.append(action_full)
            reward_list.append(reward_full)
            done_list.append(done_full)
            value_list.append(value_full)
            log_prob_list.append(log_prob_full)
            mask_list.append(mask_full)
            
            self.prev_action = torch.tensor(action_full[0], device=self.device)
            steps += 1
            
            if episode_done:
                won = False
                if isinstance(info, dict) and "score" in info:
                    won = info["score"][0] > info["score"][1]
                else:
                    won = self.episode_return > 0
                
                episode_returns.append(self.episode_return)
                episode_wins.append(won)
                episode_lengths.append(self.episode_steps)
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
            'worker_id': self.worker_id,
            'episode_returns': episode_returns,
            'episode_wins': episode_wins,
            'episode_lengths': episode_lengths
        }
        
    def close(self):
        if self.env is not None:
            self.env.close()


class Learner:
    def __init__(self, config: TrainingConfig, model_config: Dict, writer: Optional[SummaryWriter] = None):
        self.config = config
        self.device = torch.device(config.device)
        self.writer = writer
        self.model = create_model(model_config)
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate, 
            eps=1e-5, 
            weight_decay=1e-5
        )
        self.total_updates = config.total_steps // config.batch_size
        self.update_count = 0
        self.nan_count = 0
        print(f"Learner on {self.device}, params: {count_parameters(self.model):,}")
        
    def get_weights(self) -> Dict[str, np.ndarray]:
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}
    
    def update(self, trajectories: List[Dict], global_step: int = 0) -> Dict[str, float]:
        self.model.train()
        batch = self._prepare_batch(trajectories)
        if batch is None:
            return {}
        
        if torch.isnan(batch['obs']).any() or torch.isinf(batch['obs']).any():
            self.nan_count += 1
            return {'nan_skipped': 1.0}
        
        advantages, returns = self._compute_gae(batch)
        
        total_loss, policy_loss_sum, value_loss_sum, entropy_sum = 0.0, 0.0, 0.0, 0.0
        clip_fraction_sum, approx_kl_sum, explained_var_sum = 0.0, 0.0, 0.0
        num_updates, skipped_minibatches = 0, 0
        
        indices = np.arange(len(advantages))
        for epoch in range(self.config.num_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]
                
                mb_obs = batch['obs'][mb_indices]
                mb_features = batch['features'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_old_log_probs = batch['log_probs'][mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_stage_ids = torch.zeros(len(mb_indices), dtype=torch.long, device=self.device)
                
                try:
                    log_probs, entropy, values = self.model.evaluate_actions(
                        mb_obs, mb_features, mb_stage_ids, mb_actions
                    )
                except ValueError:
                    self.nan_count += 1
                    skipped_minibatches += 1
                    continue
                
                if torch.isnan(log_probs).any() or torch.isnan(values).any():
                    self.nan_count += 1
                    skipped_minibatches += 1
                    continue
                
                ratio = torch.exp(log_probs - mb_old_log_probs)
                ratio = ratio.clamp(min=0.01, max=100.0)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_epsilon, 1 + self.config.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()
                
                loss = policy_loss + self.config.value_coeff * value_loss + self.config.entropy_coeff * entropy_loss
                
                if torch.isnan(loss) or torch.isinf(loss):
                    self.nan_count += 1
                    skipped_minibatches += 1
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    self.nan_count += 1
                    skipped_minibatches += 1
                    self.optimizer.zero_grad()
                    continue
                
                self.optimizer.step()
                
                total_loss += loss.item()
                policy_loss_sum += policy_loss.item()
                value_loss_sum += value_loss.item()
                entropy_sum += -entropy_loss.item()
                
                with torch.no_grad():
                    clip_fraction = ((ratio - 1).abs() > self.config.clip_epsilon).float().mean()
                    clip_fraction_sum += clip_fraction.item()
                    approx_kl = ((ratio - 1) - (ratio.log())).mean()
                    approx_kl_sum += approx_kl.item()
                    var_y = mb_returns.var()
                    explained_var = 1 - (mb_returns - values).var() / var_y if var_y > 1e-6 else 0.0
                    explained_var_sum += float(explained_var)
                
                num_updates += 1
        
        self._update_lr()
        self.update_count += 1
        
        if num_updates == 0:
            return {'nan_skipped': float(skipped_minibatches)}
        
        stats = {
            'loss/total': total_loss / num_updates,
            'loss/policy': policy_loss_sum / num_updates,
            'loss/value': value_loss_sum / num_updates,
            'loss/entropy': entropy_sum / num_updates,
            'ppo/clip_fraction': clip_fraction_sum / num_updates,
            'ppo/approx_kl': approx_kl_sum / num_updates,
            'ppo/explained_variance': explained_var_sum / num_updates,
            'train/lr': self.optimizer.param_groups[0]['lr'],
            'train/nan_count': self.nan_count,
            'train/skipped_mb': skipped_minibatches,
            'train/grad_norm': float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm
        }
        
        if self.writer is not None and global_step > 0:
            for key, value in stats.items():
                self.writer.add_scalar(key, value, global_step)
        
        return stats
    
    def _prepare_batch(self, trajectories: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        if not trajectories:
            return None
        
        all_obs, all_features, all_actions, all_rewards, all_dones = [], [], [], [], []
        all_values, all_log_probs = [], []
        
        for traj in trajectories:
            obs = traj['obs']
            features = traj['features']
            actions = traj['actions']
            rewards = traj['rewards']
            dones = traj['dones']
            values = traj['values']
            log_probs = traj['log_probs']
            masks = traj['agent_masks']
            
            T, A = masks.shape
            mask_flat = masks.reshape(-1) > 0
            
            all_obs.append(obs.reshape(-1, obs.shape[-1])[mask_flat])
            all_features.append(features.reshape(-1, features.shape[-1])[mask_flat])
            all_actions.append(actions.reshape(-1)[mask_flat])
            all_rewards.append(rewards.reshape(-1)[mask_flat])
            all_dones.append(dones.reshape(-1)[mask_flat])
            all_values.append(values.reshape(-1)[mask_flat])
            all_log_probs.append(log_probs.reshape(-1)[mask_flat])
        
        if not all_obs:
            return None
        
        return {
            'obs': torch.from_numpy(np.concatenate(all_obs)).float().to(self.device),
            'features': torch.from_numpy(np.concatenate(all_features)).float().to(self.device),
            'actions': torch.from_numpy(np.concatenate(all_actions)).long().to(self.device),
            'rewards': torch.from_numpy(np.concatenate(all_rewards)).float().to(self.device),
            'dones': torch.from_numpy(np.concatenate(all_dones)).float().to(self.device),
            'values': torch.from_numpy(np.concatenate(all_values)).float().to(self.device),
            'log_probs': torch.from_numpy(np.concatenate(all_log_probs)).float().to(self.device)
        }
    
    def _compute_gae(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        rewards = batch['rewards']
        values = batch['values']
        dones = batch['dones']
        
        N = len(rewards)
        advantages = torch.zeros(N, device=self.device)
        returns = torch.zeros(N, device=self.device)
        
        last_gae = 0.0
        last_value = values[-1]
        
        for t in reversed(range(N)):
            next_value = last_value if t == N - 1 else values[t + 1]
            next_value = next_value * (1 - dones[t])
            delta = rewards[t] + self.config.gamma * next_value - values[t]
            last_gae = delta + self.config.gamma * self.config.gae_lambda * (1 - dones[t]) * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + values[t]
        
        adv_mean = advantages.mean()
        adv_std = advantages.std()
        if adv_std > 1e-8:
            advantages = (advantages - adv_mean) / adv_std
        
        return advantages, returns
    
    def _update_lr(self):
        if self.config.lr_schedule == "constant":
            return
        
        progress = self.update_count / max(1, self.total_updates)
        
        if self.config.lr_schedule == "linear":
            lr = self.config.learning_rate * (1 - progress)
        elif self.config.lr_schedule == "cosine":
            lr = self.config.learning_rate * 0.5 * (1 + np.cos(np.pi * progress))
        elif self.config.lr_schedule == "cosine_restarts":
            restart_period = 1000
            cycle = self.update_count % restart_period
            cycle_progress = cycle / restart_period
            min_lr = self.config.learning_rate * 0.1
            lr = min_lr + 0.5 * (self.config.learning_rate - min_lr) * (1 + np.cos(np.pi * cycle_progress))
        else:
            lr = self.config.learning_rate
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(lr, 1e-7)
            
    def save_checkpoint(self, path: Path, extra: Dict = None):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count,
            'nan_count': self.nan_count
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: Path) -> Dict:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        self.nan_count = checkpoint.get('nan_count', 0)
        return checkpoint


class Direct11v11Trainer:
    def __init__(self, config: TrainingConfig, model_config: Dict, resume_from: Optional[str] = None):
        self.config = config
        self.model_config = model_config
        self.resume_from = resume_from
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.learner: Optional[Learner] = None
        self.workers: List = []
        self.writer: Optional[SummaryWriter] = None
        
        self.total_episodes = 0
        self.total_steps = 0
        self.start_time = None
        
        # Stats tracking
        self.episode_returns = []
        self.episode_wins = []
        self.episode_lengths = []
        self.ema_win_rate = 0.0
        self.peak_win_rate = 0.0
        
    def setup(self):
        print("=" * 60)
        print("DIRECT 11v11 TRAINER (No Curriculum)")
        print(f"Env: {self.config.env_name}")
        print(f"Agents: {self.config.num_agents}")
        if self.resume_from:
            print(f"Resume: {self.resume_from}")
        print("=" * 60)
        
        run_name = f"11v11_direct_{time.strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=self.log_dir / run_name)
        print(f"TensorBoard: {self.log_dir / run_name}")
        
        self.learner = Learner(self.config, self.model_config, self.writer)
        
        if self.resume_from:
            checkpoint_path = Path(self.resume_from)
            if checkpoint_path.exists():
                print(f"Loading: {checkpoint_path}")
                checkpoint_data = self.learner.load_checkpoint(checkpoint_path)
                self.total_steps = checkpoint_data.get('total_steps', 0)
                self.total_episodes = checkpoint_data.get('total_episodes', 0)
                self.ema_win_rate = checkpoint_data.get('ema_win_rate', 0.0)
                self.peak_win_rate = checkpoint_data.get('peak_win_rate', 0.0)
                print(f"  steps={self.total_steps:,}, episodes={self.total_episodes}, updates={self.learner.update_count}")
                print(f"  win_rate={self.ema_win_rate:.1%}, peak={self.peak_win_rate:.1%}")
        
        print(f"Creating {self.config.num_workers} workers...")
        self.workers = [
            SamplerWorker.remote(
                worker_id=i,
                model_config=self.model_config,
                env_name=self.config.env_name,
                num_agents=self.config.num_agents,
                max_steps=self.config.max_episode_steps
            )
            for i in range(self.config.num_workers)
        ]
        
        self._sync_weights()
        print("Setup complete.\n")
        
    def _sync_weights(self):
        weights = self.learner.get_weights()
        ray.get([w.set_weights.remote(weights) for w in self.workers])
        
    def _aggregate_episode_stats(self, trajectories: List[Dict]):
        for traj in trajectories:
            for ret, won, length in zip(
                traj.get('episode_returns', []),
                traj.get('episode_wins', []),
                traj.get('episode_lengths', [])
            ):
                self.episode_returns.append(ret)
                self.episode_wins.append(1.0 if won else 0.0)
                self.episode_lengths.append(length)
                
                # Update EMA
                alpha = 0.02
                self.ema_win_rate = (1 - alpha) * self.ema_win_rate + alpha * (1.0 if won else 0.0)
                if self.ema_win_rate > self.peak_win_rate:
                    self.peak_win_rate = self.ema_win_rate
                
    def _log_episode_stats(self):
        if self.writer is None or not self.episode_returns:
            return
        
        recent_returns = self.episode_returns[-100:]
        recent_wins = self.episode_wins[-100:]
        recent_lengths = self.episode_lengths[-100:]
        
        self.writer.add_scalar('episode/return_mean', np.mean(recent_returns), self.total_steps)
        self.writer.add_scalar('episode/return_max', np.max(recent_returns), self.total_steps)
        self.writer.add_scalar('episode/win_rate', np.mean(recent_wins), self.total_steps)
        self.writer.add_scalar('episode/ema_win_rate', self.ema_win_rate, self.total_steps)
        self.writer.add_scalar('episode/peak_win_rate', self.peak_win_rate, self.total_steps)
        self.writer.add_scalar('episode/length_mean', np.mean(recent_lengths), self.total_steps)
        
        # Keep buffer manageable
        if len(self.episode_returns) > 1000:
            self.episode_returns = self.episode_returns[-500:]
            self.episode_wins = self.episode_wins[-500:]
            self.episode_lengths = self.episode_lengths[-500:]
        
    def train(self):
        print("Starting training...")
        self.start_time = time.time()
        
        pending = {w.collect_trajectory.remote(self.config.trajectory_length): w for w in self.workers}
        trajectories_buffer = []
        update_count = self.learner.update_count
        
        while True:
            if self.total_steps >= self.config.total_steps:
                print("\n⚠️ Max steps reached")
                break
            
            # Check win rate target
            if self.ema_win_rate >= 0.65 and self.total_episodes > 1000:
                print(f"\n🎉 TARGET REACHED! Win rate: {self.ema_win_rate:.1%}")
                break
            
            done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
            
            for ref in done_refs:
                worker = pending.pop(ref)
                try:
                    trajectory = ray.get(ref)
                    trajectories_buffer.append(trajectory)
                    
                    traj_steps = len(trajectory['obs']) * (trajectory['agent_masks'].sum() / len(trajectory['obs']))
                    self.total_steps += int(traj_steps)
                    self.total_episodes += len(trajectory['episode_returns'])
                    
                    self._aggregate_episode_stats([trajectory])
                except Exception as e:
                    print(f"Worker error: {e}")
                
                pending[worker.collect_trajectory.remote(self.config.trajectory_length)] = worker
            
            total_transitions = sum(
                len(t['obs']) * t['agent_masks'].sum() / len(t['obs']) 
                for t in trajectories_buffer
            )
            
            if total_transitions >= self.config.batch_size:
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
        print(f"Steps: {self.total_steps:,}")
        print(f"Episodes: {self.total_episodes:,}")
        print(f"Win Rate: {self.ema_win_rate:.1%}")
        print(f"Peak: {self.peak_win_rate:.1%}")
        
        if self.writer is not None:
            self.writer.close()
        
        print("\nDone.")
        
    def _log_progress(self, update_count: int, stats: Dict):
        elapsed = time.time() - self.start_time
        sps = self.total_steps / elapsed if elapsed > 0 else 0
        
        steps_m = self.total_steps / 1e6
        sps_k = sps / 1e3
        loss_val = stats.get('loss/total', 0)
        
        recent_wr = np.mean(self.episode_wins[-100:]) if self.episode_wins else 0
        recent_ret = np.mean(self.episode_returns[-100:]) if self.episode_returns else 0
        
        print(f"[{update_count}] {steps_m:.2f}M | {sps_k:.0f}k sps | Loss:{loss_val:.3f}")
        print(f"  Win: {recent_wr:.0%} (ema:{self.ema_win_rate:.0%}, peak:{self.peak_win_rate:.0%}) | Return: {recent_ret:.2f}")
                      
    def _save_checkpoint(self, update_count: int, final: bool = False):
        suffix = "final" if final else f"update_{update_count}"
        path = self.checkpoint_dir / f"checkpoint_{suffix}.pt"
        
        self.learner.save_checkpoint(path, extra={
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'ema_win_rate': self.ema_win_rate,
            'peak_win_rate': self.peak_win_rate
        })
        print(f"Saved: {path}")
        
    def close(self):
        if self.writer is not None:
            self.writer.close()
        for worker in self.workers:
            try:
                ray.get(worker.close.remote())
            except:
                pass
        ray.shutdown()


def main():
    # Configuration
    RESUME_FROM = None  # Or path to checkpoint
    CHECKPOINT_DIR = "./checkpoints_ippo_mamba"
    LOG_DIR = "./logs_ippo_mamba"
    NUM_WORKERS = 24
    
    model_config = {
        'd_model': 256,
        'mamba_d_state': 64,
        'mamba_layers': 4,
        'encoder_hidden': [256, 256],
        'policy_hidden': [256],
        'value_hidden': [256],
        'use_distributional': True,
        'dropout': 0.0,
        'num_stages': 1,
        'feature_dim': 32
    }
    
    # Paper settings from Light-MALib!
    config = TrainingConfig(
        env_name="academy_single_goal_versus_lazy",  # 11v11 easy scenario
        num_agents=11,
        max_episode_steps=1000,
        num_workers=NUM_WORKERS,
        trajectory_length=128,
        batch_size=4096,
        minibatch_size=1024,
        num_epochs=5,           # Paper: 5
        learning_rate=5e-4,     # Paper: 5e-4
        lr_schedule="cosine_restarts",
        gamma=1.0,              # Paper: 1.0 (CRITICAL!)
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coeff=0.0001,   # Paper: 0.0001
        value_coeff=0.5,
        max_grad_norm=0.5,
        total_steps=10_000_000,  # Paper: 2M should be enough
        log_interval=10,
        checkpoint_interval=100,
        weight_sync_interval=5,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR
    )
    
    ray.init(num_cpus=NUM_WORKERS + 2, num_gpus=1, ignore_reinit_error=True)
    
    trainer = Direct11v11Trainer(config, model_config, resume_from=RESUME_FROM)
    
    try:
        trainer.setup()
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()