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
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import ray
import gfootball.env as football_env

@dataclass
class ModelConfig:
    obs_dim: int = 460
    d_model: int = 64
    mamba_state: int = 8
    num_mamba_layers: int = 4
    mamba_expand: int = 2
    num_actions: int = 19
    action_emb_dim: int = 16
    encoder_hidden: List[int] = None
    policy_hidden: List[int] = None
    value_hidden: List[int] = None
    use_distributional: bool = True
    v_min: float = -10.0
    v_max: float = 10.0
    num_atoms: int = 51
    num_stages: int = 8
    stage_emb_dim: int = 8
    dropout: float = 0.0
    
    def __post_init__(self):
        if self.encoder_hidden is None:
            self.encoder_hidden = [256, 128]
        if self.policy_hidden is None:
            self.policy_hidden = [128, 64]
        if self.value_hidden is None:
            self.value_hidden = [128, 64]

class MambaBlock(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.dt_rank = max(1, d_model // 16)
        
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 
            kernel_size=4, padding=3, 
            groups=self.d_inner, bias=True
        )
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        dt_min, dt_max = 0.001, 0.1
        inv_dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        )
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt.log().clamp(min=-4.0, max=2.0))
        
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A).clamp(min=-5.0, max=2.0))
        
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        nn.init.orthogonal_(self.out_proj.weight, gain=0.5)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"MambaBlock expects 3D input (B,L,D), got {x.dim()}D")
            
        B, L, D = x.shape
        
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)
        
        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[:, :, :L]
        x_in = x_in.transpose(1, 2)
        x_in = F.silu(x_in)
        
        x_dbl = self.x_proj(x_in)
        dt, B_param, C_param = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        
        dt = self.dt_proj(dt)
        dt = F.softplus(dt).clamp(min=1e-4, max=10.0)
        
        A = -torch.exp(self.A_log.float().clamp(min=-5.0, max=2.0))
        
        y, h_new = self._selective_scan(x_in, dt, A, B_param, C_param, h)
        
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y, h_new
    
    def _selective_scan(
        self, 
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        h: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        B_size, L, d_inner = x.shape
        d_state = self.d_state
        
        if h is None:
            h = torch.zeros(B_size, d_inner, d_state, device=x.device, dtype=x.dtype)
        
        dA = torch.exp(
            (dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)).clamp(min=-20.0, max=0.0)
        )
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)
        
        outputs = []
        for t in range(L):
            h = dA[:, t] * h + dB[:, t] * x[:, t:t+1, :].transpose(1, 2)
            h = h.clamp(min=-50.0, max=50.0)
            C_t = C[:, t, :].unsqueeze(1)
            y_t = (h * C_t).sum(-1)
            outputs.append(y_t)
            
        y = torch.stack(outputs, dim=1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        
        return y, h

class MambaEncoder(nn.Module):
    def __init__(
        self, 
        d_model: int, 
        d_state: int, 
        num_layers: int, 
        expand: int = 2, 
        dropout: float = 0.0
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, expand, dropout=dropout) 
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.num_layers = num_layers
        self.d_state = d_state
        self.d_inner = d_model * expand
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
            
        new_hidden_states = []
        for i, layer in enumerate(self.layers):
            residual = x
            x, h_new = layer(x, hidden_states[i])
            x = residual + x
            new_hidden_states.append(h_new)
            
        x = self.norm(x)
        return x, new_hidden_states

class GFootballPolicyValueNet(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.stage_embedding = nn.Embedding(config.num_stages, config.stage_emb_dim)
        self.action_embedding = nn.Embedding(config.num_actions, config.action_emb_dim)
        
        obs_input_dim = config.obs_dim + config.stage_emb_dim + config.action_emb_dim
        encoder_layers = []
        in_dim = obs_input_dim
        for hidden_dim in config.encoder_hidden:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.SiLU()
            ])
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, config.d_model))
        self.obs_encoder = nn.Sequential(*encoder_layers)
        
        self.mamba = MambaEncoder(
            d_model=config.d_model,
            d_state=config.mamba_state,
            num_layers=config.num_mamba_layers,
            expand=config.mamba_expand,
            dropout=config.dropout
        )
        
        policy_layers = []
        in_dim = config.d_model
        for hidden_dim in config.policy_hidden:
            policy_layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
        policy_layers.append(nn.Linear(in_dim, config.num_actions))
        self.policy_head = nn.Sequential(*policy_layers)
        
        value_layers = []
        in_dim = config.d_model
        for hidden_dim in config.value_hidden:
            value_layers.extend([nn.Linear(in_dim, hidden_dim), nn.SiLU()])
            in_dim = hidden_dim
            
        if config.use_distributional:
            value_layers.append(nn.Linear(in_dim, config.num_atoms))
            self.register_buffer(
                'value_support',
                torch.linspace(config.v_min, config.v_max, config.num_atoms)
            )
        else:
            value_layers.append(nn.Linear(in_dim, 1))
            self.value_support = None
            
        self.value_head = nn.Sequential(*value_layers)
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
                
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
    
    def _normalize_obs(self, obs: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        if obs.dim() == 1:
            return obs.unsqueeze(0).unsqueeze(1), True
        elif obs.dim() == 2:
            return obs.unsqueeze(1), True
        elif obs.dim() == 3:
            return obs, False
        else:
            raise ValueError(f"obs must be 1D, 2D, or 3D, got {obs.dim()}D")
    
    def _normalize_index(
        self, 
        idx: Optional[torch.Tensor], 
        B: int, 
        L: int, 
        device: torch.device, 
        default_val: int = 0
    ) -> torch.Tensor:
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
        
    def forward(
        self, 
        obs: torch.Tensor, 
        stage_idx: torch.Tensor, 
        prev_action: Optional[torch.Tensor] = None, 
        hidden_state: Optional[List[torch.Tensor]] = None, 
        return_hidden: bool = False
    ) -> Dict[str, torch.Tensor]:
        
        obs, squeeze_output = self._normalize_obs(obs)
        B, L, _ = obs.shape
        device = obs.device
        
        stage_idx = self._normalize_index(stage_idx, B, L, device, default_val=0)
        prev_action = self._normalize_index(prev_action, B, L, device, default_val=0)
        
        stage_emb = self.stage_embedding(stage_idx)
        action_emb = self.action_embedding(prev_action)
        
        x = torch.cat([obs, stage_emb, action_emb], dim=-1)
        x = self.obs_encoder(x)
        
        x, new_hidden = self.mamba(x, hidden_state)
        
        logits = self.policy_head(x)
        
        if self.config.use_distributional:
            value_logits = self.value_head(x)
            value_probs = F.softmax(value_logits, dim=-1)
            value = (value_probs * self.value_support).sum(-1, keepdim=True)
        else:
            value = self.value_head(x)
            value_logits = None
            
        log_probs = F.log_softmax(logits, dim=-1)
        
        if squeeze_output:
            logits = logits.squeeze(1)
            value = value.squeeze(1)
            log_probs = log_probs.squeeze(1)
            if value_logits is not None:
                value_logits = value_logits.squeeze(1)
                
        result = {
            'logits': logits,
            'value': value.squeeze(-1) if value.dim() > 1 else value,
            'log_probs': log_probs
        }
        
        if value_logits is not None:
            result['value_logits'] = value_logits
        if return_hidden:
            result['hidden_state'] = new_hidden
            
        return result
    
    def get_action(
        self, 
        obs: torch.Tensor, 
        stage_idx: torch.Tensor, 
        prev_action: Optional[torch.Tensor] = None, 
        hidden_state: Optional[List[torch.Tensor]] = None, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[List[torch.Tensor]]]:
        
        output = self.forward(obs, stage_idx, prev_action, hidden_state, return_hidden=True)
        logits = output['logits']
        
        logits = logits.clamp(min=-20.0, max=20.0)
        
        dist = Categorical(logits=logits)
        
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
            
        log_prob = dist.log_prob(action)
        return action, log_prob, output['value'], output.get('hidden_state')
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        stage_idx: torch.Tensor, 
        actions: torch.Tensor, 
        prev_action: Optional[torch.Tensor] = None, 
        hidden_state: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        output = self.forward(obs, stage_idx, prev_action, hidden_state)
        logits = output['logits']
        
        logits = logits.clamp(min=-20.0, max=20.0)
        
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, output['value']
    
    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        d_inner = self.config.d_model * self.config.mamba_expand
        return [
            torch.zeros(batch_size, d_inner, self.config.mamba_state, device=device) 
            for _ in range(self.config.num_mamba_layers)
        ]

def create_model(config_dict: Optional[Dict] = None) -> GFootballPolicyValueNet:
    if config_dict is None:
        config_dict = {}
    config = ModelConfig(**config_dict)
    return GFootballPolicyValueNet(config)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@dataclass
class StageConfig:
    stage_id: int
    env_name: str
    representation: str = "simple115v2"
    left_agents: int = 1
    right_agents: int = 0
    max_steps: int = 3000

@dataclass 
class TrainingConfig:
    stages: List[StageConfig] = field(default_factory=list)
    # Curriculum settings - reward-based progression
    stage_unlock_win_rate: float = 0.7  # Win rate required to unlock next stage
    stage_unlock_min_episodes: int = 100  # Minimum episodes before unlock check
    stage_mastery_win_rate: float = 0.9  # Win rate to consider stage "mastered"
    final_stage_target_win_rate: float = 0.5  # Target for final stage to end training
    max_steps_without_progress: int = 500_000  # Stop if no progress for this many steps
    num_workers: int = 20
    envs_per_worker: int = 1
    trajectory_length: int = 128
    queue_size: int = 64
    batch_size: int = 2048
    minibatch_size: int = 512
    num_epochs: int = 4
    learning_rate: float = 3e-4
    lr_schedule: str = "cosine_restarts"  # Better for long training
    max_grad_norm: float = 0.5
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    use_vtrace: bool = False
    vtrace_rho_max: float = 1.0
    vtrace_c_max: float = 1.0
    total_steps: int = 100_000_000  # High limit, will stop based on curriculum
    log_interval: int = 10
    eval_interval: int = 50
    checkpoint_interval: int = 100
    weight_sync_interval: int = 5
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if not self.stages:
            self.stages = get_default_stages()

def get_default_stages() -> List[StageConfig]:
    return [
        StageConfig(0, "academy_empty_goal_close", "simple115v2", 1, 0, 400),
        StageConfig(1, "academy_run_to_score_with_keeper", "simple115v2", 1, 0, 400),
        StageConfig(2, "academy_pass_and_shoot_with_keeper", "simple115v2", 2, 0, 400),
        StageConfig(3, "academy_3_vs_1_with_keeper", "simple115v2", 3, 0, 400),
        StageConfig(4, "academy_single_goal_versus_lazy", "simple115v2", 3, 0, 1000),
        StageConfig(5, "11_vs_11_easy_stochastic", "simple115v2", 3, 0, 3000),
        StageConfig(6, "11_vs_11_easy_stochastic", "simple115v2", 5, 0, 3000),
        StageConfig(7, "11_vs_11_stochastic", "simple115v2", 11, 0, 3000),
    ]

@dataclass
class StageBaseline:
    stage_id: int
    episode_return_mean: float = 0.0
    episode_return_std: float = 1.0
    step_reward_mean: float = 0.0
    step_reward_std: float = 0.01
    win_rate: float = 0.0
    calibrated: bool = False
    
    def normalize_return(self, raw_return: float) -> float:
        if not self.calibrated or self.episode_return_std < 1e-6:
            return raw_return
        return (raw_return - self.episode_return_mean) / self.episode_return_std
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict) -> "StageBaseline":
        return cls(**d)

@ray.remote
def _calibrate_stage_batch(stage_dict: Dict, num_episodes: int, worker_id: int) -> Dict:
    stage = StageConfig(**stage_dict)
    env = football_env.create_environment(
        env_name=stage.env_name,
        representation=stage.representation,
        number_of_left_players_agent_controls=stage.left_agents,
        number_of_right_players_agent_controls=stage.right_agents,
        write_goal_dumps=False,
        write_full_episode_dumps=False,
        render=False,
        write_video=False,
    )
    
    returns, wins, step_rewards, lengths = [], [], [], []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_steps = 0
        
        while not done and ep_steps < stage.max_steps:
            if isinstance(env.action_space, list):
                actions = [s.sample() for s in env.action_space]
            else:
                actions = env.action_space.sample()
                
            obs, reward, done, info = env.step(actions)
            
            if isinstance(reward, (list, np.ndarray)):
                step_reward = float(sum(reward))
            else:
                step_reward = float(reward)
                
            step_rewards.append(step_reward)
            ep_return += step_reward
            ep_steps += 1
            
        returns.append(ep_return)
        lengths.append(ep_steps)
        
        won = False
        if isinstance(info, dict) and "score" in info:
            won = info["score"][0] > info["score"][1]
        else:
            won = ep_return > 0
        wins.append(1.0 if won else 0.0)
        
    env.close()
    
    return {
        'stage_id': stage.stage_id,
        'worker_id': worker_id,
        'returns': returns,
        'wins': wins,
        'step_rewards': step_rewards,
        'lengths': lengths
    }

class BaselineCalibrator:
    def __init__(self, stages: List[StageConfig], save_path: Path):
        self.stages = stages
        self.save_path = save_path
        self.baselines = {s.stage_id: StageBaseline(s.stage_id) for s in stages}
        
    def load(self) -> bool:
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
        data = {str(k): v.to_dict() for k, v in self.baselines.items()}
        with open(self.save_path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def calibrate(self, num_episodes: int = 100, num_workers: int = 8):
        print(f"Calibrating baselines ({num_episodes} eps/stage, {num_workers} workers)...")
        episodes_per_worker = max(1, num_episodes // num_workers)
        
        futures = []
        for stage in self.stages:
            stage_dict = asdict(stage)
            for worker_id in range(num_workers):
                futures.append(_calibrate_stage_batch.remote(stage_dict, episodes_per_worker, worker_id))
                
        stage_data = {s.stage_id: {'returns': [], 'wins': [], 'step_rewards': [], 'lengths': []} for s in self.stages}
        
        total = len(futures)
        completed = 0
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
    """
    Reward-based curriculum controller.
    - Stages unlock when win_rate >= unlock_threshold
    - Training ends when final stage reaches target win_rate
    - Focuses training on stages that need improvement
    """
    def __init__(
        self, 
        stages: List[Dict], 
        baselines: Dict[int, Dict],
        unlock_win_rate: float = 0.7,
        unlock_min_episodes: int = 100,
        mastery_win_rate: float = 0.9,
        final_target_win_rate: float = 0.5,
    ):
        self.stages = [StageConfig(**s) if isinstance(s, dict) else s for s in stages]
        self.baselines = {
            int(k): StageBaseline.from_dict(v) if isinstance(v, dict) else v 
            for k, v in baselines.items()
        }
        self.num_stages = len(self.stages)
        
        # Unlock thresholds
        self.unlock_win_rate = unlock_win_rate
        self.unlock_min_episodes = unlock_min_episodes
        self.mastery_win_rate = mastery_win_rate
        self.final_target_win_rate = final_target_win_rate
        
        # State
        self.episode_count = 0
        self.unlocked_stages = {0}  # Start with only stage 0 unlocked
        self.mastered_stages = set()
        self.highest_unlocked = 0
        self.last_unlock_episode = 0
        
        self.stage_stats = {
            s.stage_id: {
                'returns': [], 
                'wins': [], 
                'episodes': 0, 
                'ema_return': 0.0, 
                'ema_win': 0.0,
                'recent_wins': [],  # Last N wins for unlock check
            } 
            for s in self.stages
        }
        
    def get_stage(self) -> Dict:
        """Select a stage to train on, prioritizing stages that need work."""
        available = sorted(self.unlocked_stages)
        
        if len(available) == 1:
            return asdict(self.stages[available[0]])
        
        # Calculate weights: prioritize stages with lower win rates
        weights = []
        for sid in available:
            stats = self.stage_stats[sid]
            
            if sid in self.mastered_stages:
                # Mastered stages get low weight but still sampled occasionally
                w = 0.1
            elif stats['episodes'] < 20:
                # Not enough data, high priority to explore
                w = 3.0
            else:
                # Weight inversely proportional to win rate
                win_rate = stats['ema_win']
                if win_rate >= self.mastery_win_rate:
                    w = 0.2
                elif win_rate >= self.unlock_win_rate:
                    w = 0.5
                else:
                    # Focus on stages below unlock threshold
                    w = 2.0 - win_rate
                    
            # Boost weight for highest unlocked stage (frontier)
            if sid == self.highest_unlocked and sid not in self.mastered_stages:
                w *= 1.5
                
            weights.append(max(0.05, w))
        
        weights = np.array(weights)
        weights /= weights.sum()
        
        chosen_idx = np.random.choice(len(available), p=weights)
        return asdict(self.stages[available[chosen_idx]])
    
    def report_episode(self, stage_id: int, episode_return: float, won: bool):
        """Report episode result and check for stage unlocks."""
        self.episode_count += 1
        stats = self.stage_stats[stage_id]
        
        # Update stats
        stats['returns'].append(episode_return)
        stats['wins'].append(1.0 if won else 0.0)
        stats['episodes'] += 1
        
        # Track recent wins (last 100 episodes)
        stats['recent_wins'].append(1.0 if won else 0.0)
        if len(stats['recent_wins']) > 100:
            stats['recent_wins'].pop(0)
        
        # EMA update
        alpha = 0.02  # Slower EMA for more stable estimates
        if stats['episodes'] == 1:
            stats['ema_return'] = episode_return
            stats['ema_win'] = float(won)
        else:
            stats['ema_return'] = (1 - alpha) * stats['ema_return'] + alpha * episode_return
            stats['ema_win'] = (1 - alpha) * stats['ema_win'] + alpha * float(won)
        
        # Check for mastery
        if stats['episodes'] >= self.unlock_min_episodes:
            recent_win_rate = np.mean(stats['recent_wins']) if stats['recent_wins'] else 0
            if recent_win_rate >= self.mastery_win_rate:
                self.mastered_stages.add(stage_id)
        
        # Check for unlock of next stage
        self._check_unlock(stage_id)
        
    def _check_unlock(self, stage_id: int):
        """Check if completing this stage should unlock the next one."""
        next_stage = stage_id + 1
        
        # Already unlocked or no more stages
        if next_stage in self.unlocked_stages or next_stage >= self.num_stages:
            return
            
        stats = self.stage_stats[stage_id]
        
        # Need minimum episodes
        if stats['episodes'] < self.unlock_min_episodes:
            return
            
        # Check recent win rate (more reliable than EMA for unlocking)
        recent_win_rate = np.mean(stats['recent_wins']) if stats['recent_wins'] else 0
        
        if recent_win_rate >= self.unlock_win_rate:
            self.unlocked_stages.add(next_stage)
            self.highest_unlocked = max(self.highest_unlocked, next_stage)
            self.last_unlock_episode = self.episode_count
            print(f"\n🔓 STAGE {next_stage} UNLOCKED! (Stage {stage_id} win rate: {recent_win_rate:.1%})\n")
            
    def is_training_complete(self) -> bool:
        """Check if training should end."""
        final_stage = self.num_stages - 1
        
        # Final stage must be unlocked
        if final_stage not in self.unlocked_stages:
            return False
            
        stats = self.stage_stats[final_stage]
        
        # Need minimum episodes on final stage
        if stats['episodes'] < self.unlock_min_episodes:
            return False
            
        # Check if final stage target reached
        recent_win_rate = np.mean(stats['recent_wins']) if stats['recent_wins'] else 0
        return recent_win_rate >= self.final_target_win_rate
    
    def get_progress_summary(self) -> str:
        """Get a summary string of curriculum progress."""
        lines = []
        for sid in range(self.num_stages):
            stats = self.stage_stats[sid]
            status = "🔒"
            if sid in self.mastered_stages:
                status = "⭐"
            elif sid in self.unlocked_stages:
                status = "🔓"
                
            if stats['episodes'] > 0:
                recent_wr = np.mean(stats['recent_wins']) if stats['recent_wins'] else 0
                lines.append(f"{status} S{sid}: {recent_wr:.0%} ({stats['episodes']} eps)")
            else:
                lines.append(f"{status} S{sid}: --")
        return " | ".join(lines)
            
    def get_stats(self) -> Dict:
        return {
            'episode_count': self.episode_count,
            'unlocked_stages': list(self.unlocked_stages),
            'mastered_stages': list(self.mastered_stages),
            'highest_unlocked': self.highest_unlocked,
            'training_complete': self.is_training_complete(),
            'stage_stats': {
                sid: {
                    'episodes': s['episodes'],
                    'ema_return': s['ema_return'],
                    'ema_win': s['ema_win'],
                    'recent_win_rate': np.mean(s['recent_wins']) if s['recent_wins'] else 0,
                    'normalized_return': self.baselines[sid].normalize_return(s['ema_return'])
                } 
                for sid, s in self.stage_stats.items()
            }
        }

@ray.remote
class SamplerWorker:
    MAX_AGENTS = 11
    OBS_DIM = 460
    
    def __init__(
        self, 
        worker_id: int, 
        model_config: Dict, 
        stages: List[Dict], 
        baselines: Dict[int, Dict]
    ):
        self.worker_id = worker_id
        self.stages = {s['stage_id']: StageConfig(**s) for s in stages}
        self.baselines = {int(k): StageBaseline.from_dict(v) for k, v in baselines.items()}
        
        self.device = torch.device('cpu')
        self.model = create_model(model_config)
        self.model.to(self.device)
        self.model.eval()
        
        self.env = None
        self.current_stage: Optional[StageConfig] = None
        self.current_obs = None
        self.hidden_state = None
        self.prev_action = None
        self.episode_return = 0.0
        self.episode_steps = 0
        
    def set_weights(self, weights: Dict[str, np.ndarray]):
        state_dict = {k: torch.from_numpy(v.copy()) for k, v in weights.items()}
        self.model.load_state_dict(state_dict)
        
    def collect_trajectory(self, trajectory_length: int, curriculum_controller) -> Dict:
        obs_list, action_list, reward_list, done_list = [], [], [], []
        value_list, log_prob_list, stage_list, mask_list = [], [], [], []
        episode_returns, episode_wins = [], []
        episode_lengths = []
        episode_stages = []
        
        steps = 0
        while steps < trajectory_length:
            if self.env is None or self._should_switch_stage():
                stage_dict = ray.get(curriculum_controller.get_stage.remote())
                self._setup_env(StageConfig(**stage_dict))
            
            obs_tensor = self._get_obs_tensor()
            stage_tensor = torch.tensor(self.current_stage.stage_id, device=self.device)
            
            with torch.no_grad():
                action, log_prob, value, self.hidden_state = self.model.get_action(
                    obs_tensor,
                    stage_tensor,
                    prev_action=self.prev_action,
                    hidden_state=self.hidden_state,
                )
            
            action_np = action.cpu().numpy()
            log_prob_np = log_prob.cpu().numpy()
            value_np = value.cpu().numpy()
            
            action_int = int(action_np) if action_np.ndim == 0 else int(action_np.item())
            log_prob_float = float(log_prob_np) if log_prob_np.ndim == 0 else float(log_prob_np.item())
            value_float = float(value_np) if value_np.ndim == 0 else float(value_np.item())
            
            num_agents = self.current_stage.left_agents
            action_full = np.zeros(self.MAX_AGENTS, dtype=np.int64)
            log_prob_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            value_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            mask_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            
            for i in range(num_agents):
                action_full[i] = action_int
                log_prob_full[i] = log_prob_float
                value_full[i] = value_float
                mask_full[i] = 1.0
                
            if num_agents == 1:
                env_action = action_int
            else:
                env_action = [action_int] * num_agents
                
            raw_obs, reward, done, info = self.env.step(env_action)
            self._update_obs(raw_obs)
            
            if isinstance(reward, (list, np.ndarray)):
                step_reward = float(sum(reward))
            else:
                step_reward = float(reward)
                
            self.episode_return += step_reward
            self.episode_steps += 1
            
            terminated = bool(done)
            truncated = self.episode_steps >= self.current_stage.max_steps
            episode_done = terminated or truncated
            
            reward_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            done_full = np.zeros(self.MAX_AGENTS, dtype=np.float32)
            
            if num_agents > 0:
                per_agent_reward = step_reward / num_agents
                for i in range(num_agents):
                    reward_full[i] = per_agent_reward
                    done_full[i] = float(episode_done)
            
            obs_padded = np.zeros((self.MAX_AGENTS, self.OBS_DIM), dtype=np.float32)
            obs_padded[:num_agents] = self.current_obs[:num_agents]
            
            obs_list.append(obs_padded)
            action_list.append(action_full)
            reward_list.append(reward_full)
            done_list.append(done_full)
            value_list.append(value_full)
            log_prob_list.append(log_prob_full)
            stage_list.append(self.current_stage.stage_id)
            mask_list.append(mask_full)
            
            self.prev_action = torch.tensor(action_int, device=self.device)
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
                episode_stages.append(self.current_stage.stage_id)
                
                ray.get(curriculum_controller.report_episode.remote(
                    self.current_stage.stage_id, self.episode_return, won
                ))
                self._reset_episode()
                
        return {
            'obs': np.array(obs_list, dtype=np.float32),
            'actions': np.array(action_list, dtype=np.int64),
            'rewards': np.array(reward_list, dtype=np.float32),
            'dones': np.array(done_list, dtype=np.float32),
            'values': np.array(value_list, dtype=np.float32),
            'log_probs': np.array(log_prob_list, dtype=np.float32),
            'stage_ids': np.array(stage_list, dtype=np.int64),
            'agent_masks': np.array(mask_list, dtype=np.float32),
            'worker_id': self.worker_id,
            'episode_returns': episode_returns,
            'episode_wins': episode_wins,
            'episode_lengths': episode_lengths,
            'episode_stages': episode_stages,
        }
        
    def _setup_env(self, stage: StageConfig):
        if self.env is not None:
            self.env.close()
            
        self.current_stage = stage
        self.env = football_env.create_environment(
            env_name=stage.env_name,
            representation=stage.representation,
            number_of_left_players_agent_controls=stage.left_agents,
            number_of_right_players_agent_controls=stage.right_agents,
            stacked=True,
            write_goal_dumps=False,
            write_full_episode_dumps=False,
            render=False,
            write_video=False,
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
                
    def _get_obs_tensor(self) -> torch.Tensor:
        return torch.from_numpy(self.current_obs[0]).float().to(self.device)
        
    def _should_switch_stage(self) -> bool:
        return self.episode_steps == 0 and np.random.random() < 0.1
        
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
        self.stats = defaultdict(list)
        
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
            print(f"WARNING: NaN/Inf in observations (count: {self.nan_count}), skipping batch")
            return {'nan_skipped': 1.0}
            
        advantages, returns = self._compute_gae(batch)
        
        total_loss = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        clip_fraction_sum = 0.0
        approx_kl_sum = 0.0
        explained_var_sum = 0.0
        num_updates = 0
        skipped_minibatches = 0
        
        indices = np.arange(len(advantages))
        
        for epoch in range(self.config.num_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, len(indices), self.config.minibatch_size):
                end = start + self.config.minibatch_size
                mb_indices = indices[start:end]
                
                mb_obs = batch['obs'][mb_indices]
                mb_actions = batch['actions'][mb_indices]
                mb_old_log_probs = batch['log_probs'][mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_stage_ids = batch['stage_ids'][mb_indices]
                mb_old_values = batch['values'][mb_indices]
                
                try:
                    log_probs, entropy, values = self.model.evaluate_actions(
                        mb_obs, mb_stage_ids, mb_actions
                    )
                except ValueError as e:
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
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config.clip_epsilon, 
                    1 + self.config.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(values, mb_returns)
                entropy_loss = -entropy.mean()
                
                loss = (
                    policy_loss + 
                    self.config.value_coeff * value_loss + 
                    self.config.entropy_coeff * entropy_loss
                )
                
                if torch.isnan(loss) or torch.isinf(loss):
                    self.nan_count += 1
                    skipped_minibatches += 1
                    continue
                
                self.optimizer.zero_grad()
                loss.backward()
                
                grad_norm = nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.max_grad_norm
                )
                
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
                    if var_y > 1e-6:
                        explained_var = 1 - (mb_returns - values).var() / var_y
                    else:
                        explained_var = torch.tensor(0.0)
                    explained_var_sum += explained_var.item()
                    
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
            'train/grad_norm': float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
        }
        
        # TensorBoard logging
        if self.writer is not None and global_step > 0:
            for key, value in stats.items():
                self.writer.add_scalar(key, value, global_step)
        
        return stats
    
    def _prepare_batch(self, trajectories: List[Dict]) -> Optional[Dict[str, torch.Tensor]]:
        if not trajectories:
            return None
            
        all_obs, all_actions, all_rewards, all_dones = [], [], [], []
        all_values, all_log_probs, all_stage_ids = [], [], []
        
        for traj in trajectories:
            T = len(traj['obs'])
            max_agents = traj['obs'].shape[1]
            
            for t in range(T):
                for a in range(max_agents):
                    if traj['agent_masks'][t, a] > 0:
                        all_obs.append(traj['obs'][t, a])
                        all_actions.append(traj['actions'][t, a])
                        all_rewards.append(traj['rewards'][t, a])
                        all_dones.append(traj['dones'][t, a])
                        all_values.append(traj['values'][t, a])
                        all_log_probs.append(traj['log_probs'][t, a])
                        all_stage_ids.append(traj['stage_ids'][t])
                        
        if not all_obs:
            return None
            
        return {
            'obs': torch.from_numpy(np.array(all_obs)).float().to(self.device),
            'actions': torch.from_numpy(np.array(all_actions)).long().to(self.device),
            'rewards': torch.from_numpy(np.array(all_rewards)).float().to(self.device),
            'dones': torch.from_numpy(np.array(all_dones)).float().to(self.device),
            'values': torch.from_numpy(np.array(all_values)).float().to(self.device),
            'log_probs': torch.from_numpy(np.array(all_log_probs)).float().to(self.device),
            'stage_ids': torch.from_numpy(np.array(all_stage_ids)).long().to(self.device),
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
            if t == N - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]
                
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
            # Cosine annealing with warm restarts every 1000 updates
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
            'nan_count': self.nan_count,
        }
        if extra:
            checkpoint.update(extra)
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: Path) -> Dict:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint.get('update_count', 0)
        self.nan_count = checkpoint.get('nan_count', 0)
        return checkpoint

class IMPALATrainer:
    def __init__(self, config: TrainingConfig, model_config: Dict):
        self.config = config
        self.model_config = model_config
        
        self.log_dir = Path(config.log_dir)
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.baselines_path = self.checkpoint_dir / "baselines.json"
        self.calibrator = BaselineCalibrator(config.stages, self.baselines_path)
        
        self.learner: Optional[Learner] = None
        self.curriculum = None
        self.workers: List = []
        self.writer: Optional[SummaryWriter] = None
        
        self.total_episodes = 0
        self.total_steps = 0
        self.start_time = None
        
        # Progress tracking
        self.last_highest_unlocked = 0
        self.last_progress_step = 0
        
        # Rolling stats for episode metrics
        self.episode_returns_buffer = defaultdict(list)
        self.episode_wins_buffer = defaultdict(list)
        self.episode_lengths_buffer = defaultdict(list)
        
    def setup(self):
        print("=" * 60)
        print("IMPALA TRAINER SETUP (Reward-Based Curriculum)")
        print("=" * 60)
        
        # Initialize TensorBoard
        run_name = f"gfootball_mamba_{time.strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=self.log_dir / run_name)
        print(f"TensorBoard logs: {self.log_dir / run_name}")
        
        # Log hyperparameters
        hparams = {
            'num_workers': self.config.num_workers,
            'batch_size': self.config.batch_size,
            'minibatch_size': self.config.minibatch_size,
            'learning_rate': self.config.learning_rate,
            'gamma': self.config.gamma,
            'gae_lambda': self.config.gae_lambda,
            'clip_epsilon': self.config.clip_epsilon,
            'entropy_coeff': self.config.entropy_coeff,
            'value_coeff': self.config.value_coeff,
            'd_model': self.model_config.get('d_model', 64),
            'num_mamba_layers': self.model_config.get('num_mamba_layers', 4),
            'stage_unlock_win_rate': self.config.stage_unlock_win_rate,
            'stage_mastery_win_rate': self.config.stage_mastery_win_rate,
        }
        self.writer.add_hparams(hparams, {})
        
        if not self.calibrator.load():
            self.calibrator.calibrate(
                num_episodes=100, 
                num_workers=min(self.config.num_workers, 8)
            )
        else:
            print("Loaded existing baselines")
        
        stages_dict = [asdict(s) for s in self.config.stages]
        baselines_dict = {k: v.to_dict() for k, v in self.calibrator.baselines.items()}
        
        self.curriculum = CurriculumController.remote(
            stages_dict, 
            baselines_dict,
            unlock_win_rate=self.config.stage_unlock_win_rate,
            unlock_min_episodes=self.config.stage_unlock_min_episodes,
            mastery_win_rate=self.config.stage_mastery_win_rate,
            final_target_win_rate=self.config.final_stage_target_win_rate,
        )
        self.learner = Learner(self.config, self.model_config, self.writer)
        
        print(f"Creating {self.config.num_workers} workers...")
        self.workers = [
            SamplerWorker.remote(
                worker_id=i,
                model_config=self.model_config,
                stages=stages_dict,
                baselines=baselines_dict
            ) 
            for i in range(self.config.num_workers)
        ]
        
        self._sync_weights()
        print(f"\nCurriculum settings:")
        print(f"  Unlock threshold: {self.config.stage_unlock_win_rate:.0%} win rate")
        print(f"  Mastery threshold: {self.config.stage_mastery_win_rate:.0%} win rate")
        print(f"  Final stage target: {self.config.final_stage_target_win_rate:.0%} win rate")
        print(f"  Min episodes for unlock: {self.config.stage_unlock_min_episodes}")
        print("Setup complete.\n")
        
    def _sync_weights(self):
        weights = self.learner.get_weights()
        ray.get([w.set_weights.remote(weights) for w in self.workers])
        
    def _aggregate_episode_stats(self, trajectories: List[Dict]):
        """Aggregate episode statistics from trajectories."""
        for traj in trajectories:
            for i, (ret, won, length, stage_id) in enumerate(zip(
                traj.get('episode_returns', []),
                traj.get('episode_wins', []),
                traj.get('episode_lengths', []),
                traj.get('episode_stages', [])
            )):
                self.episode_returns_buffer[stage_id].append(ret)
                self.episode_wins_buffer[stage_id].append(1.0 if won else 0.0)
                self.episode_lengths_buffer[stage_id].append(length)
                
    def _log_episode_stats(self):
        """Log aggregated episode statistics to TensorBoard."""
        if self.writer is None:
            return
            
        # Per-stage metrics
        all_returns = []
        all_wins = []
        all_lengths = []
        
        for stage_id in sorted(self.episode_returns_buffer.keys()):
            returns = self.episode_returns_buffer[stage_id]
            wins = self.episode_wins_buffer[stage_id]
            lengths = self.episode_lengths_buffer[stage_id]
            
            if returns:
                self.writer.add_scalar(f'episode/return_stage_{stage_id}', np.mean(returns), self.total_steps)
                self.writer.add_scalar(f'episode/win_rate_stage_{stage_id}', np.mean(wins), self.total_steps)
                self.writer.add_scalar(f'episode/length_stage_{stage_id}', np.mean(lengths), self.total_steps)
                
                all_returns.extend(returns)
                all_wins.extend(wins)
                all_lengths.extend(lengths)
        
        # Global metrics
        if all_returns:
            self.writer.add_scalar('episode/return_mean', np.mean(all_returns), self.total_steps)
            self.writer.add_scalar('episode/return_std', np.std(all_returns), self.total_steps)
            self.writer.add_scalar('episode/return_min', np.min(all_returns), self.total_steps)
            self.writer.add_scalar('episode/return_max', np.max(all_returns), self.total_steps)
            self.writer.add_scalar('episode/win_rate', np.mean(all_wins), self.total_steps)
            self.writer.add_scalar('episode/length_mean', np.mean(all_lengths), self.total_steps)
            
            # Histogram of returns
            self.writer.add_histogram('episode/return_dist', np.array(all_returns), self.total_steps)
        
        # Clear buffers
        self.episode_returns_buffer.clear()
        self.episode_wins_buffer.clear()
        self.episode_lengths_buffer.clear()
        
    def train(self):
        print("Starting training (will stop when curriculum complete)...")
        self.start_time = time.time()
        self.last_progress_step = 0
        
        pending = {
            w.collect_trajectory.remote(self.config.trajectory_length, self.curriculum): w 
            for w in self.workers
        }
        
        trajectories_buffer = []
        update_count = 0
        
        while True:
            # Check stopping conditions
            curriculum_stats = ray.get(self.curriculum.get_stats.remote())
            
            # Stop if curriculum complete (all stages solved)
            if curriculum_stats.get('training_complete', False):
                print("\n" + "=" * 60)
                print("🎉 TRAINING COMPLETE - All stages solved!")
                print("=" * 60)
                break
                
            # Stop if max steps reached
            if self.total_steps >= self.config.total_steps:
                print("\n" + "=" * 60)
                print("⚠️ Max steps reached without completing curriculum")
                print("=" * 60)
                break
                
            # Stop if no progress for too long
            highest_unlocked = curriculum_stats.get('highest_unlocked', 0)
            if highest_unlocked > self.last_highest_unlocked:
                self.last_highest_unlocked = highest_unlocked
                self.last_progress_step = self.total_steps
            elif self.total_steps - self.last_progress_step > self.config.max_steps_without_progress:
                print("\n" + "=" * 60)
                print(f"⚠️ No progress for {self.config.max_steps_without_progress:,} steps, stopping")
                print("=" * 60)
                break
            
            done_refs, _ = ray.wait(list(pending.keys()), num_returns=1)
            
            for ref in done_refs:
                worker = pending.pop(ref)
                try:
                    trajectory = ray.get(ref)
                    trajectories_buffer.append(trajectory)
                    
                    traj_steps = len(trajectory['obs']) * (
                        trajectory['agent_masks'].sum() / len(trajectory['obs'])
                    )
                    self.total_steps += int(traj_steps)
                    self.total_episodes += len(trajectory['episode_returns'])
                    
                    # Aggregate episode stats
                    self._aggregate_episode_stats([trajectory])
                    
                except Exception as e:
                    print(f"Worker error: {e}")
                    
                pending[worker.collect_trajectory.remote(
                    self.config.trajectory_length, self.curriculum
                )] = worker
                
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
                    
                    # Log throughput and curriculum
                    if self.writer is not None:
                        elapsed = time.time() - self.start_time
                        sps = self.total_steps / elapsed if elapsed > 0 else 0
                        self.writer.add_scalar('throughput/steps_per_second', sps, self.total_steps)
                        self.writer.add_scalar('throughput/episodes', self.total_episodes, self.total_steps)
                        self.writer.add_scalar('throughput/updates', update_count, self.total_steps)
                        self.writer.add_scalar('curriculum/highest_unlocked', curriculum_stats.get('highest_unlocked', 0), self.total_steps)
                        self.writer.add_scalar('curriculum/num_unlocked', len(curriculum_stats.get('unlocked_stages', [0])), self.total_steps)
                        self.writer.add_scalar('curriculum/num_mastered', len(curriculum_stats.get('mastered_stages', [])), self.total_steps)
                    
                if update_count % self.config.checkpoint_interval == 0:
                    self._save_checkpoint(update_count)
                    
        self._save_checkpoint(update_count, final=True)
        
        # Print final summary
        final_stats = ray.get(self.curriculum.get_stats.remote())
        print("\n=== FINAL CURRICULUM STATUS ===")
        print(ray.get(self.curriculum.get_progress_summary.remote()))
        
        if self.writer is not None:
            self.writer.close()
            
        print("\nTraining finished.")
        
    def _log_progress(self, update_count: int, stats: Dict):
        elapsed = time.time() - self.start_time
        sps = self.total_steps / elapsed if elapsed > 0 else 0
        curriculum_stats = ray.get(self.curriculum.get_stats.remote())
        
        loss_str = f"Loss: {stats.get('loss/total', 0):.4f}"
        nan_str = f"NaN: {stats.get('train/nan_count', 0)}" if stats.get('train/nan_count', 0) > 0 else ""
        
        print(f"[{update_count}] Steps: {self.total_steps:,} | Ep: {self.total_episodes} | SPS: {int(sps)} | {loss_str} {nan_str}")
        
        # Show curriculum progress
        unlocked = curriculum_stats.get('unlocked_stages', [0])
        mastered = curriculum_stats.get('mastered_stages', [])
        print(f"        Unlocked: {sorted(unlocked)} | Mastered: {sorted(mastered)}")
        
        stage_stats = curriculum_stats.get('stage_stats', {})
        if stage_stats:
            # Show recent win rates for unlocked stages
            active = []
            for sid in sorted(unlocked):
                if sid in stage_stats:
                    s = stage_stats[sid]
                    recent_wr = s.get('recent_win_rate', s['ema_win'])
                    marker = "⭐" if sid in mastered else ""
                    active.append(f"S{sid}:{recent_wr:.0%}{marker}")
            print(f"        Win rates: {', '.join(active)}")
            
            # Log curriculum stats to TensorBoard
            if self.writer is not None:
                for sid, s in stage_stats.items():
                    self.writer.add_scalar(f'curriculum/ema_return_stage_{sid}', s['ema_return'], self.total_steps)
                    self.writer.add_scalar(f'curriculum/ema_win_stage_{sid}', s['ema_win'], self.total_steps)
                    self.writer.add_scalar(f'curriculum/recent_win_rate_stage_{sid}', s.get('recent_win_rate', 0), self.total_steps)
                    self.writer.add_scalar(f'curriculum/normalized_return_stage_{sid}', s['normalized_return'], self.total_steps)
                    self.writer.add_scalar(f'curriculum/episodes_stage_{sid}', s['episodes'], self.total_steps)
                      
    def _save_checkpoint(self, update_count: int, final: bool = False):
        suffix = "final" if final else f"update_{update_count}"
        path = self.checkpoint_dir / f"checkpoint_{suffix}.pt"
        curriculum_stats = ray.get(self.curriculum.get_stats.remote())
        
        self.learner.save_checkpoint(path, extra={
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'curriculum_stats': curriculum_stats
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
    model_config = {
        'd_model': 128,
        'mamba_state': 16,
        'num_mamba_layers': 4,
        'mamba_expand': 2,
        'encoder_hidden': [256, 128],
        'policy_hidden': [128, 64],
        'value_hidden': [128, 64],
        'use_distributional': True,
        'dropout': 0.0,
    }
    
    config = TrainingConfig(
        stages=get_default_stages(),
        stage_unlock_win_rate=0.7,       # 70% win rate to unlock next stage
        stage_unlock_min_episodes=100,    # Min episodes before unlock check
        stage_mastery_win_rate=0.9,       # 90% to consider stage mastered
        final_stage_target_win_rate=0.5,  # 50% on final stage to complete
        max_steps_without_progress=100_000_000,  # Stop if stuck
        num_workers=24,
        trajectory_length=128,
        batch_size=2048,
        minibatch_size=512,
        num_epochs=3,
        learning_rate=3e-4,
        lr_schedule="cosine_restarts",
        gamma=0.998,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coeff=0.01,
        value_coeff=0.5,
        max_grad_norm=0.5,
        total_steps=1_000_000_000,
        log_interval=10,
        checkpoint_interval=100,
        weight_sync_interval=5,
    )
    
    ray.init(
        num_cpus=22,
        num_gpus=1,
        object_store_memory=3 * 1024 * 1024 * 1024,
        _system_config={
            "object_spilling_threshold": 0.8,
        },
        ignore_reinit_error=True,
    )
    
    trainer = IMPALATrainer(config, model_config)
    
    try:
        trainer.setup()
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        trainer.close()

if __name__ == "__main__":
    main()