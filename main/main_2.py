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
    d_model: int = 256  # WeKick: 256
    lstm_hidden: int = 256  # WeKick: 256
    lstm_layers: int = 1
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
            self.encoder_hidden = [256, 256]  # WeKick: 256 dim
        if self.policy_hidden is None:
            self.policy_hidden = [256]
        if self.value_hidden is None:
            self.value_hidden = [256]


class LSTMEncoder(nn.Module):
    """LSTM encoder like WeKick (32 steps, 256 hidden)."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim
        
    def forward(
        self, 
        x: torch.Tensor, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, L, D)
        B, L, D = x.shape
        
        if hidden_state is None:
            h0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device, dtype=x.dtype)
            c0 = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device, dtype=x.dtype)
            hidden_state = (h0, c0)
        
        output, (hn, cn) = self.lstm(x, hidden_state)
        output = self.layer_norm(output)
        
        return output, (hn, cn)
    
    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

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
                nn.ReLU()  # WeKick style
            ])
            in_dim = hidden_dim
        self.obs_encoder = nn.Sequential(*encoder_layers)
        
        # LSTM like WeKick
        self.lstm = LSTMEncoder(
            input_dim=in_dim,
            hidden_dim=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.dropout
        )
        
        policy_layers = []
        in_dim = self.lstm.output_dim
        for hidden_dim in config.policy_hidden:
            policy_layers.extend([nn.Linear(in_dim, hidden_dim), nn.ReLU()])
            in_dim = hidden_dim
        policy_layers.append(nn.Linear(in_dim, config.num_actions))
        self.policy_head = nn.Sequential(*policy_layers)
        
        # Multi-Head Value (MHV) like WeKick - vectorized for efficiency
        self.num_value_heads = 5
        value_hidden = config.value_hidden[0] if config.value_hidden else 256
        
        # Shared first layer, then parallel heads
        self.value_fc1 = nn.Linear(self.lstm.output_dim, value_hidden * self.num_value_heads)
        
        if config.use_distributional:
            self.value_fc2 = nn.Linear(value_hidden * self.num_value_heads, config.num_atoms * self.num_value_heads)
            self.register_buffer(
                'value_support',
                torch.linspace(config.v_min, config.v_max, config.num_atoms)
            )
            self.value_out_dim = config.num_atoms
        else:
            self.value_fc2 = nn.Linear(value_hidden * self.num_value_heads, self.num_value_heads)
            self.value_support = None
            self.value_out_dim = 1
            
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.orthogonal_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1 for better gradient flow
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)
                
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
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
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
        
        x, new_hidden = self.lstm(x, hidden_state)
        
        logits = self.policy_head(x)
        
        # Multi-Head Value - vectorized
        # x: [B, L, lstm_hidden]
        v = F.relu(self.value_fc1(x))  # [B, L, hidden * num_heads]
        v = self.value_fc2(v)  # [B, L, out_dim * num_heads]
        
        # Reshape to separate heads
        B_size, L_size = v.shape[:2]
        v = v.view(B_size, L_size, self.num_value_heads, self.value_out_dim)  # [B, L, num_heads, out_dim]
        
        if self.config.use_distributional:
            value_logits = v.mean(dim=2)  # [B, L, num_atoms] - average over heads
            value_probs = F.softmax(value_logits, dim=-1)
            value = (value_probs * self.value_support).sum(-1, keepdim=True)
        else:
            value = v.mean(dim=2)  # [B, L, 1] - average over heads
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
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        
        output = self.forward(obs, stage_idx, prev_action, hidden_state, return_hidden=True)
        logits = output['logits']
        
        # NaN protection
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
    
    def evaluate_actions(
        self, 
        obs: torch.Tensor, 
        stage_idx: torch.Tensor, 
        actions: torch.Tensor, 
        prev_action: Optional[torch.Tensor] = None, 
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        output = self.forward(obs, stage_idx, prev_action, hidden_state)
        logits = output['logits']
        
        logits = logits.clamp(min=-20.0, max=20.0)
        
        dist = Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy, output['value']
    
    def get_initial_hidden_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.lstm.get_initial_hidden_state(batch_size, device)

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
    lr_schedule: str = "cosine_restarts"
    max_grad_norm: float = 0.5
    gamma: float = 0.998
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    si_lambda: float = 0.1
    use_vtrace: bool = False
    vtrace_rho_max: float = 1.0
    vtrace_c_max: float = 1.0
    total_steps: int = 100_000_000
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
        StageConfig(1, "academy_empty_goal", "simple115v2", 1, 0, 400),
        StageConfig(2, "academy_run_to_score", "simple115v2", 1, 0, 400),
        StageConfig(3, "academy_run_to_score_with_keeper", "simple115v2", 1, 0, 400),
        StageConfig(4, "academy_pass_and_shoot_with_keeper", "simple115v2", 2, 0, 400),
        StageConfig(5, "academy_3_vs_1_with_keeper", "simple115v2", 3, 0, 400),
        StageConfig(6, "academy_counterattack_easy", "simple115v2", 4, 0, 600),
        StageConfig(7, "academy_counterattack_hard", "simple115v2", 4, 0, 600),
        StageConfig(8, "academy_single_goal_versus_lazy", "simple115v2", 11, 0, 1000),
        StageConfig(9, "11_vs_11_easy_stochastic", "simple115v2", 11, 0, 3000),
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
    T_LEARN = 0.65
    T_UNLOCK = 0.70
    T_MASTERY = 0.90
    MIN_EPS_UNLOCK = 100
    LP_WINDOW = 200
    STALENESS_COEFF = 0.1
    MIN_WEIGHT = 0.05
    
    def __init__(
        self, 
        stages: List[Dict], 
        baselines: Dict[int, Dict],
        final_target_win_rate: float = 0.5,
        initial_state: Optional[Dict] = None,
    ):
        self.stages = [StageConfig(**s) if isinstance(s, dict) else s for s in stages]
        self.baselines = {
            int(k): StageBaseline.from_dict(v) if isinstance(v, dict) else v 
            for k, v in baselines.items()
        }
        self.num_stages = len(self.stages)
        self.final_target_win_rate = final_target_win_rate
        
        if initial_state:
            self._restore_state(initial_state)
        else:
            self._init_fresh()
    
    def _init_fresh(self):
        self.episode_count = 0
        self.unlocked_stages = {0}
        self.learned_stages = set()
        self.mastered_stages = set()
        self.stage_stats = {
            s.stage_id: {
                'episodes': 0,
                'ema_return': 0.0,
                'ema_win': 0.0,
                'ema_win_slow': 0.0,
                'peak_win': 0.0,
                'recent_wins': [],
                'last_sampled': 0,
                'lp_history': [],
            } 
            for s in self.stages
        }
    
    def _restore_state(self, state: Dict):
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
                    'episodes': ss.get('episodes', 0),
                    'ema_return': ss.get('ema_return', 0.0),
                    'ema_win': ss.get('ema_win', 0.0),
                    'ema_win_slow': ss.get('ema_win_slow', ss.get('ema_win', 0.0)),
                    'peak_win': ss.get('peak_win', 0.0),
                    'recent_wins': [],
                    'last_sampled': ss.get('last_sampled', 0),
                    'lp_history': [],
                }
            else:
                self.stage_stats[s.stage_id] = {
                    'episodes': 0, 'ema_return': 0.0, 'ema_win': 0.0,
                    'ema_win_slow': 0.0, 'peak_win': 0.0, 'recent_wins': [],
                    'last_sampled': 0, 'lp_history': [],
                }
        
        print(f"  Restored: ep={self.episode_count}, learned={sorted(self.learned_stages)}, mastered={sorted(self.mastered_stages)}")
    
    def _compute_learning_progress(self, sid: int) -> float:
        stats = self.stage_stats[sid]
        lp = stats['ema_win'] - stats['ema_win_slow']
        return lp
    
    def _compute_staleness(self, sid: int) -> float:
        stats = self.stage_stats[sid]
        if stats['episodes'] == 0:
            return 0.0
        episodes_since = self.episode_count - stats['last_sampled']
        return min(episodes_since / 500.0, 2.0)
    
    def _compute_weight(self, sid: int) -> float:
        stats = self.stage_stats[sid]
        
        if stats['episodes'] < 50:
            base = 1.0
        else:
            lp = self._compute_learning_progress(sid)
            lp_score = max(0.0, lp) * 10.0
            
            forgetting = max(0.0, stats['peak_win'] - stats['ema_win'])
            forgetting_score = forgetting * 5.0
            
            if sid in self.mastered_stages:
                base = 0.1 + forgetting_score
            elif sid in self.learned_stages:
                base = 0.3 + lp_score + forgetting_score
            else:
                if stats['ema_win'] < 0.1:
                    base = 0.5 + lp_score
                elif stats['ema_win'] < 0.3:
                    base = 0.8 + lp_score
                else:
                    base = 1.0 + lp_score
        
        staleness = self._compute_staleness(sid)
        staleness_bonus = self.STALENESS_COEFF * staleness
        
        return max(self.MIN_WEIGHT, base + staleness_bonus)
    
    def get_stage(self) -> Dict:
        available = sorted(self.unlocked_stages)
        
        if len(available) == 1:
            self.stage_stats[available[0]]['last_sampled'] = self.episode_count
            return asdict(self.stages[available[0]])
        
        weights = {}
        for sid in available:
            weights[sid] = self._compute_weight(sid)
        
        total = sum(weights.values())
        probs = {k: v/total for k, v in weights.items()}
        
        stages_list = list(probs.keys())
        prob_values = [probs[s] for s in stages_list]
        chosen = np.random.choice(stages_list, p=prob_values)
        
        self.stage_stats[chosen]['last_sampled'] = self.episode_count
        
        return asdict(self.stages[chosen])
    
    def report_episode(self, stage_id: int, episode_return: float, won: bool):
        self.episode_count += 1
        stats = self.stage_stats[stage_id]
        
        stats['episodes'] += 1
        
        stats['recent_wins'].append(1.0 if won else 0.0)
        if len(stats['recent_wins']) > 100:
            stats['recent_wins'].pop(0)
        
        alpha_fast = 0.02
        alpha_slow = 0.005
        
        if stats['episodes'] == 1:
            stats['ema_return'] = episode_return
            stats['ema_win'] = float(won)
            stats['ema_win_slow'] = float(won)
        else:
            stats['ema_return'] = (1 - alpha_fast) * stats['ema_return'] + alpha_fast * episode_return
            stats['ema_win'] = (1 - alpha_fast) * stats['ema_win'] + alpha_fast * float(won)
            stats['ema_win_slow'] = (1 - alpha_slow) * stats['ema_win_slow'] + alpha_slow * float(won)
        
        if stats['ema_win'] > stats['peak_win']:
            stats['peak_win'] = stats['ema_win']
        
        lp = self._compute_learning_progress(stage_id)
        stats['lp_history'].append(lp)
        if len(stats['lp_history']) > self.LP_WINDOW:
            stats['lp_history'].pop(0)
        
        self._check_learned(stage_id)
        self._check_unlock(stage_id)
        self._check_mastery(stage_id)
    
    def _check_learned(self, stage_id: int):
        if stage_id in self.learned_stages:
            return
        
        stats = self.stage_stats[stage_id]
        
        if stats['episodes'] < 100:
            return
        
        if stats['ema_win'] < self.T_LEARN:
            return
        
        recent_wr = np.mean(stats['recent_wins']) if len(stats['recent_wins']) >= 50 else 0
        if recent_wr < 0.5:
            return
        
        self.learned_stages.add(stage_id)
        print(f"\n📚 STAGE {stage_id} LEARNED! (ema={stats['ema_win']:.1%})\n")
    
    def _check_unlock(self, stage_id: int):
        next_stage = stage_id + 1
        
        if next_stage >= self.num_stages:
            return
        if next_stage in self.unlocked_stages:
            return
        
        stats = self.stage_stats[stage_id]
        
        if stats['episodes'] < self.MIN_EPS_UNLOCK:
            return
        
        recent_wr = np.mean(stats['recent_wins']) if stats['recent_wins'] else 0
        
        if recent_wr >= self.T_UNLOCK:
            self.unlocked_stages.add(next_stage)
            print(f"\n🔓 STAGE {next_stage} UNLOCKED! (Stage {stage_id}: {recent_wr:.1%})\n")
    
    def _check_mastery(self, stage_id: int):
        if stage_id in self.mastered_stages:
            return
        
        stats = self.stage_stats[stage_id]
        
        if stats['episodes'] < 200:
            return
        
        recent_wr = np.mean(stats['recent_wins']) if stats['recent_wins'] else 0
        
        if recent_wr >= self.T_MASTERY:
            self.mastered_stages.add(stage_id)
            print(f"\n⭐ STAGE {stage_id} MASTERED! ({recent_wr:.1%})\n")
    
    def is_training_complete(self) -> bool:
        final_stage = self.num_stages - 1
        
        if final_stage not in self.learned_stages:
            return False
        
        stats = self.stage_stats[final_stage]
        recent_wr = np.mean(stats['recent_wins']) if stats['recent_wins'] else 0
        
        return recent_wr >= self.final_target_win_rate
    
    def get_progress_summary(self) -> str:
        lines = []
        for sid in sorted(self.unlocked_stages):
            stats = self.stage_stats[sid]
            
            if sid in self.mastered_stages:
                status = "⭐"
            elif sid in self.learned_stages:
                status = "📚"
            else:
                status = "🔓"
            
            if stats['episodes'] > 0:
                recent_wr = np.mean(stats['recent_wins']) if stats['recent_wins'] else 0
                lp = self._compute_learning_progress(sid)
                weight = self._compute_weight(sid)
                lp_str = f"LP{lp:+.0%}" if abs(lp) > 0.01 else ""
                lines.append(f"{status}S{sid}:{recent_wr:.0%} w={weight:.1f} {lp_str}")
            else:
                lines.append(f"{status}S{sid}:--")
        
        return " | ".join(lines)
    
    def get_stats(self) -> Dict:
        weights = {sid: self._compute_weight(sid) for sid in self.unlocked_stages}
        total_w = sum(weights.values())
        sample_probs = {sid: w/total_w for sid, w in weights.items()}
        
        return {
            'episode_count': self.episode_count,
            'unlocked_stages': list(self.unlocked_stages),
            'learned_stages': list(self.learned_stages),
            'mastered_stages': list(self.mastered_stages),
            'highest_unlocked': max(self.unlocked_stages) if self.unlocked_stages else 0,
            'training_complete': self.is_training_complete(),
            'sample_probs': sample_probs,
            'stage_stats': {
                str(sid): {
                    'episodes': s['episodes'],
                    'ema_return': s['ema_return'],
                    'ema_win': s['ema_win'],
                    'ema_win_slow': s['ema_win_slow'],
                    'peak_win': s['peak_win'],
                    'learning_progress': self._compute_learning_progress(sid),
                    'forgetting': max(0.0, s['peak_win'] - s['ema_win']),
                    'staleness': self._compute_staleness(sid),
                    'weight': self._compute_weight(sid),
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
        
        self.si_omega = {}
        self.si_prev_params = {}
        self.si_running_sum = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.si_omega[name] = torch.zeros_like(param, device=self.device)
                self.si_prev_params[name] = param.data.clone()
                self.si_running_sum[name] = torch.zeros_like(param, device=self.device)
        
        self.si_lambda = config.si_lambda
        self.si_epsilon = 1e-3
        
        print(f"Learner on {self.device}, params: {count_parameters(self.model):,}, SI lambda: {self.si_lambda}")
        
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
                
                si_loss = 0.0
                if self.si_lambda > 0:
                    for name, param in self.model.named_parameters():
                        if param.requires_grad and name in self.si_omega:
                            si_loss += (self.si_omega[name] * (param - self.si_prev_params[name]).pow(2)).sum()
                    loss = loss + self.si_lambda * si_loss
                
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
                
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.si_running_sum:
                        delta = param.data - self.si_prev_params[name]
                        if param.grad is not None:
                            self.si_running_sum[name] += -param.grad.data * delta
                
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
        
        if self.update_count % 100 == 0:
            self._consolidate_si()
        
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
    
    def _consolidate_si(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.si_omega:
                delta = param.data - self.si_prev_params[name]
                delta_norm = delta.pow(2) + self.si_epsilon
                self.si_omega[name] += self.si_running_sum[name] / delta_norm
                self.si_omega[name] = torch.clamp(self.si_omega[name], min=0.0)
                self.si_prev_params[name] = param.data.clone()
                self.si_running_sum[name].zero_()
    
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
            'nan_count': self.nan_count,
            'si_omega': {k: v.cpu() for k, v in self.si_omega.items()},
            'si_prev_params': {k: v.cpu() for k, v in self.si_prev_params.items()},
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
        
        if 'si_omega' in checkpoint:
            for name in self.si_omega:
                if name in checkpoint['si_omega']:
                    self.si_omega[name] = checkpoint['si_omega'][name].to(self.device)
        if 'si_prev_params' in checkpoint:
            for name in self.si_prev_params:
                if name in checkpoint['si_prev_params']:
                    self.si_prev_params[name] = checkpoint['si_prev_params'][name].to(self.device)
        
        return checkpoint

class IMPALATrainer:
    def __init__(self, config: TrainingConfig, model_config: Dict, resume_from: Optional[str] = None):
        self.config = config
        self.model_config = model_config
        self.resume_from = resume_from
        
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
        
        self.last_highest_unlocked = 0
        self.last_progress_step = 0
        
        self.episode_returns_buffer = defaultdict(list)
        self.episode_wins_buffer = defaultdict(list)
        self.episode_lengths_buffer = defaultdict(list)
        
        # Checkpoint data (will be loaded if resuming)
        self.checkpoint_data = None
        
    def setup(self):
        print("=" * 60)
        print("IMPALA TRAINER SETUP (Reward-Based Curriculum)")
        if self.resume_from:
            print(f"RESUMING FROM: {self.resume_from}")
        print("=" * 60)
        
        # Initialize TensorBoard
        run_name = f"gfootball_mamba_{time.strftime('%Y%m%d_%H%M%S')}"
        if self.resume_from:
            run_name = f"gfootball_mamba_resumed_{time.strftime('%Y%m%d_%H%M%S')}"
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
            'd_model': self.model_config.get('d_model', 256),
            'lstm_hidden': self.model_config.get('lstm_hidden', 256),
            'si_lambda': self.config.si_lambda,
            'resumed': self.resume_from is not None,
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
        
        self.learner = Learner(self.config, self.model_config, self.writer)
        
        curriculum_initial_state = None
        if self.resume_from:
            checkpoint_path = Path(self.resume_from)
            if checkpoint_path.exists():
                print(f"\nLoading checkpoint: {checkpoint_path}")
                self.checkpoint_data = self.learner.load_checkpoint(checkpoint_path)
                
                self.total_steps = self.checkpoint_data.get('total_steps', 0)
                self.total_episodes = self.checkpoint_data.get('total_episodes', 0)
                curriculum_initial_state = self.checkpoint_data.get('curriculum_stats', None)
                
                print(f"  Restored: steps={self.total_steps:,}, episodes={self.total_episodes}, updates={self.learner.update_count}")
            else:
                print(f"WARNING: Checkpoint not found: {checkpoint_path}")
        
        self.curriculum = CurriculumController.remote(
            stages_dict, 
            baselines_dict,
            final_target_win_rate=self.config.final_stage_target_win_rate,
            initial_state=curriculum_initial_state,
        )
        
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
        
        if curriculum_initial_state:
            self.last_highest_unlocked = len(curriculum_initial_state.get('learned_stages', []))
            self.last_progress_step = self.total_steps
        
        print(f"\nCurriculum settings (Learning Progress + Staleness):")
        print(f"  T_LEARN:   {CurriculumController.T_LEARN:.0%}")
        print(f"  T_UNLOCK:  {CurriculumController.T_UNLOCK:.0%}")
        print(f"  T_MASTERY: {CurriculumController.T_MASTERY:.0%}")
        print(f"  LP_WINDOW: {CurriculumController.LP_WINDOW}")
        print(f"  STALENESS_COEFF: {CurriculumController.STALENESS_COEFF}")
        
        if self.resume_from:
            print(f"\nResumed training state:")
            print(f"  Total steps: {self.total_steps:,}")
            print(f"  Total episodes: {self.total_episodes}")
            print(f"  Learner updates: {self.learner.update_count}")
            
        print("Setup complete.\n")
        
    def _sync_weights(self):
        weights = self.learner.get_weights()
        ray.get([w.set_weights.remote(weights) for w in self.workers])
        
    def _aggregate_episode_stats(self, trajectories: List[Dict]):
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
        if self.writer is None:
            return
            
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
        
        if all_returns:
            self.writer.add_scalar('episode/return_mean', np.mean(all_returns), self.total_steps)
            self.writer.add_scalar('episode/return_std', np.std(all_returns), self.total_steps)
            self.writer.add_scalar('episode/return_min', np.min(all_returns), self.total_steps)
            self.writer.add_scalar('episode/return_max', np.max(all_returns), self.total_steps)
            self.writer.add_scalar('episode/win_rate', np.mean(all_wins), self.total_steps)
            self.writer.add_scalar('episode/length_mean', np.mean(all_lengths), self.total_steps)
            self.writer.add_histogram('episode/return_dist', np.array(all_returns), self.total_steps)
        
        self.episode_returns_buffer.clear()
        self.episode_wins_buffer.clear()
        self.episode_lengths_buffer.clear()
        
    def train(self):
        print("Starting training (will stop when curriculum complete)...")
        self.start_time = time.time()
        if self.last_progress_step == 0:
            self.last_progress_step = self.total_steps
        
        pending = {
            w.collect_trajectory.remote(self.config.trajectory_length, self.curriculum): w 
            for w in self.workers
        }
        
        trajectories_buffer = []
        update_count = self.learner.update_count  # Start from restored count
        
        while True:
            curriculum_stats = ray.get(self.curriculum.get_stats.remote())
            
            if curriculum_stats.get('training_complete', False):
                print("\n" + "=" * 60)
                print("🎉 TRAINING COMPLETE - All stages solved!")
                print("=" * 60)
                break
                
            if self.total_steps >= self.config.total_steps:
                print("\n" + "=" * 60)
                print("⚠️ Max steps reached without completing curriculum")
                print("=" * 60)
                break
                
            highest_unlocked = curriculum_stats.get('highest_unlocked', 0)
            num_learned = len(curriculum_stats.get('learned_stages', []))
            if num_learned > self.last_highest_unlocked:
                self.last_highest_unlocked = num_learned
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
        
        learned = curriculum_stats.get('learned_stages', [])
        mastered = curriculum_stats.get('mastered_stages', [])
        sample_probs = curriculum_stats.get('sample_probs', {})
        
        print(f"[{update_count}] Steps: {self.total_steps:,} | Ep: {self.total_episodes} | SPS: {int(sps)} | {loss_str} {nan_str}")
        print(f"        Learned: {sorted(learned)} | Mastered: {sorted(mastered)}")
        
        stage_stats = curriculum_stats.get('stage_stats', {})
        if stage_stats:
            active = []
            unlocked = curriculum_stats.get('unlocked_stages', [0])
            for sid in sorted(unlocked):
                sid_str = str(sid)
                if sid_str in stage_stats:
                    s = stage_stats[sid_str]
                    if s['episodes'] > 0:
                        recent_wr = s.get('recent_win_rate', s['ema_win'])
                        lp = s.get('learning_progress', 0)
                        prob = sample_probs.get(sid, 0) * 100
                        
                        marker = ""
                        if sid in mastered:
                            marker = "⭐"
                        elif sid in learned:
                            marker = "📚"
                        
                        lp_str = f"LP{lp:+.0%}" if abs(lp) > 0.02 else ""
                        active.append(f"S{sid}:{recent_wr:.0%}{marker}({prob:.0f}%){lp_str}")
            print(f"        {', '.join(active)}")
            
            if self.writer is not None:
                for sid_str, s in stage_stats.items():
                    sid = int(sid_str)
                    self.writer.add_scalar(f'curriculum/ema_win_stage_{sid}', s['ema_win'], self.total_steps)
                    self.writer.add_scalar(f'curriculum/peak_win_stage_{sid}', s['peak_win'], self.total_steps)
                    self.writer.add_scalar(f'curriculum/learning_progress_stage_{sid}', s.get('learning_progress', 0), self.total_steps)
                    self.writer.add_scalar(f'curriculum/forgetting_stage_{sid}', s['forgetting'], self.total_steps)
                    self.writer.add_scalar(f'curriculum/weight_stage_{sid}', s.get('weight', 0), self.total_steps)
                    self.writer.add_scalar(f'curriculum/staleness_stage_{sid}', s.get('staleness', 0), self.total_steps)
                    self.writer.add_scalar(f'curriculum/episodes_stage_{sid}', s['episodes'], self.total_steps)
                
                for sid, prob in sample_probs.items():
                    self.writer.add_scalar(f'curriculum/sample_prob_stage_{sid}', prob, self.total_steps)
                
                self.writer.add_scalar('curriculum/num_learned', len(learned), self.total_steps)
                self.writer.add_scalar('curriculum/num_mastered', len(mastered), self.total_steps)
                      
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
    RESUME_FROM = None  # "./checkpoints/checkpoint_update_XXX.pt" für Resume
    CHECKPOINT_DIR = "./checkpoints"
    LOG_DIR = "./logs"
    NUM_WORKERS = 24
    
    model_config = {
        'd_model': 256,  # WeKick: 256
        'lstm_hidden': 256,  # WeKick: 256
        'lstm_layers': 1,
        'encoder_hidden': [256, 256],  # WeKick: 256 dim each
        'policy_hidden': [256],
        'value_hidden': [256],
        'use_distributional': True,
        'dropout': 0.0,
        'num_stages': 10,
    }
    
    config = TrainingConfig(
        stages=get_default_stages(),
        final_stage_target_win_rate=0.5,
        max_steps_without_progress=100_000_000,
        num_workers=NUM_WORKERS,
        trajectory_length=128,
        batch_size=4096,
        minibatch_size=1024,
        num_epochs=3,
        learning_rate=1e-4,  # WeKick: 1e-4
        lr_schedule="cosine_restarts",
        gamma=0.998,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        entropy_coeff=0.01,
        value_coeff=0.5,
        si_lambda=0.1,
        max_grad_norm=0.5,
        total_steps=1_000_000_000,
        log_interval=10,
        checkpoint_interval=100,
        weight_sync_interval=5,
        log_dir=LOG_DIR,
        checkpoint_dir=CHECKPOINT_DIR,
    )
    
    ray.init(
        num_cpus=NUM_WORKERS,
        num_gpus=1,
        ignore_reinit_error=True,
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
 