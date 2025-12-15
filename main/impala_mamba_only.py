import time
from pathlib import Path
from collections import deque
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.amp import autocast, GradScaler
import pyarrow.parquet as pq
import ray
import gfootball.env as football_env

FEATURE_DIM = 93
OBS_DIM = 115
NUM_ACTIONS = 19


class FeatureEngineer:
    GOAL = np.array([1.0, 0.0], dtype=np.float32)
    OWN_GOAL = np.array([-1.0, 0.0], dtype=np.float32)

    @staticmethod
    def extract(obs: np.ndarray, active_idx: int = None) -> np.ndarray:
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.reshape(1, -1)
        B = obs.shape[0]
        obs = obs[:, :115] if obs.shape[1] >= 115 else np.pad(obs, ((0, 0), (0, 115 - obs.shape[1])))
        feat = np.zeros((B, FEATURE_DIM), dtype=np.float32)
        
        left_pos = obs[:, 0:22].reshape(B, 11, 2)
        left_dir = obs[:, 22:44].reshape(B, 11, 2)
        right_pos = obs[:, 44:66].reshape(B, 11, 2)
        right_dir = obs[:, 66:88].reshape(B, 11, 2)
        ball_pos, ball_z, ball_dir = obs[:, 88:90], obs[:, 90:91], obs[:, 91:94]
        ball_owned_team = np.argmax(obs[:, 94:97], axis=1) - 1
        game_mode, sticky = obs[:, 98:105], obs[:, 105:115]
        
        if active_idx is None:
            active_idx = np.argmax(obs[:, 97:108], axis=1)
        elif isinstance(active_idx, int):
            active_idx = np.full(B, active_idx)
        
        bi = np.arange(B)
        active_pos = left_pos[bi, active_idx]
        ball_speed = np.linalg.norm(ball_dir[:, :2], axis=1)
        
        feat[:, 0:2] = ball_pos
        feat[:, 2] = np.clip(ball_z[:, 0], 0, 1)
        feat[:, 3] = np.clip(ball_speed, 0, 2)
        feat[:, 4:6] = ball_dir[:, :2]
        feat[:, 6] = ball_owned_team / 2.0
        rel_ball = ball_pos - active_pos
        feat[:, 7:9] = rel_ball
        feat[:, 9] = np.clip(np.linalg.norm(rel_ball, axis=1), 0, 2)
        feat[:, 10] = np.arctan2(rel_ball[:, 1], rel_ball[:, 0]) / np.pi
        
        goal_vec = FeatureEngineer.GOAL - ball_pos
        dist_goal = np.linalg.norm(goal_vec, axis=1)
        feat[:, 11] = np.clip(dist_goal, 0, 2)
        feat[:, 12] = np.abs(np.arctan2(goal_vec[:, 1], goal_vec[:, 0])) / np.pi
        feat[:, 13] = np.clip(0.088 / (dist_goal + 0.01), 0, 1)
        feat[:, 14] = (dist_goal < 0.35).astype(np.float32)
        feat[:, 15] = np.clip(np.linalg.norm(FeatureEngineer.OWN_GOAL - ball_pos, axis=1), 0, 2)
        right_x = right_pos[:, :, 0]
        keeper_idx = np.argmax(right_x, axis=1)
        keeper_pos = right_pos[bi, keeper_idx]
        keeper_dist = np.linalg.norm(ball_pos - keeper_pos, axis=1)
        feat[:, 16] = np.clip(keeper_dist, 0, 2)
        feat[:, 17] = np.arctan2((ball_pos - keeper_pos)[:, 1], (ball_pos - keeper_pos)[:, 0]) / np.pi
        
        left_active = np.abs(left_pos[:, :, 0]) > 0.01
        tm_dist = np.linalg.norm(left_pos - active_pos[:, None, :], axis=2)
        tm_dist[bi, active_idx] = 999.0
        tm_dist = np.where(left_active, tm_dist, 999.0)
        tm_idx = np.argsort(tm_dist, axis=1)
        for i in range(5):
            idx = tm_idx[:, i]
            valid = tm_dist[bi, idx] < 100
            feat[:, 18+i*4:20+i*4] = np.where(valid[:, None], left_pos[bi, idx] - active_pos, 0)
            feat[:, 20+i*4:22+i*4] = np.where(valid[:, None], left_dir[bi, idx], 0)
        
        right_active = np.abs(right_pos[:, :, 0]) > 0.01
        op_dist = np.linalg.norm(right_pos - active_pos[:, None, :], axis=2)
        op_dist = np.where(right_active, op_dist, 999.0)
        op_idx = np.argsort(op_dist, axis=1)
        for i in range(5):
            idx = op_idx[:, i]
            valid = op_dist[bi, idx] < 100
            feat[:, 38+i*4:40+i*4] = np.where(valid[:, None], right_pos[bi, idx] - active_pos, 0)
            feat[:, 40+i*4:42+i*4] = np.where(valid[:, None], right_dir[bi, idx], 0)
        
        ball_x = ball_pos[:, 0]
        left_x = left_pos[:, :, 0]
        feat[:, 58] = np.sum((left_x > ball_x[:, None]) & left_active, axis=1) / 11.0
        feat[:, 59] = np.sum((right_x > ball_x[:, None]) & right_active, axis=1) / 11.0
        feat[:, 60] = np.clip((feat[:, 58] - feat[:, 59]) * 2, -1, 1)
        opp_dist_active = np.where(right_active, np.linalg.norm(right_pos - active_pos[:, None, :], axis=2), 10.0)
        feat[:, 63] = np.clip(np.min(opp_dist_active, axis=1), 0, 1)
        
        sorted_rx = np.sort(right_x, axis=1)
        offside_line = np.maximum(ball_x, sorted_rx[:, 1])
        feat[:, 67] = offside_line
        feat[:, 68] = ((active_pos[:, 0] > offside_line) & (ball_owned_team == 0)).astype(np.float32)
        feat[:, 69:72] = np.column_stack([(ball_x > 0.33), ((ball_x >= -0.33) & (ball_x <= 0.33)), (ball_x < -0.33)])
        feat[:, 72:74] = np.column_stack([(ball_pos[:, 1] > 0.2), (ball_pos[:, 1] < -0.2)])
        
        feat[:, 74:76] = sticky[:, 8:10]
        sticky_dir = sticky[:, :8]
        sticky_active = np.any(sticky_dir > 0, axis=1)
        angle = np.argmax(sticky_dir, axis=1) * (2 * np.pi / 8)
        feat[:, 76:78] = np.column_stack([np.where(sticky_active, np.cos(angle), 0), np.where(sticky_active, np.sin(angle), 0)])
        feat[:, 78:85] = game_mode
        feat[:, 85:93] = 0
        
        return feat[0] if squeeze else feat


class ExpertBuffer:
    def __init__(self, parquet_path: str, rollout_len: int = 128):
        self.rollout_len = rollout_len
        self.rollouts = []
        self.returns = []
        
        if parquet_path and Path(parquet_path).exists():
            self._load_parquet(parquet_path)
    
    def _load_parquet(self, parquet_path):
        print(f"Loading expert data from {parquet_path}...")
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        obs_all = np.array([np.frombuffer(b, dtype=np.float32) for b in df['obs']])
        actions = df['action'].values.astype(np.int64)
        rewards = df['reward'].values.astype(np.float32)
        active = df['active'].values.astype(np.int64)
        episode_ids = df['episode_id'].values
        scores = df['score'].values
        
        print(f"Computing features for {len(obs_all)} samples...")
        feat_eng = FeatureEngineer()
        feat_all = np.array([feat_eng.extract(o, a) for o, a in zip(obs_all, active)])
        
        unique_eps = np.unique(episode_ids)
        print(f"Processing {len(unique_eps)} episodes...")
        
        for ep_id in unique_eps:
            mask = episode_ids == ep_id
            ep_obs = obs_all[mask]
            ep_feat = feat_all[mask]
            ep_act = actions[mask]
            ep_rew = rewards[mask]
            ep_score = scores[mask][0]
            
            num_rollouts = len(ep_obs) // self.rollout_len
            for i in range(num_rollouts):
                start = i * self.rollout_len
                end = start + self.rollout_len
                
                prev_acts = np.zeros(self.rollout_len, dtype=np.int64)
                prev_acts[0] = NUM_ACTIONS
                prev_acts[1:] = ep_act[start:end-1]
                
                rollout = {
                    'obs': ep_obs[start:end].astype(np.float32),
                    'feat': ep_feat[start:end].astype(np.float32),
                    'act': ep_act[start:end],
                    'rew': ep_rew[start:end],
                    'done': np.zeros(self.rollout_len, dtype=np.float32),
                    'lp': np.zeros(self.rollout_len, dtype=np.float32),
                    'prev_act': prev_acts,
                    'bootstrap': 0.0,
                }
                rollout['done'][-1] = 1.0 if i == num_rollouts - 1 else 0.0
                
                self.rollouts.append(rollout)
                self.returns.append(ep_score)
        
        print(f"Loaded {len(self.rollouts)} expert rollouts, avg return: {np.mean(self.returns):.1f}")
    
    def sample(self, n: int) -> List[dict]:
        if not self.rollouts:
            return []
        n = min(n, len(self.rollouts))
        weights = np.array(self.returns) + 0.1
        probs = weights / weights.sum()
        idx = np.random.choice(len(self.rollouts), size=n, replace=False, p=probs)
        return [self.rollouts[i] for i in idx]
    
    def __len__(self):
        return len(self.rollouts)


class GoldenMemory:
    def __init__(self, capacity: int = 256, max_uses: int = 8):
        self.capacity = capacity
        self.max_uses = max_uses
        self.buffer = []
        self.returns = []
        self.wins = []
        self.uses = []
        
    def add(self, rollout: dict, ret: float, won: bool):
        if ret < 0.5 and not won:
            return False
        
        if len(self.buffer) >= self.capacity:
            worst_idx = int(np.argmin(self.returns))
            if ret <= self.returns[worst_idx]:
                return False
            self.buffer.pop(worst_idx)
            self.returns.pop(worst_idx)
            self.wins.pop(worst_idx)
            self.uses.pop(worst_idx)
        
        self.buffer.append(rollout.copy())
        self.returns.append(ret)
        self.wins.append(won)
        self.uses.append(0)
        return True
    
    def sample(self, n: int) -> List[dict]:
        if not self.buffer:
            return []
        
        valid = [i for i, u in enumerate(self.uses) if u < self.max_uses]
        if not valid:
            self._cleanup()
            return []
        
        n = min(n, len(valid))
        weights = np.array([self.returns[i] for i in valid]) + 0.1
        weights = np.array([w * 2.0 if self.wins[valid[j]] else w for j, w in enumerate(weights)])
        probs = weights / weights.sum()
        idx = np.random.choice(valid, size=n, replace=False, p=probs)
        
        for i in idx:
            self.uses[i] += 1
        
        return [self.buffer[i] for i in idx]
    
    def _cleanup(self):
        keep = [i for i, u in enumerate(self.uses) if u < self.max_uses]
        self.buffer = [self.buffer[i] for i in keep]
        self.returns = [self.returns[i] for i in keep]
        self.wins = [self.wins[i] for i in keep]
        self.uses = [self.uses[i] for i in keep]
    
    def stats(self):
        if not self.buffer:
            return {'size': 0, 'wins': 0, 'ret_mean': 0, 'ret_max': 0, 'fresh': 0}
        fresh = sum(1 for u in self.uses if u < self.max_uses)
        return {
            'size': len(self.buffer),
            'wins': sum(self.wins),
            'ret_mean': np.mean(self.returns),
            'ret_max': np.max(self.returns),
            'fresh': fresh
        }
    
    def __len__(self):
        return len(self.buffer)


class PopArtValueHead(nn.Module):
    def __init__(self, input_dim: int, beta: float = 1e-3):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(input_dim, 1)
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sigma', torch.ones(1))
        self.register_buffer('nu', torch.ones(1))
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)
    
    def denormalize(self, normalized):
        return normalized * self.sigma + self.mu
    
    def normalize_target(self, targets):
        return (targets - self.mu) / self.sigma

    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor):
        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()
        t_mean = targets.mean()
        t_sq_mean = (targets ** 2).mean()
        self.mu.mul_(1 - self.beta).add_(self.beta * t_mean)
        self.nu.mul_(1 - self.beta).add_(self.beta * t_sq_mean)
        var = torch.clamp(self.nu - self.mu ** 2, min=1e-4)
        self.sigma.copy_(torch.sqrt(var))
        self.linear.weight.data.mul_(old_sigma / self.sigma)
        self.linear.bias.data.copy_(
            (self.linear.bias.data * old_sigma + old_mu - self.mu) / self.sigma
        )


class SoftClamp(nn.Module):
    """Soft Clipping with preserved gradients using tanh."""
    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.range = (max_val - min_val) / 2
        self.center = (max_val + min_val) / 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tanh-based soft clipping: gradients are preserved!
        return self.center + self.range * torch.tanh((x - self.center) / self.range)


class MambaBlock(nn.Module):
    """Mamba selective state space block with numerical stability."""
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        
        # A_log with bounded initialization
        self.A_log_diag = nn.Parameter(torch.log(torch.linspace(1.0, d_state, d_state)).clamp(max=2.0))
        
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Soft clamps for numerical stability
        self.dt_clamp = SoftClamp(1e-4, 0.5)
        self.h_clamp = SoftClamp(-10.0, 10.0)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.B_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.C_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        
        # dt_proj: small weights and bias so softplus outputs small values
        nn.init.normal_(self.dt_proj.weight, std=0.001)
        nn.init.constant_(self.dt_proj.bias, -4.6)  # softplus(-4.6) ≈ 0.01

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_in, z = xz.chunk(2, dim=-1)
        
        # dt with soft clamp for stability
        dt = self.dt_clamp(F.softplus(self.dt_proj(x_in)))
        
        B_t = self.B_proj(x_in)
        C_t = self.C_proj(x_in)
        
        # A with bounded range
        A_diag = -torch.exp(self.A_log_diag.clamp(max=2.0))
        
        outputs = []
        for t in range(L):
            # Discretization
            dt_t = dt[:, t, :].unsqueeze(-1)  # [B, D, 1]
            dA = torch.exp(dt_t * A_diag.unsqueeze(0).unsqueeze(0))  # [B, D, d_state]
            dB = dt_t * B_t[:, t, :].unsqueeze(1)  # [B, D, d_state]
            
            # State update with soft clamp
            h = h * dA + x_in[:, t, :].unsqueeze(-1) * dB
            h = self.h_clamp(h)  # Prevent explosion
            
            # Output
            y_t = (h * C_t[:, t, :].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        out = y * F.silu(z) + x_in * self.D
        out = x + self.dropout(self.out_proj(out))
        
        return out, h


class Net(nn.Module):
    def __init__(self, d_model: int = 128, d_state: int = 32, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        
        self.action_emb = nn.Embedding(NUM_ACTIONS + 1, 16)
        
        input_dim = OBS_DIM + FEATURE_DIM + 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
        self.policy = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )
        self.value = PopArtValueHead(d_model)
        
        self._init()
        print(f"Net (Mamba): {sum(p.numel() for p in self.parameters()):,} params")

    def _init(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)

    def init_hidden(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize hidden state for all Mamba layers."""
        return [torch.zeros(batch_size, self.d_model, self.d_state, device=device) 
                for _ in range(self.num_layers)]

    def forward(self, obs, feat, prev_actions=None, hidden=None):
        squeeze = obs.dim() == 2
        if squeeze:
            obs, feat = obs.unsqueeze(1), feat.unsqueeze(1)
        
        B, L, _ = obs.shape
        
        if prev_actions is None:
            prev_actions = torch.full((B,), NUM_ACTIONS, dtype=torch.long, device=obs.device)
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.unsqueeze(1).expand(-1, L)
        
        x = torch.cat([obs, feat, self.action_emb(prev_actions)], dim=-1)
        x = self.encoder(x)
        
        if hidden is None:
            hidden = self.init_hidden(B, obs.device)
        
        new_hidden = []
        for i, layer in enumerate(self.mamba_layers):
            x, h_new = layer(x, hidden[i])
            new_hidden.append(h_new)
        
        x = self.final_norm(x)
        
        logits = self.policy(x)
        values_norm = self.value(x)
        
        if squeeze:
            logits, values_norm = logits.squeeze(1), values_norm.squeeze(1)
        return logits, values_norm, new_hidden

    def get_action(self, obs, feat, prev_actions, hidden=None):
        logits, values_norm, hidden = self.forward(obs, feat, prev_actions, hidden)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        values = self.value.denormalize(values_norm)
        return actions, dist.log_prob(actions), values, hidden


def vtrace(behavior_lp, target_lp, rewards, values, bootstrap, dones, gamma=0.99, rho_bar=1.0, c_bar=1.0):
    B, T = rewards.shape
    rhos = torch.exp((target_lp - behavior_lp).clamp(-20, 20))
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)
    not_done = 1.0 - dones
    values_tp1 = torch.cat([values[:, 1:], bootstrap.unsqueeze(1)], dim=1)
    deltas = clipped_rhos * (rewards + gamma * values_tp1 * not_done - values)
    
    vs_minus_v = torch.zeros_like(rewards)
    acc = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        acc = deltas[:, t] + gamma * cs[:, t] * acc * not_done[:, t]
        vs_minus_v[:, t] = acc
    
    vs = values + vs_minus_v
    vs_tp1 = torch.cat([vs[:, 1:], bootstrap.unsqueeze(1)], dim=1)
    advantages = clipped_rhos * (rewards + gamma * vs_tp1 * not_done - values)
    return vs, advantages, clipped_rhos


@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, wid, d_model, d_state, num_layers, rollout_len, env_name, reward_type="scoring"):
        self.wid = wid
        self.rollout_len = rollout_len
        self.feat_eng = FeatureEngineer()
        
        self.env = football_env.create_environment(
            env_name=env_name,
            representation="simple115v2",
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0,
            rewards=reward_type,
            render=False
        )
        self.model = Net(d_model, d_state, num_layers)
        self.model.eval()
        self._reset()

    def set_weights(self, w):
        self.model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in w.items()})

    def _reset(self):
        obs = self.env.reset()
        obs = np.array(obs).flatten()[:OBS_DIM].astype(np.float32)
        self.obs = obs
        self.feat = self.feat_eng.extract(obs)
        self.ep_ret, self.ep_len = 0.0, 0
        self.prev_act = torch.tensor([NUM_ACTIONS], dtype=torch.long)
        self.hidden = self.model.init_hidden(1, torch.device('cpu'))

    def collect(self):
        data = {k: [] for k in ['obs', 'feat', 'prev_act', 'act', 'lp', 'rew', 'done']}
        episodes = []
        max_rew, min_rew = -999, 999
        
        for _ in range(self.rollout_len):
            with torch.no_grad():
                act, lp, _, self.hidden = self.model.get_action(
                    torch.from_numpy(self.obs).float().unsqueeze(0),
                    torch.from_numpy(self.feat).float().unsqueeze(0),
                    self.prev_act,
                    self.hidden
                )
            
            data['obs'].append(self.obs.copy())
            data['feat'].append(self.feat.copy())
            data['prev_act'].append(self.prev_act.item())
            data['act'].append(act.item())
            data['lp'].append(lp.item())
            self.prev_act = act.clone()
            
            obs, rew, done, info = self.env.step([act.item()])
            rew = float(rew)
            self.ep_ret += rew
            self.ep_len += 1
            max_rew, min_rew = max(max_rew, rew), min(min_rew, rew)
            
            ep_done = bool(done) or self.ep_len >= 3000
            data['rew'].append(rew)
            data['done'].append(float(ep_done))
            
            if ep_done:
                won = self.ep_ret > 0
                episodes.append({'return': self.ep_ret, 'won': won, 'length': self.ep_len})
                self._reset()
            else:
                obs = np.array(obs).flatten()[:OBS_DIM].astype(np.float32)
                self.obs = obs
                self.feat = self.feat_eng.extract(obs)
        
        with torch.no_grad():
            _, _, bootstrap, _ = self.model.get_action(
                torch.from_numpy(self.obs).float().unsqueeze(0),
                torch.from_numpy(self.feat).float().unsqueeze(0),
                self.prev_act,
                self.hidden
            )
        
        return {
            'obs': np.array(data['obs'], dtype=np.float32),
            'feat': np.array(data['feat'], dtype=np.float32),
            'prev_act': np.array(data['prev_act'], dtype=np.int64),
            'act': np.array(data['act'], dtype=np.int64),
            'lp': np.array(data['lp'], dtype=np.float32),
            'rew': np.array(data['rew'], dtype=np.float32),
            'done': np.array(data['done'], dtype=np.float32),
            'bootstrap': bootstrap.item(),
            'episodes': episodes,
            'max_rew': max_rew,
            'min_rew': min_rew
        }

    def close(self):
        self.env.close()


class Learner:
    def __init__(self, num_workers=24, rollout_len=128, batch_size=32,
                 lr=5e-4, gamma=0.997, entropy_coeff=0.005, value_coeff=0.5, sil_coeff=0.5,
                 d_model=128, d_state=32, num_layers=2, env_name="11_vs_11_easy_stochastic",
                 checkpoint_dir="./checkpoints", reward_type="scoring",
                 expert_parquet=None, warmstart_path=None):
        
        self.num_workers = num_workers
        self.rollout_len, self.batch_size = rollout_len, batch_size
        self.gamma, self.entropy_coeff, self.value_coeff = gamma, entropy_coeff, value_coeff
        self.sil_coeff = sil_coeff
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        
        self.model = Net(d_model, d_state, num_layers).to(self.device)
        
        if warmstart_path and Path(warmstart_path).exists():
            print(f"Loading warmstart from {warmstart_path}...")
            ckpt = torch.load(warmstart_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model'], strict=False)
            print(f"Loaded warmstart (val_acc: {ckpt.get('val_acc', '?')}%)")
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.scaler = GradScaler('cuda')
        
        self.expert = ExpertBuffer(expert_parquet, rollout_len) if expert_parquet else None
        self.golden = GoldenMemory(capacity=batch_size * 8, max_uses=8)
        
        ray.init(ignore_reinit_error=True, num_cpus=num_workers + 4)
        self.workers = [Worker.remote(i, d_model, d_state, num_layers, rollout_len, env_name, reward_type)
                        for i in range(num_workers)]
        
        self.total_steps, self.updates, self.start = 0, 0, None
        self.returns, self.wins, self.lengths = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
        self.max_rew, self.min_rew = -999, 999
        self.pending, self.queue = {}, []
        
        samples = batch_size * rollout_len
        print(f"\n{'='*70}")
        print(f"IMPALA + SIL (Mamba) | {self.device} | {num_workers}W")
        print(f"Batch: {batch_size} x {rollout_len} = {samples:,} samples/update")
        print(f"Model: d_model={d_model}, d_state={d_state}, layers={num_layers}")
        print(f"LR: {lr} | γ: {gamma} | Ent: {entropy_coeff} | Val: {value_coeff} | SIL: {sil_coeff}")
        print(f"Rewards: {reward_type}")
        print(f"Expert Buffer: {len(self.expert)} rollouts" if self.expert else "Expert Buffer: None")
        print(f"Golden Memory: cap={batch_size*8}, max_uses=8")
        print(f"{'='*70}\n")

    def _weights(self):
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def _broadcast(self):
        ray.get([w.set_weights.remote(self._weights()) for w in self.workers])

    def _prepare_batch(self, rollouts):
        B, T = len(rollouts), self.rollout_len
        
        obs = torch.from_numpy(np.stack([r['obs'] for r in rollouts])).float().to(self.device)
        feat = torch.from_numpy(np.stack([r['feat'] for r in rollouts])).float().to(self.device)
        prev_act = torch.from_numpy(np.stack([r['prev_act'] for r in rollouts])).long().to(self.device)
        act = torch.from_numpy(np.stack([r['act'] for r in rollouts])).long().to(self.device)
        beh_lp = torch.from_numpy(np.stack([r['lp'] for r in rollouts])).float().to(self.device)
        rew = torch.from_numpy(np.stack([r['rew'] for r in rollouts])).float().to(self.device)
        done = torch.from_numpy(np.stack([r['done'] for r in rollouts])).float().to(self.device)
        bootstrap = torch.tensor([r['bootstrap'] for r in rollouts], dtype=torch.float32, device=self.device)
        
        return obs, feat, prev_act, act, beh_lp, rew, done, bootstrap, T

    def _update_vtrace(self, rollouts):
        obs, feat, prev_act, act, beh_lp, rew, done, bootstrap, T = self._prepare_batch(rollouts)
        
        with autocast('cuda'):
            logits, values_norm, _ = self.model.forward(obs, feat, prev_act)
            dist = Categorical(logits=logits)
            target_lp = dist.log_prob(act)
            entropy = dist.entropy()
            values = self.model.value.denormalize(values_norm)
        
        with torch.no_grad():
            vs, adv, rhos = vtrace(beh_lp, target_lp.float().detach(), rew,
                                   values.float().detach(), bootstrap.float(), done, self.gamma)
            self.model.value.update_stats(vs)
            vs_norm = self.model.value.normalize_target(vs)
            mean_rho, max_rho = rhos.mean().item(), rhos.max().item()
        
        with autocast('cuda'):
            policy_loss = -(target_lp * adv.detach()).mean()
            value_loss = F.mse_loss(values_norm, vs_norm.detach())
            ent_loss = -entropy.mean()
            loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * ent_loss
        
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = sum(p.grad.norm(2).item() ** 2 for p in self.model.parameters() if p.grad is not None) ** 0.5
        grad_clipped = nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            'loss': loss.item(), 'pi': policy_loss.item(), 'v': value_loss.item(),
            'ent': entropy.mean().item(), 'rho': mean_rho, 'rho_max': max_rho,
            'grad': grad_norm, 'grad_clip': grad_clipped.item() if torch.is_tensor(grad_clipped) else grad_clipped,
            'adv_mean': adv.mean().item(), 'adv_max': adv.max().item()
        }

    def _update_sil(self, rollouts):
        if len(rollouts) == 0:
            return None
        
        obs, feat, prev_act, act, _, rew, done, bootstrap, T = self._prepare_batch(rollouts)
        
        with autocast('cuda'):
            logits, values_norm, _ = self.model.forward(obs, feat, prev_act)
            dist = Categorical(logits=logits)
            target_lp = dist.log_prob(act)
            values = self.model.value.denormalize(values_norm)
        
        with torch.no_grad():
            mc_returns = torch.zeros_like(rew)
            running = bootstrap.clone()
            for t in reversed(range(T)):
                running = torch.where(done[:, t].bool(), rew[:, t], rew[:, t] + self.gamma * running)
                mc_returns[:, t] = running
            
            sil_adv = torch.clamp(mc_returns - values.detach(), min=0)
        
        mask = sil_adv > 0
        if not mask.any():
            return {'sil_loss': 0.0, 'sil_pi': 0.0, 'sil_v': 0.0, 'sil_adv': 0.0, 'sil_frac': 0.0}
        
        with autocast('cuda'):
            sil_policy_loss = -(target_lp[mask] * sil_adv[mask]).mean()
            sil_value_loss = 0.5 * (sil_adv[mask] ** 2).mean()
            sil_loss = self.sil_coeff * (sil_policy_loss + sil_value_loss)
        
        self.optimizer.zero_grad()
        self.scaler.scale(sil_loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            'sil_loss': sil_loss.item(), 'sil_pi': sil_policy_loss.item(),
            'sil_v': sil_value_loss.item(), 'sil_adv': sil_adv[mask].mean().item(),
            'sil_frac': mask.float().mean().item()
        }

    def train(self, max_time=3600, target_wr=80):
        print(f"Training for {max_time}s or {target_wr}% WR...\n")
        self.start = time.time()
        self._broadcast()
        
        for w in self.workers:
            self.pending[w.collect.remote()] = w
        
        while time.time() - self.start < max_time:
            while len(self.queue) < self.batch_size:
                done_refs, _ = ray.wait(list(self.pending.keys()), num_returns=1)
                for ref in done_refs:
                    w = self.pending.pop(ref)
                    r = ray.get(ref)
                    self.queue.append(r)
                    self.total_steps += self.rollout_len
                    self.max_rew = max(self.max_rew, r['max_rew'])
                    self.min_rew = min(self.min_rew, r['min_rew'])
                    for ep in r['episodes']:
                        self.returns.append(ep['return'])
                        self.wins.append(float(ep['won']))
                        self.lengths.append(ep['length'])
                        self.golden.add(r, ep['return'], ep['won'])
                    
                    w.set_weights.remote(self._weights())
                    self.pending[w.collect.remote()] = w
            
            fresh_batch = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            stats_vt = self._update_vtrace(fresh_batch)
            
            stats_sil = None
            stats_exp = None
            
            if self.expert and len(self.expert) >= self.batch_size // 2:
                expert_batch = self.expert.sample(self.batch_size // 2)
                stats_exp = self._update_sil(expert_batch)
            
            if len(self.golden) >= 16:
                golden_batch = self.golden.sample(min(self.batch_size // 2, len(self.golden)))
                if golden_batch:
                    stats_sil = self._update_sil(golden_batch)
            
            self.updates += 1
            
            if self.updates % 10 == 0:
                elapsed = time.time() - self.start
                sps = self.total_steps / elapsed
                wr = np.mean(self.wins) * 100 if self.wins else 0
                ret = np.mean(self.returns) if self.returns else 0
                ret_max = np.max(self.returns) if self.returns else 0
                gm = self.golden.stats()
                
                sil_str = f"SIL L:{stats_sil['sil_loss']:.2f} p:{stats_sil['sil_pi']:.2f} v:{stats_sil['sil_v']:.2f} adv:{stats_sil['sil_adv']:.2f}({stats_sil['sil_frac']:.0%})" if stats_sil else "SIL:--"
                exp_str = f"EXP L:{stats_exp['sil_loss']:.2f} p:{stats_exp['sil_pi']:.2f} v:{stats_exp['sil_v']:.2f} adv:{stats_exp['sil_adv']:.2f}({stats_exp['sil_frac']:.0%})" if stats_exp else "EXP:--"
                
                print(f"[{self.updates:4d}] {self.total_steps/1e6:.1f}M {sps/1e3:.0f}k/s {elapsed/60:.0f}m | W:{wr:4.0f}% R:{ret:+.1f}({ret_max:+.0f}) rw:[{self.min_rew:.1f},{self.max_rew:.1f}] | VT L:{stats_vt['loss']:.2f} p:{stats_vt['pi']:+.2f} v:{stats_vt['v']:.2f} H:{stats_vt['ent']:.2f} rho:{stats_vt['rho']:.1f}/{stats_vt['rho_max']:.1f} adv:{stats_vt['adv_mean']:.3f}/{stats_vt['adv_max']:.2f} | {exp_str} | {sil_str} | GM:{gm['size']}({gm['fresh']}) ret:{gm['ret_mean']:.1f}/{gm['ret_max']:.1f} | mu:{self.model.value.mu.item():.2f} sig:{self.model.value.sigma.item():.2f} | grad:{stats_vt['grad']:.1f}->{stats_vt['grad_clip']:.1f} scale:{self.scaler.get_scale():.0f}")
            
            if self.updates % 100 == 0:
                self._save()
            
            wr = np.mean(self.wins) * 100 if self.wins else 0
            if wr >= target_wr and len(self.wins) >= 50:
                print(f"\n{wr:.0f}% WR reached in {(time.time()-self.start)/60:.1f}m!")
                self._save("winner")
                return True
        
        print(f"\nTime limit. Final WR: {np.mean(self.wins)*100:.0f}%")
        self._save("final")
        return False

    def _save(self, name=None):
        name = name or f"u{self.updates}"
        path = self.checkpoint_dir / f"ckpt_{name}.pt"
        torch.save({
            'model': self.model.state_dict(),
            'opt': self.optimizer.state_dict(),
            'steps': self.total_steps,
            'updates': self.updates,
            'wr': np.mean(self.wins) if self.wins else 0,
            'config': {
                'd_model': self.d_model,
                'd_state': self.d_state,
                'num_layers': self.num_layers,
            }
        }, path)
        print(f"  Saved {path}")

    def close(self):
        for ref in self.pending:
            try:
                ray.cancel(ref)
            except:
                pass
        for w in self.workers:
            try:
                ray.get(w.close.remote(), timeout=2)
            except:
                pass
        ray.shutdown()


if __name__ == "__main__":
    learner = Learner(
        num_workers=32,
        env_name="11_vs_11_easy_stochastic",
        reward_type="scoring",
        
        rollout_len=512,
        batch_size=64,
        lr=0.00001,
        gamma=0.9997,
        entropy_coeff=0.01,
        value_coeff=0.5,
        sil_coeff=0.5,
        
        # Mamba config
        d_model=128,
        d_state=32,
        num_layers=2,
        
        checkpoint_dir="./checkpoints_mamba",
        expert_parquet=r"C:\clones\rlib_gfootball\main\expert.parquet",
        warmstart_path=r"C:\clones\rlib_gfootball\main\bc_mamba_warmstart.pt",
    )
    
    try:
        learner.train(max_time=360000, target_wr=95)
    except KeyboardInterrupt:
        print("\nStopped!")
    finally:
        learner.close()