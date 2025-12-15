import time
import math
import pickle
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.amp import autocast, GradScaler
import trueskill
import pyarrow.parquet as pq
import ray
import gfootball.env as football_env

FEATURE_DIM = 93
OBS_DIM = 115
NUM_ACTIONS = 19

TS_ENV = trueskill.TrueSkill(mu=25.0, sigma=8.333, beta=4.166, tau=0.083, draw_probability=0.05)

def ts_win_probability(player: trueskill.Rating, opponent: trueskill.Rating) -> float:
    delta_mu = player.mu - opponent.mu
    denom = math.sqrt(2 * (TS_ENV.beta ** 2) + player.sigma ** 2 + opponent.sigma ** 2)
    return 0.5 * (1 + math.erf(delta_mu / (denom * math.sqrt(2))))

@dataclass
class PoolMember:
    name: str
    policy_type: str
    rating: trueskill.Rating
    frozen: bool = True
    weights_path: Optional[str] = None
    games_played: int = 0
    wins: int = 0
    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games_played)

class PolicyPool:
    def __init__(self, checkpoint_dir: Path, max_snapshots: int = 10):
        self.checkpoint_dir = checkpoint_dir
        self.max_snapshots = max_snapshots
        self.members: Dict[str, PoolMember] = {}
        self.wr_vs: Dict[str, deque] = {}
        self._init_anchors()

    def _init_anchors(self):
        self.members["random"] = PoolMember(name="random", policy_type="random", rating=TS_ENV.create_rating(mu=5.0, sigma=1.0), frozen=True)
        self.members["lazy"] = PoolMember(name="lazy", policy_type="lazy", rating=TS_ENV.create_rating(mu=8.0, sigma=1.0), frozen=True)
        self.members["bot_easy"] = PoolMember(name="bot_easy", policy_type="bot_easy", rating=TS_ENV.create_rating(mu=20.0, sigma=6.0), frozen=False)
        self.members["bot_medium"] = PoolMember(name="bot_medium", policy_type="bot_medium", rating=TS_ENV.create_rating(mu=30.0, sigma=6.0), frozen=False)
        self.members["bot_hard"] = PoolMember(name="bot_hard", policy_type="bot_hard", rating=TS_ENV.create_rating(mu=40.0, sigma=6.0), frozen=False)
        for name in self.members:
            self.wr_vs[name] = deque(maxlen=100)

    def add_snapshot(self, name: str, weights_path: str, rating: trueskill.Rating):
        snapshots = [m for m in self.members.values() if m.policy_type == "checkpoint"]
        if len(snapshots) >= self.max_snapshots:
            weakest = min(snapshots, key=lambda m: m.rating.mu)
            del self.members[weakest.name]
            if weakest.name in self.wr_vs:
                del self.wr_vs[weakest.name]
        self.members[name] = PoolMember(name=name, policy_type="checkpoint", rating=TS_ENV.create_rating(mu=rating.mu, sigma=rating.sigma), frozen=False, weights_path=weights_path)
        self.wr_vs[name] = deque(maxlen=100)

    def sample_opponent(self, current_rating: trueskill.Rating, force_bot: str = None) -> str:
        if force_bot and force_bot in self.members:
            return force_bot
        weights = {}
        for name, member in self.members.items():
            if len(self.wr_vs[name]) >= 10:
                wr = np.mean(list(self.wr_vs[name]))
            else:
                wr = ts_win_probability(current_rating, member.rating)
            if wr < 0.1:
                weight = 0.05
            elif wr < 0.3:
                weight = 0.1 + (wr - 0.1) * 2
            elif wr < 0.7:
                weight = 0.5
            elif wr < 0.9:
                weight = 0.5 - (wr - 0.7) * 2
            else:
                weight = 0.05
            if member.policy_type in ["random", "lazy"]:
                weight = max(weight, 0.03)
            if member.policy_type.startswith("bot_"):
                weight = max(weight, 0.08)
            weights[name] = weight
        total = sum(weights.values())
        probs = {k: v / total for k, v in weights.items()}
        names = list(probs.keys())
        p = [probs[n] for n in names]
        return np.random.choice(names, p=p)

    def report_game(self, opponent_name: str, current_rating: trueskill.Rating, won: bool, drawn: bool = False) -> Tuple[trueskill.Rating, trueskill.Rating]:
        member = self.members[opponent_name]
        member.games_played += 1
        self.wr_vs[opponent_name].append(1.0 if won else 0.0)
        if drawn:
            new_current, new_opponent = TS_ENV.rate_1vs1(current_rating, member.rating, drawn=True)
        elif won:
            new_current, new_opponent = TS_ENV.rate_1vs1(current_rating, member.rating)
        else:
            new_opponent, new_current = TS_ENV.rate_1vs1(member.rating, current_rating)
            member.wins += 1
        if not member.frozen:
            member.rating = new_opponent
        return new_current, member.rating

    def get_rating_gaps(self, current_rating: trueskill.Rating) -> Dict[str, float]:
        gaps = {}
        for name in ["random", "lazy", "bot_easy", "bot_medium", "bot_hard"]:
            if name in self.members:
                gaps[name] = current_rating.mu - self.members[name].rating.mu
        return gaps

    def get_stats(self) -> Dict:
        return {
            'num_members': len(self.members),
            'num_snapshots': sum(1 for m in self.members.values() if m.policy_type == "checkpoint"),
            'members': {name: {'type': m.policy_type, 'mu': m.rating.mu, 'sigma': m.rating.sigma, 'games': m.games_played, 'wr_against': np.mean(list(self.wr_vs[name])) if self.wr_vs[name] else 0.5} for name, m in self.members.items()}
        }

    def save(self, path: Path):
        state = {
            'members': {name: {'name': m.name, 'policy_type': m.policy_type, 'rating_mu': m.rating.mu, 'rating_sigma': m.rating.sigma, 'frozen': m.frozen, 'weights_path': m.weights_path, 'games_played': m.games_played, 'wins': m.wins} for name, m in self.members.items()},
            'wr_vs': {name: list(wr) for name, wr in self.wr_vs.items()}
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: Path):
        if not path.exists():
            return False
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.members = {}
        for name, data in state['members'].items():
            self.members[name] = PoolMember(name=data['name'], policy_type=data['policy_type'], rating=TS_ENV.create_rating(data['rating_mu'], data['rating_sigma']), frozen=data['frozen'], weights_path=data.get('weights_path'), games_played=data.get('games_played', 0), wins=data.get('wins', 0))
        self.wr_vs = {name: deque(wr, maxlen=100) for name, wr in state.get('wr_vs', {}).items()}
        for name in self.members:
            if name not in self.wr_vs:
                self.wr_vs[name] = deque(maxlen=100)
        return True

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
        old_mu, old_sigma = self.mu.clone(), self.sigma.clone()
        t_mean, t_sq_mean = targets.mean(), (targets ** 2).mean()
        self.mu.mul_(1 - self.beta).add_(self.beta * t_mean)
        self.nu.mul_(1 - self.beta).add_(self.beta * t_sq_mean)
        var = torch.clamp(self.nu - self.mu ** 2, min=1e-4)
        self.sigma.copy_(torch.sqrt(var))
        self.linear.weight.data.mul_(old_sigma / self.sigma)
        self.linear.bias.data.copy_((self.linear.bias.data * old_sigma + old_mu - self.mu) / self.sigma)

class Net(nn.Module):
    def __init__(self, d_model: int = 128, lstm_hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.action_emb = nn.Embedding(NUM_ACTIONS + 1, 16)
        input_dim = OBS_DIM + FEATURE_DIM + 16
        self.encoder = nn.Sequential(nn.Linear(input_dim, d_model), nn.LayerNorm(d_model), nn.ReLU(), nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.ReLU())
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=1, batch_first=True)
        self.policy = nn.Sequential(nn.Linear(lstm_hidden, 128), nn.ReLU(), nn.Linear(128, NUM_ACTIONS))
        self.value = PopArtValueHead(lstm_hidden)
        self._init()
        print(f"Net: {sum(p.numel() for p in self.parameters()):,} params")
    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.lstm_hidden, device=device), torch.zeros(1, batch_size, self.lstm_hidden, device=device))
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
        x, hidden = self.lstm(x, hidden)
        logits = self.policy(x)
        values_norm = self.value(x)
        if squeeze:
            logits, values_norm = logits.squeeze(1), values_norm.squeeze(1)
        return logits, values_norm, hidden
    def get_action(self, obs, feat, prev_actions, hidden=None, deterministic=False):
        logits, values_norm, hidden = self.forward(obs, feat, prev_actions, hidden)
        if deterministic:
            actions = logits.argmax(dim=-1)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
        else:
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        values = self.value.denormalize(values_norm)
        return actions, log_probs, values, hidden

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
class SelfPlayWorker:
    def __init__(self, wid: int, d_model: int, lstm_hidden: int, rollout_len: int):
        self.wid = wid
        self.rollout_len = rollout_len
        self.feat_eng = FeatureEngineer()
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.model = Net(d_model, lstm_hidden)
        self.model.eval()
        self.opponent_model = None
        self.opponent_type = None
        self.env = None
        self.current_env_type = None
        self.obs = None
        self.feat = None
        self.ep_ret = 0.0
        self.ep_len = 0
        self.prev_act = None
        self.hidden = None
        self.opp_obs = None
        self.opp_feat = None
        self.opp_prev_act = None
        self.opp_hidden = None
    def set_weights(self, weights: dict):
        self.model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})
    def set_opponent(self, opponent_type: str, weights: dict = None):
        prev_type = self.opponent_type
        self.opponent_type = opponent_type
        if opponent_type == "checkpoint" and weights is not None:
            if self.opponent_model is None:
                self.opponent_model = Net(self.d_model, self.lstm_hidden)
                self.opponent_model.eval()
            self.opponent_model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})
            if prev_type != "checkpoint":
                self.opp_prev_act = torch.tensor([NUM_ACTIONS], dtype=torch.long)
                self.opp_hidden = None
    def _get_env_name_for_opponent(self, opponent_type: str) -> Tuple[str, int, int]:
        if opponent_type in ["random", "lazy", "checkpoint"]:
            return "11_vs_11_stochastic", 1, 1
        elif opponent_type == "bot_easy":
            return "11_vs_11_easy_stochastic", 1, 0
        elif opponent_type == "bot_medium":
            return "11_vs_11_stochastic", 1, 0
        elif opponent_type == "bot_hard":
            return "11_vs_11_hard_stochastic", 1, 0
        else:
            return "11_vs_11_easy_stochastic", 1, 0
    def _create_env(self, opponent_type: str) -> bool:
        new_env_name, new_left, new_right = self._get_env_name_for_opponent(opponent_type)
        env_key = (new_env_name, new_left, new_right)
        self.current_env_type = opponent_type
        if hasattr(self, '_env_key') and self._env_key == env_key:
            return False
        if self.env is not None:
            self.env.close()
        self.env = football_env.create_environment(env_name=new_env_name, representation="simple115v2", number_of_left_players_agent_controls=new_left, number_of_right_players_agent_controls=new_right, rewards="scoring", render=False)
        self._env_key = env_key
        return True
    def _reset(self):
        raw_obs = self.env.reset()
        if self.current_env_type in ["random", "lazy", "checkpoint"]:
            if isinstance(raw_obs, list) and len(raw_obs) == 2:
                left_obs = np.array(raw_obs[0]).flatten()[:OBS_DIM].astype(np.float32)
                right_obs = np.array(raw_obs[1]).flatten()[:OBS_DIM].astype(np.float32)
            else:
                left_obs = np.array(raw_obs).flatten()[:OBS_DIM].astype(np.float32)
                right_obs = left_obs.copy()
            self.obs = left_obs
            self.feat = self.feat_eng.extract(left_obs)
            self.opp_obs = right_obs
            self.opp_feat = self.feat_eng.extract(right_obs)
            self.opp_prev_act = torch.tensor([NUM_ACTIONS], dtype=torch.long)
            if self.opponent_model is not None:
                self.opp_hidden = self.opponent_model.init_hidden(1, torch.device('cpu'))
            else:
                self.opp_hidden = None
        else:
            self.obs = np.array(raw_obs).flatten()[:OBS_DIM].astype(np.float32)
            self.feat = self.feat_eng.extract(self.obs)
        self.ep_ret = 0.0
        self.ep_len = 0
        self.prev_act = torch.tensor([NUM_ACTIONS], dtype=torch.long)
        self.hidden = self.model.init_hidden(1, torch.device('cpu'))
    def _get_opponent_action(self) -> int:
        if self.opponent_type == "random":
            return np.random.randint(0, NUM_ACTIONS)
        elif self.opponent_type == "lazy":
            return 0
        elif self.opponent_type == "checkpoint":
            with torch.no_grad():
                act, _, _, self.opp_hidden = self.opponent_model.get_action(torch.from_numpy(self.opp_obs).float().unsqueeze(0), torch.from_numpy(self.opp_feat).float().unsqueeze(0), self.opp_prev_act, self.opp_hidden)
            self.opp_prev_act = act.clone()
            return act.item()
        else:
            return None
    def collect(self, opponent_type: str, opponent_weights: dict = None) -> dict:
        self.set_opponent(opponent_type, opponent_weights)
        env_changed = self._create_env(opponent_type)
        if env_changed or self.obs is None:
            self._reset()
        data = {k: [] for k in ['obs', 'feat', 'prev_act', 'act', 'lp', 'rew', 'done']}
        episodes = []
        max_rew, min_rew = -999, 999
        is_selfplay = opponent_type in ["random", "lazy", "checkpoint"]
        for _ in range(self.rollout_len):
            with torch.no_grad():
                act, lp, _, self.hidden = self.model.get_action(torch.from_numpy(self.obs).float().unsqueeze(0), torch.from_numpy(self.feat).float().unsqueeze(0), self.prev_act, self.hidden)
            data['obs'].append(self.obs.copy())
            data['feat'].append(self.feat.copy())
            data['prev_act'].append(self.prev_act.item())
            data['act'].append(act.item())
            data['lp'].append(lp.item())
            self.prev_act = act.clone()
            if is_selfplay:
                opp_act = self._get_opponent_action()
                env_action = [act.item(), opp_act]
            else:
                env_action = [act.item()]
            raw_obs, rew, done, info = self.env.step(env_action)
            if isinstance(rew, (list, np.ndarray)):
                rew = float(rew[0])
            else:
                rew = float(rew)
            if isinstance(done, (list, np.ndarray)):
                done = done[0]
            self.ep_ret += rew
            self.ep_len += 1
            max_rew, min_rew = max(max_rew, rew), min(min_rew, rew)
            ep_done = bool(done) or self.ep_len >= 3000
            data['rew'].append(rew)
            data['done'].append(float(ep_done))
            if ep_done:
                score = None
                if isinstance(info, dict) and 'score' in info:
                    score = info['score']
                elif isinstance(info, list) and len(info) > 0 and isinstance(info[0], dict) and 'score' in info[0]:
                    score = info[0]['score']
                if score is not None:
                    won = score[0] > score[1]
                    drawn = score[0] == score[1]
                else:
                    won = self.ep_ret > 0
                    drawn = abs(self.ep_ret) < 0.01
                episodes.append({'return': self.ep_ret, 'won': won, 'drawn': drawn, 'length': self.ep_len, 'opponent': opponent_type})
                self._reset()
            else:
                if is_selfplay:
                    if isinstance(raw_obs, list) and len(raw_obs) == 2:
                        self.obs = np.array(raw_obs[0]).flatten()[:OBS_DIM].astype(np.float32)
                        self.opp_obs = np.array(raw_obs[1]).flatten()[:OBS_DIM].astype(np.float32)
                    else:
                        self.obs = np.array(raw_obs).flatten()[:OBS_DIM].astype(np.float32)
                        self.opp_obs = self.obs.copy()
                    self.feat = self.feat_eng.extract(self.obs)
                    self.opp_feat = self.feat_eng.extract(self.opp_obs)
                else:
                    self.obs = np.array(raw_obs).flatten()[:OBS_DIM].astype(np.float32)
                    self.feat = self.feat_eng.extract(self.obs)
        with torch.no_grad():
            _, _, bootstrap, _ = self.model.get_action(torch.from_numpy(self.obs).float().unsqueeze(0), torch.from_numpy(self.feat).float().unsqueeze(0), self.prev_act, self.hidden)
        return {'obs': np.array(data['obs'], dtype=np.float32), 'feat': np.array(data['feat'], dtype=np.float32), 'prev_act': np.array(data['prev_act'], dtype=np.int64), 'act': np.array(data['act'], dtype=np.int64), 'lp': np.array(data['lp'], dtype=np.float32), 'rew': np.array(data['rew'], dtype=np.float32), 'done': np.array(data['done'], dtype=np.float32), 'bootstrap': bootstrap.item(), 'episodes': episodes, 'max_rew': max_rew, 'min_rew': min_rew, 'opponent_type': opponent_type}
    def close(self):
        if self.env is not None:
            self.env.close()

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
                rollout = {'obs': ep_obs[start:end].astype(np.float32), 'feat': ep_feat[start:end].astype(np.float32), 'act': ep_act[start:end], 'rew': ep_rew[start:end], 'done': np.zeros(self.rollout_len, dtype=np.float32), 'lp': np.zeros(self.rollout_len, dtype=np.float32), 'prev_act': prev_acts, 'bootstrap': 0.0}
                rollout['done'][-1] = 1.0 if i == num_rollouts - 1 else 0.0
                self.rollouts.append(rollout)
                self.returns.append(ep_score)
        print(f"Loaded {len(self.rollouts)} expert rollouts, avg return: {np.mean(self.returns):.1f}")
    def sample(self, n: int) -> List[dict]:
        if not self.rollouts:
            return []
        n = min(n, len(self.rollouts))
        weights = np.array(self.returns) - np.min(self.returns) + 0.1
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
        self.opp_ratings = []
    def add(self, rollout: dict, ret: float, won: bool, opp_rating: float = 25.0):
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
            self.opp_ratings.pop(worst_idx)
        self.buffer.append(rollout.copy())
        self.returns.append(ret)
        self.wins.append(won)
        self.uses.append(0)
        self.opp_ratings.append(opp_rating)
        return True
    def sample(self, n: int) -> List[dict]:
        if not self.buffer:
            return []
        valid = [i for i, u in enumerate(self.uses) if u < self.max_uses]
        if not valid:
            self._cleanup()
            return []
        n = min(n, len(valid))
        rets = np.array([self.returns[i] for i in valid])
        opp_rats = np.array([self.opp_ratings[i] for i in valid])
        weights = rets - rets.min() + 0.1
        weights = weights * (1.0 + opp_rats / 50.0)
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
        self.opp_ratings = [self.opp_ratings[i] for i in keep]
    def stats(self):
        if not self.buffer:
            return {'size': 0, 'wins': 0, 'ret_mean': 0, 'ret_max': 0, 'fresh': 0, 'opp_mean': 0}
        fresh = sum(1 for u in self.uses if u < self.max_uses)
        return {'size': len(self.buffer), 'wins': sum(self.wins), 'ret_mean': np.mean(self.returns), 'ret_max': np.max(self.returns), 'fresh': fresh, 'opp_mean': np.mean(self.opp_ratings)}
    def __len__(self):
        return len(self.buffer)

class SelfPlayLearner:
    def __init__(self, num_workers: int = 24, rollout_len: int = 512, batch_size: int = 64, lr: float = 5e-4, gamma: float = 0.997, entropy_coeff: float = 0.01, value_coeff: float = 0.5, sil_coeff: float = 0.5, d_model: int = 128, lstm_hidden: int = 128, checkpoint_dir: str = "./checkpoints_selfplay", warmstart_path: str = None, expert_parquet: str = None, snapshot_interval: int = 500, max_snapshots: int = 10):
        self.num_workers = num_workers
        self.rollout_len = rollout_len
        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.sil_coeff = sil_coeff
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.snapshot_interval = snapshot_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.model = Net(d_model, lstm_hidden).to(self.device)
        if warmstart_path and Path(warmstart_path).exists():
            print(f"Loading warmstart from {warmstart_path}...")
            ckpt = torch.load(warmstart_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model'], strict=False)
            print(f"Loaded warmstart")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.scaler = GradScaler('cuda')
        self.expert = ExpertBuffer(expert_parquet, rollout_len) if expert_parquet else None
        self.golden = GoldenMemory(capacity=batch_size * 8, max_uses=8)
        self.current_rating = TS_ENV.create_rating(mu=25.0, sigma=8.333)
        self.pool = PolicyPool(self.checkpoint_dir, max_snapshots=max_snapshots)
        pool_path = self.checkpoint_dir / "pool.pkl"
        if pool_path.exists():
            print("Loading existing pool...")
            self.pool.load(pool_path)
        if warmstart_path and Path(warmstart_path).exists():
            warmstart_snapshot = self.checkpoint_dir / "snapshot_warmstart.pt"
            if not warmstart_snapshot.exists():
                torch.save({'model': self.model.state_dict()}, warmstart_snapshot)
            self.pool.add_snapshot("warmstart", str(warmstart_snapshot), TS_ENV.create_rating(mu=25.0, sigma=8.0))
        ray.init(ignore_reinit_error=True, num_cpus=num_workers + 4)
        self.workers = [SelfPlayWorker.remote(i, d_model, lstm_hidden, rollout_len) for i in range(num_workers)]
        self.total_steps = 0
        self.updates = 0
        self.start = None
        self.returns = deque(maxlen=100)
        self.wins = deque(maxlen=100)
        self.lengths = deque(maxlen=100)
        self.max_rew, self.min_rew = -999, 999
        self.best_rating_vs_hard = -float('inf')
        self.pending = {}
        self.queue = []
        self.rollout_count = 0
        self._print_config(lr)

    def _print_config(self, lr):
        print(f"\n{'='*70}")
        print(f"IMPALA + Self-Play + SIL | {self.device} | {self.num_workers}W")
        print(f"Batch: {self.batch_size} x {self.rollout_len} = {self.batch_size * self.rollout_len:,} samples/update")
        print(f"LR: {lr} | γ: {self.gamma} | Ent: {self.entropy_coeff} | Val: {self.value_coeff} | SIL: {self.sil_coeff}")
        print(f"Snapshot interval: {self.snapshot_interval} updates")
        print(f"Pool: {self.pool.get_stats()['num_members']} members")
        if self.expert:
            print(f"Expert Buffer: {len(self.expert)} rollouts")
        print(f"Golden Memory: cap={self.batch_size * 8}")
        print(f"Current Rating: μ={self.current_rating.mu:.1f}, σ={self.current_rating.sigma:.2f}")
        print(f"{'='*70}\n")

    def _weights(self) -> dict:
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def _get_opponent_weights(self, opponent_name: str) -> Optional[dict]:
        member = self.pool.members.get(opponent_name)
        if member is None or member.policy_type != "checkpoint":
            return None
        if member.weights_path and Path(member.weights_path).exists():
            ckpt = torch.load(member.weights_path, map_location='cpu', weights_only=False)
            return {k: v.numpy() for k, v in ckpt['model'].items()}
        return None

    def _prepare_batch(self, rollouts):
        B = len(rollouts)
        T = rollouts[0]['obs'].shape[0]
        obs = torch.from_numpy(np.stack([r['obs'] for r in rollouts])).float().to(self.device)
        feat = torch.from_numpy(np.stack([r['feat'] for r in rollouts])).float().to(self.device)
        prev_act = torch.from_numpy(np.stack([r['prev_act'] for r in rollouts])).long().to(self.device)
        act = torch.from_numpy(np.stack([r['act'] for r in rollouts])).long().to(self.device)
        beh_lp = torch.from_numpy(np.stack([r['lp'] for r in rollouts])).float().to(self.device)
        rew = torch.from_numpy(np.stack([r['rew'] for r in rollouts])).float().to(self.device)
        done = torch.from_numpy(np.stack([r['done'] for r in rollouts])).float().to(self.device)
        bootstrap = torch.tensor([r['bootstrap'] for r in rollouts], dtype=torch.float32, device=self.device)
        return obs, feat, prev_act, act, beh_lp, rew, done, bootstrap, T

    def _update(self, rollouts) -> dict:
        obs, feat, prev_act, act, beh_lp, rew, done, bootstrap, T = self._prepare_batch(rollouts)
        with autocast('cuda'):
            logits, values_norm, _ = self.model.forward(obs, feat, prev_act)
            dist = Categorical(logits=logits)
            target_lp = dist.log_prob(act)
            entropy = dist.entropy()
            values = self.model.value.denormalize(values_norm)
        with torch.no_grad():
            vs, adv, rhos = vtrace(beh_lp, target_lp.float().detach(), rew, values.float().detach(), bootstrap.float(), done, self.gamma)
            self.model.value.update_stats(vs)
            vs_norm = self.model.value.normalize_target(vs)
            mean_rho = rhos.mean().item()
            max_rho = rhos.max().item()
            adv_mean = adv.mean().item()
            adv_max = adv.max().item()
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
        return {'loss': loss.item(), 'pi': policy_loss.item(), 'v': value_loss.item(), 'ent': entropy.mean().item(), 'rho': mean_rho, 'rho_max': max_rho, 'adv_mean': adv_mean, 'adv_max': adv_max, 'grad': grad_norm, 'grad_clip': grad_clipped.item() if torch.is_tensor(grad_clipped) else grad_clipped}

    def _update_sil(self, rollouts) -> Optional[dict]:
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
        return {'sil_loss': sil_loss.item(), 'sil_pi': sil_policy_loss.item(), 'sil_v': sil_value_loss.item(), 'sil_adv': sil_adv[mask].mean().item(), 'sil_frac': mask.float().mean().item()}

    def _maybe_save_snapshot(self):
        if self.updates % self.snapshot_interval != 0:
            return
        pool_ratings = [m.rating.mu for m in self.pool.members.values() if m.policy_type == "checkpoint"]
        if pool_ratings:
            median_rating = np.median(pool_ratings)
            if self.current_rating.mu < median_rating:
                return
        name = f"snapshot_u{self.updates}"
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save({'model': self.model.state_dict()}, path)
        rating_copy = TS_ENV.create_rating(mu=self.current_rating.mu, sigma=self.current_rating.sigma)
        self.pool.add_snapshot(name, str(path), rating_copy)
        print(f"  📸 Saved snapshot: {name} (μ={self.current_rating.mu:.1f})")

    def _log_progress(self, stats: dict, stats_exp: dict = None, stats_sil: dict = None):
        elapsed = time.time() - self.start
        sps = self.total_steps / elapsed
        wr = np.mean(list(self.wins)) * 100 if self.wins else 0
        ret = np.mean(list(self.returns)) if self.returns else 0
        ret_max = np.max(list(self.returns)) if self.returns else 0
        gaps = self.pool.get_rating_gaps(self.current_rating)
        gap_easy = gaps.get('bot_easy', 0)
        gap_med = gaps.get('bot_medium', 0)
        gap_hard = gaps.get('bot_hard', 0)
        pool_stats = self.pool.get_stats()
        gm = self.golden.stats()
        bot_wr = self._get_bot_winrates()
        exp_str = f"EXP L:{stats_exp['sil_loss']:.2f} p:{stats_exp['sil_pi']:.2f} v:{stats_exp['sil_v']:.2f} adv:{stats_exp['sil_adv']:.2f}({stats_exp['sil_frac']:.0%})" if stats_exp else "EXP:--"
        sil_str = f"SIL L:{stats_sil['sil_loss']:.2f} p:{stats_sil['sil_pi']:.2f} v:{stats_sil['sil_v']:.2f} adv:{stats_sil['sil_adv']:.2f}({stats_sil['sil_frac']:.0%})" if stats_sil else "SIL:--"
        print(f"[{self.updates:4d}] {self.total_steps/1e6:.1f}M {sps/1e3:.0f}k/s {elapsed/60:.0f}m | W:{wr:4.0f}% R:{ret:+.1f}({ret_max:+.0f}) | μ={self.current_rating.mu:.1f} σ={self.current_rating.sigma:.2f} | E:{bot_wr['easy']:2.0f}%({bot_wr['easy_n']}) M:{bot_wr['medium']:2.0f}%({bot_wr['medium_n']}) H:{bot_wr['hard']:2.0f}%({bot_wr['hard_n']}) | VT L:{stats['loss']:.2f} p:{stats['pi']:+.2f} v:{stats['v']:.2f} H:{stats['ent']:.2f} ρ:{stats['rho']:.1f}/{stats['rho_max']:.1f} | {exp_str} | {sil_str} | GM:{gm['size']}({gm['fresh']}) opp:{gm['opp_mean']:.0f} | ∇:{stats['grad']:.1f}→{stats['grad_clip']:.1f} | Pool:{pool_stats['num_snapshots']}snap")

    def _get_bot_winrates(self) -> dict:
        result = {'easy': 0, 'medium': 0, 'hard': 0, 'easy_n': 0, 'medium_n': 0, 'hard_n': 0}
        for name, wr_key in [('bot_easy', 'easy'), ('bot_medium', 'medium'), ('bot_hard', 'hard')]:
            if name in self.pool.wr_vs:
                n = len(self.pool.wr_vs[name])
                result[f'{wr_key}_n'] = n
                if n > 0:
                    result[wr_key] = np.mean(list(self.pool.wr_vs[name])) * 100
        return result

    def train(self, max_time: int = 3600):
        print(f"Training for {max_time}s...\n")
        self.start = time.time()
        current_weights = self._weights()
        ray.get([w.set_weights.remote(current_weights) for w in self.workers])
        for w in self.workers:
            opponent = self.pool.sample_opponent(self.current_rating)
            opp_weights = self._get_opponent_weights(opponent)
            self.pending[w.collect.remote(opponent, opp_weights)] = (w, opponent)
        while time.time() - self.start < max_time:
            while len(self.queue) < self.batch_size:
                done_refs, _ = ray.wait(list(self.pending.keys()), num_returns=1)
                for ref in done_refs:
                    w, opponent_name = self.pending.pop(ref)
                    rollout = ray.get(ref)
                    self.queue.append(rollout)
                    self.total_steps += self.rollout_len
                    self.max_rew = max(self.max_rew, rollout['max_rew'])
                    self.min_rew = min(self.min_rew, rollout['min_rew'])
                    for ep in rollout['episodes']:
                        self.returns.append(ep['return'])
                        self.wins.append(float(ep['won']))
                        self.lengths.append(ep['length'])
                        self.current_rating, _ = self.pool.report_game(opponent_name, self.current_rating, won=ep['won'], drawn=ep.get('drawn', False))
                    if rollout['episodes']:
                        best_ep = max(rollout['episodes'], key=lambda e: e['return'])
                        opp_mu = self.pool.members[opponent_name].rating.mu if opponent_name in self.pool.members else 25.0
                        self.golden.add(rollout, best_ep['return'], best_ep['won'], opp_mu)
                    w.set_weights.remote(self._weights())
                    self.rollout_count += 1
                    if self.rollout_count % 5 == 0:
                        forced_bots = ['bot_easy', 'bot_medium', 'bot_hard']
                        next_opponent = forced_bots[(self.rollout_count // 5) % 3]
                    else:
                        next_opponent = self.pool.sample_opponent(self.current_rating)
                    opp_weights = self._get_opponent_weights(next_opponent)
                    self.pending[w.collect.remote(next_opponent, opp_weights)] = (w, next_opponent)
            batch = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            stats = self._update(batch)
            self.updates += 1
            stats_exp = None
            if self.expert and len(self.expert) >= self.batch_size // 2:
                expert_batch = self.expert.sample(self.batch_size // 2)
                stats_exp = self._update_sil(expert_batch)
            stats_sil = None
            if self.golden:
                golden_batch = self.golden.sample(min(self.batch_size // 2, len(self.golden)))
                if golden_batch:
                    stats_sil = self._update_sil(golden_batch)
            self._maybe_save_snapshot()
            if self.updates % 10 == 0:
                self._log_progress(stats, stats_exp, stats_sil)
            if self.updates % 100 == 0:
                self._save_checkpoint()
                gaps = self.pool.get_rating_gaps(self.current_rating)
                if gaps.get('bot_hard', -999) > self.best_rating_vs_hard + 1.0:
                    self.best_rating_vs_hard = gaps['bot_hard']
                    self._save_checkpoint("best_vs_hard")
        print(f"\nTime limit reached.")
        self._save_checkpoint("final")

    def _save_checkpoint(self, name: str = None):
        name = name or f"u{self.updates}"
        path = self.checkpoint_dir / f"ckpt_{name}.pt"
        torch.save({'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'updates': self.updates, 'total_steps': self.total_steps, 'rating_mu': self.current_rating.mu, 'rating_sigma': self.current_rating.sigma}, path)
        self.pool.save(self.checkpoint_dir / "pool.pkl")
        print(f"  💾 Saved {path}")

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
    learner = SelfPlayLearner(
        num_workers=32,
        rollout_len=512,
        batch_size=64,
        lr=0.0005,
        gamma=0.997,
        entropy_coeff=0.01,
        value_coeff=0.5,
        sil_coeff=0.5,
        d_model=128,
        lstm_hidden=128,
        checkpoint_dir="./checkpoints_selfplay",
        warmstart_path=r"C:\clones\rlib_gfootball\checkpoints_selfplay\ckpt_u2800.pt",
        expert_parquet=r"C:\clones\rlib_gfootball\main\expert.parquet",
        snapshot_interval=5000,
        max_snapshots=15,
    )
    try:
        learner.train(max_time=360000)
    except KeyboardInterrupt:
        print("\nStopped!")
    finally:
        learner.close()