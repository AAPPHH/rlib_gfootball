import os, time, threading
from dataclasses import dataclass
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam
import ray

@dataclass
class Config:
    env_name: str = "11_vs_11_hard_stochastic"
    num_agents: int = 10
    action_dim: int = 19
    obs_dim: int = 217
    num_workers: int = 24
    buffer_capacity: int = 250
    batch_size: int = 80
    buffer_max_usage: int = 3
    rollout_length: int = 3000
    sample_length: int = 1000
    policy_update_interval: int = 10
    trainer_pull_interval: float = 0.05
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    optimizer_eps: float = 1e-5
    actor_hidden: Tuple[int, ...] = (256, 128, 64)
    critic_hidden: Tuple[int, ...] = (256, 128, 64)
    ppo_epochs: int = 5
    gae_lambda: float = 0.95
    gamma: float = 1.0
    entropy_coef: float = 0.0001
    clip_param: float = 0.2
    kl_early_stop: float = 0.01
    max_grad_norm: float = 0.5
    value_clip: float = 0.2
    use_popart: bool = True
    popart_beta: float = 0.99999
    use_feature_norm: bool = True
    use_orthogonal_init: bool = True
    init_gain: float = 1.0
    max_iterations: int = 2000
    target_win_rate: float = 0.8
    save_interval: int = 100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_name: str = "grf_i9_4080"
    reward_type: str = "simple"

class RewardShaper:
    def __init__(self, num_agents=10, goal_weight=1.0, lose_weight=1.0):
        self.num_agents = num_agents
        self.goal_weight = goal_weight
        self.lose_weight = lose_weight
        self.last_score = [0, 0]
    def reset(self):
        self.last_score = [0, 0]
    def compute(self, obs):
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        info = {'goal_scored': False, 'goal_conceded': False}
        score = obs.get('score', [0, 0])
        if score[0] > self.last_score[0]:
            info['goal_scored'] = True
            rewards += self.goal_weight
        if score[1] > self.last_score[1]:
            info['goal_conceded'] = True
            rewards -= self.lose_weight
        self.last_score = list(score)
        return rewards, info

class FeatureEncoder:
    def __init__(self, num_left=11, num_right=11):
        self.num_left = num_left
        self.num_right = num_right
        self.num_closest = 5
    @property
    def obs_dim(self):
        return 217
    def encode(self, obs, player_idx):
        features = []
        ball = np.asarray(obs.get('ball', np.zeros(3)), dtype=np.float32)
        ball_dir = np.asarray(obs.get('ball_direction', np.zeros(3)), dtype=np.float32)
        ball_owned_team = obs.get('ball_owned_team', -1)
        ball_owned_player = obs.get('ball_owned_player', -1)
        ball_ownership = np.array([float(ball_owned_team == -1), float(ball_owned_team == 0), float(ball_owned_team == 1), ball_owned_player / 11.0 if ball_owned_player >= 0 else 0.0], dtype=np.float32)
        features.extend([ball, ball_dir, ball_ownership])
        left_team = np.asarray(obs.get('left_team', np.zeros((self.num_left, 2))), dtype=np.float32)
        left_team_dir = np.asarray(obs.get('left_team_direction', np.zeros((self.num_left, 2))), dtype=np.float32)
        right_team = np.asarray(obs.get('right_team', np.zeros((self.num_right, 2))), dtype=np.float32)
        right_team_dir = np.asarray(obs.get('right_team_direction', np.zeros((self.num_right, 2))), dtype=np.float32)
        controlled_idx = min(player_idx + 1, self.num_left - 1)
        player_pos = left_team[controlled_idx]
        player_dir = left_team_dir[controlled_idx]
        rel_ball = ball[:2] - player_pos
        dist_to_ball = np.linalg.norm(rel_ball)
        angle_to_ball = np.arctan2(rel_ball[1], rel_ball[0]) if dist_to_ball > 0 else 0.0
        player_state = np.concatenate([player_pos, player_dir, rel_ball, [dist_to_ball, angle_to_ball, controlled_idx / 11.0, ball[2]]])
        features.append(player_state)
        teammate_features = []
        for i in range(self.num_left):
            if i == controlled_idx: continue
            rel_pos = left_team[i] - player_pos
            teammate_features.append(np.concatenate([rel_pos, left_team_dir[i]]))
        while len(teammate_features) < 10:
            teammate_features.append(np.zeros(4, dtype=np.float32))
        features.append(np.array(teammate_features[:10]).flatten())
        opponent_features = []
        for i in range(self.num_right):
            rel_pos = right_team[i] - player_pos
            opponent_features.append(np.concatenate([rel_pos, right_team_dir[i]]))
        while len(opponent_features) < 11:
            opponent_features.append(np.zeros(4, dtype=np.float32))
        features.append(np.array(opponent_features[:11]).flatten())
        teammate_dists = [(i, np.linalg.norm(left_team[i] - player_pos)) for i in range(self.num_left) if i != controlled_idx]
        teammate_dists.sort(key=lambda x: x[1])
        closest_teammates = [np.concatenate([left_team[idx] - player_pos, left_team_dir[idx]]) for idx, _ in teammate_dists[:self.num_closest]]
        while len(closest_teammates) < self.num_closest:
            closest_teammates.append(np.zeros(4, dtype=np.float32))
        features.append(np.array(closest_teammates).flatten())
        opponent_dists = [(i, np.linalg.norm(right_team[i] - player_pos)) for i in range(self.num_right)]
        opponent_dists.sort(key=lambda x: x[1])
        closest_opponents = [np.concatenate([right_team[idx] - player_pos, right_team_dir[idx]]) for idx, _ in opponent_dists[:self.num_closest]]
        while len(closest_opponents) < self.num_closest:
            closest_opponents.append(np.zeros(4, dtype=np.float32))
        features.append(np.array(closest_opponents).flatten())
        active = obs.get('active', player_idx)
        game_mode = obs.get('game_mode', 0)
        game_mode_onehot = np.zeros(7, dtype=np.float32)
        if 0 <= game_mode < 7: game_mode_onehot[game_mode] = 1.0
        score = obs.get('score', [0, 0])
        steps_left = obs.get('steps_left', 3000)
        ball_zone = np.zeros(3, dtype=np.float32)
        if ball[0] < -0.33: ball_zone[0] = 1.0
        elif ball[0] < 0.33: ball_zone[1] = 1.0
        else: ball_zone[2] = 1.0
        game_state = np.concatenate([[active / 11.0], game_mode_onehot, [score[0] / 10.0, score[1] / 10.0, (score[0] - score[1]) / 10.0], [steps_left / 3000.0], ball_zone, [float(steps_left < 1500)]])
        features.append(game_state)
        features.append(np.ones(19, dtype=np.float32))
        all_features = np.concatenate(features)
        if len(all_features) < self.obs_dim:
            all_features = np.pad(all_features, (0, self.obs_dim - len(all_features)))
        return all_features[:self.obs_dim].astype(np.float32)

class ActionMasking:
    def __init__(self, thresh=0.03):
        self.thresh = thresh
        self.ball_actions = np.array([9, 10, 11, 12, 17])
    def get_mask(self, obs, player_idx):
        mask = np.ones(19, dtype=np.float32)
        ball = np.asarray(obs.get('ball', np.zeros(3)))[:2]
        ball_owned_team = obs.get('ball_owned_team', -1)
        ball_owned_player = obs.get('ball_owned_player', -1)
        left_team = np.asarray(obs.get('left_team', np.zeros((11, 2))))
        game_mode = obs.get('game_mode', 0)
        controlled_idx = min(player_idx + 1, 10)
        player_pos = left_team[controlled_idx]
        dist = np.linalg.norm(ball - player_pos)
        if ball_owned_team == 0:
            if ball_owned_player != controlled_idx and dist > self.thresh:
                mask[self.ball_actions] = 0.0
        elif ball_owned_team == 1:
            mask[self.ball_actions] = 0.0
            if dist > self.thresh * 3: mask[16] = 0.0
        else:
            if dist > self.thresh:
                mask[self.ball_actions] = 0.0
                mask[16] = 0.0
        if ball[0] < 0.6: mask[12] = 0.0
        if game_mode == 1: mask[[9,10,11]] = 0.0
        elif game_mode in (2, 4, 6): mask[12] = 0.0
        if mask.sum() == 0: mask[0] = 1.0
        return mask

class PopArt:
    def __init__(self, beta=0.99999):
        self.beta = beta
        self.mean = 0.0
        self.mean_sq = 1.0
        self.std = 1.0
    def update(self, targets):
        targets = np.array(targets).flatten()
        if len(targets) == 0: return
        batch_mean = np.mean(targets)
        batch_mean_sq = np.mean(targets ** 2)
        self.mean = self.beta * self.mean + (1 - self.beta) * batch_mean
        self.mean_sq = self.beta * self.mean_sq + (1 - self.beta) * batch_mean_sq
        self.std = np.sqrt(np.maximum(self.mean_sq - self.mean ** 2, 1e-8))
    def normalize(self, targets):
        return (targets - self.mean) / (self.std + 1e-8)

class RunningMeanStd:
    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon
    def update(self, x):
        if len(x) == 0: return
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        self.var = m2 / total_count
        self.count = total_count

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=(256, 128, 64), use_orthogonal=True, gain=1.0):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, act_dim)
        if use_orthogonal:
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=gain)
                    nn.init.constant_(m.bias, 0)
            nn.init.orthogonal_(self.head.weight, gain=0.01)
            nn.init.constant_(self.head.bias, 0)
    def forward(self, x, mask=None):
        logits = self.head(self.net(x))
        if mask is not None: logits = logits.masked_fill(mask == 0, float('-inf'))
        return logits, F.softmax(logits, dim=-1)
    def act(self, x, mask=None, deterministic=False):
        logits, probs = self.forward(x, mask)
        dist = Categorical(probs)
        action = probs.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy()
    def evaluate(self, x, action, mask=None):
        _, probs = self.forward(x, mask)
        dist = Categorical(probs)
        return dist.log_prob(action), dist.entropy()

class Critic(nn.Module):
    def __init__(self, obs_dim, hidden=(256, 128, 64), use_orthogonal=True, gain=1.0):
        super().__init__()
        layers = []
        prev_dim = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev_dim, 1)
        if use_orthogonal:
            for m in self.net:
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=gain)
                    nn.init.constant_(m.bias, 0)
            nn.init.orthogonal_(self.head.weight, gain=1.0)
            nn.init.constant_(self.head.bias, 0)
    def forward(self, x):
        return self.head(self.net(x))

class Policy(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.actor = Actor(cfg.obs_dim, cfg.action_dim, cfg.actor_hidden, cfg.use_orthogonal_init, cfg.init_gain)
        self.critic = Critic(cfg.obs_dim, cfg.critic_hidden, cfg.use_orthogonal_init, cfg.init_gain)
        self.obs_rms = RunningMeanStd((cfg.obs_dim,))
        self.popart = PopArt(cfg.popart_beta) if cfg.use_popart else None
    def normalize_obs(self, x):
        mean = torch.tensor(self.obs_rms.mean, device=x.device, dtype=x.dtype)
        std = torch.sqrt(torch.tensor(self.obs_rms.var, device=x.device, dtype=x.dtype)) + 1e-8
        return (x - mean) / std
    def update_obs_rms(self, obs):
        self.obs_rms.update(obs)
    def get_value(self, x):
        if self.cfg.use_feature_norm: x = self.normalize_obs(x)
        return self.critic(x)
    def act_batch(self, x, mask=None, deterministic=False):
        if self.cfg.use_feature_norm: x = self.normalize_obs(x)
        actions, log_probs, _ = self.actor.act(x, mask, deterministic)
        values = self.critic(x).squeeze(-1)
        return actions, log_probs, values
    def evaluate(self, x, action, mask=None):
        if self.cfg.use_feature_norm: x = self.normalize_obs(x)
        log_prob, entropy = self.actor.evaluate(x, action, mask)
        value = self.critic(x)
        return log_prob, entropy, value
    def get_weights(self):
        return {
            'actor': {k: v.cpu().clone() for k, v in self.actor.state_dict().items()},
            'critic': {k: v.cpu().clone() for k, v in self.critic.state_dict().items()},
            'obs_rms_mean': self.obs_rms.mean.copy(), 'obs_rms_var': self.obs_rms.var.copy(),
            'obs_rms_count': self.obs_rms.count,
            'popart_mean': self.popart.mean if self.popart else 0.0,
            'popart_mean_sq': self.popart.mean_sq if self.popart else 1.0,
            'popart_std': self.popart.std if self.popart else 1.0
        }
    def set_weights(self, w):
        self.actor.load_state_dict(w['actor'])
        self.critic.load_state_dict(w['critic'])
        self.obs_rms.mean = w['obs_rms_mean']
        self.obs_rms.var = w['obs_rms_var']
        self.obs_rms.count = w['obs_rms_count']
        if self.popart:
            self.popart.mean = w['popart_mean']
            self.popart.mean_sq = w['popart_mean_sq']
            self.popart.std = w['popart_std']

def compute_gae(rewards, values, dones, next_value, gamma=1.0, gae_lambda=0.95):
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    last_value = next_value
    for t in reversed(range(T)):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * last_value - values[t]
            gae = delta + gamma * gae_lambda * gae
        advantages[t] = gae
        last_value = values[t]
    return advantages, advantages + values

@ray.remote
class PolicyServer:
    def __init__(self):
        self.weights = None
        self.version = 0
        self.lock = threading.Lock()
    def get_weights(self):
        with self.lock: return self.weights, self.version
    def set_weights(self, weights, version):
        with self.lock: self.weights, self.version = weights, version

@ray.remote
class DataServer:
    def __init__(self, capacity=250, max_usage=3):
        self.capacity = capacity
        self.max_usage = max_usage
        self.buffer = []
        self.usage = []
        self.times = []
        self.t = 0
        self.lock = threading.Lock()
        self.written = 0
    def push(self, samples):
        with self.lock:
            for s in samples:
                if len(self.buffer) >= self.capacity:
                    max_u = max(self.usage)
                    cands = [i for i, u in enumerate(self.usage) if u == max_u]
                    idx = min(cands, key=lambda i: self.times[i])
                    self.buffer.pop(idx)
                    self.usage.pop(idx)
                    self.times.pop(idx)
                self.buffer.append(s)
                self.usage.append(0)
                self.times.append(self.t)
                self.t += 1
                self.written += 1
    def sample(self, n):
        with self.lock:
            if not self.buffer: return []
            n = min(n, len(self.buffer))
            max_t = max(1, self.t)
            prio = [max((self.max_usage - self.usage[i]) + (self.times[i] / max_t) * 0.5, 0.01) for i in range(len(self.buffer))]
            total = sum(prio)
            probs = [p / total for p in prio]
            indices = np.random.choice(len(self.buffer), size=n, replace=False, p=probs)
            for idx in indices: self.usage[idx] += 1
            result = [self.buffer[i] for i in indices]
            i = 0
            while i < len(self.buffer):
                if self.usage[i] >= self.max_usage:
                    self.buffer.pop(i)
                    self.usage.pop(i)
                    self.times.pop(i)
                else: i += 1
            return result
    def size(self):
        with self.lock: return len(self.buffer)
    def stats(self):
        with self.lock: return {'size': len(self.buffer), 'written': self.written}

@ray.remote(num_cpus=1)
class RolloutWorker:
    def __init__(self, wid, cfg, policy_server, data_server):
        self.wid = wid
        self.cfg = cfg
        self.policy_server = policy_server
        self.data_server = data_server
        self.encoder = FeatureEncoder()
        self.masking = ActionMasking()
        self.shaper = RewardShaper(cfg.num_agents)
        self.env = None
        self.policy = Policy(cfg)
        self.policy.eval()
        self.pv = -1
        self.wins = self.losses = self.draws = 0
        self.goals_for = self.goals_against = 0
        self.games = self.steps = 0
        self.running = False
    def create_env(self):
        try:
            import gfootball.env as grf
            self.env = grf.create_environment(
                env_name=self.cfg.env_name, representation='raw', rewards='scoring',
                number_of_left_players_agent_controls=self.cfg.num_agents,
                number_of_right_players_agent_controls=0, render=False,
                write_video=False, write_full_episode_dumps=False, write_goal_dumps=False, logdir='')
            return True
        except: return False
    def _mock(self):
        return {'ball': np.random.uniform(-1, 1, 3).astype(np.float32),
                'ball_direction': np.zeros(3, dtype=np.float32),
                'ball_owned_team': np.random.choice([-1, 0, 1]),
                'ball_owned_player': np.random.randint(0, 11),
                'left_team': np.random.uniform(-1, 1, (11, 2)).astype(np.float32),
                'left_team_direction': np.zeros((11, 2), dtype=np.float32),
                'right_team': np.random.uniform(-1, 1, (11, 2)).astype(np.float32),
                'right_team_direction': np.zeros((11, 2), dtype=np.float32),
                'active': 0, 'game_mode': 0, 'score': [0, 0], 'steps_left': 3000}
    def _update_policy(self):
        w, v = ray.get(self.policy_server.get_weights.remote())
        if w is not None and v > self.pv:
            self.policy.set_weights(w)
            self.pv = v
    def _reset(self):
        self.shaper.reset()
        raw = self.env.reset() if self.env else [self._mock() for _ in range(self.cfg.num_agents)]
        obs, masks = [], []
        for i in range(self.cfg.num_agents):
            o = raw[i] if isinstance(raw, list) else raw
            obs.append(self.encoder.encode(o, i))
            masks.append(self.masking.get_mask(o, i))
        return np.stack(obs), np.stack(masks), raw
    def _step(self, actions):
        if self.env:
            raw, _, done, info = self.env.step(actions.tolist())
        else:
            raw = [self._mock() for _ in range(self.cfg.num_agents)]
            done = np.random.random() < 0.001
            info = {'score': [0, 0]}
        obs, masks = [], []
        for i in range(self.cfg.num_agents):
            o = raw[i] if isinstance(raw, list) else raw
            obs.append(self.encoder.encode(o, i))
            masks.append(self.masking.get_mask(o, i))
        ref = raw[0] if isinstance(raw, list) else raw
        rewards, rinfo = self.shaper.compute(ref)
        return np.stack(obs), np.stack(masks), rewards, done, info, raw, rinfo['goal_scored'] or rinfo['goal_conceded']
    def run(self, max_steps=None):
        self.running = True
        self._update_policy()
        obs, masks, raw = self._reset()
        seg = {'obs': [], 'actions': [], 'masks': [], 'log_probs': [], 'values': [], 'rewards': [], 'dones': []}
        uc = 0
        while self.running and (max_steps is None or self.steps < max_steps):
            with torch.no_grad():
                obs_t = torch.FloatTensor(obs)
                masks_t = torch.FloatTensor(masks)
                actions_t, lp_t, val_t = self.policy.act_batch(obs_t, masks_t)
                actions, lp, val = actions_t.numpy(), lp_t.numpy(), val_t.numpy()
            next_obs, next_masks, rewards, done, info, raw, goal_event = self._step(actions)
            seg['obs'].append(obs.copy())
            seg['actions'].append(actions.copy())
            seg['masks'].append(masks.copy())
            seg['log_probs'].append(lp.copy())
            seg['values'].append(val.copy())
            seg['rewards'].append(rewards.copy())
            should_seg = goal_event or done
            seg['dones'].append(np.array([should_seg] * self.cfg.num_agents))
            if should_seg and seg['obs']:
                sample = {k: np.stack(v) for k, v in seg.items()}
                sample['is_terminal'] = done
                ray.get(self.data_server.push.remote([sample]))
                seg = {'obs': [], 'actions': [], 'masks': [], 'log_probs': [], 'values': [], 'rewards': [], 'dones': []}
            if done:
                self.games += 1
                score = info.get('score', [0, 0])
                self.goals_for += score[0]
                self.goals_against += score[1]
                if score[0] > score[1]: self.wins += 1
                elif score[0] < score[1]: self.losses += 1
                else: self.draws += 1
                obs, masks, raw = self._reset()
            else:
                obs, masks = next_obs, next_masks
            self.steps += 1
            uc += 1
            if uc >= self.cfg.policy_update_interval:
                self._update_policy()
                uc = 0
        if seg['obs']:
            sample = {k: np.stack(v) for k, v in seg.items()}
            sample['is_terminal'] = False
            ray.get(self.data_server.push.remote([sample]))
        return self.steps
    def stop(self): self.running = False
    def get_stats(self):
        total = self.wins + self.losses + self.draws
        return {'wid': self.wid, 'wr': self.wins / max(total, 1), 'games': total,
                'w': self.wins, 'l': self.losses, 'd': self.draws,
                'gf': self.goals_for, 'ga': self.goals_against, 'steps': self.steps, 'pv': self.pv}

class PPOTrainer:
    def __init__(self, policy, cfg):
        self.policy = policy
        self.cfg = cfg
        self.actor_opt = Adam(policy.actor.parameters(), lr=cfg.actor_lr, eps=cfg.optimizer_eps)
        self.critic_opt = Adam(policy.critic.parameters(), lr=cfg.critic_lr, eps=cfg.optimizer_eps)
    def train_step(self, obs, actions, masks, old_lp, adv, ret, old_val):
        dev = self.cfg.device
        obs, actions, masks = obs.to(dev), actions.to(dev), masks.to(dev)
        old_lp, adv, ret, old_val = old_lp.to(dev), adv.to(dev), ret.to(dev), old_val.to(dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        stats = {'pl': 0, 'vl': 0, 'ent': 0, 'kl': 0, 'ep': 0}
        for _ in range(self.cfg.ppo_epochs):
            lp, ent, val = self.policy.evaluate(obs, actions, masks)
            val = val.squeeze(-1)
            ratio = torch.exp(lp - old_lp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1 - self.cfg.clip_param, 1 + self.cfg.clip_param) * adv
            pl = -torch.min(s1, s2).mean()
            if self.cfg.value_clip > 0:
                vc = old_val + torch.clamp(val - old_val, -self.cfg.value_clip, self.cfg.value_clip)
                vl = 0.5 * torch.max((val - ret)**2, (vc - ret)**2).mean()
            else:
                vl = 0.5 * ((val - ret)**2).mean()
            el = -ent.mean()
            loss = pl + 0.5 * vl + self.cfg.entropy_coef * el
            with torch.no_grad(): kl = ((ratio - 1) - torch.log(ratio)).mean().item()
            if kl > self.cfg.kl_early_stop: break
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
            self.actor_opt.step()
            self.critic_opt.step()
            stats['pl'] += pl.item()
            stats['vl'] += vl.item()
            stats['ent'] += -el.item()
            stats['kl'] = kl
            stats['ep'] += 1
        if stats['ep'] > 0:
            stats['pl'] /= stats['ep']
            stats['vl'] /= stats['ep']
            stats['ent'] /= stats['ep']
        return stats
    def update(self, samples):
        if not samples: return {}
        all_obs, all_act, all_mask, all_lp, all_val, all_adv, all_ret = [], [], [], [], [], [], []
        for s in samples:
            obs, act, mask, lp, val, rew, done = s['obs'], s['actions'], s['masks'], s['log_probs'], s['values'], s['rewards'], s['dones']
            is_term = s.get('is_terminal', True)
            T, N = obs.shape[:2]
            self.policy.update_obs_rms(obs.reshape(-1, self.cfg.obs_dim))
            with torch.no_grad():
                last_obs = torch.FloatTensor(obs[-1]).to(self.cfg.device)
                nv = self.policy.get_value(last_obs).squeeze(-1).cpu().numpy()
                if is_term: nv = np.zeros_like(nv)
            for i in range(N):
                adv, ret = compute_gae(rew[:, i], val[:, i], done[:, i], nv[i], self.cfg.gamma, self.cfg.gae_lambda)
                all_obs.append(obs[:, i])
                all_act.append(act[:, i])
                all_mask.append(mask[:, i])
                all_lp.append(lp[:, i])
                all_val.append(val[:, i])
                all_adv.append(adv)
                all_ret.append(ret)
        all_obs = torch.FloatTensor(np.concatenate(all_obs))
        all_act = torch.LongTensor(np.concatenate(all_act))
        all_mask = torch.FloatTensor(np.concatenate(all_mask))
        all_lp = torch.FloatTensor(np.concatenate(all_lp))
        all_val = torch.FloatTensor(np.concatenate(all_val))
        all_adv = torch.FloatTensor(np.concatenate(all_adv))
        all_ret = torch.FloatTensor(np.concatenate(all_ret))
        if self.policy.popart:
            self.policy.popart.update(all_ret.numpy())
            all_ret = torch.FloatTensor(self.policy.popart.normalize(all_ret.numpy()))
        return self.train_step(all_obs, all_act, all_mask, all_lp, all_adv, all_ret, all_val)

@ray.remote(num_gpus=1)
class AsyncTrainer:
    def __init__(self, cfg, policy_server, data_server):
        self.cfg = cfg
        self.policy_server = policy_server
        self.data_server = data_server
        self.policy = Policy(cfg).to(cfg.device)
        self.trainer = PPOTrainer(self.policy, cfg)
        self.version = 0
        self.iters = 0
        self.running = False
        self.last_stats = {}
    def _push(self):
        self.version += 1
        ray.get(self.policy_server.set_weights.remote(self.policy.get_weights(), self.version))
    def run(self, max_iter=None):
        self.running = True
        self._push()
        while self.running and (max_iter is None or self.iters < max_iter):
            bs = ray.get(self.data_server.size.remote())
            if bs < self.cfg.batch_size:
                time.sleep(self.cfg.trainer_pull_interval)
                continue
            samples = ray.get(self.data_server.sample.remote(self.cfg.batch_size))
            if not samples:
                time.sleep(self.cfg.trainer_pull_interval)
                continue
            self.policy.train()
            self.last_stats = self.trainer.update(samples)
            self.policy.eval()
            self._push()
            self.iters += 1
        return self.iters
    def stop(self): self.running = False
    def get_stats(self): return {'iters': self.iters, 'version': self.version, 'last': self.last_stats}
    def save(self, path):
        ckpt = {'iter': self.iters, 'version': self.version, 'policy': self.policy.state_dict(),
                'actor_opt': self.trainer.actor_opt.state_dict(), 'critic_opt': self.trainer.critic_opt.state_dict(),
                'obs_rms_mean': self.policy.obs_rms.mean, 'obs_rms_var': self.policy.obs_rms.var, 'obs_rms_count': self.policy.obs_rms.count}
        if self.policy.popart:
            ckpt['popart'] = {'mean': self.policy.popart.mean, 'mean_sq': self.policy.popart.mean_sq, 'std': self.policy.popart.std}
        torch.save(ckpt, path)

class GRFTrainer:
    def __init__(self, cfg=None):
        self.cfg = cfg or Config()
        self.wr_history = []
        self.best_wr = 0.0
        self.start = None
        self.total_steps = 0
        ray.init(ignore_reinit_error=True, logging_level=40)
        self.policy_server = PolicyServer.remote()
        self.data_server = DataServer.remote(self.cfg.buffer_capacity, self.cfg.buffer_max_usage)
        self.trainer = AsyncTrainer.remote(self.cfg, self.policy_server, self.data_server)
        print(f"Creating {self.cfg.num_workers} envs...", end=" ", flush=True)
        self.workers = [RolloutWorker.remote(i, self.cfg, self.policy_server, self.data_server) for i in range(self.cfg.num_workers)]
        results = ray.get([w.create_env.remote() for w in self.workers])
        print(f"{sum(results)}/{len(results)} ready")
    def _fmt_wr(self, wr):
        if wr >= 0.8: return f"\033[92m{wr:.0%}\033[0m"
        if wr >= 0.5: return f"\033[93m{wr:.0%}\033[0m"
        return f"\033[91m{wr:.0%}\033[0m"
    def _spark(self, vals, w=12):
        if len(vals) < 2: return ""
        chars = "▁▂▃▄▅▆▇█"
        v = vals[-w:]
        mn, mx = min(v), max(v)
        if mx == mn: return chars[4] * len(v)
        return "".join(chars[int((x-mn)/(mx-mn)*7)] for x in v)
    def _log(self, it, ts):
        ws = ray.get([w.get_stats.remote() for w in self.workers])
        ds = ray.get(self.data_server.stats.remote())
        wins = sum(s['w'] for s in ws)
        losses = sum(s['l'] for s in ws)
        draws = sum(s['d'] for s in ws)
        games = wins + losses + draws
        wr = wins / max(games, 1)
        gf = sum(s['gf'] for s in ws)
        ga = sum(s['ga'] for s in ws)
        gd = gf - ga
        steps = sum(s['steps'] for s in ws)
        self.total_steps = steps
        self.wr_history.append(wr)
        is_best = wr > self.best_wr
        if is_best: self.best_wr = wr
        elapsed = time.time() - self.start
        sps = int(steps / elapsed) if elapsed > 0 else 0
        pl = ts.get('last', {}).get('pl', 0)
        vl = ts.get('last', {}).get('vl', 0)
        ent = ts.get('last', {}).get('ent', 0)
        star = "⭐" if is_best else ""
        trend = self._spark(self.wr_history)
        wr_fmt = self._fmt_wr(wr)
        gd_col = "\033[92m" if gd > 0 else "\033[91m" if gd < 0 else ""
        gd_rst = "\033[0m" if gd != 0 else ""
        print(f"[{it:>4}] {steps/1e6:.1f}M | {sps//1000}k sps | LR:{self.cfg.actor_lr:.1e} | Buf:{ds['size']:>3}")
        print(f"  W:{wins} L:{losses} D:{draws} | WR:{wr_fmt}{star} | GD:{gd_col}{gd:+}{gd_rst} | PL:{pl:.3f} VL:{vl:.3f} Ent:{ent:.4f}")
        if trend: print(f"  Trend: {trend}")
    def _save(self, name):
        save_dir = f"checkpoints/{self.cfg.experiment_name}"
        os.makedirs(save_dir, exist_ok=True)
        ray.get(self.trainer.save.remote(f"{save_dir}/{name}.pt"))
        print(f"Saved: {save_dir}/{name}.pt")
    def train(self):
        self.start = time.time()
        print(f"{'='*60}")
        print(f"GRF IPPO | {self.cfg.env_name} | W:{self.cfg.num_workers} B:{self.cfg.buffer_capacity} BS:{self.cfg.batch_size}")
        print(f"{'='*60}")
        for w in self.workers: w.run.remote(self.cfg.rollout_length * 100)
        self.trainer.run.remote(self.cfg.max_iterations)
        last_log = time.time()
        last_save = 0
        try:
            while True:
                ts = ray.get(self.trainer.get_stats.remote())
                it = ts['iters']
                if it >= self.cfg.max_iterations:
                    print(f"\n✓ Max iterations reached")
                    break
                if time.time() - last_log > 10:
                    self._log(it, ts)
                    last_log = time.time()
                    ws = ray.get([w.get_stats.remote() for w in self.workers[:5]])
                    wr = sum(s['w'] for s in ws) / max(sum(s['games'] for s in ws), 1)
                    if wr > self.best_wr:
                        self.best_wr = wr
                        self._save("best")
                    if wr >= self.cfg.target_win_rate:
                        print(f"\n🎉 Target {self.cfg.target_win_rate:.0%} reached!")
                        break
                if it > 0 and it - last_save >= self.cfg.save_interval:
                    self._save(f"ckpt_{it}")
                    last_save = it
                time.sleep(2)
        except KeyboardInterrupt:
            print(f"\n⚠ Interrupted")
        finally:
            ray.get(self.trainer.stop.remote())
            ray.get([w.stop.remote() for w in self.workers])
            self._save("final")
        elapsed = time.time() - self.start
        print(f"{'='*60}")
        print(f"Done | {elapsed/3600:.1f}h | {self.total_steps/1e6:.1f}M steps | Best: {self._fmt_wr(self.best_wr)}")
        print(f"{'='*60}")
        ray.shutdown()

if __name__ == "__main__":
    cfg = Config()
    trainer = GRFTrainer(cfg)
    trainer.train()