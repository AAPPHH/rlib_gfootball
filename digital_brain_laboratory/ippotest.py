import os, time, threading, csv
from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Dict, List
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
    num_workers: int = 150
    episode_length: int = 3000
    sample_length: int = 1000
    buffer_capacity: int = 1000
    batch_size: int = 300
    actor_lr: float = 5e-4
    critic_lr: float = 5e-4
    optimizer_eps: float = 1e-5
    actor_hidden: Tuple[int, ...] = (256, 128, 64)
    critic_hidden: Tuple[int, ...] = (256, 128, 64)
    ppo_epochs: int = 5
    gae_lambda: float = 0.95
    gamma: float = 1.0
    clip_param: float = 0.2
    value_clip: float = 0.2
    kl_early_stop: float = 0.01
    max_grad_norm: float = 0.5
    entropy_coef: float = 0.0001
    use_feature_norm: bool = True
    use_popart: bool = True
    popart_beta: float = 0.99999
    use_orthogonal_init: bool = True
    init_gain: float = 1.0
    max_iterations: int = 2000
    target_win_rate: float = 0.80
    save_interval: int = 50
    log_interval: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    experiment_name: str = "db_football_repro"
    log_dir: str = ""

class CSVLogger:
    def __init__(self, log_dir: str, experiment_name: str):
        if not log_dir:
            log_dir = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(log_dir, f"{experiment_name}_{timestamp}.csv")
        self.fields = ['timestamp', 'iteration', 'elapsed_hours', 'total_steps', 'steps_per_sec', 'wins', 'losses', 'draws', 'games', 'win_rate', 'goals_for', 'goals_against', 'goal_diff', 'buffer_size', 'policy_loss', 'value_loss', 'entropy', 'kl_div', 'best_win_rate']
        self.file = open(self.filepath, 'w', newline='')
        self.writer = csv.DictWriter(self.file, fieldnames=self.fields)
        self.writer.writeheader()
        self.file.flush()
        print(f"CSV: {self.filepath}")

    def log(self, data: dict):
        row = {f: data.get(f, '') for f in self.fields}
        row['timestamp'] = datetime.now().isoformat()
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()

class EncoderBasic:
    def __init__(self):
        self.num_left = 11
        self.num_right = 11

    def encode(self, obs: Dict, player_idx: int) -> np.ndarray:
        features = []
        ball = np.asarray(obs.get('ball', [0, 0, 0]), dtype=np.float32)
        ball_dir = np.asarray(obs.get('ball_direction', [0, 0, 0]), dtype=np.float32)
        ball_owned_team = obs.get('ball_owned_team', -1)
        ball_owned_player = obs.get('ball_owned_player', -1)
        ball_owned = np.array([float(ball_owned_team == -1), float(ball_owned_team == 0), float(ball_owned_team == 1), ball_owned_player / 11.0 if ball_owned_player >= 0 else 0.0], dtype=np.float32)
        features.extend([ball, ball_dir, ball_owned])
        left_team = np.asarray(obs.get('left_team', np.zeros((11, 2))), dtype=np.float32)
        left_dir = np.asarray(obs.get('left_team_direction', np.zeros((11, 2))), dtype=np.float32)
        right_team = np.asarray(obs.get('right_team', np.zeros((11, 2))), dtype=np.float32)
        right_dir = np.asarray(obs.get('right_team_direction', np.zeros((11, 2))), dtype=np.float32)
        ctrl_idx = min(player_idx + 1, 10)
        player_pos = left_team[ctrl_idx]
        player_dir = left_dir[ctrl_idx]
        rel_ball = ball[:2] - player_pos
        dist_ball = np.linalg.norm(rel_ball)
        angle_ball = np.arctan2(rel_ball[1], rel_ball[0]) if dist_ball > 1e-6 else 0.0
        player_state = np.concatenate([player_pos, player_dir, rel_ball, [dist_ball, angle_ball, ctrl_idx / 11.0, ball[2]]])
        features.append(player_state)
        teammate_feats = []
        for i in range(11):
            if i != ctrl_idx:
                teammate_feats.append(np.concatenate([left_team[i] - player_pos, left_dir[i]]))
        while len(teammate_feats) < 10:
            teammate_feats.append(np.zeros(4, dtype=np.float32))
        features.append(np.array(teammate_feats[:10]).flatten())
        opponent_feats = []
        for i in range(11):
            opponent_feats.append(np.concatenate([right_team[i] - player_pos, right_dir[i]]))
        features.append(np.array(opponent_feats[:11]).flatten())
        tm_dists = [(i, np.linalg.norm(left_team[i] - player_pos)) for i in range(11) if i != ctrl_idx]
        tm_dists.sort(key=lambda x: x[1])
        closest_tm = [np.concatenate([left_team[i] - player_pos, left_dir[i]]) for i, _ in tm_dists[:5]]
        while len(closest_tm) < 5:
            closest_tm.append(np.zeros(4, dtype=np.float32))
        features.append(np.array(closest_tm).flatten())
        op_dists = [(i, np.linalg.norm(right_team[i] - player_pos)) for i in range(11)]
        op_dists.sort(key=lambda x: x[1])
        closest_op = [np.concatenate([right_team[i] - player_pos, right_dir[i]]) for i, _ in op_dists[:5]]
        while len(closest_op) < 5:
            closest_op.append(np.zeros(4, dtype=np.float32))
        features.append(np.array(closest_op).flatten())
        active = obs.get('active', player_idx)
        game_mode = obs.get('game_mode', 0)
        gm_onehot = np.zeros(7, dtype=np.float32)
        if 0 <= game_mode < 7:
            gm_onehot[game_mode] = 1.0
        score = obs.get('score', [0, 0])
        steps_left = obs.get('steps_left', 3000)
        ball_zone = np.zeros(3, dtype=np.float32)
        if ball[0] < -0.33:
            ball_zone[0] = 1.0
        elif ball[0] < 0.33:
            ball_zone[1] = 1.0
        else:
            ball_zone[2] = 1.0
        game_state = np.concatenate([[active / 11.0], gm_onehot, [score[0] / 10.0, score[1] / 10.0, (score[0] - score[1]) / 10.0], [steps_left / 3000.0], ball_zone, [float(steps_left < 1500)]])
        features.append(game_state)
        features.append(np.ones(19, dtype=np.float32))
        all_feats = np.concatenate(features)
        if len(all_feats) < 217:
            all_feats = np.pad(all_feats, (0, 217 - len(all_feats)))
        return all_feats[:217].astype(np.float32)

class ActionMasking:
    def __init__(self):
        self.thresh = 0.03
        self.ball_actions = np.array([9, 10, 11, 12, 17])

    def get_mask(self, obs: Dict, player_idx: int) -> np.ndarray:
        mask = np.ones(19, dtype=np.float32)
        ball = np.asarray(obs.get('ball', [0, 0, 0]))[:2]
        ball_owned_team = obs.get('ball_owned_team', -1)
        ball_owned_player = obs.get('ball_owned_player', -1)
        left_team = np.asarray(obs.get('left_team', np.zeros((11, 2))))
        game_mode = obs.get('game_mode', 0)
        ctrl_idx = min(player_idx + 1, 10)
        dist = np.linalg.norm(ball - left_team[ctrl_idx])
        if ball_owned_team == 0:
            if ball_owned_player != ctrl_idx and dist > self.thresh:
                mask[self.ball_actions] = 0.0
        elif ball_owned_team == 1:
            mask[self.ball_actions] = 0.0
            if dist > self.thresh * 3:
                mask[16] = 0.0
        else:
            if dist > self.thresh:
                mask[self.ball_actions] = 0.0
                mask[16] = 0.0
        if ball[0] < 0.6:
            mask[12] = 0.0
        if game_mode == 1:
            mask[[9, 10, 11]] = 0.0
        elif game_mode in (2, 4, 6):
            mask[12] = 0.0
        if mask.sum() == 0:
            mask[0] = 1.0
        return mask

class RewardShaper:
    def __init__(self, num_agents: int = 10):
        self.num_agents = num_agents
        self.last_score = [0, 0]

    def reset(self):
        self.last_score = [0, 0]

    def compute(self, obs: Dict) -> Tuple[np.ndarray, bool]:
        rewards = np.zeros(self.num_agents, dtype=np.float32)
        score = obs.get('score', [0, 0])
        goal_event = False
        if score[0] > self.last_score[0]:
            rewards += 1.0
            goal_event = True
        if score[1] > self.last_score[1]:
            rewards -= 1.0
            goal_event = True
        self.last_score = list(score)
        return rewards, goal_event

class PopArt:
    def __init__(self, beta: float = 0.99999):
        self.beta = beta
        self.mean = 0.0
        self.mean_sq = 1.0
        self.std = 1.0

    def update(self, targets: np.ndarray):
        targets = np.asarray(targets).flatten()
        if len(targets) == 0:
            return
        self.mean = self.beta * self.mean + (1 - self.beta) * np.mean(targets)
        self.mean_sq = self.beta * self.mean_sq + (1 - self.beta) * np.mean(targets ** 2)
        self.std = np.sqrt(max(self.mean_sq - self.mean ** 2, 1e-8))

    def normalize(self, targets: np.ndarray) -> np.ndarray:
        return (targets - self.mean) / (self.std + 1e-8)

class RunningMeanStd:
    def __init__(self, shape: Tuple[int, ...]):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x: np.ndarray):
        if len(x) == 0:
            return
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean += delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total
        self.count = total

class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: Tuple[int, ...], gain: float):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, act_dim)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.head.weight, gain=0.01)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x, mask=None):
        logits = self.head(self.net(x))
        if mask is not None:
            logits = logits.masked_fill(mask == 0, float('-inf'))
        return logits, F.softmax(logits, dim=-1)

    def act(self, x, mask=None):
        _, probs = self.forward(x, mask)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action)

    def evaluate(self, x, action, mask=None):
        _, probs = self.forward(x, mask)
        dist = Categorical(probs)
        return dist.log_prob(action), dist.entropy()

class Critic(nn.Module):
    def __init__(self, obs_dim: int, hidden: Tuple[int, ...], gain: float):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=gain)
                nn.init.constant_(m.bias, 0)
        nn.init.orthogonal_(self.head.weight, gain=1.0)
        nn.init.constant_(self.head.bias, 0)

    def forward(self, x):
        return self.head(self.net(x))

class Policy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.actor = Actor(cfg.obs_dim, cfg.action_dim, cfg.actor_hidden, cfg.init_gain)
        self.critic = Critic(cfg.obs_dim, cfg.critic_hidden, cfg.init_gain)
        self.obs_rms = RunningMeanStd((cfg.obs_dim,))
        self.popart = PopArt(cfg.popart_beta) if cfg.use_popart else None

    def normalize_obs(self, x):
        mean = torch.tensor(self.obs_rms.mean, device=x.device, dtype=x.dtype)
        std = torch.sqrt(torch.tensor(self.obs_rms.var, device=x.device, dtype=x.dtype)) + 1e-8
        return (x - mean) / std

    def act(self, x, mask=None):
        if self.cfg.use_feature_norm:
            x = self.normalize_obs(x)
        actions, log_probs = self.actor.act(x, mask)
        values = self.critic(x).squeeze(-1)
        return actions, log_probs, values

    def evaluate(self, x, action, mask=None):
        if self.cfg.use_feature_norm:
            x = self.normalize_obs(x)
        log_prob, entropy = self.actor.evaluate(x, action, mask)
        value = self.critic(x)
        return log_prob, entropy, value

    def get_value(self, x):
        if self.cfg.use_feature_norm:
            x = self.normalize_obs(x)
        return self.critic(x)

    def get_weights(self):
        return {
            'actor': {k: v.cpu().clone() for k, v in self.actor.state_dict().items()},
            'critic': {k: v.cpu().clone() for k, v in self.critic.state_dict().items()},
            'obs_rms': (self.obs_rms.mean.copy(), self.obs_rms.var.copy(), self.obs_rms.count),
            'popart': (self.popart.mean, self.popart.mean_sq, self.popart.std) if self.popart else None
        }

    def set_weights(self, w):
        self.actor.load_state_dict(w['actor'])
        self.critic.load_state_dict(w['critic'])
        self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count = w['obs_rms']
        if self.popart and w['popart']:
            self.popart.mean, self.popart.mean_sq, self.popart.std = w['popart']

def compute_gae(rewards, values, dones, next_val, gamma=1.0, lam=0.95):
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    last_val = next_val
    for t in reversed(range(T)):
        if dones[t]:
            delta = rewards[t] - values[t]
            gae = delta
        else:
            delta = rewards[t] + gamma * last_val - values[t]
            gae = delta + gamma * lam * gae
        adv[t] = gae
        last_val = values[t]
    return adv, adv + values

@ray.remote
class PolicyServer:
    def __init__(self):
        self.weights = None
        self.version = 0
        self.lock = threading.Lock()

    def get(self):
        with self.lock:
            return self.weights, self.version

    def set(self, weights, version):
        with self.lock:
            self.weights = weights
            self.version = version

@ray.remote
class DataServer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.usage = []
        self.times = []
        self.t = 0
        self.lock = threading.Lock()
        self.written = 0

    def push(self, samples: List[Dict]):
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

    def sample(self, n: int) -> List[Dict]:
        with self.lock:
            if not self.buffer:
                return []
            n = min(n, len(self.buffer))
            max_t = max(1, self.t)
            prio = [max((3 - self.usage[i]) + (self.times[i] / max_t) * 0.5, 0.01) for i in range(len(self.buffer))]
            total = sum(prio)
            probs = [p / total for p in prio]
            indices = np.random.choice(len(self.buffer), size=n, replace=False, p=probs)
            for idx in indices:
                self.usage[idx] += 1
            result = [self.buffer[i] for i in indices]
            i = 0
            while i < len(self.buffer):
                if self.usage[i] >= 3:
                    self.buffer.pop(i)
                    self.usage.pop(i)
                    self.times.pop(i)
                else:
                    i += 1
            return result

    def size(self):
        with self.lock:
            return len(self.buffer)

    def stats(self):
        with self.lock:
            return {'size': len(self.buffer), 'written': self.written}

@ray.remote(num_cpus=1)
class RolloutWorker:
    def __init__(self, wid: int, cfg: Config, policy_server, data_server):
        self.wid = wid
        self.cfg = cfg
        self.policy_server = policy_server
        self.data_server = data_server
        self.encoder = EncoderBasic()
        self.masking = ActionMasking()
        self.shaper = RewardShaper(cfg.num_agents)
        self.env = None
        self.policy = Policy(cfg)
        self.policy.eval()
        self.pv = -1
        self.wins = self.losses = self.draws = 0
        self.gf = self.ga = 0
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
        except Exception as e:
            print(f"Worker {self.wid} env failed: {e}")
            return False

    def _try_sync(self):
        try:
            ref = self.policy_server.get.remote()
            ready, _ = ray.wait([ref], timeout=0)
            if ready:
                w, v = ray.get(ref)
                if w and v > self.pv:
                    self.policy.set_weights(w)
                    self.pv = v
        except:
            pass

    def _encode(self, raw):
        obs, masks = [], []
        for i in range(self.cfg.num_agents):
            o = raw[i] if isinstance(raw, list) else raw
            obs.append(self.encoder.encode(o, i))
            masks.append(self.masking.get_mask(o, i))
        return np.stack(obs), np.stack(masks)

    def _make_samples(self, ep_obs, ep_act, ep_mask, ep_lp, ep_val, ep_rew, ep_done, is_terminal):
        T = len(ep_obs)
        samples = []
        for start in range(0, T, self.cfg.sample_length):
            end = min(start + self.cfg.sample_length, T)
            if end - start < 10:
                continue
            samples.append({
                'obs': np.stack(ep_obs[start:end]),
                'act': np.stack(ep_act[start:end]),
                'mask': np.stack(ep_mask[start:end]),
                'lp': np.stack(ep_lp[start:end]),
                'val': np.stack(ep_val[start:end]),
                'rew': np.stack(ep_rew[start:end]),
                'done': np.stack(ep_done[start:end]),
                'terminal': is_terminal and end == T
            })
        return samples

    def run(self):
        self.running = True
        self._try_sync()
        self.shaper.reset()
        raw = self.env.reset()
        obs, masks = self._encode(raw)
        ep_obs, ep_act, ep_mask, ep_lp, ep_val, ep_rew, ep_done = [], [], [], [], [], [], []
        while self.running:
            with torch.no_grad():
                act, lp, val = self.policy.act(torch.FloatTensor(obs), torch.FloatTensor(masks))
                act, lp, val = act.numpy(), lp.numpy(), val.numpy()
            raw, _, done, info = self.env.step(act.tolist())
            next_obs, next_masks = self._encode(raw)
            ref = raw[0] if isinstance(raw, list) else raw
            rew, goal_event = self.shaper.compute(ref)
            ep_obs.append(obs)
            ep_act.append(act)
            ep_mask.append(masks)
            ep_lp.append(lp)
            ep_val.append(val)
            ep_rew.append(rew)
            ep_done.append(np.array([goal_event or done] * self.cfg.num_agents))
            if done:
                self.games += 1
                score = info.get('score', [0, 0])
                self.gf += score[0]
                self.ga += score[1]
                if score[0] > score[1]:
                    self.wins += 1
                elif score[0] < score[1]:
                    self.losses += 1
                else:
                    self.draws += 1
                if ep_obs:
                    samples = self._make_samples(ep_obs, ep_act, ep_mask, ep_lp, ep_val, ep_rew, ep_done, True)
                    if samples:
                        self.data_server.push.remote(samples)
                ep_obs, ep_act, ep_mask, ep_lp, ep_val, ep_rew, ep_done = [], [], [], [], [], [], []
                self.shaper.reset()
                raw = self.env.reset()
                obs, masks = self._encode(raw)
                self._try_sync()
            else:
                obs, masks = next_obs, next_masks
            self.steps += 1

    def stop(self):
        self.running = False

    def get_stats(self):
        total = self.wins + self.losses + self.draws
        return {'wid': self.wid, 'wr': self.wins / max(total, 1), 'games': total, 'w': self.wins, 'l': self.losses, 'd': self.draws, 'gf': self.gf, 'ga': self.ga, 'steps': self.steps, 'pv': self.pv}

class PPOTrainer:
    def __init__(self, policy: Policy, cfg: Config):
        self.policy = policy
        self.cfg = cfg
        self.actor_opt = Adam(policy.actor.parameters(), lr=cfg.actor_lr, eps=cfg.optimizer_eps)
        self.critic_opt = Adam(policy.critic.parameters(), lr=cfg.critic_lr, eps=cfg.optimizer_eps)

    def update(self, samples: List[Dict]):
        if not samples:
            return {}
        all_obs, all_act, all_mask, all_lp, all_val, all_adv, all_ret = [], [], [], [], [], [], []
        for s in samples:
            obs, act, mask, lp, val = s['obs'], s['act'], s['mask'], s['lp'], s['val']
            rew, done, terminal = s['rew'], s['done'], s.get('terminal', True)
            T, N = obs.shape[:2]
            self.policy.obs_rms.update(obs.reshape(-1, self.cfg.obs_dim))
            with torch.no_grad():
                nv = self.policy.get_value(torch.FloatTensor(obs[-1]).to(self.cfg.device))
                nv = nv.squeeze(-1).cpu().numpy()
                if terminal:
                    nv = np.zeros_like(nv)
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
        all_adv = np.concatenate(all_adv)
        all_ret = np.concatenate(all_ret)
        if self.policy.popart:
            self.policy.popart.update(all_ret)
            all_ret = self.policy.popart.normalize(all_ret)
        all_adv = (all_adv - all_adv.mean()) / (all_adv.std() + 1e-8)
        all_adv = torch.FloatTensor(all_adv)
        all_ret = torch.FloatTensor(all_ret)
        dev = self.cfg.device
        n = all_obs.shape[0]
        mb_size = min(8192, n)
        stats = {'pl': 0, 'vl': 0, 'ent': 0, 'kl': 0, 'epochs': 0}
        for _ in range(self.cfg.ppo_epochs):
            idx = np.random.permutation(n)
            epoch_kl = 0
            batches = 0
            for start in range(0, n, mb_size):
                end = min(start + mb_size, n)
                mb_idx = idx[start:end]
                mb_obs = all_obs[mb_idx].to(dev)
                mb_act = all_act[mb_idx].to(dev)
                mb_mask = all_mask[mb_idx].to(dev)
                mb_old_lp = all_lp[mb_idx].to(dev)
                mb_adv = all_adv[mb_idx].to(dev)
                mb_ret = all_ret[mb_idx].to(dev)
                mb_old_val = all_val[mb_idx].to(dev)
                new_lp, ent, new_val = self.policy.evaluate(mb_obs, mb_act, mb_mask)
                new_val = new_val.squeeze(-1)
                ratio = torch.exp(new_lp - mb_old_lp)
                s1 = ratio * mb_adv
                s2 = torch.clamp(ratio, 1 - self.cfg.clip_param, 1 + self.cfg.clip_param) * mb_adv
                pl = -torch.min(s1, s2).mean()
                val_clipped = mb_old_val + torch.clamp(new_val - mb_old_val, -self.cfg.value_clip, self.cfg.value_clip)
                vl = 0.5 * torch.max((new_val - mb_ret)**2, (val_clipped - mb_ret)**2).mean()
                loss = pl + 0.5 * vl - self.cfg.entropy_coef * ent.mean()
                self.actor_opt.zero_grad()
                self.critic_opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.actor_opt.step()
                self.critic_opt.step()
                with torch.no_grad():
                    kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                epoch_kl += kl
                batches += 1
                stats['pl'] += pl.item()
                stats['vl'] += vl.item()
                stats['ent'] += ent.mean().item()
                stats['epochs'] += 1
            if batches > 0 and epoch_kl / batches > self.cfg.kl_early_stop:
                break
        if stats['epochs'] > 0:
            stats['pl'] /= stats['epochs']
            stats['vl'] /= stats['epochs']
            stats['ent'] /= stats['epochs']
        stats['kl'] = epoch_kl / max(batches, 1)
        return stats

@ray.remote(num_gpus=1)
class AsyncTrainer:
    def __init__(self, cfg: Config, policy_server, data_server):
        self.cfg = cfg
        self.policy_server = policy_server
        self.data_server = data_server
        self.policy = Policy(cfg).to(cfg.device)
        self.trainer = PPOTrainer(self.policy, cfg)
        self.version = 0
        self.iters = 0
        self.running = False
        self.last_stats = {}

    def run(self):
        self.running = True
        self.version += 1
        ray.get(self.policy_server.set.remote(self.policy.get_weights(), self.version))
        while self.running:
            size = ray.get(self.data_server.size.remote())
            if size < self.cfg.batch_size:
                time.sleep(0.02)
                continue
            samples = ray.get(self.data_server.sample.remote(self.cfg.batch_size))
            if not samples:
                time.sleep(0.02)
                continue
            self.policy.train()
            self.last_stats = self.trainer.update(samples)
            self.policy.eval()
            torch.cuda.empty_cache()
            self.version += 1
            ray.get(self.policy_server.set.remote(self.policy.get_weights(), self.version))
            self.iters += 1
            if self.iters >= self.cfg.max_iterations:
                break

    def stop(self):
        self.running = False

    def get_stats(self):
        return {'iters': self.iters, 'version': self.version, 'last': self.last_stats}

    def save(self, path: str):
        torch.save({
            'iter': self.iters, 'version': self.version, 'policy': self.policy.state_dict(),
            'actor_opt': self.trainer.actor_opt.state_dict(), 'critic_opt': self.trainer.critic_opt.state_dict(),
            'obs_rms': (self.policy.obs_rms.mean, self.policy.obs_rms.var, self.policy.obs_rms.count),
            'popart': (self.policy.popart.mean, self.policy.popart.mean_sq, self.policy.popart.std) if self.policy.popart else None
        }, path)

class DBFootballTrainer:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.wr_history = []
        self.best_wr = 0.0
        self.start = None
        self.total_steps = 0
        self.logger = CSVLogger(self.cfg.log_dir, self.cfg.experiment_name)
        self.policy_server = PolicyServer.remote()
        self.data_server = DataServer.remote(self.cfg.buffer_capacity)
        self.trainer = AsyncTrainer.remote(self.cfg, self.policy_server, self.data_server)
        print(f"Creating {self.cfg.num_workers} workers...", end=" ", flush=True)
        self.workers = [RolloutWorker.remote(i, self.cfg, self.policy_server, self.data_server) for i in range(self.cfg.num_workers)]
        results = ray.get([w.create_env.remote() for w in self.workers])
        print(f"{sum(results)}/{len(results)} ready")

    def _log(self, it: int, ts: Dict):
        ws = ray.get([w.get_stats.remote() for w in self.workers])
        ds = ray.get(self.data_server.stats.remote())
        wins = sum(s['w'] for s in ws)
        losses = sum(s['l'] for s in ws)
        draws = sum(s['d'] for s in ws)
        games = wins + losses + draws
        wr = wins / max(games, 1)
        gf = sum(s['gf'] for s in ws)
        ga = sum(s['ga'] for s in ws)
        steps = sum(s['steps'] for s in ws)
        self.total_steps = steps
        self.wr_history.append(wr)
        if wr > self.best_wr:
            self.best_wr = wr
        elapsed = time.time() - self.start
        sps = int(steps / elapsed) if elapsed > 0 else 0
        last = ts.get('last', {})
        self.logger.log({
            'iteration': it, 'elapsed_hours': elapsed / 3600, 'total_steps': steps,
            'steps_per_sec': sps, 'wins': wins, 'losses': losses, 'draws': draws,
            'games': games, 'win_rate': wr, 'goals_for': gf, 'goals_against': ga,
            'goal_diff': gf - ga, 'buffer_size': ds['size'],
            'policy_loss': last.get('pl', 0), 'value_loss': last.get('vl', 0),
            'entropy': last.get('ent', 0), 'kl_div': last.get('kl', 0), 'best_win_rate': self.best_wr
        })
        wr_col = "\033[92m" if wr >= 0.8 else "\033[93m" if wr >= 0.5 else "\033[91m"
        gd = gf - ga
        gd_col = "\033[92m" if gd > 0 else "\033[91m" if gd < 0 else ""
        print(f"[{it:>4}] {steps/1e6:.2f}M | {sps//1000}k/s | Buf:{ds['size']} | W:{wins} L:{losses} D:{draws} | WR:{wr_col}{wr:.1%}\033[0m | GD:{gd_col}{gd:+}\033[0m | PL:{last.get('pl',0):.3f} VL:{last.get('vl',0):.3f}")

    def _save(self, name: str):
        save_dir = f"checkpoints/{self.cfg.experiment_name}"
        os.makedirs(save_dir, exist_ok=True)
        ray.get(self.trainer.save.remote(f"{save_dir}/{name}.pt"))
        print(f"Saved: {save_dir}/{name}.pt")

    def train(self):
        self.start = time.time()
        print("=" * 70)
        print(f"DB-Football | {self.cfg.env_name} | W:{self.cfg.num_workers} Buf:{self.cfg.buffer_capacity} BS:{self.cfg.batch_size}")
        print("=" * 70)
        for w in self.workers:
            w.run.remote()
        self.trainer.run.remote()
        last_log = time.time()
        last_save = 0
        try:
            while True:
                ts = ray.get(self.trainer.get_stats.remote())
                it = ts['iters']
                if it >= self.cfg.max_iterations:
                    print(f"\nMax iterations reached")
                    break
                if time.time() - last_log > self.cfg.log_interval:
                    self._log(it, ts)
                    last_log = time.time()
                    if self.best_wr >= self.cfg.target_win_rate:
                        print(f"\nTarget {self.cfg.target_win_rate:.0%} reached!")
                        break
                if it > 0 and it - last_save >= self.cfg.save_interval:
                    self._save(f"iter_{it}")
                    last_save = it
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nInterrupted")
        finally:
            ray.get(self.trainer.stop.remote())
            ray.get([w.stop.remote() for w in self.workers])
            self._save("final")
            self.logger.close()
        elapsed = time.time() - self.start
        print("=" * 70)
        print(f"Done | {elapsed/3600:.1f}h | {self.total_steps/1e6:.1f}M steps | Best WR: {self.best_wr:.1%}")
        print("=" * 70)

if __name__ == "__main__":
    ray.init()
    trainer = DBFootballTrainer(Config())
    trainer.train()