import time
import math
import pickle
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
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

FEATURE_DIM, OBS_DIM, NUM_ACTIONS = 93, 115, 19
trueskill.setup(backend='mpmath')
TS_ENV = trueskill.TrueSkill(mu=25.0, sigma=8.333, beta=4.166, tau=0.083, draw_probability=0.05)

@dataclass
class LeagueMember:
    name: str
    member_type: str
    rating: trueskill.Rating
    weights_path: Optional[str] = None
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    env_name: Optional[str] = None
    controls_right: bool = False
    recent_wins: deque = field(default_factory=lambda: deque(maxlen=100))
    recent_draws: deque = field(default_factory=lambda: deque(maxlen=100))
    rating_history: deque = field(default_factory=lambda: deque(maxlen=500))
    rating_update_factor: float = 1.0
    created_idx: int = 0
    @property
    def win_rate(self) -> float:
        return self.wins / max(1, self.games_played)
    @property
    def recent_win_rate(self) -> float:
        return np.mean(list(self.recent_wins)) if self.recent_wins else 0.0
    @property
    def recent_draw_rate(self) -> float:
        return np.mean(list(self.recent_draws)) if self.recent_draws else 0.0
    @property
    def recent_loss_rate(self) -> float:
        return 1.0 - self.recent_win_rate - self.recent_draw_rate
    @property
    def conservative_skill(self) -> float:
        return self.rating.mu - 3 * self.rating.sigma
    def record_game(self, won: bool, drawn: bool = False):
        self.games_played += 1
        if drawn:
            self.draws += 1
            self.recent_wins.append(0.0)
            self.recent_draws.append(1.0)
        elif won:
            self.wins += 1
            self.recent_wins.append(1.0)
            self.recent_draws.append(0.0)
        else:
            self.losses += 1
            self.recent_wins.append(0.0)
            self.recent_draws.append(0.0)
        self.rating_history.append(self.rating.mu)

class PureLeague:
    def __init__(self, checkpoint_dir: Path, d_model: int = 512, lstm_hidden: int = 512, max_snapshots: int = 15, snapshot_on_rating_gain: float = 2.0, snapshot_on_champion_wins: int = 5, bot_floor: float = 0.25, latest_snapshot_prob: float = 0.5, min_skill_spacing: float = 3.0, min_policy_distance: float = 0.1):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.d_model, self.lstm_hidden = d_model, lstm_hidden
        self.max_snapshots = max_snapshots
        self.snapshot_on_rating_gain = snapshot_on_rating_gain
        self.snapshot_on_champion_wins = snapshot_on_champion_wins
        self.bot_floor = bot_floor
        self.latest_snapshot_prob = latest_snapshot_prob
        self.snapshot_counter = 0
        self.min_skill_spacing = min_skill_spacing
        self.min_policy_distance = min_policy_distance
        self.members: Dict[str, LeagueMember] = {}
        self.champion: Optional[str] = None
        self.last_snapshot_rating = 25.0
        self.wins_vs_champion = 0
        self.total_games = 0
        self.policy_fingerprints: Dict[str, np.ndarray] = {}
        self._init_bots()

    def _init_bots(self):
        bot_configs = [
            ("random", 5.0, 2.0, "11_vs_11_stochastic", False),
            ("lazy", 8.0, 2.0, "11_vs_11_stochastic", False),
            ("bot_easy", 18.0, 2.0, "11_vs_11_easy_stochastic", True),
            ("bot_medium", 28.0, 2.0, "11_vs_11_stochastic", True),
            ("bot_hard", 38.0, 2.0, "11_vs_11_hard_stochastic", True),
        ]
        for name, mu, sigma, env_name, controls_right in bot_configs:
            self.members[name] = LeagueMember(
                name=name, member_type="bot", rating=TS_ENV.create_rating(mu=mu, sigma=sigma),
                env_name=env_name, controls_right=controls_right, rating_update_factor=0.0
            )
        self._update_champion()

    def _update_champion(self):
        best_name, best_skill = None, -np.inf
        for name, member in self.members.items():
            if member.conservative_skill > best_skill:
                best_skill, best_name = member.conservative_skill, name
        if best_name != self.champion:
            old = self.champion
            self.champion, self.wins_vs_champion = best_name, 0
            if old:
                print(f"  👑 Champion: {best_name} (μ={self.members[best_name].rating.mu:.1f}) < {old}")

    def win_probability(self, player: trueskill.Rating, opponent: trueskill.Rating) -> float:
        delta = player.mu - opponent.mu
        denom = math.sqrt(2 * (TS_ENV.beta ** 2) + player.sigma ** 2 + opponent.sigma ** 2)
        return 0.5 * (1 + math.erf(delta / (denom * math.sqrt(2))))

    def select_opponent(self, current_rating: trueskill.Rating, force_bot: str = None) -> Tuple[str, str]:
        if force_bot and force_bot in self.members:
            return force_bot, "eval"
        snapshots = [(n, m) for n, m in self.members.items() if m.member_type == "snapshot"]
        r = np.random.random()
        if r < self.bot_floor or not snapshots:
            bots = [n for n, m in self.members.items() if m.member_type == "bot"]
            weights = []
            for name in bots:
                member = self.members[name]
                if len(member.recent_wins) >= 5:
                    agent_score = 1 - member.recent_win_rate - member.recent_draw_rate * 0.5
                    weight = agent_score * (1 - agent_score) + 0.1
                else:
                    weight = 0.5
                weights.append(weight)
            return np.random.choice(bots, p=np.array(weights) / sum(weights)), "bot"
        if r < self.bot_floor + self.latest_snapshot_prob:
            newest = max(snapshots, key=lambda x: x[1].created_idx)
            return newest[0], "latest"
        candidates, weights = list(self.members.keys()), []
        for name in candidates:
            member = self.members[name]
            win_prob = self.win_probability(current_rating, member.rating)
            if len(member.recent_wins) >= 5:
                agent_score = 1 - member.recent_win_rate - member.recent_draw_rate * 0.5
                p = win_prob * 0.5 + agent_score * 0.5
            else:
                p = win_prob
            weight = p * (1 - p) + 0.05
            if member.games_played < 50:
                weight *= 1.5
            weights.append(weight)
        return np.random.choice(candidates, p=np.array(weights) / sum(weights)), "pfsp"

    def report_game(self, opponent_name: str, current_rating: trueskill.Rating, won: bool, drawn: bool = False) -> trueskill.Rating:
        if opponent_name not in self.members:
            return current_rating
        member = self.members[opponent_name]
        self.total_games += 1
        if drawn:
            new_current, new_opponent = TS_ENV.rate_1vs1(current_rating, member.rating, drawn=True)
        elif won:
            new_current, new_opponent = TS_ENV.rate_1vs1(current_rating, member.rating)
        else:
            new_opponent, new_current = TS_ENV.rate_1vs1(member.rating, current_rating)
        if member.member_type == "snapshot" and member.rating_update_factor > 0:
            if member.rating_update_factor < 1.0:
                new_mu = member.rating.mu + (new_opponent.mu - member.rating.mu) * member.rating_update_factor
                new_sigma = member.rating.sigma + (new_opponent.sigma - member.rating.sigma) * member.rating_update_factor
                member.rating = TS_ENV.create_rating(mu=new_mu, sigma=new_sigma)
            else:
                member.rating = new_opponent
        member.record_game(won=not won, drawn=drawn)
        if opponent_name == self.champion and won:
            self.wins_vs_champion += 1
        self._update_champion()
        return new_current

    def compute_policy_fingerprint(self, weights: dict, num_samples: int = 100) -> np.ndarray:
        rng = np.random.RandomState(42)
        test_obs = rng.randn(num_samples, OBS_DIM).astype(np.float32) * 0.5
        test_feat = rng.randn(num_samples, FEATURE_DIM).astype(np.float32) * 0.5
        model = Net(self.d_model, self.lstm_hidden)
        model.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
        model.eval()
        with torch.no_grad():
            logits, _, _ = model.forward(torch.from_numpy(test_obs), torch.from_numpy(test_feat), torch.full((num_samples,), NUM_ACTIONS, dtype=torch.long))
            return F.softmax(logits, dim=-1).numpy().mean(axis=0)

    def policy_distance(self, fp1: np.ndarray, fp2: np.ndarray) -> float:
        fp1, fp2 = np.clip(fp1, 1e-8, 1.0), np.clip(fp2, 1e-8, 1.0)
        m = 0.5 * (fp1 + fp2)
        return float(np.sqrt(0.5 * (np.sum(fp1 * np.log(fp1 / m)) + np.sum(fp2 * np.log(fp2 / m)))))

    def should_snapshot(self, current_rating: trueskill.Rating, current_weights: dict = None) -> bool:
        rating_ok = current_rating.mu - self.last_snapshot_rating >= self.snapshot_on_rating_gain
        champ_ok = self.wins_vs_champion >= self.snapshot_on_champion_wins
        if not rating_ok and not champ_ok:
            return False
        if current_weights is None or not self.policy_fingerprints:
            return True
        current_fp = self.compute_policy_fingerprint(current_weights)
        min_dist = min((self.policy_distance(current_fp, fp) for fp in self.policy_fingerprints.values()), default=float('inf'))
        if min_dist < self.min_policy_distance:
            print(f"  ⏭️ Snapshot skipped: Policy too similar (dist={min_dist:.3f} < {self.min_policy_distance})")
            return False
        return True

    def _ensure_skill_spacing(self):
        snapshots = [(n, m) for n, m in self.members.items() if m.member_type == "snapshot"]
        if len(snapshots) < 3:
            return
        snapshots.sort(key=lambda x: x[1].conservative_skill, reverse=True)
        keep = set()
        if self.champion and self.members[self.champion].member_type == "snapshot":
            keep.add(self.champion)
        last_kept_skill = float('inf')
        for name, member in snapshots:
            if last_kept_skill - member.conservative_skill >= self.min_skill_spacing or name in keep:
                keep.add(name)
                last_kept_skill = member.conservative_skill
        for name, _ in sorted(snapshots, key=lambda x: x[1].created_idx, reverse=True)[:2]:
            keep.add(name)
        removed = []
        for name, member in snapshots:
            if name not in keep:
                if member.weights_path:
                    Path(member.weights_path).unlink(missing_ok=True)
                if name in self.policy_fingerprints:
                    del self.policy_fingerprints[name]
                del self.members[name]
                removed.append(name)
        if removed:
            print(f"  🧹 Skill-Spacing: Removed {len(removed)} snapshots")

    def add_snapshot(self, name: str, weights_path: str, weights: dict, rating: trueskill.Rating) -> bool:
        self._maybe_prune_snapshots()
        self._ensure_skill_spacing()
        self.policy_fingerprints[name] = self.compute_policy_fingerprint(weights)
        self.snapshot_counter += 1
        self.members[name] = LeagueMember(
            name=name, member_type="snapshot",
            rating=TS_ENV.create_rating(mu=rating.mu, sigma=rating.sigma),
            weights_path=weights_path, rating_update_factor=1.0,
            created_idx=self.snapshot_counter
        )
        self.last_snapshot_rating = rating.mu
        self.wins_vs_champion = 0
        self._update_champion()
        print(f"  📸 Snapshot: {name} (μ={rating.mu:.1f})")
        return True

    def _maybe_prune_snapshots(self):
        snapshots = [m for m in self.members.values() if m.member_type == "snapshot"]
        if len(snapshots) < self.max_snapshots:
            return
        keep = set()
        if self.champion and self.members[self.champion].member_type == "snapshot":
            keep.add(self.champion)
        for m in sorted(snapshots, key=lambda m: m.conservative_skill, reverse=True)[:5]:
            keep.add(m.name)
        for m in sorted(snapshots, key=lambda m: m.created_idx, reverse=True)[:3]:
            keep.add(m.name)
        for m in snapshots:
            if m.name not in keep:
                if m.weights_path:
                    Path(m.weights_path).unlink(missing_ok=True)
                if m.name in self.policy_fingerprints:
                    del self.policy_fingerprints[m.name]
                del self.members[m.name]

    def get_member_weights(self, name: str) -> Optional[Dict]:
        member = self.members.get(name)
        if member is None or member.member_type != "snapshot":
            return None
        if member.weights_path and Path(member.weights_path).exists():
            ckpt = torch.load(member.weights_path, map_location='cpu', weights_only=False)
            return {k: v.numpy() for k, v in ckpt['model'].items()}
        return None

    def get_env_config(self, name: str) -> Tuple[str, int, int]:
        member = self.members.get(name)
        if member is None:
            return "11_vs_11_stochastic", 1, 0
        if member.member_type == "bot" and member.controls_right:
            return member.env_name, 1, 0
        return "11_vs_11_stochastic", 1, 1

    def get_stats(self) -> Dict:
        bots = {n: m for n, m in self.members.items() if m.member_type == "bot"}
        snaps = {n: m for n, m in self.members.items() if m.member_type == "snapshot"}
        total = sum(m.games_played for m in self.members.values())
        if total > 0:
            probs = [m.games_played / total for m in self.members.values() if m.games_played > 0]
            entropy = -sum(p * np.log(p + 1e-8) for p in probs)
            max_entropy = np.log(len([m for m in self.members.values() if m.games_played > 0]) + 1e-8)
            diversity = entropy / max_entropy if max_entropy > 0 else 0
        else:
            diversity = 0
        return {
            'total_games': self.total_games, 'champion': self.champion,
            'champion_mu': self.members[self.champion].rating.mu if self.champion else 0,
            'num_snapshots': len(snaps), 'wins_vs_champion': self.wins_vs_champion, 'diversity': diversity,
            'bots': {n: {'mu': m.rating.mu, 'games': m.games_played, 'wr': 1 - m.recent_win_rate - m.recent_draw_rate,
                         'dr': m.recent_draw_rate, 'lr': m.recent_win_rate} for n, m in bots.items()}
        }

    def get_ranking(self) -> List[Tuple[str, float, str]]:
        ranked = sorted(self.members.values(), key=lambda m: m.conservative_skill, reverse=True)
        return [(m.name, m.conservative_skill, m.member_type) for m in ranked]

    def print_ranking(self, top_n: int = 10):
        print(f"\n{'='*70}\n LIGA RANKING (Top {top_n})\n{'='*70}")
        for i, (name, skill, mtype) in enumerate(self.get_ranking()[:top_n], 1):
            m = self.members[name]
            champ = "👑" if name == self.champion else "  "
            icon = "🤖" if mtype == "bot" else "📸"
            print(f"{champ}{i:2d}. {icon} {name:25s} μ={m.rating.mu:5.1f} σ={m.rating.sigma:4.2f} skill={skill:5.1f} games={m.games_played:4d} agent_wr={(1 - m.recent_win_rate - m.recent_draw_rate)*100:4.0f}%")
        print("="*70)

    def save(self, path: Path = None):
        path = path or (self.checkpoint_dir / "league.pkl")
        state = {
            'members': {name: {
                'name': m.name, 'member_type': m.member_type, 'rating_mu': m.rating.mu, 'rating_sigma': m.rating.sigma,
                'weights_path': m.weights_path, 'games_played': m.games_played, 'wins': m.wins, 'losses': m.losses,
                'draws': m.draws, 'env_name': m.env_name, 'controls_right': m.controls_right,
                'rating_update_factor': m.rating_update_factor, 'recent_wins': list(m.recent_wins),
                'recent_draws': list(m.recent_draws), 'created_idx': m.created_idx
            } for name, m in self.members.items()},
            'champion': self.champion, 'last_snapshot_rating': self.last_snapshot_rating,
            'wins_vs_champion': self.wins_vs_champion, 'total_games': self.total_games,
            'snapshot_counter': self.snapshot_counter,
            'policy_fingerprints': {k: v.tolist() for k, v in self.policy_fingerprints.items()}
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load(self, path: Path = None) -> bool:
        path = path or (self.checkpoint_dir / "league.pkl")
        if not path.exists():
            return False
        with open(path, 'rb') as f:
            state = pickle.load(f)
        self.members = {}
        for name, d in state['members'].items():
            m = LeagueMember(
                name=d['name'], member_type=d['member_type'],
                rating=TS_ENV.create_rating(d['rating_mu'], d['rating_sigma']),
                weights_path=d.get('weights_path'), games_played=d.get('games_played', 0),
                wins=d.get('wins', 0), losses=d.get('losses', 0), draws=d.get('draws', 0),
                env_name=d.get('env_name'), controls_right=d.get('controls_right', False),
                rating_update_factor=d.get('rating_update_factor', 1.0),
                created_idx=d.get('created_idx', 0)
            )
            m.recent_wins = deque(d.get('recent_wins', []), maxlen=100)
            m.recent_draws = deque(d.get('recent_draws', []), maxlen=100)
            self.members[name] = m
        self.champion = state.get('champion')
        self.last_snapshot_rating = state.get('last_snapshot_rating', 25.0)
        self.wins_vs_champion = state.get('wins_vs_champion', 0)
        self.total_games = state.get('total_games', 0)
        self.snapshot_counter = state.get('snapshot_counter', max((m.created_idx for m in self.members.values()), default=0))
        self.policy_fingerprints = {k: np.array(v) for k, v in state.get('policy_fingerprints', {}).items()}
        print(f"  📂 League loaded: {len(self.members)} members, champion={self.champion}")
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
        feat[:, 0:2] = ball_pos
        feat[:, 2] = np.clip(ball_z[:, 0], 0, 1)
        feat[:, 3] = np.clip(np.linalg.norm(ball_dir[:, :2], axis=1), 0, 2)
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
        op_dist = np.where(right_active, np.linalg.norm(right_pos - active_pos[:, None, :], axis=2), 999.0)
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
        offside_line = np.maximum(ball_x, np.sort(right_x, axis=1)[:, 1])
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
        self.mu.mul_(1 - self.beta).add_(self.beta * targets.mean())
        self.nu.mul_(1 - self.beta).add_(self.beta * (targets ** 2).mean())
        self.sigma.copy_(torch.sqrt(torch.clamp(self.nu - self.mu ** 2, min=1e-4)))
        self.linear.weight.data.mul_(old_sigma / self.sigma)
        self.linear.bias.data.copy_((self.linear.bias.data * old_sigma + old_mu - self.mu) / self.sigma)

class Net(nn.Module):
    def __init__(self, d_model: int = 512, lstm_hidden: int = 512):
        super().__init__()
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.action_emb = nn.Embedding(NUM_ACTIONS + 1, 16)
        self.encoder = nn.Sequential(
            nn.Linear(OBS_DIM + FEATURE_DIM + 16, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=1, batch_first=True)
        self.policy = nn.Sequential(
            nn.Linear(lstm_hidden, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(lstm_hidden, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU()
        )
        self.value = PopArtValueHead(128)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.lstm_hidden, device=device),
                torch.zeros(1, batch_size, self.lstm_hidden, device=device))

    def forward(self, obs, feat, prev_actions=None, hidden=None):
        squeeze = obs.dim() == 2
        if squeeze:
            obs, feat = obs.unsqueeze(1), feat.unsqueeze(1)
        B, L, _ = obs.shape
        if prev_actions is None:
            prev_actions = torch.full((B,), NUM_ACTIONS, dtype=torch.long, device=obs.device)
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.unsqueeze(1).expand(-1, L)
        x = self.encoder(torch.cat([obs, feat, self.action_emb(prev_actions)], dim=-1))
        if hidden is None:
            hidden = self.init_hidden(B, obs.device)
        x, hidden = self.lstm(x, hidden)
        logits = self.policy(x)
        values_norm = self.value(self.value_mlp(x))
        if squeeze:
            logits, values_norm = logits.squeeze(1), values_norm.squeeze(1)
        return logits, values_norm, hidden

    def get_action(self, obs, feat, prev_actions, hidden=None, deterministic=False):
        logits, values_norm, hidden = self.forward(obs, feat, prev_actions, hidden)
        dist = Categorical(logits=logits)
        actions = logits.argmax(dim=-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), self.value.denormalize(values_norm), hidden

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
    def __init__(self, wid: int, d_model: int, lstm_hidden: int, rollout_len: int = 512, goal_traj_len: int = 128):
        self.wid = wid
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.rollout_len = rollout_len
        self.goal_traj_len = goal_traj_len
        self.feat_eng = FeatureEngineer()
        self.model = Net(d_model, lstm_hidden)
        self.model.eval()
        self.opponent_model = None
        self.opponent_type = None
        self.env = None
        self.current_env_key = None
        self.opp_obs = None
        self.opp_feat = None
        self.opp_prev_act = None
        self.opp_hidden = None
        self.obs = None
        self.feat = None
        self.prev_act = None
        self.hidden = None
        self.ep_ret = 0.0
        self.ep_len = 0
        self.ep_score = [0, 0]
        self.rolling = {k: deque(maxlen=goal_traj_len) for k in ['obs', 'feat', 'prev_act', 'act', 'lp']}

    def set_weights(self, weights: dict):
        self.model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})

    def set_opponent(self, opponent_type: str, weights: dict = None):
        prev_type = self.opponent_type
        self.opponent_type = opponent_type
        if weights is not None:
            if self.opponent_model is None:
                self.opponent_model = Net(self.d_model, self.lstm_hidden)
                self.opponent_model.eval()
            self.opponent_model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})
            if prev_type != "snapshot":
                self.opp_prev_act = torch.tensor([NUM_ACTIONS], dtype=torch.long)
                self.opp_hidden = None

    def _create_env(self, env_name: str, left: int, right: int, rewards: str = "scoring") -> bool:
        env_key = (env_name, left, right, rewards)
        if self.current_env_key == env_key and self.env is not None:
            return False
        if self.env is not None:
            self.env.close()
        self.env = football_env.create_environment(
            env_name=env_name, representation="simple115v2",
            number_of_left_players_agent_controls=left,
            number_of_right_players_agent_controls=right,
            rewards=rewards, render=False
        )
        self.current_env_key = env_key
        return True

    def _reset(self):
        raw_obs = self.env.reset()
        _, left, right, _ = self.current_env_key
        if right > 0:
            if isinstance(raw_obs, list) and len(raw_obs) == 2:
                left_obs = np.array(raw_obs[0]).flatten()[:OBS_DIM].astype(np.float32)
                right_obs = np.array(raw_obs[1]).flatten()[:OBS_DIM].astype(np.float32)
            else:
                left_obs = np.array(raw_obs).flatten()[:OBS_DIM].astype(np.float32)
                right_obs = left_obs.copy()
            self.obs, self.feat = left_obs, self.feat_eng.extract(left_obs)
            self.opp_obs = right_obs
            self.opp_feat = self.feat_eng.extract(right_obs)
            self.opp_prev_act = torch.tensor([NUM_ACTIONS], dtype=torch.long)
            if self.opponent_model is not None:
                self.opp_hidden = self.opponent_model.init_hidden(1, torch.device('cpu'))
        else:
            self.obs = np.array(raw_obs).flatten()[:OBS_DIM].astype(np.float32)
            self.feat = self.feat_eng.extract(self.obs)
        self.prev_act = torch.tensor([NUM_ACTIONS], dtype=torch.long)
        self.hidden = self.model.init_hidden(1, torch.device('cpu'))
        self.ep_ret, self.ep_len = 0.0, 0
        self.ep_score = [0, 0]
        for buf in self.rolling.values():
            buf.clear()

    def _get_opponent_action(self) -> Optional[int]:
        if self.opponent_type == "random":
            return np.random.randint(0, NUM_ACTIONS)
        elif self.opponent_type == "lazy":
            return 0
        elif self.opponent_type == "snapshot" and self.opponent_model is not None:
            with torch.no_grad():
                act, _, _, self.opp_hidden = self.opponent_model.get_action(
                    torch.from_numpy(self.opp_obs).float().unsqueeze(0),
                    torch.from_numpy(self.opp_feat).float().unsqueeze(0),
                    self.opp_prev_act, self.opp_hidden
                )
            self.opp_prev_act = act.clone()
            return act.item()
        return None

    def collect(self, env_config: Tuple[str, int, int], opponent_type: str, opponent_weights: dict = None, rewards: str = "scoring") -> dict:
        env_name, left, right = env_config
        self.set_opponent(opponent_type, opponent_weights)
        env_changed = self._create_env(env_name, left, right, rewards)
        if env_changed or self.obs is None:
            self._reset()
        is_selfplay = right > 0
        data = {k: [] for k in ['obs', 'feat', 'prev_act', 'act', 'lp', 'rew', 'done']}
        episodes = []
        goal_trajectories = []
        for _ in range(self.rollout_len):
            with torch.no_grad():
                act, lp, _, self.hidden = self.model.get_action(
                    torch.from_numpy(self.obs).float().unsqueeze(0),
                    torch.from_numpy(self.feat).float().unsqueeze(0),
                    self.prev_act, self.hidden
                )
            data['obs'].append(self.obs.copy())
            data['feat'].append(self.feat.copy())
            data['prev_act'].append(self.prev_act.item())
            data['act'].append(act.item())
            data['lp'].append(lp.item())
            self.rolling['obs'].append(self.obs.copy())
            self.rolling['feat'].append(self.feat.copy())
            self.rolling['prev_act'].append(self.prev_act.item())
            self.rolling['act'].append(act.item())
            self.rolling['lp'].append(lp.item())
            self.prev_act = act.clone()
            env_action = [act.item(), self._get_opponent_action()] if is_selfplay else [act.item()]
            raw_obs, rew, done, info = self.env.step(env_action)
            rew = float(rew[0]) if isinstance(rew, (list, np.ndarray)) else float(rew)
            done = done[0] if isinstance(done, (list, np.ndarray)) else done
            self.ep_ret += rew
            self.ep_len += 1
            if rew > 0.5:
                self.ep_score[0] += 1
                if len(self.rolling['obs']) >= self.goal_traj_len // 4:
                    traj_len = len(self.rolling['obs'])
                    traj_rew = np.zeros(traj_len, dtype=np.float32)
                    traj_done = np.zeros(traj_len, dtype=np.float32)
                    traj_rew[-1] = 1.0
                    traj_done[-1] = 1.0
                    goal_trajectories.append({
                        'obs': np.array(list(self.rolling['obs']), dtype=np.float32),
                        'feat': np.array(list(self.rolling['feat']), dtype=np.float32),
                        'prev_act': np.array(list(self.rolling['prev_act']), dtype=np.int64),
                        'act': np.array(list(self.rolling['act']), dtype=np.int64),
                        'lp': np.array(list(self.rolling['lp']), dtype=np.float32),
                        'rew': traj_rew, 'done': traj_done, 'bootstrap': 0.0
                    })
                for buf in self.rolling.values():
                    buf.clear()
            elif rew < -0.5:
                self.ep_score[1] += 1
                for buf in self.rolling.values():
                    buf.clear()
            data['rew'].append(rew)
            data['done'].append(float(bool(done)))
            if bool(done):
                episodes.append({
                    'return': self.ep_ret, 'won': self.ep_score[0] > self.ep_score[1],
                    'drawn': self.ep_score[0] == self.ep_score[1], 'length': self.ep_len,
                    'opponent': opponent_type, 'goals_scored': self.ep_score[0]
                })
                self._reset()
            elif is_selfplay:
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
            _, _, bootstrap, _ = self.model.get_action(
                torch.from_numpy(self.obs).float().unsqueeze(0),
                torch.from_numpy(self.feat).float().unsqueeze(0),
                self.prev_act, self.hidden
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
            'opponent_type': opponent_type,
            'goal_trajectories': goal_trajectories
        }

    def close(self):
        if self.env is not None:
            self.env.close()

class ExpertBufferGPU:
    def __init__(self, parquet_path: str, device: torch.device, window_len: int = 512):
        self.device = device
        self.window_len = window_len
        self.num_episodes = 0
        self.episode_slices = []
        self.window_slices = []
        self.obs = self.feat = self.act = self.rew = self.done = self.prev_act = None
        if parquet_path and Path(parquet_path).exists():
            self._load_parquet(parquet_path)

    def _load_parquet(self, parquet_path):
        print(f"Loading expert data to GPU from {parquet_path}...")
        df = pq.read_table(parquet_path).to_pandas()
        obs_all = np.array([np.frombuffer(b, dtype=np.float32) for b in df['obs']])
        actions = df['action'].values.astype(np.int64)
        rewards = df['reward'].values.astype(np.float32)
        active = df['active'].values.astype(np.int64)
        episode_ids = df['episode_id'].values
        print(f"Computing features for {len(obs_all)} samples...")
        feat_all = np.array([FeatureEngineer.extract(o, a) for o, a in zip(obs_all, active)])
        unique_eps = np.unique(episode_ids)
        print(f"Processing {len(unique_eps)} episodes...")
        all_obs, all_feat, all_act, all_rew, all_done, all_prev_act = [], [], [], [], [], []
        current_idx = 0
        for ep_id in unique_eps:
            mask = episode_ids == ep_id
            ep_obs, ep_feat, ep_act, ep_rew = obs_all[mask], feat_all[mask], actions[mask], rewards[mask]
            ep_len = len(ep_obs)
            prev_acts = np.zeros(ep_len, dtype=np.int64)
            prev_acts[0] = NUM_ACTIONS
            prev_acts[1:] = ep_act[:-1]
            ep_done = np.zeros(ep_len, dtype=np.float32)
            ep_done[-1] = 1.0
            all_obs.append(ep_obs)
            all_feat.append(ep_feat)
            all_act.append(ep_act)
            all_rew.append(ep_rew)
            all_done.append(ep_done)
            all_prev_act.append(prev_acts)
            self.episode_slices.append((current_idx, current_idx + ep_len))
            current_idx += ep_len
        self.obs = torch.from_numpy(np.concatenate(all_obs)).float().to(self.device)
        self.feat = torch.from_numpy(np.concatenate(all_feat)).float().to(self.device)
        self.act = torch.from_numpy(np.concatenate(all_act)).long().to(self.device)
        self.rew = torch.from_numpy(np.concatenate(all_rew)).float().to(self.device)
        self.done = torch.from_numpy(np.concatenate(all_done)).float().to(self.device)
        self.prev_act = torch.from_numpy(np.concatenate(all_prev_act)).long().to(self.device)
        self.num_episodes = len(unique_eps)
        for start, end in self.episode_slices:
            s = start
            while s < end:
                self.window_slices.append((s, min(s + self.window_len, end)))
                s += self.window_len
        vram_mb = (self.obs.numel()*4 + self.feat.numel()*4 + self.act.numel()*8 + self.rew.numel()*4 + self.done.numel()*4 + self.prev_act.numel()*8) / 1e6
        print(f"Expert Buffer on GPU: {self.num_episodes} episodes, {len(self.window_slices)} windows (<= {self.window_len} steps), {len(self.obs)} steps, {vram_mb:.1f}MB VRAM")

    def sample(self, n: int) -> dict:
        if not self.window_slices:
            return None
        n = min(n, len(self.window_slices))
        idx = np.random.choice(len(self.window_slices), size=n, replace=False)
        max_len = max(self.window_slices[i][1] - self.window_slices[i][0] for i in idx)
        obs = torch.zeros(n, max_len, OBS_DIM, device=self.device)
        feat = torch.zeros(n, max_len, FEATURE_DIM, device=self.device)
        act = torch.zeros(n, max_len, dtype=torch.long, device=self.device)
        rew = torch.zeros(n, max_len, device=self.device)
        done = torch.ones(n, max_len, device=self.device)
        prev_act = torch.zeros(n, max_len, dtype=torch.long, device=self.device)
        mask = torch.zeros(n, max_len, device=self.device)
        lp = torch.zeros(n, max_len, device=self.device)
        for i, win_idx in enumerate(idx):
            start, end = self.window_slices[win_idx]
            length = end - start
            obs[i, :length] = self.obs[start:end]
            feat[i, :length] = self.feat[start:end]
            act[i, :length] = self.act[start:end]
            rew[i, :length] = self.rew[start:end]
            done[i, :length] = self.done[start:end]
            prev_act[i, :length] = self.prev_act[start:end]
            mask[i, :length] = 1.0
        return {
            'obs': obs, 'feat': feat, 'act': act, 'rew': rew, 'done': done,
            'prev_act': prev_act, 'mask': mask, 'lp': lp,
            'bootstrap': torch.zeros(n, device=self.device), 'max_len': max_len
        }

    def __len__(self):
        return len(self.window_slices)

class GoldenMemoryGPU:
    def __init__(self, capacity: int, max_uses: int, traj_len: int, device: torch.device):
        self.capacity = capacity
        self.max_uses = max_uses
        self.traj_len = traj_len
        self.device = device
        self.size = 0
        self.obs = torch.zeros(capacity, traj_len, OBS_DIM, device=device)
        self.feat = torch.zeros(capacity, traj_len, FEATURE_DIM, device=device)
        self.act = torch.zeros(capacity, traj_len, dtype=torch.long, device=device)
        self.prev_act = torch.zeros(capacity, traj_len, dtype=torch.long, device=device)
        self.rew = torch.zeros(capacity, traj_len, device=device)
        self.done = torch.ones(capacity, traj_len, device=device)
        self.mask = torch.zeros(capacity, traj_len, device=device)
        self.lp = torch.zeros(capacity, traj_len, device=device)
        self.uses = torch.zeros(capacity, dtype=torch.long, device='cpu')
        self.opp_ratings = torch.zeros(capacity, device='cpu')
        self.lengths = torch.zeros(capacity, dtype=torch.long, device='cpu')

    def add(self, trajectory: dict, opp_rating: float = 25.0):
        traj_len = min(trajectory['obs'].shape[0], self.traj_len)
        if self.size >= self.capacity:
            expired = (self.uses >= self.max_uses).nonzero(as_tuple=True)[0]
            if len(expired) > 0:
                idx = expired[0].item()
            else:
                idx = self.opp_ratings[:self.size].argmin().item()
                if opp_rating <= self.opp_ratings[idx].item():
                    return False
        else:
            idx = self.size
            self.size += 1
        self.obs[idx].zero_()
        self.feat[idx].zero_()
        self.act[idx].zero_()
        self.prev_act[idx].zero_()
        self.rew[idx].zero_()
        self.done[idx].fill_(1.0)
        self.mask[idx].zero_()
        self.lp[idx].zero_()
        self.obs[idx, :traj_len] = torch.from_numpy(trajectory['obs'][:traj_len].copy()).to(self.device)
        self.feat[idx, :traj_len] = torch.from_numpy(trajectory['feat'][:traj_len].copy()).to(self.device)
        self.act[idx, :traj_len] = torch.from_numpy(trajectory['act'][:traj_len].copy()).to(self.device)
        self.prev_act[idx, :traj_len] = torch.from_numpy(trajectory['prev_act'][:traj_len].copy()).to(self.device)
        self.rew[idx, :traj_len] = torch.from_numpy(trajectory['rew'][:traj_len].copy()).to(self.device)
        self.done[idx, :traj_len] = torch.from_numpy(trajectory['done'][:traj_len].copy()).to(self.device)
        self.mask[idx, :traj_len] = 1.0
        self.uses[idx] = 0
        self.opp_ratings[idx] = opp_rating
        self.lengths[idx] = traj_len
        return True

    def sample(self, n: int) -> dict:
        if self.size == 0:
            return None
        valid_idx = (self.uses[:self.size] < self.max_uses).nonzero(as_tuple=True)[0]
        if len(valid_idx) == 0:
            return None
        n = min(n, len(valid_idx))
        weights = 1.0 + self.opp_ratings[valid_idx] / 50.0
        probs = (weights / weights.sum()).numpy()
        chosen = np.random.choice(len(valid_idx), size=n, replace=False, p=probs)
        idx = valid_idx[chosen]
        for i in idx:
            self.uses[i] += 1
        return {
            'obs': self.obs[idx], 'feat': self.feat[idx], 'act': self.act[idx],
            'prev_act': self.prev_act[idx], 'rew': self.rew[idx], 'done': self.done[idx],
            'mask': self.mask[idx], 'lp': self.lp[idx],
            'bootstrap': torch.zeros(n, device=self.device), 'max_len': self.traj_len
        }

    def stats(self):
        if self.size == 0:
            return {'size': 0, 'fresh': 0, 'avg_opp': 0}
        fresh = (self.uses[:self.size] < self.max_uses).sum().item()
        return {'size': self.size, 'fresh': fresh, 'avg_opp': self.opp_ratings[:self.size].mean().item()}

    def __len__(self):
        return self.size

class LeagueLearner:
    def __init__(self, num_workers: int = 24, rollout_len: int = 512, batch_size: int = 24, lr: float = 5e-4, gamma: float = 1.0,
                 entropy_coeff: float = 0.01, value_coeff: float = 0.5, sil_coeff: float = 0.5,
                 d_model: int = 512, lstm_hidden: int = 512, checkpoint_dir: str = "./checkpoints_league",
                 warmstart_path: str = None, expert_parquet: str = None, max_snapshots: int = 15,
                 snapshot_on_rating_gain: float = 2.0, snapshot_on_champion_wins: int = 5,
                 bot_floor: float = 0.25, latest_snapshot_prob: float = 0.5, expert_threshold: float = 0.7,
                 min_skill_spacing: float = 3.0,
                 min_policy_distance: float = 0.1, goal_traj_len: int = 128, golden_capacity: int = 2048,
                 golden_max_uses: int = 5):
        self.num_workers = num_workers
        self.rollout_len = rollout_len
        self.batch_size = batch_size
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.sil_coeff = sil_coeff
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.expert_threshold = expert_threshold
        self.expert_disabled = False
        self.goal_traj_len = goal_traj_len
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.model = Net(d_model, lstm_hidden).to(self.device)
        if warmstart_path and Path(warmstart_path).exists():
            print(f"Loading warmstart from {warmstart_path}...")
            ckpt = torch.load(warmstart_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt['model'], strict=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.scaler = GradScaler('cuda')
        self.expert = ExpertBufferGPU(expert_parquet, self.device, rollout_len) if expert_parquet else None
        self.golden = GoldenMemoryGPU(golden_capacity, golden_max_uses, goal_traj_len, self.device)
        self.current_rating = TS_ENV.create_rating(mu=25.0, sigma=8.333)
        self.league = PureLeague(
            checkpoint_dir=self.checkpoint_dir, d_model=d_model, lstm_hidden=lstm_hidden,
            max_snapshots=max_snapshots, snapshot_on_rating_gain=snapshot_on_rating_gain,
            snapshot_on_champion_wins=snapshot_on_champion_wins, bot_floor=bot_floor,
            latest_snapshot_prob=latest_snapshot_prob,
            min_skill_spacing=min_skill_spacing, min_policy_distance=min_policy_distance
        )
        if (self.checkpoint_dir / "league.pkl").exists():
            self.league.load()
        if warmstart_path and Path(warmstart_path).exists():
            snap_path = self.checkpoint_dir / "snapshot_warmstart.pt"
            if not snap_path.exists():
                torch.save({'model': self.model.state_dict()}, snap_path)
            if "warmstart" not in self.league.members:
                self.league.add_snapshot("warmstart", str(snap_path), self._weights(), TS_ENV.create_rating(mu=25.0, sigma=8.0))
        ray.init(ignore_reinit_error=True, num_cpus=num_workers + 4)
        self.workers = [SelfPlayWorker.remote(i, d_model, lstm_hidden, rollout_len, goal_traj_len) for i in range(num_workers)]
        self.total_steps = 0
        self.total_matches = 0
        self.updates = 0
        self.start = None
        self.ep_wins = deque(maxlen=100)
        self.ep_draws = deque(maxlen=100)
        self.ep_losses = deque(maxlen=100)
        self.returns = deque(maxlen=100)
        self.lengths = deque(maxlen=100)
        self.returns_vs_hard = deque(maxlen=100)
        self.pending = {}
        self.queue = []
        self.eval_cycle = 0
        self.eval_bots = ["random", "lazy", "bot_easy", "bot_medium", "bot_hard"]
        self._print_config(lr, golden_capacity, golden_max_uses)

    def _print_config(self, lr, golden_capacity, golden_max_uses):
        print(f"\n{'='*70}")
        print(f"IMPALA + PFSP League (GPU Buffers) | {self.device} | {self.num_workers}W")
        print(f"Batch: {self.batch_size} x {self.rollout_len} = {self.batch_size * self.rollout_len:,} samples/update | Goal Traj: {self.goal_traj_len} steps")
        print(f"Golden: capacity={golden_capacity} max_uses={golden_max_uses}")
        print(f"LR: {lr} | γ: {self.gamma} | Ent: {self.entropy_coeff} | Val: {self.value_coeff} | SIL: {self.sil_coeff}")
        print(f"League: max_snap={self.league.max_snapshots} bot_floor={self.league.bot_floor:.0%} latest_snap={self.league.latest_snapshot_prob:.0%} skill_spacing={self.league.min_skill_spacing} policy_dist={self.league.min_policy_distance}")
        if self.expert:
            print(f"Expert Buffer (GPU): {len(self.expert)} windows | threshold: {self.expert_threshold}")
        print(f"Current Rating: μ={self.current_rating.mu:.1f}, σ={self.current_rating.sigma:.2f}")
        print(f"{'='*70}\n")

    def _weights(self) -> dict:
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def _dispatch(self, w):
        self.eval_cycle += 1
        force_bot = self.eval_bots[self.eval_cycle % 50] if self.eval_cycle % 50 < len(self.eval_bots) else None
        opponent, _ = self.league.select_opponent(self.current_rating, force_bot=force_bot)
        opp_weights = self.league.get_member_weights(opponent)
        opp_type = "snapshot" if opp_weights else opponent
        self.pending[w.collect.remote(self.league.get_env_config(opponent), opp_type, opp_weights)] = (w, opponent)

    def _prepare_batch(self, rollouts):
        max_len = max(r['obs'].shape[0] for r in rollouts)
        B = len(rollouts)
        def pad(arr, ml, pv=0):
            if arr.shape[0] == ml:
                return arr
            if arr.ndim == 1:
                return np.pad(arr, (0, ml - arr.shape[0]), constant_values=pv)
            return np.pad(arr, ((0, ml - arr.shape[0]), (0, 0)), constant_values=pv)
        obs = torch.from_numpy(np.stack([pad(r['obs'], max_len) for r in rollouts])).float().to(self.device)
        feat = torch.from_numpy(np.stack([pad(r['feat'], max_len) for r in rollouts])).float().to(self.device)
        prev_act = torch.from_numpy(np.stack([pad(r['prev_act'], max_len) for r in rollouts])).long().to(self.device)
        act = torch.from_numpy(np.stack([pad(r['act'], max_len) for r in rollouts])).long().to(self.device)
        beh_lp = torch.from_numpy(np.stack([pad(r['lp'], max_len) for r in rollouts])).float().to(self.device)
        rew = torch.from_numpy(np.stack([pad(r['rew'], max_len) for r in rollouts])).float().to(self.device)
        done = torch.from_numpy(np.stack([pad(r['done'], max_len, 1.0) for r in rollouts])).float().to(self.device)
        bootstrap = torch.tensor([r['bootstrap'] for r in rollouts], dtype=torch.float32, device=self.device)
        mask = torch.from_numpy(np.stack([
            np.concatenate([np.ones(r['obs'].shape[0]), np.zeros(max_len - r['obs'].shape[0])])
            for r in rollouts
        ])).float().to(self.device)
        return {'obs': obs, 'feat': feat, 'prev_act': prev_act, 'act': act, 'lp': beh_lp,
                'rew': rew, 'done': done, 'bootstrap': bootstrap, 'mask': mask, 'max_len': max_len}

    def _update(self, rollouts) -> dict:
        batch = self._prepare_batch(rollouts)
        obs, feat, prev_act, act, beh_lp = batch['obs'], batch['feat'], batch['prev_act'], batch['act'], batch['lp']
        rew, done, bootstrap, mask, T = batch['rew'], batch['done'], batch['bootstrap'], batch['mask'], batch['max_len']
        with autocast('cuda'):
            logits, values_norm, _ = self.model.forward(obs, feat, prev_act)
            dist = Categorical(logits=logits)
            target_lp = dist.log_prob(act)
            entropy = dist.entropy()
            values = self.model.value.denormalize(values_norm)
        with torch.no_grad():
            vs, adv, rhos = vtrace(beh_lp, target_lp.float().detach(), rew, values.float().detach(), bootstrap.float(), done, self.gamma)
            self.model.value.update_stats(vs[mask.bool()])
            vs_norm = self.model.value.normalize_target(vs)
            mean_rho = rhos[mask.bool()].mean().item()
            max_rho = rhos[mask.bool()].max().item()
        with autocast('cuda'):
            policy_loss = -((target_lp * adv.detach()) * mask).sum() / mask.sum()
            value_loss = ((values_norm - vs_norm.detach()) ** 2 * mask).sum() / mask.sum()
            ent_loss = -(entropy * mask).sum() / mask.sum()
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
            'ent': (entropy * mask).sum().item() / mask.sum().item(),
            'rho': mean_rho, 'rho_max': max_rho, 'grad': grad_norm,
            'grad_clip': grad_clipped.item() if torch.is_tensor(grad_clipped) else grad_clipped
        }

    def _update_sil_gpu(self, batch: dict) -> Optional[dict]:
        if batch is None:
            return None
        obs, feat, prev_act, act = batch['obs'], batch['feat'], batch['prev_act'], batch['act']
        rew, done, bootstrap, mask, T = batch['rew'], batch['done'], batch['bootstrap'], batch['mask'], batch['max_len']
        with autocast('cuda'):
            logits, values_norm, _ = self.model.forward(obs, feat, prev_act)
            target_lp = Categorical(logits=logits).log_prob(act)
            values = self.model.value.denormalize(values_norm)
        with torch.no_grad():
            mc_returns = torch.zeros_like(rew)
            running = bootstrap.clone()
            for t in reversed(range(T)):
                running = torch.where(done[:, t].bool(), rew[:, t], rew[:, t] + self.gamma * running)
                mc_returns[:, t] = running
            sil_adv = torch.clamp(mc_returns - values.detach(), min=0)
        sil_mask = (sil_adv > 0) & (mask > 0)
        if not sil_mask.any():
            return {'sil_loss': 0.0, 'sil_pi': 0.0, 'sil_v': 0.0, 'sil_adv': 0.0, 'sil_frac': 0.0}
        with autocast('cuda'):
            sil_policy_loss = -(target_lp[sil_mask] * sil_adv[sil_mask]).mean()
            sil_value_loss = 0.5 * (sil_adv[sil_mask] ** 2).mean()
            sil_loss = self.sil_coeff * (sil_policy_loss + sil_value_loss)
        self.optimizer.zero_grad()
        self.scaler.scale(sil_loss).backward()
        self.scaler.unscale_(self.optimizer)
        nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {
            'sil_loss': sil_loss.item(), 'sil_pi': sil_policy_loss.item(), 'sil_v': sil_value_loss.item(),
            'sil_adv': sil_adv[sil_mask].mean().item(), 'sil_frac': sil_mask.float().sum().item() / mask.sum().item()
        }

    def _maybe_snapshot(self):
        current_weights = self._weights()
        if self.league.should_snapshot(self.current_rating, current_weights):
            name = f"snap_u{self.updates}_r{self.current_rating.mu:.0f}"
            path = self.checkpoint_dir / f"{name}.pt"
            torch.save({'model': self.model.state_dict()}, path)
            self.league.add_snapshot(name, str(path), current_weights, self.current_rating)

    def _log_progress(self, stats: dict, stats_exp: dict = None, stats_sil: dict = None):
        elapsed = time.time() - self.start
        sps = self.total_steps / elapsed
        mpm = self.total_matches / (elapsed / 60) if elapsed > 0 else 0
        n_eps = len(self.ep_wins)
        if n_eps > 0:
            win_rate = sum(self.ep_wins) / n_eps * 100
            draw_rate = sum(self.ep_draws) / n_eps * 100
            loss_rate = sum(self.ep_losses) / n_eps * 100
        else:
            win_rate, draw_rate, loss_rate = 0, 0, 0
        ret = np.mean(list(self.returns)) if self.returns else 0
        ret_max = np.max(list(self.returns)) if self.returns else 0
        avg_vs_hard = np.mean(list(self.returns_vs_hard)) if len(self.returns_vs_hard) >= 10 else 0
        league_stats = self.league.get_stats()
        gm = self.golden.stats()
        bot_stats = league_stats['bots']
        e_wr = bot_stats.get('bot_easy', {}).get('wr', 0) * 100
        e_dr = bot_stats.get('bot_easy', {}).get('dr', 0) * 100
        e_lr = bot_stats.get('bot_easy', {}).get('lr', 0) * 100
        e_n = bot_stats.get('bot_easy', {}).get('games', 0)
        m_wr = bot_stats.get('bot_medium', {}).get('wr', 0) * 100
        m_dr = bot_stats.get('bot_medium', {}).get('dr', 0) * 100
        m_lr = bot_stats.get('bot_medium', {}).get('lr', 0) * 100
        m_n = bot_stats.get('bot_medium', {}).get('games', 0)
        h_wr = bot_stats.get('bot_hard', {}).get('wr', 0) * 100
        h_dr = bot_stats.get('bot_hard', {}).get('dr', 0) * 100
        h_lr = bot_stats.get('bot_hard', {}).get('lr', 0) * 100
        h_n = bot_stats.get('bot_hard', {}).get('games', 0)
        if stats_exp:
            exp_str = f"EXP L:{stats_exp['sil_loss']:.2f} p:{stats_exp['sil_pi']:.2f} v:{stats_exp['sil_v']:.2f} adv:{stats_exp['sil_adv']:.2f}({stats_exp['sil_frac']:.0%})"
        elif self.expert_disabled:
            exp_str = "EXP:OFF"
        else:
            exp_str = "EXP:--"
        if stats_sil:
            sil_str = f"SIL L:{stats_sil['sil_loss']:.2f} adv:{stats_sil['sil_adv']:.2f}({stats_sil['sil_frac']:.0%})"
        else:
            sil_str = "SIL:--"
        champ = league_stats['champion'] or "none"
        print(f"[{self.updates:4d}] {self.total_steps/1e6:.1f}M {sps/1e3:.0f}k/s {mpm:.0f}m/min {elapsed/60:.0f}m | W/D/L:{win_rate:.0f}/{draw_rate:.0f}/{loss_rate:.0f}% R:{ret:+.1f}({ret_max:+.0f}) | μ={self.current_rating.mu:.1f} σ={self.current_rating.sigma:.2f} | E:{e_wr:.0f}/{e_dr:.0f}/{e_lr:.0f}({e_n}) M:{m_wr:.0f}/{m_dr:.0f}/{m_lr:.0f}({m_n}) H:{h_wr:.0f}/{h_dr:.0f}/{h_lr:.0f}({h_n}) RvH:{avg_vs_hard:+.1f} | VT L:{stats['loss']:.2f} p:{stats['pi']:+.3f} v:{stats['v']:.2f} H:{stats['ent']:.2f} ρ:{stats['rho']:.1f}/{stats['rho_max']:.1f} | {exp_str} | {sil_str} | GM:{gm['size']}({gm['fresh']}) | ∇:{stats['grad']:.1f}→{stats['grad_clip']:.1f} | {league_stats['num_snapshots']}snap 🏆{champ}")

    def train(self, max_time: int = 3600):
        print(f"Training for {max_time}s...\n")
        self.start = time.time()
        ray.get([w.set_weights.remote(self._weights()) for w in self.workers])
        for w in self.workers:
            self._dispatch(w)
        while time.time() - self.start < max_time:
            while len(self.queue) < self.batch_size:
                done_refs, _ = ray.wait(list(self.pending.keys()), num_returns=1)
                for ref in done_refs:
                    w, opponent_name = self.pending.pop(ref)
                    rollout = ray.get(ref)
                    self.queue.append(rollout)
                    self.total_steps += rollout['obs'].shape[0]
                    for ep in rollout['episodes']:
                        self.total_matches += 1
                        self.returns.append(ep['return'])
                        self.lengths.append(ep['length'])
                        if ep['won']:
                            self.ep_wins.append(1)
                            self.ep_draws.append(0)
                            self.ep_losses.append(0)
                        elif ep['drawn']:
                            self.ep_wins.append(0)
                            self.ep_draws.append(1)
                            self.ep_losses.append(0)
                        else:
                            self.ep_wins.append(0)
                            self.ep_draws.append(0)
                            self.ep_losses.append(1)
                        if opponent_name == "bot_hard":
                            self.returns_vs_hard.append(ep['return'])
                        self.current_rating = self.league.report_game(opponent_name, self.current_rating, won=ep['won'], drawn=ep['drawn'])
                    opp_mu = self.league.members[opponent_name].rating.mu if opponent_name in self.league.members else 25.0
                    for traj in rollout.get('goal_trajectories', []):
                        self.golden.add(traj, opp_mu)
                    w.set_weights.remote(self._weights())
                    self._dispatch(w)
            batch = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            stats = self._update(batch)
            self.updates += 1
            stats_exp = None
            if self.expert and len(self.expert) >= self.batch_size // 2 and not self.expert_disabled:
                h_wr = self.league.get_stats()['bots'].get('bot_hard', {}).get('wr', 0)
                if h_wr < self.expert_threshold:
                    stats_exp = self._update_sil_gpu(self.expert.sample(self.batch_size // 2))
                else:
                    self.expert_disabled = True
                    print(f"  🎓 Expert disabled (H:{h_wr*100:.0f}%)")
            stats_sil = None
            if len(self.golden) >= self.batch_size // 4:
                golden_batch = self.golden.sample(min(self.batch_size // 2, len(self.golden)))
                if golden_batch:
                    stats_sil = self._update_sil_gpu(golden_batch)
            self._maybe_snapshot()
            if self.updates % 10 == 0:
                self._log_progress(stats, stats_exp, stats_sil)
            if self.updates % 100 == 0:
                self._save_checkpoint()
            if self.updates % 500 == 0:
                self.league.print_ranking()
        print(f"\nTime limit reached.")
        self._save_checkpoint("final")
        self.league.print_ranking()

    def _save_checkpoint(self, name: str = None):
        name = name or f"u{self.updates}"
        path = self.checkpoint_dir / f"ckpt_{name}.pt"
        torch.save({
            'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict(),
            'updates': self.updates, 'total_steps': self.total_steps, 'total_matches': self.total_matches,
            'rating_mu': self.current_rating.mu, 'rating_sigma': self.current_rating.sigma
        }, path)
        self.league.save()
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
    learner = LeagueLearner(
        num_workers=28, rollout_len=512, batch_size=64, lr=0.0001, gamma=0.999,
        entropy_coeff=0.01, value_coeff=0.2, sil_coeff=0.5,
        d_model=512, lstm_hidden=512,
        checkpoint_dir="./checkpoints_league_v77",
        warmstart_path=r"C:\clones\rlib_gfootball\bc_warmstart_v2.pt",
        expert_parquet=r"C:\clones\rlib_gfootball\main\expert.parquet",
        max_snapshots=15, snapshot_on_rating_gain=2.0, snapshot_on_champion_wins=5,
        bot_floor=0.25, latest_snapshot_prob=0.5, expert_threshold=0.7,
        min_skill_spacing=3.0, min_policy_distance=0.02,
        goal_traj_len=128, golden_capacity=2048, golden_max_uses=5,
    )
    try:
        learner.train(max_time=24*3600)
    except KeyboardInterrupt:
        print("\nStopped!")
    finally:
        learner.close()