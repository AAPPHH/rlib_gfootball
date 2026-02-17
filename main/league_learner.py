"""IMPALA league training system with TrueSkill matchmaking.

Implements a self-play league with:
- V-trace off-policy correction
- Self-Imitation Learning (SIL) from golden memory
- Expert behavioral cloning warmstart
- TrueSkill-based opponent selection and snapshotting
"""

import math
import pickle
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import trueskill
from torch.amp import GradScaler, autocast
from torch.distributions import Categorical

import gfootball.env as football_env
from feature_engineer import FEATURE_DIM, OBS_DIM, FeatureEngineer
from net import NUM_ACTIONS, Net

TS_ENV = trueskill.TrueSkill(
    mu=25.0, sigma=8.333, beta=4.166, tau=0.083, draw_probability=0.05
)


@dataclass
class LeagueMember:
    """A single member in the league (bot, snapshot, or agent).

    Attributes:
        name: Unique identifier.
        member_type: One of ``"bot"`` or ``"snapshot"``.
        rating: Current TrueSkill rating.
        weights_path: Path to the saved model weights (snapshots only).
        games_played: Total number of games played.
        wins: Total wins.
        losses: Total losses.
        draws: Total draws.
        env_name: GFootball environment variant for bots.
        controls_right: Whether this member controls the right team.
        recent_results: Rolling window of recent game outcomes.
        rating_history: Rolling window of rating history.
        rating_update_factor: Damping factor for rating updates (< 1 for bots).
    """

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
    recent_results: deque = field(default_factory=lambda: deque(maxlen=100))
    rating_history: deque = field(default_factory=lambda: deque(maxlen=500))
    rating_update_factor: float = 1.0

    @property
    def win_rate(self) -> float:
        """Overall win rate."""
        return self.wins / max(1, self.games_played)

    @property
    def recent_win_rate(self) -> float:
        """Win rate over the recent results window."""
        return np.mean(list(self.recent_results)) if self.recent_results else 0.5

    @property
    def conservative_skill(self) -> float:
        """Conservative skill estimate (mu - 3*sigma)."""
        return self.rating.mu - 3 * self.rating.sigma

    def record_game(self, won: bool, drawn: bool = False) -> None:
        """Record the outcome of a game.

        Args:
            won: Whether this member won.
            drawn: Whether the game was a draw.
        """
        self.games_played += 1
        if drawn:
            self.draws += 1
            self.recent_results.append(0.5)
        elif won:
            self.wins += 1
            self.recent_results.append(1.0)
        else:
            self.losses += 1
            self.recent_results.append(0.0)
        self.rating_history.append(self.rating.mu)


class PureLeague:
    """TrueSkill-based league with opponent selection and snapshotting.

    Manages a pool of opponents (built-in bots and past agent snapshots)
    and provides matchmaking via skill-based, champion-focused, and
    exploration-based selection strategies.

    Args:
        checkpoint_dir: Directory for saving snapshots and league state.
        max_snapshots: Maximum number of snapshots to retain.
        snapshot_on_rating_gain: Minimum rating gain to trigger a snapshot.
        snapshot_on_champion_wins: Wins vs champion to trigger a snapshot.
        skill_matched_prob: Probability of skill-matched opponent selection.
        champion_prob: Probability of selecting the current champion.
        exploration_prob: Probability of exploration-based selection.
        bot_rating_factor: Damping factor for bot rating updates.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        max_snapshots: int = 15,
        snapshot_on_rating_gain: float = 2.0,
        snapshot_on_champion_wins: int = 5,
        skill_matched_prob: float = 0.35,
        champion_prob: float = 0.50,
        exploration_prob: float = 0.15,
        bot_rating_factor: float = 0.3,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.max_snapshots = max_snapshots
        self.snapshot_on_rating_gain = snapshot_on_rating_gain
        self.snapshot_on_champion_wins = snapshot_on_champion_wins
        self.skill_matched_prob = skill_matched_prob
        self.champion_prob = champion_prob
        self.exploration_prob = exploration_prob
        self.bot_rating_factor = bot_rating_factor
        self.members: Dict[str, LeagueMember] = {}
        self.champion: Optional[str] = None
        self.last_snapshot_rating: float = 25.0
        self.wins_vs_champion: int = 0
        self.total_games: int = 0
        self._init_bots()

    def _init_bots(self) -> None:
        """Initialize built-in bot opponents with preset ratings."""
        bot_configs = [
            ("random", 5.0, 2.0, "11_vs_11_stochastic", False),
            ("lazy", 8.0, 2.0, "11_vs_11_stochastic", False),
            ("bot_easy", 18.0, 5.0, "11_vs_11_easy_stochastic", True),
            ("bot_medium", 28.0, 5.0, "11_vs_11_stochastic", True),
            ("bot_hard", 38.0, 5.0, "11_vs_11_hard_stochastic", True),
        ]
        for name, mu, sigma, env_name, controls_right in bot_configs:
            self.members[name] = LeagueMember(
                name=name,
                member_type="bot",
                rating=TS_ENV.create_rating(mu=mu, sigma=sigma),
                env_name=env_name,
                controls_right=controls_right,
                rating_update_factor=self.bot_rating_factor,
            )
        self._update_champion()

    def _update_champion(self) -> None:
        """Recalculate and update the current champion."""
        best_name, best_skill = None, -np.inf
        for name, member in self.members.items():
            skill = member.conservative_skill
            if skill > best_skill:
                best_skill, best_name = skill, name
        if best_name != self.champion:
            old = self.champion
            self.champion = best_name
            self.wins_vs_champion = 0
            if old:
                print(
                    f"  \U0001f451 Champion: {best_name} "
                    f"(\u03bc={self.members[best_name].rating.mu:.1f}) < {old}"
                )

    def match_quality(self, r1: trueskill.Rating, r2: trueskill.Rating) -> float:
        """Compute TrueSkill match quality between two ratings.

        Args:
            r1: First player's rating.
            r2: Second player's rating.

        Returns:
            Match quality in ``[0, 1]``.
        """
        return trueskill.quality_1vs1(r1, r2)

    def win_probability(
        self, player: trueskill.Rating, opponent: trueskill.Rating
    ) -> float:
        """Estimate win probability for the player.

        Args:
            player: Player's rating.
            opponent: Opponent's rating.

        Returns:
            Estimated probability of the player winning.
        """
        delta = player.mu - opponent.mu
        denom = math.sqrt(2 * (TS_ENV.beta**2) + player.sigma**2 + opponent.sigma**2)
        return 0.5 * (1 + math.erf(delta / (denom * math.sqrt(2))))

    def select_opponent(
        self, current_rating: trueskill.Rating, force_bot: str = None
    ) -> Tuple[str, str]:
        """Select an opponent using the configured strategy mix.

        Args:
            current_rating: The agent's current rating.
            force_bot: If set, force selection of this specific bot.

        Returns:
            Tuple of ``(opponent_name, selection_reason)``.
        """
        if force_bot and force_bot in self.members:
            return force_bot, "eval"
        cutoff = current_rating.mu - 15
        strong = [
            n
            for n, m in self.members.items()
            if m.rating.mu > cutoff or m.member_type == "bot"
        ]
        if len(strong) < 3:
            strong = list(self.members.keys())
        r = np.random.random()
        if r < self.champion_prob and self.champion:
            return self.champion, "champion"
        if r < self.champion_prob + self.skill_matched_prob:
            return self._select_skill_matched(current_rating, strong)
        return self._select_exploration(strong)

    def _select_skill_matched(
        self, current_rating: trueskill.Rating, candidates: List[str]
    ) -> Tuple[str, str]:
        """Select an opponent with similar skill level.

        Args:
            current_rating: The agent's current rating.
            candidates: List of candidate opponent names.

        Returns:
            Tuple of ``(opponent_name, "skill_matched")``.
        """
        qualities = np.array(
            [
                self.match_quality(current_rating, self.members[n].rating)
                for n in candidates
            ]
        )
        weights = np.exp(qualities / 0.3)
        probs = weights / weights.sum()
        return np.random.choice(candidates, p=probs), "skill_matched"

    def _select_exploration(self, candidates: List[str]) -> Tuple[str, str]:
        """Select a less-played opponent for exploration.

        Args:
            candidates: List of candidate opponent names.

        Returns:
            Tuple of ``(opponent_name, "exploration")``.
        """
        games = np.array([self.members[n].games_played for n in candidates])
        weights = 1.0 / (games + 1)
        probs = weights / weights.sum()
        return np.random.choice(candidates, p=probs), "exploration"

    def report_game(
        self,
        opponent_name: str,
        current_rating: trueskill.Rating,
        won: bool,
        drawn: bool = False,
    ) -> trueskill.Rating:
        """Report a game result and update ratings.

        Args:
            opponent_name: Name of the opponent.
            current_rating: The agent's rating before the game.
            won: Whether the agent won.
            drawn: Whether the game was a draw.

        Returns:
            The agent's updated rating.
        """
        if opponent_name not in self.members:
            return current_rating
        member = self.members[opponent_name]
        self.total_games += 1
        if drawn:
            new_current, new_opponent = TS_ENV.rate_1vs1(
                current_rating, member.rating, drawn=True
            )
        elif won:
            new_current, new_opponent = TS_ENV.rate_1vs1(current_rating, member.rating)
        else:
            new_opponent, new_current = TS_ENV.rate_1vs1(member.rating, current_rating)
        if member.rating_update_factor < 1.0:
            mu_delta = new_opponent.mu - member.rating.mu
            sigma_delta = new_opponent.sigma - member.rating.sigma
            new_mu = member.rating.mu + mu_delta * member.rating_update_factor
            new_sigma = member.rating.sigma + sigma_delta * member.rating_update_factor
            member.rating = TS_ENV.create_rating(mu=new_mu, sigma=new_sigma)
        else:
            member.rating = new_opponent
        member.record_game(won=not won, drawn=drawn)
        if opponent_name == self.champion and won:
            self.wins_vs_champion += 1
        self._update_champion()
        return new_current

    def should_snapshot(self, current_rating: trueskill.Rating) -> bool:
        """Check whether the agent should create a new snapshot.

        Args:
            current_rating: The agent's current rating.

        Returns:
            ``True`` if a snapshot should be created.
        """
        if (
            current_rating.mu - self.last_snapshot_rating
            >= self.snapshot_on_rating_gain
        ):
            return True
        if self.wins_vs_champion >= self.snapshot_on_champion_wins:
            return True
        return False

    def add_snapshot(
        self, name: str, weights_path: str, rating: trueskill.Rating
    ) -> bool:
        """Add a new agent snapshot to the league.

        Args:
            name: Unique snapshot name.
            weights_path: Path to the saved model weights.
            rating: The agent's rating at snapshot time.

        Returns:
            ``True`` on success.
        """
        self._maybe_prune_snapshots()
        self.members[name] = LeagueMember(
            name=name,
            member_type="snapshot",
            rating=TS_ENV.create_rating(mu=rating.mu, sigma=rating.sigma),
            weights_path=weights_path,
            rating_update_factor=1.0,
        )
        self.last_snapshot_rating = rating.mu
        self.wins_vs_champion = 0
        self._update_champion()
        print(f"  \U0001f4f8 Snapshot: {name} (\u03bc={rating.mu:.1f})")
        return True

    def _maybe_prune_snapshots(self) -> None:
        """Remove old snapshots if the pool exceeds the maximum size."""
        snapshots = [m for m in self.members.values() if m.member_type == "snapshot"]
        if len(snapshots) < self.max_snapshots:
            return
        keep = set()
        if self.champion and self.members[self.champion].member_type == "snapshot":
            keep.add(self.champion)
        by_skill = sorted(snapshots, key=lambda m: m.conservative_skill, reverse=True)
        for m in by_skill[:5]:
            keep.add(m.name)
        by_name = sorted(snapshots, key=lambda m: m.name, reverse=True)
        for m in by_name[:3]:
            keep.add(m.name)
        for m in snapshots:
            if m.name not in keep:
                if m.weights_path:
                    Path(m.weights_path).unlink(missing_ok=True)
                del self.members[m.name]

    def get_member_weights(self, name: str) -> Optional[Dict]:
        """Load and return model weights for a snapshot member.

        Args:
            name: Member name.

        Returns:
            Dictionary of numpy weight arrays, or ``None`` if unavailable.
        """
        member = self.members.get(name)
        if member is None or member.member_type != "snapshot":
            return None
        if member.weights_path and Path(member.weights_path).exists():
            ckpt = torch.load(
                member.weights_path, map_location="cpu", weights_only=False
            )
            return {k: v.numpy() for k, v in ckpt["model"].items()}
        return None

    def get_env_config(self, name: str) -> Tuple[str, int, int]:
        """Get the environment configuration for playing against a member.

        Args:
            name: Member name.

        Returns:
            Tuple of ``(env_name, num_left_players, num_right_players)``.
        """
        member = self.members.get(name)
        if member is None:
            return "11_vs_11_stochastic", 1, 0
        if member.member_type == "bot" and member.controls_right:
            return member.env_name, 1, 0
        return "11_vs_11_stochastic", 1, 1

    def get_stats(self) -> Dict:
        """Return summary statistics of the league.

        Returns:
            Dictionary with game counts, champion info, and per-bot stats.
        """
        bots = {n: m for n, m in self.members.items() if m.member_type == "bot"}
        snaps = {n: m for n, m in self.members.items() if m.member_type == "snapshot"}
        return {
            "total_games": self.total_games,
            "champion": self.champion,
            "champion_mu": (
                self.members[self.champion].rating.mu if self.champion else 0
            ),
            "num_snapshots": len(snaps),
            "wins_vs_champion": self.wins_vs_champion,
            "bots": {
                n: {
                    "mu": m.rating.mu,
                    "games": m.games_played,
                    "wr": m.recent_win_rate,
                }
                for n, m in bots.items()
            },
        }

    def get_ranking(self) -> List[Tuple[str, float, str]]:
        """Return league members sorted by conservative skill.

        Returns:
            List of ``(name, conservative_skill, member_type)`` tuples.
        """
        ranked = sorted(
            self.members.values(),
            key=lambda m: m.conservative_skill,
            reverse=True,
        )
        return [(m.name, m.conservative_skill, m.member_type) for m in ranked]

    def print_ranking(self, top_n: int = 10) -> None:
        """Print a formatted league ranking table.

        Args:
            top_n: Number of top members to display.
        """
        print(f"\n{'=' * 70}\n LIGA RANKING (Top {top_n})\n{'=' * 70}")
        for i, (name, skill, mtype) in enumerate(self.get_ranking()[:top_n], 1):
            m = self.members[name]
            champ = "\U0001f451" if name == self.champion else "  "
            icon = "\U0001f916" if mtype == "bot" else "\U0001f4f8"
            print(
                f"{champ}{i:2d}. {icon} {name:25s} "
                f"\u03bc={m.rating.mu:5.1f} \u03c3={m.rating.sigma:4.2f} "
                f"skill={skill:5.1f} games={m.games_played:4d} "
                f"wr={m.recent_win_rate * 100:4.0f}%"
            )
        print("=" * 70)

    def save(self, path: Path = None) -> None:
        """Serialize the league state to disk.

        Args:
            path: Output file path. Defaults to ``league.pkl`` in the
                checkpoint directory.
        """
        path = path or (self.checkpoint_dir / "league.pkl")
        state = {
            "members": {
                name: {
                    "name": m.name,
                    "member_type": m.member_type,
                    "rating_mu": m.rating.mu,
                    "rating_sigma": m.rating.sigma,
                    "weights_path": m.weights_path,
                    "games_played": m.games_played,
                    "wins": m.wins,
                    "losses": m.losses,
                    "draws": m.draws,
                    "env_name": m.env_name,
                    "controls_right": m.controls_right,
                    "rating_update_factor": m.rating_update_factor,
                    "recent_results": list(m.recent_results),
                }
                for name, m in self.members.items()
            },
            "champion": self.champion,
            "last_snapshot_rating": self.last_snapshot_rating,
            "wins_vs_champion": self.wins_vs_champion,
            "total_games": self.total_games,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: Path = None) -> bool:
        """Load league state from disk.

        Args:
            path: Input file path. Defaults to ``league.pkl`` in the
                checkpoint directory.

        Returns:
            ``True`` if the file was loaded successfully.
        """
        path = path or (self.checkpoint_dir / "league.pkl")
        if not path.exists():
            return False
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.members = {}
        for name, d in state["members"].items():
            m = LeagueMember(
                name=d["name"],
                member_type=d["member_type"],
                rating=TS_ENV.create_rating(d["rating_mu"], d["rating_sigma"]),
                weights_path=d.get("weights_path"),
                games_played=d.get("games_played", 0),
                wins=d.get("wins", 0),
                losses=d.get("losses", 0),
                draws=d.get("draws", 0),
                env_name=d.get("env_name"),
                controls_right=d.get("controls_right", False),
                rating_update_factor=d.get("rating_update_factor", 1.0),
            )
            m.recent_results = deque(d.get("recent_results", []), maxlen=100)
            self.members[name] = m
        self.champion = state.get("champion")
        self.last_snapshot_rating = state.get("last_snapshot_rating", 25.0)
        self.wins_vs_champion = state.get("wins_vs_champion", 0)
        self.total_games = state.get("total_games", 0)
        print(
            f"  \U0001f4c2 League loaded: {len(self.members)} members, "
            f"champion={self.champion}"
        )
        return True


def vtrace(
    behavior_lp: torch.Tensor,
    target_lp: torch.Tensor,
    rewards: torch.Tensor,
    values: torch.Tensor,
    bootstrap: torch.Tensor,
    dones: torch.Tensor,
    gamma: float = 0.99,
    rho_bar: float = 1.0,
    c_bar: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute V-trace targets and advantages.

    Args:
        behavior_lp: Log-probs under the behavior policy, shape ``(B, T)``.
        target_lp: Log-probs under the target policy, shape ``(B, T)``.
        rewards: Rewards, shape ``(B, T)``.
        values: Value estimates, shape ``(B, T)``.
        bootstrap: Bootstrap values for the last step, shape ``(B,)``.
        dones: Done flags, shape ``(B, T)``.
        gamma: Discount factor.
        rho_bar: Importance weight clipping threshold for targets.
        c_bar: Importance weight clipping threshold for trace coefficients.

    Returns:
        Tuple of ``(vs_targets, advantages, clipped_rhos)``.
    """
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
    """Ray remote actor that collects rollouts via self-play.

    Each worker maintains its own environment instance, agent model copy,
    and optional opponent model for snapshot-based self-play.

    Args:
        wid: Worker identifier.
        d_model: Network encoder hidden dimension.
        lstm_hidden: LSTM hidden dimension.
        rollout_len: Number of steps per rollout.
    """

    def __init__(self, wid: int, d_model: int, lstm_hidden: int, rollout_len: int):
        self.wid = wid
        self.rollout_len = rollout_len
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.feat_eng = FeatureEngineer()
        self.model = Net(d_model, lstm_hidden)
        self.model.eval()
        self.opponent_model, self.opponent_type = None, None
        self.env, self.current_env_key = None, None
        self.obs, self.feat = None, None
        self.ep_ret, self.ep_len = 0.0, 0
        self.prev_act, self.hidden = None, None
        self.ep_score = [0, 0]
        self.opp_obs, self.opp_feat = None, None
        self.opp_prev_act, self.opp_hidden = None, None

    def set_weights(self, weights: dict) -> None:
        """Update the agent's model weights.

        Args:
            weights: Dictionary mapping parameter names to numpy arrays.
        """
        self.model.load_state_dict(
            {k: torch.from_numpy(v.copy()) for k, v in weights.items()}
        )

    def set_opponent(self, opponent_type: str, weights: dict = None) -> None:
        """Configure the opponent for self-play.

        Args:
            opponent_type: Type of opponent (``"random"``, ``"lazy"``,
                ``"snapshot"``, or a bot name).
            weights: Model weights for snapshot opponents.
        """
        prev_type = self.opponent_type
        self.opponent_type = opponent_type
        if weights is not None:
            if self.opponent_model is None:
                self.opponent_model = Net(self.d_model, self.lstm_hidden)
                self.opponent_model.eval()
            self.opponent_model.load_state_dict(
                {k: torch.from_numpy(v.copy()) for k, v in weights.items()}
            )
            if prev_type != "snapshot":
                self.opp_prev_act = torch.tensor([NUM_ACTIONS], dtype=torch.long)
                self.opp_hidden = None

    def _create_env(
        self, env_name: str, left: int, right: int
    ) -> Tuple[bool, Optional[dict]]:
        """Create or reuse a GFootball environment.

        Args:
            env_name: Environment scenario name.
            left: Number of left-team agent-controlled players.
            right: Number of right-team agent-controlled players.

        Returns:
            Tuple of ``(env_changed, partial_episode_info)``.
        """
        env_key = (env_name, left, right)
        if self.current_env_key == env_key:
            return False, None
        partial_ep = None
        if self.env is not None and self.ep_len > 0:
            won = self.ep_score[0] > self.ep_score[1]
            drawn = self.ep_score[0] == self.ep_score[1]
            partial_ep = {
                "return": self.ep_ret,
                "won": won,
                "drawn": drawn,
                "length": self.ep_len,
                "partial": True,
            }
            self.env.close()
        elif self.env is not None:
            self.env.close()
        self.env = football_env.create_environment(
            env_name=env_name,
            representation="simple115v2",
            number_of_left_players_agent_controls=left,
            number_of_right_players_agent_controls=right,
            rewards="scoring",
            render=False,
        )
        self.current_env_key = env_key
        return True, partial_ep

    def _reset(self) -> None:
        """Reset the environment and internal state for a new episode."""
        raw_obs = self.env.reset()
        _, left, right = self.current_env_key
        if right > 0:
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
                self.opp_hidden = self.opponent_model.init_hidden(
                    1, torch.device("cpu")
                )
        else:
            self.obs = np.array(raw_obs).flatten()[:OBS_DIM].astype(np.float32)
            self.feat = self.feat_eng.extract(self.obs)
        self.ep_ret, self.ep_len = 0.0, 0
        self.ep_score = [0, 0]
        self.prev_act = torch.tensor([NUM_ACTIONS], dtype=torch.long)
        self.hidden = self.model.init_hidden(1, torch.device("cpu"))

    def _get_opponent_action(self) -> Optional[int]:
        """Compute the opponent's action based on opponent type.

        Returns:
            Action index, or ``None`` if the opponent is a built-in bot.
        """
        if self.opponent_type == "random":
            return np.random.randint(0, NUM_ACTIONS)
        elif self.opponent_type == "lazy":
            return 0
        elif self.opponent_type == "snapshot" and self.opponent_model is not None:
            with torch.no_grad():
                act, _, _, self.opp_hidden = self.opponent_model.get_action(
                    torch.from_numpy(self.opp_obs).float().unsqueeze(0),
                    torch.from_numpy(self.opp_feat).float().unsqueeze(0),
                    self.opp_prev_act,
                    self.opp_hidden,
                )
            self.opp_prev_act = act.clone()
            return act.item()
        return None

    def collect(
        self,
        env_config: Tuple[str, int, int],
        opponent_type: str,
        opponent_weights: dict = None,
    ) -> dict:
        """Collect a single rollout of experience.

        Args:
            env_config: Tuple of ``(env_name, num_left, num_right)``.
            opponent_type: Type of opponent to play against.
            opponent_weights: Model weights for snapshot opponents.

        Returns:
            Dictionary containing observations, actions, log-probs, rewards,
            done flags, bootstrap value, and completed episode summaries.
        """
        env_name, left, right = env_config
        prev_opponent = self.opponent_type
        self.set_opponent(opponent_type, opponent_weights)
        env_changed, partial_ep = self._create_env(env_name, left, right)
        if env_changed or self.obs is None:
            self._reset()
        is_selfplay = right > 0
        data = {k: [] for k in ["obs", "feat", "prev_act", "act", "lp", "rew", "done"]}
        episodes = []
        if partial_ep and prev_opponent:
            partial_ep["opponent"] = prev_opponent
            episodes.append(partial_ep)
        for _ in range(self.rollout_len):
            with torch.no_grad():
                act, lp, _, self.hidden = self.model.get_action(
                    torch.from_numpy(self.obs).float().unsqueeze(0),
                    torch.from_numpy(self.feat).float().unsqueeze(0),
                    self.prev_act,
                    self.hidden,
                )
            data["obs"].append(self.obs.copy())
            data["feat"].append(self.feat.copy())
            data["prev_act"].append(self.prev_act.item())
            data["act"].append(act.item())
            data["lp"].append(lp.item())
            self.prev_act = act.clone()
            if is_selfplay:
                opp_act = self._get_opponent_action()
                env_action = [act.item(), opp_act]
            else:
                env_action = [act.item()]
            raw_obs, rew, done, info = self.env.step(env_action)
            rew = float(rew[0]) if isinstance(rew, (list, np.ndarray)) else float(rew)
            done = done[0] if isinstance(done, (list, np.ndarray)) else done
            self.ep_ret += rew
            if rew > 0.5:
                self.ep_score[0] += 1
            elif rew < -0.5:
                self.ep_score[1] += 1
            self.ep_len += 1
            ep_done = bool(done) or self.ep_len >= 3000
            data["rew"].append(rew)
            data["done"].append(float(ep_done))
            if ep_done:
                won = self.ep_score[0] > self.ep_score[1]
                drawn = self.ep_score[0] == self.ep_score[1]
                episodes.append(
                    {
                        "return": self.ep_ret,
                        "won": won,
                        "drawn": drawn,
                        "length": self.ep_len,
                        "opponent": opponent_type,
                    }
                )
                self._reset()
            else:
                if is_selfplay:
                    if isinstance(raw_obs, list) and len(raw_obs) == 2:
                        self.obs = (
                            np.array(raw_obs[0]).flatten()[:OBS_DIM].astype(np.float32)
                        )
                        self.opp_obs = (
                            np.array(raw_obs[1]).flatten()[:OBS_DIM].astype(np.float32)
                        )
                    else:
                        self.obs = (
                            np.array(raw_obs).flatten()[:OBS_DIM].astype(np.float32)
                        )
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
                self.prev_act,
                self.hidden,
            )
        return {
            "obs": np.array(data["obs"], dtype=np.float32),
            "feat": np.array(data["feat"], dtype=np.float32),
            "prev_act": np.array(data["prev_act"], dtype=np.int64),
            "act": np.array(data["act"], dtype=np.int64),
            "lp": np.array(data["lp"], dtype=np.float32),
            "rew": np.array(data["rew"], dtype=np.float32),
            "done": np.array(data["done"], dtype=np.float32),
            "bootstrap": bootstrap.item(),
            "episodes": episodes,
            "opponent_type": opponent_type,
        }

    def close(self) -> None:
        """Close the environment."""
        if self.env is not None:
            self.env.close()


class ExpertBuffer:
    """Buffer of expert demonstration rollouts loaded from parquet.

    Prioritizes sampling of higher-return episodes.

    Args:
        parquet_path: Path to the expert parquet file.
        rollout_len: Number of steps per rollout chunk.
    """

    def __init__(self, parquet_path: str, rollout_len: int = 128):
        self.rollout_len = rollout_len
        self.rollouts: List[dict] = []
        self.returns: List[float] = []
        if parquet_path and Path(parquet_path).exists():
            self._load_parquet(parquet_path)

    def _load_parquet(self, parquet_path: str) -> None:
        """Load and chunk expert episodes from a parquet file.

        Args:
            parquet_path: Path to the parquet file.
        """
        print(f"Loading expert data from {parquet_path}...")
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        obs_all = np.array([np.frombuffer(b, dtype=np.float32) for b in df["obs"]])
        actions = df["action"].values.astype(np.int64)
        rewards = df["reward"].values.astype(np.float32)
        active = df["active"].values.astype(np.int64)
        episode_ids = df["episode_id"].values
        scores = df["score"].values
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
                end = (i + 1) * self.rollout_len
                prev_acts = np.zeros(self.rollout_len, dtype=np.int64)
                prev_acts[0] = NUM_ACTIONS
                prev_acts[1:] = ep_act[start : end - 1]
                rollout = {
                    "obs": ep_obs[start:end].astype(np.float32),
                    "feat": ep_feat[start:end].astype(np.float32),
                    "act": ep_act[start:end],
                    "rew": ep_rew[start:end],
                    "done": np.zeros(self.rollout_len, dtype=np.float32),
                    "lp": np.zeros(self.rollout_len, dtype=np.float32),
                    "prev_act": prev_acts,
                    "bootstrap": 0.0,
                }
                rollout["done"][-1] = 1.0 if i == num_rollouts - 1 else 0.0
                self.rollouts.append(rollout)
                self.returns.append(ep_score)
        print(
            f"Loaded {len(self.rollouts)} expert rollouts, "
            f"avg return: {np.mean(self.returns):.1f}"
        )

    def sample(self, n: int) -> List[dict]:
        """Sample rollouts weighted by return.

        Args:
            n: Number of rollouts to sample.

        Returns:
            List of rollout dictionaries.
        """
        if not self.rollouts:
            return []
        n = min(n, len(self.rollouts))
        weights = np.array(self.returns) - np.min(self.returns) + 0.1
        probs = weights / weights.sum()
        idx = np.random.choice(len(self.rollouts), size=n, replace=False, p=probs)
        return [self.rollouts[i] for i in idx]

    def __len__(self) -> int:
        return len(self.rollouts)


class GoldenMemory:
    """Replay buffer of the agent's best self-play experiences.

    Stores high-return rollouts with limited replay counts and
    priority sampling based on return and opponent strength.

    Args:
        capacity: Maximum number of rollouts to store.
        max_uses: Maximum number of times a rollout can be replayed.
    """

    def __init__(self, capacity: int = 256, max_uses: int = 8):
        self.capacity = capacity
        self.max_uses = max_uses
        self.buffer: List[dict] = []
        self.returns: List[float] = []
        self.wins: List[bool] = []
        self.uses: List[int] = []
        self.opp_ratings: List[float] = []

    def add(
        self,
        rollout: dict,
        ret: float,
        won: bool,
        opp_rating: float = 25.0,
    ) -> bool:
        """Add a rollout to the buffer if it meets quality criteria.

        Args:
            rollout: Rollout data dictionary.
            ret: Episode return.
            won: Whether the episode was won.
            opp_rating: Opponent's TrueSkill mu at game time.

        Returns:
            ``True`` if the rollout was added.
        """
        if ret < -0.5 and not won:
            return False
        if len(self.buffer) >= self.capacity:
            worst_idx = int(np.argmin(self.returns))
            if ret <= self.returns[worst_idx]:
                return False
            for lst in [
                self.buffer,
                self.returns,
                self.wins,
                self.uses,
                self.opp_ratings,
            ]:
                lst.pop(worst_idx)
        self.buffer.append(rollout.copy())
        self.returns.append(ret)
        self.wins.append(won)
        self.uses.append(0)
        self.opp_ratings.append(opp_rating)
        return True

    def sample(self, n: int) -> List[dict]:
        """Sample rollouts with priority weighting.

        Priority is based on return magnitude, opponent rating, and
        whether the episode was won.

        Args:
            n: Number of rollouts to sample.

        Returns:
            List of rollout dictionaries.
        """
        if not self.buffer:
            return []
        valid = [i for i, u in enumerate(self.uses) if u < self.max_uses]
        if not valid:
            self._cleanup()
            return []
        n = min(n, len(valid))
        rets = np.array([self.returns[i] for i in valid])
        opp_rats = np.array([self.opp_ratings[i] for i in valid])
        weights = (rets - rets.min() + 0.1) * (1.0 + opp_rats / 50.0)
        weights = np.array(
            [w * 2.0 if self.wins[valid[j]] else w for j, w in enumerate(weights)]
        )
        probs = weights / weights.sum()
        idx = np.random.choice(valid, size=n, replace=False, p=probs)
        for i in idx:
            self.uses[i] += 1
        return [self.buffer[i] for i in idx]

    def _cleanup(self) -> None:
        """Remove exhausted rollouts that have exceeded max_uses."""
        keep = [i for i, u in enumerate(self.uses) if u < self.max_uses]
        for attr in ["buffer", "returns", "wins", "uses", "opp_ratings"]:
            setattr(self, attr, [getattr(self, attr)[i] for i in keep])

    def stats(self) -> dict:
        """Return summary statistics of the buffer.

        Returns:
            Dictionary with size, win count, mean return, and fresh count.
        """
        if not self.buffer:
            return {"size": 0, "wins": 0, "ret_mean": 0, "fresh": 0}
        return {
            "size": len(self.buffer),
            "wins": sum(self.wins),
            "ret_mean": np.mean(self.returns),
            "fresh": sum(1 for u in self.uses if u < self.max_uses),
        }

    def __len__(self) -> int:
        return len(self.buffer)


class LeagueLearner:
    """Main training orchestrator combining IMPALA, SIL, and league play.

    Coordinates distributed rollout collection via Ray workers, performs
    V-trace policy gradient updates, self-imitation learning from golden
    memory and expert demonstrations, and manages the league lifecycle.

    Args:
        num_workers: Number of Ray rollout workers.
        rollout_len: Steps per rollout.
        batch_size: Number of rollouts per training batch.
        lr: Learning rate.
        gamma: Discount factor.
        entropy_coeff: Entropy bonus coefficient.
        value_coeff: Value loss coefficient.
        sil_coeff: Self-imitation learning loss coefficient.
        d_model: Network encoder hidden dimension.
        lstm_hidden: LSTM hidden dimension.
        checkpoint_dir: Directory for checkpoints.
        warmstart_path: Path to a pre-trained checkpoint for warmstarting.
        expert_parquet: Path to expert demonstration parquet file.
        max_snapshots: Maximum league snapshots.
        snapshot_on_rating_gain: Rating gain threshold for snapshotting.
        snapshot_on_champion_wins: Champion wins threshold for snapshotting.
        skill_matched_prob: Probability of skill-matched opponent selection.
        champion_prob: Probability of champion opponent selection.
        exploration_prob: Probability of exploration opponent selection.
        bot_rating_factor: Damping for bot rating updates.
        expert_threshold: Win rate vs hard bot to disable expert guidance.
    """

    def __init__(
        self,
        num_workers: int = 24,
        rollout_len: int = 512,
        batch_size: int = 64,
        lr: float = 5e-4,
        gamma: float = 0.997,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        sil_coeff: float = 0.5,
        d_model: int = 128,
        lstm_hidden: int = 128,
        checkpoint_dir: str = "./checkpoints_league",
        warmstart_path: str = None,
        expert_parquet: str = None,
        max_snapshots: int = 15,
        snapshot_on_rating_gain: float = 2.0,
        snapshot_on_champion_wins: int = 5,
        skill_matched_prob: float = 0.35,
        champion_prob: float = 0.50,
        exploration_prob: float = 0.15,
        bot_rating_factor: float = 0.3,
        expert_threshold: float = 0.7,
    ):
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)

        self.model = Net(d_model, lstm_hidden).to(self.device)
        if warmstart_path and Path(warmstart_path).exists():
            print(f"Loading warmstart from {warmstart_path}...")
            ckpt = torch.load(
                warmstart_path,
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(ckpt["model"], strict=False)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, eps=1e-5)
        self.scaler = GradScaler("cuda")
        self.expert = (
            ExpertBuffer(expert_parquet, rollout_len) if expert_parquet else None
        )
        self.golden = GoldenMemory(capacity=batch_size * 8, max_uses=8)
        self.current_rating = TS_ENV.create_rating(mu=25.0, sigma=8.333)

        self.league = PureLeague(
            checkpoint_dir=self.checkpoint_dir,
            max_snapshots=max_snapshots,
            snapshot_on_rating_gain=snapshot_on_rating_gain,
            snapshot_on_champion_wins=snapshot_on_champion_wins,
            skill_matched_prob=skill_matched_prob,
            champion_prob=champion_prob,
            exploration_prob=exploration_prob,
            bot_rating_factor=bot_rating_factor,
        )
        league_path = self.checkpoint_dir / "league.pkl"
        if league_path.exists():
            self.league.load()

        if warmstart_path and Path(warmstart_path).exists():
            snap_path = self.checkpoint_dir / "snapshot_warmstart.pt"
            if not snap_path.exists():
                torch.save({"model": self.model.state_dict()}, snap_path)
            if "warmstart" not in self.league.members:
                self.league.add_snapshot(
                    "warmstart",
                    str(snap_path),
                    TS_ENV.create_rating(mu=25.0, sigma=8.0),
                )

        ray.init(ignore_reinit_error=True, num_cpus=num_workers + 4)
        self.workers = [
            SelfPlayWorker.remote(i, d_model, lstm_hidden, rollout_len)
            for i in range(num_workers)
        ]
        self.total_steps = 0
        self.updates = 0
        self.start: Optional[float] = None
        self.returns: deque = deque(maxlen=100)
        self.wins: deque = deque(maxlen=100)
        self.lengths: deque = deque(maxlen=100)
        self.returns_vs_hard: deque = deque(maxlen=100)
        self.pending: dict = {}
        self.queue: list = []
        self.eval_cycle = 0
        self.eval_bots = [
            "random",
            "lazy",
            "bot_easy",
            "bot_medium",
            "bot_hard",
        ]
        self._print_config(lr)

    def _print_config(self, lr: float) -> None:
        """Print the training configuration summary.

        Args:
            lr: Learning rate.
        """
        print(f"\n{'=' * 70}")
        print(f"IMPALA + Pure League | {self.device} | {self.num_workers}W")
        print(
            f"Batch: {self.batch_size} x {self.rollout_len} = "
            f"{self.batch_size * self.rollout_len:,} samples/update"
        )
        print(
            f"LR: {lr} | \u03b3: {self.gamma} | Ent: {self.entropy_coeff} "
            f"| Val: {self.value_coeff} | SIL: {self.sil_coeff}"
        )
        print(
            f"League: max_snap={self.league.max_snapshots} "
            f"skill_match={self.league.skill_matched_prob:.0%} "
            f"champ={self.league.champion_prob:.0%}"
        )
        if self.expert:
            print(
                f"Expert Buffer: {len(self.expert)} rollouts "
                f"| threshold: {self.expert_threshold}"
            )
        print(
            f"Current Rating: \u03bc={self.current_rating.mu:.1f}, "
            f"\u03c3={self.current_rating.sigma:.2f}"
        )
        print(f"{'=' * 70}\n")

    def _weights(self) -> dict:
        """Extract model weights as a dict of numpy arrays.

        Returns:
            Dictionary mapping parameter names to numpy arrays.
        """
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def _prepare_batch(self, rollouts: List[dict]) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
    ]:
        """Stack rollouts into batched tensors on the training device.

        Args:
            rollouts: List of rollout dictionaries.

        Returns:
            Tuple of ``(obs, feat, prev_act, act, beh_lp, rew, done,
            bootstrap, T)``.
        """
        B, T = len(rollouts), rollouts[0]["obs"].shape[0]
        obs = (
            torch.from_numpy(np.stack([r["obs"] for r in rollouts]))
            .float()
            .to(self.device)
        )
        feat = (
            torch.from_numpy(np.stack([r["feat"] for r in rollouts]))
            .float()
            .to(self.device)
        )
        prev_act = (
            torch.from_numpy(np.stack([r["prev_act"] for r in rollouts]))
            .long()
            .to(self.device)
        )
        act = (
            torch.from_numpy(np.stack([r["act"] for r in rollouts]))
            .long()
            .to(self.device)
        )
        beh_lp = (
            torch.from_numpy(np.stack([r["lp"] for r in rollouts]))
            .float()
            .to(self.device)
        )
        rew = (
            torch.from_numpy(np.stack([r["rew"] for r in rollouts]))
            .float()
            .to(self.device)
        )
        done = (
            torch.from_numpy(np.stack([r["done"] for r in rollouts]))
            .float()
            .to(self.device)
        )
        bootstrap = torch.tensor(
            [r["bootstrap"] for r in rollouts],
            dtype=torch.float32,
            device=self.device,
        )
        return obs, feat, prev_act, act, beh_lp, rew, done, bootstrap, T

    def _update(self, rollouts: List[dict]) -> dict:
        """Perform a single V-trace IMPALA update.

        Args:
            rollouts: Batch of rollout dictionaries.

        Returns:
            Dictionary of training metrics.
        """
        obs, feat, prev_act, act, beh_lp, rew, done, bootstrap, T = self._prepare_batch(
            rollouts
        )
        with autocast("cuda"):
            logits, values_norm, _ = self.model.forward(obs, feat, prev_act)
            dist = Categorical(logits=logits)
            target_lp = dist.log_prob(act)
            entropy = dist.entropy()
            values = self.model.value.denormalize(values_norm)
        with torch.no_grad():
            vs, adv, rhos = vtrace(
                beh_lp,
                target_lp.float().detach(),
                rew,
                values.float().detach(),
                bootstrap.float(),
                done,
                self.gamma,
            )
            self.model.value.update_stats(vs)
            vs_norm = self.model.value.normalize_target(vs)
            mean_rho, max_rho = rhos.mean().item(), rhos.max().item()
        with autocast("cuda"):
            policy_loss = -(target_lp * adv.detach()).mean()
            value_loss = F.mse_loss(values_norm, vs_norm.detach())
            ent_loss = -entropy.mean()
            loss = (
                policy_loss
                + self.value_coeff * value_loss
                + self.entropy_coeff * ent_loss
            )
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = (
            sum(
                p.grad.norm(2).item() ** 2
                for p in self.model.parameters()
                if p.grad is not None
            )
            ** 0.5
        )
        grad_clipped = nn.utils.clip_grad_norm_(self.model.parameters(), 40.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {
            "loss": loss.item(),
            "pi": policy_loss.item(),
            "v": value_loss.item(),
            "ent": entropy.mean().item(),
            "rho": mean_rho,
            "rho_max": max_rho,
            "grad": grad_norm,
            "grad_clip": (
                grad_clipped.item() if torch.is_tensor(grad_clipped) else grad_clipped
            ),
        }

    def _update_sil(self, rollouts: List[dict]) -> Optional[dict]:
        """Perform a Self-Imitation Learning update.

        Args:
            rollouts: Batch of rollout dictionaries (expert or golden).

        Returns:
            Dictionary of SIL metrics, or ``None`` if rollouts is empty.
        """
        if not rollouts:
            return None
        obs, feat, prev_act, act, _, rew, done, bootstrap, T = self._prepare_batch(
            rollouts
        )
        with autocast("cuda"):
            logits, values_norm, _ = self.model.forward(obs, feat, prev_act)
            dist = Categorical(logits=logits)
            target_lp = dist.log_prob(act)
            values = self.model.value.denormalize(values_norm)
        with torch.no_grad():
            mc_returns = torch.zeros_like(rew)
            running = bootstrap.clone()
            for t in reversed(range(T)):
                running = torch.where(
                    done[:, t].bool(),
                    rew[:, t],
                    rew[:, t] + self.gamma * running,
                )
                mc_returns[:, t] = running
            sil_adv = torch.clamp(mc_returns - values.detach(), min=0)
        mask = sil_adv > 0
        if not mask.any():
            return {
                "sil_loss": 0.0,
                "sil_pi": 0.0,
                "sil_v": 0.0,
                "sil_adv": 0.0,
                "sil_frac": 0.0,
            }
        with autocast("cuda"):
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
            "sil_loss": sil_loss.item(),
            "sil_pi": sil_policy_loss.item(),
            "sil_v": sil_value_loss.item(),
            "sil_adv": sil_adv[mask].mean().item(),
            "sil_frac": mask.float().mean().item(),
        }

    def _maybe_snapshot(self) -> None:
        """Create a league snapshot if the rating threshold is met."""
        if self.league.should_snapshot(self.current_rating):
            name = f"snap_u{self.updates}_r{self.current_rating.mu:.0f}"
            path = self.checkpoint_dir / f"{name}.pt"
            torch.save({"model": self.model.state_dict()}, path)
            self.league.add_snapshot(name, str(path), self.current_rating)

    def _log_progress(
        self,
        stats: dict,
        stats_exp: dict = None,
        stats_sil: dict = None,
    ) -> None:
        """Print a single-line training progress summary.

        Args:
            stats: V-trace update metrics.
            stats_exp: Expert SIL update metrics (optional).
            stats_sil: Golden memory SIL update metrics (optional).
        """
        elapsed = time.time() - self.start
        sps = self.total_steps / elapsed
        wr = np.mean(list(self.wins)) * 100 if self.wins else 0
        ret = np.mean(list(self.returns)) if self.returns else 0
        ret_max = np.max(list(self.returns)) if self.returns else 0
        avg_vs_hard = (
            np.mean(list(self.returns_vs_hard))
            if len(self.returns_vs_hard) >= 10
            else 0
        )
        league_stats = self.league.get_stats()
        gm = self.golden.stats()
        bot_stats = league_stats["bots"]
        e_wr = (1 - bot_stats.get("bot_easy", {}).get("wr", 0.5)) * 100
        m_wr = (1 - bot_stats.get("bot_medium", {}).get("wr", 0.5)) * 100
        h_wr = (1 - bot_stats.get("bot_hard", {}).get("wr", 0.5)) * 100
        e_n = bot_stats.get("bot_easy", {}).get("games", 0)
        m_n = bot_stats.get("bot_medium", {}).get("games", 0)
        h_n = bot_stats.get("bot_hard", {}).get("games", 0)
        if stats_exp:
            exp_str = (
                f"EXP L:{stats_exp['sil_loss']:.2f} "
                f"p:{stats_exp['sil_pi']:.2f} v:{stats_exp['sil_v']:.2f} "
                f"adv:{stats_exp['sil_adv']:.2f}"
                f"({stats_exp['sil_frac']:.0%})"
            )
        elif self.expert_disabled:
            exp_str = "EXP:OFF"
        else:
            exp_str = "EXP:--"
        if stats_sil:
            sil_str = (
                f"SIL L:{stats_sil['sil_loss']:.2f} "
                f"p:{stats_sil['sil_pi']:.2f} v:{stats_sil['sil_v']:.2f} "
                f"adv:{stats_sil['sil_adv']:.2f}"
                f"({stats_sil['sil_frac']:.0%})"
            )
        else:
            sil_str = "SIL:--"
        champ = league_stats["champion"] or "none"
        print(
            f"[{self.updates:4d}] {self.total_steps / 1e6:.1f}M "
            f"{sps / 1e3:.0f}k/s {elapsed / 60:.0f}m | "
            f"W:{wr:4.0f}% R:{ret:+.1f}({ret_max:+.0f}) | "
            f"\u03bc={self.current_rating.mu:.1f} "
            f"\u03c3={self.current_rating.sigma:.2f} | "
            f"E:{e_wr:2.0f}%({e_n}) M:{m_wr:2.0f}%({m_n}) "
            f"H:{h_wr:2.0f}%({h_n}) RvH:{avg_vs_hard:+.1f} | "
            f"VT L:{stats['loss']:.2f} p:{stats['pi']:+.2f} "
            f"v:{stats['v']:.2f} H:{stats['ent']:.2f} "
            f"\u03c1:{stats['rho']:.1f}/{stats['rho_max']:.1f} | "
            f"{exp_str} | {sil_str} | "
            f"GM:{gm['size']}({gm['fresh']}) | "
            f"\u2207:{stats['grad']:.1f}\u2192{stats['grad_clip']:.1f} | "
            f"League:{league_stats['num_snapshots']}snap "
            f"\U0001f3c6{champ}"
        )

    def train(self, max_time: int = 3600) -> None:
        """Run the main training loop.

        Args:
            max_time: Maximum training time in seconds.
        """
        print(f"Training for {max_time}s...\n")
        self.start = time.time()
        current_weights = self._weights()
        ray.get([w.set_weights.remote(current_weights) for w in self.workers])
        for w in self.workers:
            self.eval_cycle += 1
            if self.eval_cycle % 50 < len(self.eval_bots):
                force_bot = self.eval_bots[self.eval_cycle % 50]
                opponent, reason = self.league.select_opponent(
                    self.current_rating, force_bot=force_bot
                )
            else:
                opponent, reason = self.league.select_opponent(self.current_rating)
            env_config = self.league.get_env_config(opponent)
            opp_weights = self.league.get_member_weights(opponent)
            opp_type = "snapshot" if opp_weights else opponent
            self.pending[w.collect.remote(env_config, opp_type, opp_weights)] = (
                w,
                opponent,
            )
        while time.time() - self.start < max_time:
            while len(self.queue) < self.batch_size:
                done_refs, _ = ray.wait(list(self.pending.keys()), num_returns=1)
                for ref in done_refs:
                    w, opponent_name = self.pending.pop(ref)
                    rollout = ray.get(ref)
                    self.queue.append(rollout)
                    self.total_steps += self.rollout_len
                    for ep in rollout["episodes"]:
                        self.returns.append(ep["return"])
                        self.wins.append(float(ep["won"]))
                        self.lengths.append(ep["length"])
                        if opponent_name == "bot_hard":
                            self.returns_vs_hard.append(ep["return"])
                        self.current_rating = self.league.report_game(
                            opponent_name,
                            self.current_rating,
                            won=ep["won"],
                            drawn=ep.get("drawn", False),
                        )
                    if rollout["episodes"]:
                        best_ep = max(
                            rollout["episodes"],
                            key=lambda e: e["return"],
                        )
                        opp_mu = (
                            self.league.members[opponent_name].rating.mu
                            if opponent_name in self.league.members
                            else 25.0
                        )
                        self.golden.add(
                            rollout,
                            best_ep["return"],
                            best_ep["won"],
                            opp_mu,
                        )
                    w.set_weights.remote(self._weights())
                    self.eval_cycle += 1
                    if self.eval_cycle % 50 < len(self.eval_bots):
                        force_bot = self.eval_bots[self.eval_cycle % 50]
                        opponent, reason = self.league.select_opponent(
                            self.current_rating, force_bot=force_bot
                        )
                    else:
                        opponent, reason = self.league.select_opponent(
                            self.current_rating
                        )
                    env_config = self.league.get_env_config(opponent)
                    opp_weights = self.league.get_member_weights(opponent)
                    opp_type = "snapshot" if opp_weights else opponent
                    self.pending[
                        w.collect.remote(env_config, opp_type, opp_weights)
                    ] = (w, opponent)
            batch = self.queue[: self.batch_size]
            self.queue = self.queue[self.batch_size :]
            stats = self._update(batch)
            self.updates += 1
            stats_exp = None
            if (
                self.expert
                and len(self.expert) >= self.batch_size // 2
                and not self.expert_disabled
            ):
                bot_stats = self.league.get_stats()["bots"]
                h_wr = 1 - bot_stats.get("bot_hard", {}).get("wr", 0.5)
                if h_wr < self.expert_threshold:
                    expert_batch = self.expert.sample(self.batch_size // 2)
                    stats_exp = self._update_sil(expert_batch)
                else:
                    self.expert_disabled = True
                    print(f"  \U0001f393 Expert disabled (H:{h_wr * 100:.0f}%)")
            stats_sil = None
            if self.golden:
                golden_batch = self.golden.sample(
                    min(self.batch_size // 2, len(self.golden))
                )
                if golden_batch:
                    stats_sil = self._update_sil(golden_batch)
            self._maybe_snapshot()
            if self.updates % 10 == 0:
                self._log_progress(stats, stats_exp, stats_sil)
            if self.updates % 100 == 0:
                self._save_checkpoint()
            if self.updates % 500 == 0:
                self.league.print_ranking()
        print("\nTime limit reached.")
        self._save_checkpoint("final")
        self.league.print_ranking()

    def _save_checkpoint(self, name: str = None) -> None:
        """Save model, optimizer state, and league to disk.

        Args:
            name: Optional checkpoint name suffix. Defaults to the
                current update count.
        """
        name = name or f"u{self.updates}"
        path = self.checkpoint_dir / f"ckpt_{name}.pt"
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "updates": self.updates,
                "total_steps": self.total_steps,
                "rating_mu": self.current_rating.mu,
                "rating_sigma": self.current_rating.sigma,
            },
            path,
        )
        self.league.save()
        print(f"  \U0001f4be Saved {path}")

    def close(self) -> None:
        """Cancel pending work, close workers, and shut down Ray."""
        for ref in self.pending:
            try:
                ray.cancel(ref)
            except Exception:
                pass
        for w in self.workers:
            try:
                ray.get(w.close.remote(), timeout=2)
            except Exception:
                pass
        ray.shutdown()


if __name__ == "__main__":
    learner = LeagueLearner(
        num_workers=32,
        rollout_len=512,
        batch_size=128,
        lr=0.0005,
        gamma=0.997,
        entropy_coeff=0.01,
        value_coeff=0.5,
        sil_coeff=0.5,
        d_model=512,
        lstm_hidden=512,
        checkpoint_dir="./checkpoints_league",
        warmstart_path=r"C:\clones\rlib_gfootball\checkpoints_selfplay\ckpt_u900.pt",
        expert_parquet=r"C:\clones\rlib_gfootball\main\expert.parquet",
        max_snapshots=15,
        snapshot_on_rating_gain=2.0,
        snapshot_on_champion_wins=5,
        skill_matched_prob=0.35,
        champion_prob=0.50,
        exploration_prob=0.15,
        bot_rating_factor=0.3,
        expert_threshold=0.7,
    )
    try:
        learner.train(max_time=360000)
    except KeyboardInterrupt:
        print("\nStopped!")
    finally:
        learner.close()
