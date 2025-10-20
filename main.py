import os
from pathlib import Path
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from datetime import datetime
from dataclasses import dataclass
import pickle
import json

from gymnasium import spaces
import gfootball.env as football_env

from trueskill import Rating, rate_1vs1, quality_1vs1

import ray
from ray import tune
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.models import ModelCatalog
from model import GFootballMamba
@dataclass
class TrainingStage:
    name: str
    env_name: str
    representation: str 
    left_players: int
    right_players: int
    target_reward: float
    max_timesteps: int
    description: str = ""

TRAINING_STAGES = [
    TrainingStage("stage_1_basic_0", "academy_empty_goal_close", "simple115v2", 1, 0, 0.75, 5_000_000, "1 Spieler vor Tor"),
    TrainingStage("stage_1_basic_1", "academy_run_to_score_with_keeper", "simple115v2", 1, 0, 0.75, 10_000_000, "1 Spieler rennt zum Tor"),
    TrainingStage("stage_1_basic_2", "academy_pass_and_shoot_with_keeper", "simple115v2", 1, 0, 0.75, 10_000_000, "1 Spieler gegen Keeper"),
    TrainingStage("stage_2_1v1", "academy_3_vs_1_with_keeper", "simple115v2", 3, 0, 0.75, 20_000_000, "1v1"),
    TrainingStage("stage_3_3v3", "11_vs_11_easy_stochastic", "simple115v2", 3, 0, 1.0, 50_000_000, "3v3"),
    TrainingStage("stage_4_3v3", "11_vs_11_easy_stochastic", "simple115v2", 3, 3, 1.0, 100_000_000, "3v3"),
    TrainingStage("stage_5_3v3", "11_vs_11_stochastic", "simple115v2", 3, 3, 1.0, 500_000_000, "3v3"),
]

class PolicyPool:
    def __init__(self, save_dir: str, 
                 max_versions_per_policy: int = 25,
                 keep_top_n: int = 10,
                 active_zone_size: int = 15,
                 auto_prune_enabled: bool = True):
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.policy_versions = {
            "policy_left": [],
            "policy_right": []
        }
        
        self.ratings = {
            "policy_left": {},
            "policy_right": {}
        }
        
        self.current_versions = {
            "policy_left": 0,
            "policy_right": 0
        }
        
        self.champions = {
            "policy_left": None,
            "policy_right": None
        }
        
        self.match_history = []
        
        self.max_versions_per_policy = max_versions_per_policy
        self.keep_top_n = keep_top_n
        self.active_zone_size = active_zone_size
        self.auto_prune_enabled = auto_prune_enabled
        
        self.protected_versions = {
            "policy_left": set([0]),
            "policy_right": set([0])
        }
        
        self.load_state()

        for policy_name in ["policy_left", "policy_right"]:
            if 0 not in self.ratings[policy_name]:
                self.ratings[policy_name][0] = Rating()
                if not self.policy_versions[policy_name]:
                    self.policy_versions[policy_name].append({
                        'version_id': 0, 'iteration': 0, 'weights_path': None, 
                        'timestamp': datetime.now().isoformat()
                    })
                    self.current_versions[policy_name] = 0
                self.champions[policy_name] = 0
    
    def add_policy_version(self, policy_name: str, weights: dict, iteration: int) -> int:
        version_id = len(self.policy_versions[policy_name])
        
        weights_path = self.save_dir / f"{policy_name}_v{version_id}_iter{iteration}.pkl"
        with open(weights_path, 'wb') as f:
            pickle.dump(weights, f)
        
        self.policy_versions[policy_name].append({
            'version_id': version_id,
            'iteration': iteration,
            'weights_path': str(weights_path),
            'timestamp': datetime.now().isoformat()
        })
        
        current_rating = self.ratings[policy_name][self.current_versions[policy_name]]
        self.ratings[policy_name][version_id] = Rating(mu=current_rating.mu, sigma=current_rating.sigma + 1.0)
        self.current_versions[policy_name] = version_id
        
        self._update_champion(policy_name)
        
        if self.auto_prune_enabled:
            num_versions = len(self.policy_versions[policy_name])
            if num_versions > self.max_versions_per_policy:
                self._auto_prune(policy_name)
        
        self.save_state()
        return version_id
    
    def _update_champion(self, policy_name: str):
        best_version = None
        best_skill = -np.inf
        
        for version_id, rating in self.ratings[policy_name].items():
            conservative_skill = rating.mu - 3 * rating.sigma
            if conservative_skill > best_skill:
                best_skill = conservative_skill
                best_version = version_id
        
        old_champion = self.champions[policy_name]
        self.champions[policy_name] = best_version
        
        if best_version is not None:
            self.protected_versions[policy_name].add(best_version)
        
        if old_champion != best_version and best_version is not None:
            pass
    
    def _auto_prune(self, policy_name: str):
        versions = self.policy_versions[policy_name]
        num_versions = len(versions)
        
        if num_versions <= self.max_versions_per_policy:
            return
        
        version_skills = []
        for version_info in versions:
            vid = version_info['version_id']
            if vid in self.ratings[policy_name]:
                rating = self.ratings[policy_name][vid]
                skill = rating.mu - 3 * rating.sigma
                version_skills.append((vid, skill, version_info))
        
        version_skills.sort(key=lambda x: x[1], reverse=True)
        
        versions_to_keep = set(self.protected_versions[policy_name])
        
        for i in range(min(self.keep_top_n, len(version_skills))):
            versions_to_keep.add(version_skills[i][0])
        
        recent_versions = [v['version_id'] for v in versions[-self.active_zone_size:]]
        versions_to_keep.update(recent_versions)
        
        versions_to_keep.add(self.current_versions[policy_name])
        
        all_version_ids = set(v['version_id'] for v in versions)
        versions_to_prune = all_version_ids - versions_to_keep
        
        if not versions_to_prune:
            return
        
        pruned_count = 0
        for vid in versions_to_prune:
            version_idx = next((i for i, v in enumerate(versions) if v['version_id'] == vid), None)
            if version_idx is not None:
                version_info = versions[version_idx]
                
                weights_path = version_info.get('weights_path')
                if weights_path and Path(weights_path).exists():
                    try:
                        Path(weights_path).unlink()
                        pruned_count += 1
                    except Exception as e:
                        print(f"  Warning: Could not delete {weights_path}: {e}")
                
                del versions[version_idx]
                if vid in self.ratings[policy_name]:
                    del self.ratings[policy_name][vid]
    
    def get_active_versions(self, policy_name: str) -> List[int]:
        if not self.policy_versions[policy_name]:
            return [0]
        
        active_versions = set()
        
        version_skills = []
        for vid, rating in self.ratings[policy_name].items():
            skill = rating.mu - 3 * rating.sigma
            version_skills.append((vid, skill))
        
        version_skills.sort(key=lambda x: x[1], reverse=True)
        for i in range(min(self.keep_top_n, len(version_skills))):
            active_versions.add(version_skills[i][0])
        
        recent_vids = [v['version_id'] for v in self.policy_versions[policy_name][-self.active_zone_size:]]
        active_versions.update(recent_vids)
        
        active_versions.add(self.current_versions[policy_name])
        
        return sorted(list(active_versions))
    
    def update_ratings(self, policy1_name: str, policy1_version: int,
                       policy2_name: str, policy2_version: int,
                       left_score: float, right_score: float, num_games: int):
        rating1 = self.ratings[policy1_name][policy1_version]
        rating2 = self.ratings[policy2_name][policy2_version]
        
        avg_left = left_score / num_games
        avg_right = right_score / num_games
        score_diff = avg_left - avg_right
        
        if score_diff > 0.1:
            new_rating1, new_rating2 = rate_1vs1(rating1, rating2)
        elif score_diff < -0.1:
            new_rating2, new_rating1 = rate_1vs1(rating2, rating1)
        else:
            new_rating1, new_rating2 = rate_1vs1(rating1, rating2, drawn=True)
        
        self.ratings[policy1_name][policy1_version] = new_rating1
        self.ratings[policy2_name][policy2_version] = new_rating2
        
        self._update_champion(policy1_name)
        self._update_champion(policy2_name)
        
        match_record = {
            'timestamp': datetime.now().isoformat(),
            'policy1': f"{policy1_name}_v{policy1_version}",
            'policy2': f"{policy2_name}_v{policy2_version}",
            'left_score': left_score,
            'right_score': right_score,
            'num_games': num_games,
            'rating1_before': {'mu': rating1.mu, 'sigma': rating1.sigma},
            'rating2_before': {'mu': rating2.mu, 'sigma': rating2.sigma},
            'rating1_after': {'mu': new_rating1.mu, 'sigma': new_rating1.sigma},
            'rating2_after': {'mu': new_rating2.mu, 'sigma': new_rating2.sigma},
        }
        self.match_history.append(match_record)
        
        if len(self.match_history) > 1000:
            self.match_history = self.match_history[-1000:]
        
        self.save_state()
    
    def sample_opponent(self, policy_name: str, current_version: int, 
                        temperature: float = 0.7,
                        exploration_prob: float = 0.15,
                        champion_prob: float = 0.10) -> Tuple[str, int]:
        opponent_name = "policy_right" if policy_name == "policy_left" else "policy_left"
        
        active_versions = self.get_active_versions(opponent_name)
        
        if not active_versions:
            return opponent_name, 0
        
        if len(active_versions) == 1:
            return opponent_name, active_versions[0]
        
        current_rating = self.ratings[policy_name][current_version]
        champion_id = self.champions[opponent_name]
        
        qualities = []
        version_ids = []
        
        for vid in active_versions:
            opponent_rating = self.ratings[opponent_name][vid]
            match_quality = quality_1vs1(current_rating, opponent_rating)
            
            qualities.append(match_quality)
            version_ids.append(vid)
        
        qualities = np.array(qualities)
        
        weights = np.power(qualities, temperature)
        probs = weights / weights.sum()
        
        min_prob = exploration_prob / len(probs)
        probs = (1.0 - exploration_prob - champion_prob) * probs + min_prob
        
        if champion_id is not None and champion_id in version_ids:
            champion_idx = version_ids.index(champion_id)
            probs[champion_idx] += champion_prob
        
        probs = probs / probs.sum()
        
        selected_idx = np.random.choice(len(version_ids), p=probs)
        selected_version = version_ids[selected_idx]
        
        return opponent_name, selected_version
    
    def get_rating_stats(self) -> dict:
        stats = {}
        for policy_name in ["policy_left", "policy_right"]:
            if self.ratings[policy_name]:
                current_ver = self.current_versions[policy_name]
                rating = self.ratings[policy_name][current_ver]

                stats[f"{policy_name}_mu"] = rating.mu
                stats[f"{policy_name}_sigma"] = rating.sigma
                
                conservative_skill = rating.mu - 3 * rating.sigma
                stats[f"{policy_name}_skill"] = conservative_skill
                
                stats[f"{policy_name}_version"] = current_ver
                stats[f"{policy_name}_num_versions"] = len(self.policy_versions[policy_name])
                
                champion_id = self.champions[policy_name]
                if champion_id is not None:
                    champion_rating = self.ratings[policy_name][champion_id]
                    champion_skill = champion_rating.mu - 3 * champion_rating.sigma
                    stats[f"{policy_name}_champion_version"] = champion_id
                    stats[f"{policy_name}_champion_skill"] = champion_skill
                    stats[f"{policy_name}_is_champion"] = 1.0 if current_ver == champion_id else 0.0
        
        stats["active_zone_size_left"] = len(self.get_active_versions("policy_left"))
        stats["active_zone_size_right"] = len(self.get_active_versions("policy_right"))
        
        if len(self.match_history) >= 5:
            recent_opponents = [m['policy2'] for m in self.match_history[-10:]]
            unique_opponents = len(set(recent_opponents))
            stats["opponent_diversity"] = unique_opponents / min(10, len(recent_opponents))
        
        return stats
    
    def save_state(self):
        state = {
            'policy_versions': self.policy_versions,
            'ratings': {
                policy: {ver: {'mu': r.mu, 'sigma': r.sigma} 
                         for ver, r in versions.items()}
                for policy, versions in self.ratings.items()
            },
            'current_versions': self.current_versions,
            'champions': self.champions,
            'protected_versions': {k: list(v) for k, v in self.protected_versions.items()},
            'match_history': self.match_history[-100:],
            'config': {
                'max_versions_per_policy': self.max_versions_per_policy,
                'keep_top_n': self.keep_top_n,
                'active_zone_size': self.active_zone_size,
                'auto_prune_enabled': self.auto_prune_enabled
            }
        }
        state_path = self.save_dir / 'pool_state.json'
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load_state(self):
        state_path = self.save_dir / 'pool_state.json'
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.policy_versions = state['policy_versions']
                self.current_versions = state['current_versions']
                
                for policy, versions in state['ratings'].items():
                    for ver_str, rating_dict in versions.items():
                        ver = int(ver_str)
                        self.ratings[policy][ver] = Rating(
                            mu=rating_dict['mu'],
                            sigma=rating_dict['sigma']
                        )
                
                if 'champions' in state:
                    self.champions = state['champions']
                
                if 'protected_versions' in state:
                    self.protected_versions = {k: set(v) for k, v in state['protected_versions'].items()}
                
                if 'match_history' in state:
                    self.match_history = state['match_history']
                
                if 'config' in state:
                    cfg = state['config']
                    self.max_versions_per_policy = cfg.get('max_versions_per_policy', self.max_versions_per_policy)
                    self.keep_top_n = cfg.get('keep_top_n', self.keep_top_n)
                    self.active_zone_size = cfg.get('active_zone_size', self.active_zone_size)
                    self.auto_prune_enabled = cfg.get('auto_prune_enabled', self.auto_prune_enabled)
                
            except Exception as e:
                print(f"Warning: Could not load pool state: {e}")


class EnhancedSelfPlayCallback(DefaultCallbacks):
    def __init__(self, update_interval=25, version_save_interval=100,
                 has_left_agents=True, has_right_agents=True, 
                 save_dir="policy_pool",
                 min_games_before_rating_update=20,
                 max_versions_per_policy=50,
                 keep_top_n=10,
                 active_zone_size=15,
                 auto_prune_enabled=True):
        super().__init__()
        self.update_interval = update_interval
        self.version_save_interval = version_save_interval
        self._update_counter = 0
        self.self_play_enabled = has_left_agents and has_right_agents
        self.active_policy_name = "policy_left"
        self.min_games_before_rating_update = min_games_before_rating_update
        
        self.policy_pool = PolicyPool(
            save_dir=save_dir,
            max_versions_per_policy=max_versions_per_policy,
            keep_top_n=keep_top_n,
            active_zone_size=active_zone_size,
            auto_prune_enabled=auto_prune_enabled
        ) if self.self_play_enabled else None
        
        self.episode_outcomes = []
        self.accumulated_left_score = 0.0
        self.accumulated_right_score = 0.0

        self.current_opponent_versions = {
            "policy_left": 0,
            "policy_right": 0
        }

    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        if not self.self_play_enabled:
            return
        
        agent_rewards = episode.agent_rewards
        
        left_rewards = [r for aid, r in agent_rewards.items() if "left" in aid]
        right_rewards = [r for aid, r in agent_rewards.items() if "right" in aid]
        
        if left_rewards and right_rewards:
            left_total = sum(left_rewards) / len(left_rewards) if left_rewards else 0.0
            right_total = sum(right_rewards) / len(right_rewards) if right_rewards else 0.0
            
            self.accumulated_left_score += left_total
            self.accumulated_right_score += right_total
            self.episode_outcomes.append((left_total, right_total))

    def on_train_result(self, *, algorithm, result: dict, **kwargs): 
        if self.self_play_enabled and len(self.episode_outcomes) >= self.min_games_before_rating_update:
            left_version = self.policy_pool.current_versions["policy_left"]
            right_version = self.current_opponent_versions["policy_right"]
            
            num_games = len(self.episode_outcomes)
            
            self.policy_pool.update_ratings(
                "policy_left", left_version,
                "policy_right", right_version,
                self.accumulated_left_score,
                self.accumulated_right_score,
                num_games
            )
            
            self.episode_outcomes = []
            self.accumulated_left_score = 0.0
            self.accumulated_right_score = 0.0
        
        if self.self_play_enabled and self._update_counter > 0:
            
            if self._update_counter % self.version_save_interval == 0:
                active_policy = algorithm.get_policy(self.active_policy_name)
                if active_policy:
                    weights = active_policy.get_weights()
                    version_id = self.policy_pool.add_policy_version(
                        self.active_policy_name, weights, self._update_counter
                    )
            
            if self._update_counter % self.update_interval == 0:
                self.active_policy_name = (
                    "policy_right" if self.active_policy_name == "policy_left" else "policy_left"
                )
                
                current_version = self.policy_pool.current_versions[self.active_policy_name]
                
                opponent_name, opponent_version = self.policy_pool.sample_opponent(
                    self.active_policy_name, 
                    current_version, 
                    temperature=0.7,
                    exploration_prob=0.15,
                    champion_prob=0.10
                )
                
                self.current_opponent_versions[opponent_name] = opponent_version
                
                if opponent_version != self.policy_pool.current_versions[opponent_name]:
                    version_info = next((v for v in self.policy_pool.policy_versions[opponent_name] 
                                         if v['version_id'] == opponent_version), None)
                    
                    if version_info:
                        weights_path = version_info['weights_path']
                        
                        try:
                            with open(weights_path, 'rb') as f:
                                opponent_weights = pickle.load(f)
                            
                            opponent_policy = algorithm.get_policy(opponent_name)
                            if opponent_policy:
                                opponent_policy.set_weights(opponent_weights)
                        except Exception as e:
                            print(f"  ✗ Warning: Could not load opponent weights: {e}\n")
        
        if self.self_play_enabled:
            result.setdefault("custom_metrics", {})
            result["custom_metrics"]["active_policy"] = 1.0 if self.active_policy_name == "policy_left" else 0.0
            
            rating_stats = self.policy_pool.get_rating_stats()
            result["custom_metrics"].update(rating_stats)
        
        self._update_counter += 1


class GFootballMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        default_config = {
            "env_name": "11_vs_11_stochastic", "representation": "simple115v2",
            "rewards": "scoring,checkpoints", "number_of_left_players_agent_controls": 11,
            "number_of_right_players_agent_controls": 11, "stacked": True,
            "logdir": "/tmp/gfootball", "write_goal_dumps": False,
            "write_full_episode_dumps": False, "render": False,
            "write_video": False, "dump_frequency": 1,
        }
        self.env_config = {**default_config, **config}
        self.debug_mode = self.env_config.get("debug_mode", False)
        if self.debug_mode:
            self.env_config.update({"render": True, "write_video": True})

        self.left_players = self.env_config["number_of_left_players_agent_controls"]
        self.right_players = self.env_config["number_of_right_players_agent_controls"]
        
        creation_kwargs = self.env_config.copy()
        creation_kwargs.pop("debug_mode", None)
        self.env = football_env.create_environment(**creation_kwargs)
        
        self.agent_ids = [f"left_{i}" for i in range(self.left_players)] + \
                         [f"right_{i}" for i in range(self.right_players)]
        self._agent_ids = set(self.agent_ids)

        reset_result = self.env.reset()
        test_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        if not self.agent_ids:
            single_obs_shape = test_obs.shape
        elif len(test_obs.shape) > 1:
            single_obs_shape = test_obs[0].shape
        else:
            assert test_obs.shape[0] % len(self.agent_ids) == 0
            single_obs_shape = (test_obs.shape[0] // len(self.agent_ids),)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=single_obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(19)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        return self._split_obs(obs), {aid: {} for aid in self.agent_ids}

    def step(self, action_dict):
        actions = [action_dict.get(aid, 0) for aid in self.agent_ids]
        step_result = self.env.step(actions)
        
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, rewards, done, info = step_result
            terminated, truncated = done, False
        
        dones = {aid: done for aid in self.agent_ids}; dones["__all__"] = done
        truncs = {aid: truncated for aid in self.agent_ids}; truncs["__all__"] = truncated
        
        if self.debug_mode:
            import time; time.sleep(0.05)

        return (self._split_obs(obs), self._split_rewards(rewards), dones, 
                truncs, {aid: info for aid in self.agent_ids})

    def _split_obs(self, obs):
        if not self.agent_ids: return {}
        if len(obs.shape) == 1: obs = obs.reshape(len(self.agent_ids), -1)
        return {self.agent_ids[i]: obs[i].astype(np.float32) for i in range(len(self.agent_ids))}

    def _split_rewards(self, rewards):
        if np.isscalar(rewards): return {aid: float(rewards) for aid in self.agent_ids}
        return {self.agent_ids[i]: float(rewards[i]) for i in range(len(self.agent_ids))}

    def close(self): self.env.close()


def get_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "policy_left" if "left" in agent_id else "policy_right"


def select_policy_to_train(policy_id, worker, **kwargs):
    if hasattr(worker, 'get_metrics'):
        metrics = worker.get_metrics()
        active_policy_value = metrics.get("custom_metrics", {}).get("active_policy", 1.0)
        active_policy = "policy_left" if active_policy_value >= 0.5 else "policy_right"
        return policy_id == active_policy
    return True


def create_impala_config(stage: TrainingStage, debug_mode: bool = False, 
                         hyperparams: dict = None, 
                         policy_pool_dir: str = "policy_pool",
                         max_versions: int = 50,
                         keep_top: int = 10,
                         active_zone: int = 15,
                         auto_prune: bool = True) -> ImpalaConfig:
    config = ImpalaConfig()
    
    env_config = {
        "debug_mode": debug_mode, "representation": stage.representation, "env_name": stage.env_name,
        "number_of_left_players_agent_controls": stage.left_players,
        "number_of_right_players_agent_controls": stage.right_players,
        "rewards": "scoring,checkpoints",
    }
    config.environment("gfootball_multi", env_config=env_config)
    config.framework("torch")
    config.env_runners(num_env_runners=0 if debug_mode else 5, num_envs_per_env_runner=1,
                       num_cpus_per_env_runner=2,
                       rollout_fragment_length=8 if debug_mode else 8)
    
    if hyperparams is None:
        hyperparams = {"lr": 0.0001, "entropy_coeff": 0.008, "vf_loss_coeff": 0.5}



    config.training(
            lr=hyperparams["lr"],
            entropy_coeff=hyperparams["entropy_coeff"],
            vf_loss_coeff=hyperparams["vf_loss_coeff"],
            grad_clip=0.5,
            train_batch_size=1024 if not debug_mode else 8,
            learner_queue_size=1,
        )
    
    use_custom_model = False
    custom_model_config = {
            "custom_model": "GFootballMamba",
            "custom_model_config": {
            "d_model": 128,
            "num_layers": 5,
            "d_state": 16,
            "d_conv": 3,
            "expand": 2,
            "dropout": 0.05,
        }
    }
    
    standard_model_config = {
        "fcnet_hiddens": [256, 256],
        "lstm_cell_size": 256,
    }

    if use_custom_model:
        print("--- Using CUSTOM model: GFootballMamba ---")
        config.model.update(custom_model_config)
    else:
        print("--- Using STANDARD model: IMPALA CNN+LSTM ---")
        config.model.update(standard_model_config)

    config.resources(num_gpus=1/2, num_cpus_for_main_process=2)
    config.learners(num_learners=1, num_gpus_per_learner=1/2, num_cpus_per_learner=1)

    policies = {}
    has_left, has_right = stage.left_players > 0, stage.right_players > 0

    if has_left: policies["policy_left"] = PolicySpec()
    if has_right: policies["policy_right"] = PolicySpec()
        
    policies_to_train_fn = None
    if has_left and has_right:
        policies_to_train_fn = select_policy_to_train
    elif has_left:
        policies_to_train_fn = ["policy_left"]
    else:
        policies_to_train_fn = ["policy_right"]
    
    config.multi_agent(
        policies=policies, 
        policy_mapping_fn=get_policy_mapping_fn, 
        policies_to_train=policies_to_train_fn
    )

    config.callbacks(lambda: EnhancedSelfPlayCallback(
        update_interval=50, 
        version_save_interval=50,
        has_left_agents=has_left, 
        has_right_agents=has_right,
        save_dir=policy_pool_dir,
        min_games_before_rating_update=20,
        max_versions_per_policy=max_versions,
        keep_top_n=keep_top,
        active_zone_size=active_zone,
        auto_prune_enabled=auto_prune
    ))
    
    config.debugging(seed=42, log_level="WARN")
    return config

def short_trial_name_creator(trial):
    return trial.trial_id

def train_single_stage(stage, stage_index, debug_mode, restore_checkpoint):
    print("\n" + "="*80)
    print(f"STARTING STAGE {stage_index + 1}: {stage.name} - {stage.description}")
    if restore_checkpoint:
        print(f"  -> Initializing all trials from: {restore_checkpoint}")
    print("="*80 + "\n")

    results_path = Path(__file__).resolve().parent / "training_results_transfer_pbt"
    policy_pool_dir = results_path / f"{stage.name}_policy_pool"

    param_space = create_impala_config(
        stage=stage,
        debug_mode=debug_mode,
        hyperparams={
            "lr": tune.choice([0.0005, 0.0003, 0.0001]),
            "entropy_coeff": tune.choice([0.006, 0.008, 0.01]),
            "vf_loss_coeff": tune.choice([0.4, 0.5, 0.6]),
        },
        policy_pool_dir=str(policy_pool_dir),
        max_versions=25,
        keep_top=10,
        active_zone=15,
        auto_prune=True
    ).to_dict()

    if restore_checkpoint:
        param_space["_restore_checkpoint_path"] = restore_checkpoint

    metric_path = "env_runners/episode_return_mean"
    stop_criteria = {
        "timesteps_total": stage.max_timesteps,
    }
    
    checkpoint_config = CheckpointConfig(
        num_to_keep=None, checkpoint_frequency=50,
        checkpoint_score_attribute=metric_path,
        checkpoint_score_order="max", checkpoint_at_end=True,
    )
    
    pbt_scheduler = PopulationBasedTraining(
        time_attr="timesteps_total", metric=metric_path,
        mode="max", perturbation_interval=64_000,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-5, 1e-3),
            "entropy_coeff": tune.uniform(0.0, 0.02),
            "vf_loss_coeff": tune.uniform(0.1, 1.0),
        }
    )

    tuner = tune.Tuner(
        "IMPALA",
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=pbt_scheduler, 
            num_samples=2,
            max_concurrent_trials=2,
            trial_dirname_creator=short_trial_name_creator
        ),
        run_config=RunConfig(
            stop=stop_criteria,
            checkpoint_config=checkpoint_config,
            name=f"{stage.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            # name="stage_1_basic_1_20251019_191114",
            storage_path=str(results_path),
        ),
    )
    
    results = tuner.fit()
    if results.experiment_path:
        experiment_path = str(results.experiment_path)
        log_file = results_path / "tensorboard_commands.txt"
        tensorboard_command = f"tensorboard --logdir \"{experiment_path}\""
        
        with open(log_file, "a") as f:
            f.write(f"--- Stage: {stage.name} ---\n")
            f.write(f"{tensorboard_command}\n\n")
        
    best_result = results.get_best_result(metric=metric_path, mode="max")
    
    if best_result and best_result.checkpoint:
        checkpoint_path = str(best_result.checkpoint.path)
        print(f"\nSTAGE {stage_index + 1} COMPLETE. Best checkpoint: {checkpoint_path}")
        return checkpoint_path
    else:
        print(f"\nWarning: No best checkpoint found for stage {stage.name}.")
        return None

def train_progressive(start_stage, end_stage, debug_mode, initial_checkpoint):
    if end_stage is None: end_stage = len(TRAINING_STAGES) - 1
    
    print("\n" + "="*80)
    print("PROGRESSIVE TRAINING WITH AUTO-PRUNING + CHAMPION TRACKING")
    print(f"Stages: {start_stage + 1} to {end_stage + 1} of {len(TRAINING_STAGES)}")
    print("="*80 + "\n")
    
    current_checkpoint = initial_checkpoint
    
    for i in range(start_stage, end_stage + 1):
        stage = TRAINING_STAGES[i]
        checkpoint = train_single_stage(stage, i, debug_mode, current_checkpoint)
        if checkpoint:
            current_checkpoint = checkpoint
        else:
            print(f"Stopping progressive training: Stage {i+1} did not produce a valid checkpoint.")
            break
            
    print("\n" + "="*80)
    print("PROGRESSIVE TRAINING COMPLETE")
    if current_checkpoint:
        print(f"Final Checkpoint Path: {current_checkpoint}")
    print("="*80 + "\n")

def main():
    debug_mode = False
    use_transfer = True
    start_stage = 1
    end_stage = 5
    initial_checkpoint = None
    
    ray.init(ignore_reinit_error=True, log_to_driver=False, local_mode=debug_mode)
    register_env("gfootball_multi", lambda config: GFootballMultiAgentEnv(config))
    ModelCatalog.register_custom_model("GFootballMamba", GFootballMamba)


    if use_transfer:
        train_progressive(start_stage, end_stage, debug_mode, initial_checkpoint)
    else:
        stage = TRAINING_STAGES[start_stage]
        train_single_stage(stage, start_stage, debug_mode, initial_checkpoint)
    
    ray.shutdown()

if __name__ == "__main__":
    main()