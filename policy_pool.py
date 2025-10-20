import numpy as np
from pathlib import Path
from typing import Tuple, List
from datetime import datetime
import pickle
import json

from trueskill import Rating, rate_1vs1, quality_1vs1
from ray.rllib.algorithms.callbacks import DefaultCallbacks

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
                        print(f"  Warning: Could not delete {weights_path}: {e}")
                
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
        
        if policy1_version not in self.ratings[policy1_name]:
             self.ratings[policy1_name][policy1_version] = Rating()
        if policy2_version not in self.ratings[policy2_name]:
             self.ratings[policy2_name][policy2_version] = Rating()

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
        
        if current_version not in self.ratings[policy_name]:
             self.ratings[policy_name][current_version] = Rating()

        current_rating = self.ratings[policy_name][current_version]
        champion_id = self.champions[opponent_name]
        
        qualities = []
        version_ids = []
        
        for vid in active_versions:
            if vid not in self.ratings[opponent_name]:
                self.ratings[opponent_name][vid] = Rating()
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
                if current_ver not in self.ratings[policy_name]:
                    self.ratings[policy_name][current_ver] = Rating()
                rating = self.ratings[policy_name][current_ver]

                stats[f"{policy_name}_mu"] = rating.mu
                stats[f"{policy_name}_sigma"] = rating.sigma
                
                conservative_skill = rating.mu - 3 * rating.sigma
                stats[f"{policy_name}_skill"] = conservative_skill
                
                stats[f"{policy_name}_version"] = current_ver
                stats[f"{policy_name}_num_versions"] = len(self.policy_versions[policy_name])
                
                champion_id = self.champions[policy_name]
                if champion_id is not None:
                    if champion_id not in self.ratings[policy_name]:
                        self.ratings[policy_name][champion_id] = Rating()
                    champion_rating = self.ratings[policy_name][champion_id]
                    champion_skill = champion_rating.mu - 3 * champion_rating.sigma
                    stats[f"{policy_name}_champion_version"] = champion_id
                    stats[f"{policy_name}_champion_skill"] = champion_skill
                    stats[f"{policy_name}_is_champion"] = 1.0 if current_ver == champion_id else 0.0
        
        stats["active_zone_size_left"] = len(self.get_active_versions("policy_left"))
        stats["active_zone_size_right"] = len(self.get_active_versions("policy_right"))
        
        if len(self.match_history) >= 5:
            recent_opponents = [m['policy2'] for m in self.match_history[-10:] if 'policy2' in m]
            unique_opponents = len(set(recent_opponents))
            if recent_opponents:
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
        try:
            with open(state_path, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save pool state: {e}")

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
        
        if left_rewards or right_rewards:
            left_total = sum(left_rewards) / len(left_rewards) if left_rewards else 0.0
            right_total = sum(right_rewards) / len(right_rewards) if right_rewards else 0.0
            
            if left_rewards and right_rewards:
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
                    
                    if version_info and version_info['weights_path']:
                        weights_path = version_info['weights_path']
                        
                        try:
                            with open(weights_path, 'rb') as f:
                                opponent_weights = pickle.load(f)
                            
                            opponent_policy = algorithm.get_policy(opponent_name)
                            if opponent_policy:
                                opponent_policy.set_weights(opponent_weights)
                        except Exception as e:
                            print(f"  ✗ Warning: Could not load opponent weights: {e}\n")
                    elif opponent_version != 0:
                         print(f"  ✗ Warning: No weights path found for opponent {opponent_name} v{opponent_version}\n")

        if self.self_play_enabled:
            result.setdefault("custom_metrics", {})
            result["custom_metrics"]["active_policy"] = 1.0 if self.active_policy_name == "policy_left" else 0.0
            
            rating_stats = self.policy_pool.get_rating_stats()
            result["custom_metrics"].update(rating_stats)
        
        self._update_counter += 1