from pathlib import Path
from typing import Any, Dict, Optional, Type
import numpy as np
import os
import json
from collections import defaultdict
from gymnasium import spaces
import gfootball.env as football_env
import warnings

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.learner.learner import Learner
from ray.rllib.utils.annotations import override
import torch

from model_3 import GFootballMambaRLModule


class PPOTorchLearnerNoForeach(PPOTorchLearner):
    @override(Learner)
    def configure_optimizers_for_module(self, module_id, config=None):
        module = self._module[module_id]
        params = list(module.parameters())
        lr = self.config.lr if self.config else 5e-5
        optimizer = torch.optim.Adam(params, lr=lr, foreach=False)
        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=params,
            lr_or_lr_schedule=lr,
        )


class PPONoForeach(PPO):
    @classmethod
    @override(PPO)
    def get_default_config(cls):
        return PPOConfigNoForeach()


class PPOConfigNoForeach(PPOConfig):
    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or PPONoForeach)
    
    @override(PPOConfig)
    def get_default_learner_class(self) -> Type[Learner]:
        return PPOTorchLearnerNoForeach


CALIBRATE_ONLY = False
CALIBRATION_EPISODES = 100
CALIBRATION_WORKERS = 8
NUM_SAMPLES = 2
STOP_ITERATIONS = 500
EVAL_INTERVAL = 20
EVAL_EPISODES_PER_STAGE = 5


class StageConfig:
    def __init__(self, stage_id: int, env_name: str, representation: str,
                 left_agents: int = 1, right_agents: int = 0, max_steps: int = 3000):
        self.stage_id = stage_id
        self.env_name = env_name
        self.representation = representation
        self.left_agents = left_agents
        self.right_agents = right_agents
        self.max_steps = max_steps


STAGE_CONFIGS = [
    StageConfig(0, "academy_empty_goal_close", "simple115v2", 1, 0, 400),
    StageConfig(1, "academy_run_to_score_with_keeper", "simple115v2", 1, 0, 400),
    StageConfig(2, "academy_pass_and_shoot_with_keeper", "simple115v2", 1, 0, 400),
    StageConfig(3, "academy_3_vs_1_with_keeper", "simple115v2", 3, 0, 400),
    StageConfig(4, "academy_single_goal_versus_lazy", "simple115v2", 3, 0, 1000),
    StageConfig(5, "11_vs_11_easy_stochastic", "simple115v2", 3, 0, 3000),
    StageConfig(6, "11_vs_11_easy_stochastic", "simple115v2", 5, 0, 3000),
    StageConfig(7, "11_vs_11_stochastic", "simple115v2", 11, 0, 3000),
]


class StageBaseline:
    def __init__(self, stage_id: int):
        self.stage_id = stage_id
        self.episode_return_mean = 0.0
        self.episode_return_std = 1.0
        self.episode_length_mean = 100.0
        self.step_reward_mean = 0.0
        self.step_reward_std = 0.01
        self.win_rate = 0.0
        self.calibrated = False
        self.calibration_episodes = 0
        
    def normalize_episode_return(self, raw_return: float) -> float:
        if not self.calibrated:
            return raw_return
        improvement = raw_return - self.episode_return_mean
        if self.episode_return_std > 1e-6:
            return improvement / self.episode_return_std
        return improvement
    
    def normalize_step_reward(self, raw_reward: float) -> float:
        if not self.calibrated:
            return raw_reward
        improvement = raw_reward - self.step_reward_mean
        if self.step_reward_std > 1e-6:
            return improvement / self.step_reward_std
        return improvement
    
    def to_dict(self) -> Dict:
        return {
            "stage_id": self.stage_id,
            "episode_return_mean": self.episode_return_mean,
            "episode_return_std": self.episode_return_std,
            "episode_length_mean": self.episode_length_mean,
            "step_reward_mean": self.step_reward_mean,
            "step_reward_std": self.step_reward_std,
            "win_rate": self.win_rate,
            "calibrated": self.calibrated,
            "calibration_episodes": self.calibration_episodes
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> "StageBaseline":
        b = cls(d["stage_id"])
        b.episode_return_mean = d.get("episode_return_mean", 0.0)
        b.episode_return_std = d.get("episode_return_std", 1.0)
        b.episode_length_mean = d.get("episode_length_mean", 100.0)
        b.step_reward_mean = d.get("step_reward_mean", 0.0)
        b.step_reward_std = d.get("step_reward_std", 0.01)
        b.win_rate = d.get("win_rate", 0.0)
        b.calibrated = d.get("calibrated", False)
        b.calibration_episodes = d.get("calibration_episodes", 0)
        return b


def ensure_baseline_objects(baselines_input) -> Dict[int, StageBaseline]:
    if not baselines_input:
        return {i: StageBaseline(i) for i in range(len(STAGE_CONFIGS))}
    
    result = {}
    for k, v in baselines_input.items():
        k_int = int(k)
        if isinstance(v, StageBaseline):
            result[k_int] = v
        elif isinstance(v, dict):
            result[k_int] = StageBaseline.from_dict(v)
        else:
            raise TypeError(f"Invalid baseline type for stage {k}: {type(v)}")
    return result


@ray.remote
def calibrate_episode_batch(stage_id: int, num_episodes: int, worker_id: int) -> Dict:
    config = STAGE_CONFIGS[stage_id]
    
    env_kwargs = {
        "env_name": config.env_name,
        "representation": config.representation,
        "number_of_left_players_agent_controls": config.left_agents,
        "number_of_right_players_agent_controls": config.right_agents,
        "write_goal_dumps": False,
        "write_full_episode_dumps": False,
        "render": False,
        "write_video": False,
        "dump_frequency": 0,
    }
    
    env = football_env.create_environment(**env_kwargs)
    
    episode_returns = []
    episode_lengths = []
    all_step_rewards = []
    wins = []
    
    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        ep_return = 0.0
        ep_step_rewards = []
        steps = 0
        
        while not done and steps < config.max_steps:
            if isinstance(env.action_space, list):
                actions = [space.sample() for space in env.action_space]
            else:
                actions = env.action_space.sample()
                
            obs, reward, done, info = env.step(actions)
            
            if isinstance(reward, (list, np.ndarray)):
                step_reward = float(sum(reward))
            else:
                step_reward = float(reward)
                
            ep_step_rewards.append(step_reward)
            ep_return += step_reward
            steps += 1
            
        episode_returns.append(ep_return)
        episode_lengths.append(steps)
        all_step_rewards.extend(ep_step_rewards)
        
        won = False
        if isinstance(info, dict) and "score" in info:
            won = info["score"][0] > info["score"][1]
        else:
            won = ep_return > 0
        wins.append(1.0 if won else 0.0)
            
    env.close()
    
    return {
        "stage_id": stage_id,
        "worker_id": worker_id,
        "episode_returns": episode_returns,
        "episode_lengths": episode_lengths,
        "step_rewards": all_step_rewards,
        "wins": wins
    }


class BaselineCalibrator:
    def __init__(self, save_path: Optional[Path] = None):
        self.baselines = {i: StageBaseline(i) for i in range(len(STAGE_CONFIGS))}
        self.save_path = save_path or Path("stage_baselines.json")
        
    def save(self):
        data = {str(k): v.to_dict() for k, v in self.baselines.items()}
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)
            
    def load(self) -> bool:
        if not self.save_path.exists():
            return False
        try:
            with open(self.save_path, "r") as f:
                data = json.load(f)
            for k, v in data.items():
                self.baselines[int(k)] = StageBaseline.from_dict(v)
            return all(b.calibrated for b in self.baselines.values())
        except Exception as e:
            print(f"Failed to load baselines: {e}")
            return False
    
    def calibrate_stage(self, stage_id: int, num_episodes: int = 50) -> StageBaseline:
        config = STAGE_CONFIGS[stage_id]
        
        env_kwargs = {
            "env_name": config.env_name,
            "representation": config.representation,
            "number_of_left_players_agent_controls": config.left_agents,
            "number_of_right_players_agent_controls": config.right_agents,
            "write_goal_dumps": False,
            "write_full_episode_dumps": False,
            "render": False,
            "write_video": False,
            "dump_frequency": 0,
        }
        
        env = football_env.create_environment(**env_kwargs)
        
        episode_returns = []
        episode_lengths = []
        all_step_rewards = []
        wins = []
        
        for ep in range(num_episodes):
            obs = env.reset()
            done = False
            ep_return = 0.0
            ep_step_rewards = []
            steps = 0
            initial_score = None
            
            while not done and steps < config.max_steps:
                if isinstance(env.action_space, list):
                    actions = [space.sample() for space in env.action_space]
                else:
                    actions = env.action_space.sample()
                    
                obs, reward, done, info = env.step(actions)
                
                if initial_score is None and isinstance(info, dict) and "score" in info:
                    initial_score = list(info["score"])
                
                if isinstance(reward, (list, np.ndarray)):
                    step_reward = float(sum(reward))
                else:
                    step_reward = float(reward)
                    
                ep_step_rewards.append(step_reward)
                ep_return += step_reward
                steps += 1
                
            episode_returns.append(ep_return)
            episode_lengths.append(steps)
            all_step_rewards.extend(ep_step_rewards)
            
            won = False
            if isinstance(info, dict) and "score" in info:
                final_score = info["score"]
                won = final_score[0] > final_score[1]
            else:
                won = ep_return > 0
            wins.append(1.0 if won else 0.0)
                
        env.close()
        
        baseline = self.baselines[stage_id]
        baseline.episode_return_mean = float(np.mean(episode_returns))
        baseline.episode_return_std = float(max(np.std(episode_returns), 0.01))
        baseline.episode_length_mean = float(np.mean(episode_lengths))
        baseline.step_reward_mean = float(np.mean(all_step_rewards)) if all_step_rewards else 0.0
        baseline.step_reward_std = float(max(np.std(all_step_rewards), 0.001)) if all_step_rewards else 0.01
        baseline.win_rate = float(np.mean(wins))
        baseline.calibrated = True
        baseline.calibration_episodes = num_episodes
        
        print(f"Stage {stage_id} ({config.env_name}, {config.left_agents} agents):")
        print(f"  Episode Return: {baseline.episode_return_mean:.4f} +/- {baseline.episode_return_std:.4f}")
        print(f"  Episode Length: {baseline.episode_length_mean:.1f}")
        print(f"  Step Reward:    {baseline.step_reward_mean:.6f} +/- {baseline.step_reward_std:.6f}")
        print(f"  Win Rate:       {baseline.win_rate:.2%}")
        
        return baseline
    
    def calibrate_all(self, num_episodes_short: int = 50, num_episodes_long: int = 10):
        print("=" * 60)
        print("BASELINE CALIBRATION")
        print("=" * 60)
        
        for stage_id in range(len(STAGE_CONFIGS)):
            stage = STAGE_CONFIGS[stage_id]
            if stage.max_steps >= 1000:
                num_ep = num_episodes_long
            else:
                num_ep = num_episodes_short
            print(f"Calibrating stage {stage_id} with {num_ep} episodes...")
            self.calibrate_stage(stage_id, num_ep)
            
        self.save()
        print("=" * 60)
        print(f"Baselines saved to {self.save_path}")
        print("=" * 60)
        
        return self.baselines
    
    def calibrate_all_parallel(self, num_episodes: int = 100, num_workers: int = 8):
        print("=" * 60)
        print(f"PARALLEL BASELINE CALIBRATION ({num_workers} workers, {num_episodes} episodes/stage)")
        print("=" * 60)
        
        episodes_per_worker = max(1, num_episodes // num_workers)
        
        futures = []
        for stage_id in range(len(STAGE_CONFIGS)):
            for worker_id in range(num_workers):
                future = calibrate_episode_batch.remote(stage_id, episodes_per_worker, worker_id)
                futures.append(future)
                
        print(f"Launched {len(futures)} calibration tasks...")
        
        stage_data = {sid: {"returns": [], "lengths": [], "steps": [], "wins": []} 
                      for sid in range(len(STAGE_CONFIGS))}
        
        completed = 0
        total = len(futures)
        
        while futures:
            done, futures = ray.wait(futures, num_returns=1, timeout=None)
            for ref in done:
                result = ray.get(ref)
                sid = result["stage_id"]
                stage_data[sid]["returns"].extend(result["episode_returns"])
                stage_data[sid]["lengths"].extend(result["episode_lengths"])
                stage_data[sid]["steps"].extend(result["step_rewards"])
                stage_data[sid]["wins"].extend(result["wins"])
                completed += 1
                print(f"  Progress: {completed}/{total} tasks completed", end="\r")
                
        print()
        
        for stage_id in range(len(STAGE_CONFIGS)):
            data = stage_data[stage_id]
            config = STAGE_CONFIGS[stage_id]
            baseline = self.baselines[stage_id]
            
            baseline.episode_return_mean = float(np.mean(data["returns"]))
            baseline.episode_return_std = float(max(np.std(data["returns"]), 0.01))
            baseline.episode_length_mean = float(np.mean(data["lengths"]))
            baseline.step_reward_mean = float(np.mean(data["steps"])) if data["steps"] else 0.0
            baseline.step_reward_std = float(max(np.std(data["steps"]), 0.001)) if data["steps"] else 0.01
            baseline.win_rate = float(np.mean(data["wins"]))
            baseline.calibrated = True
            baseline.calibration_episodes = len(data["returns"])
            
            print(f"Stage {stage_id} ({config.env_name}, {config.left_agents} agents):")
            print(f"  Episodes: {baseline.calibration_episodes}")
            print(f"  Episode Return: {baseline.episode_return_mean:.4f} +/- {baseline.episode_return_std:.4f}")
            print(f"  Episode Length: {baseline.episode_length_mean:.1f}")
            print(f"  Step Reward:    {baseline.step_reward_mean:.6f} +/- {baseline.step_reward_std:.6f}")
            print(f"  Win Rate:       {baseline.win_rate:.2%}")
            
        self.save()
        print("=" * 60)
        print(f"Baselines saved to {self.save_path}")
        print("=" * 60)
        
        return self.baselines


class StageMetrics:
    def __init__(self, stage_id: int, baseline: StageBaseline, ema_alpha: float = 0.05):
        self.stage_id = stage_id
        self.baseline = baseline
        self.ema_alpha = ema_alpha
        
        self.normalized_ema = 0.0
        self.normalized_min = float('inf')
        self.normalized_max = float('-inf')
        self.win_rate_ema = baseline.win_rate
        self.raw_ema = baseline.episode_return_mean
        self.episode_count = 0
        
    def update(self, raw_return: float, won: bool):
        normalized = self.baseline.normalize_episode_return(raw_return)
        win_val = 1.0 if won else 0.0
        
        if self.episode_count == 0:
            self.normalized_ema = normalized
            self.win_rate_ema = win_val
            self.raw_ema = raw_return
        else:
            self.normalized_ema = (1 - self.ema_alpha) * self.normalized_ema + self.ema_alpha * normalized
            self.win_rate_ema = (1 - self.ema_alpha) * self.win_rate_ema + self.ema_alpha * win_val
            self.raw_ema = (1 - self.ema_alpha) * self.raw_ema + self.ema_alpha * raw_return
            
        self.episode_count += 1
        
        if normalized > self.normalized_max:
            self.normalized_max = normalized
        if normalized < self.normalized_min:
            self.normalized_min = normalized
            
    def get_retention(self) -> float:
        if self.episode_count < 5:
            return 1.0
        if self.normalized_max <= self.normalized_min:
            return 1.0
            
        range_val = self.normalized_max - self.normalized_min
        current_pos = self.normalized_ema - self.normalized_min
        return max(0.0, min(1.0, current_pos / range_val))


class UnifiedMetricsTracker:
    def __init__(self, baselines: Dict[int, StageBaseline], ema_alpha: float = 0.05):
        self.baselines = ensure_baseline_objects(baselines)
        self.stage_metrics = {
            sid: StageMetrics(sid, self.baselines[sid], ema_alpha) 
            for sid in range(len(STAGE_CONFIGS))
        }
        
    def update(self, stage_id: int, raw_return: float, won: bool):
        self.stage_metrics[stage_id].update(raw_return, won)
        
    def get_normalized_ema(self, stage_id: int) -> float:
        return self.stage_metrics[stage_id].normalized_ema
    
    def get_retention(self, stage_id: int) -> float:
        return self.stage_metrics[stage_id].get_retention()
    
    def get_all_metrics(self) -> Dict[str, float]:
        metrics = {}
        active_stages = []
        
        for sid in range(len(STAGE_CONFIGS)):
            sm = self.stage_metrics[sid]
            bl = self.baselines[sid]
            
            metrics[f"stage_{sid}/normalized_ema"] = sm.normalized_ema
            metrics[f"stage_{sid}/normalized_max"] = sm.normalized_max if sm.normalized_max > float('-inf') else 0.0
            metrics[f"stage_{sid}/raw_ema"] = sm.raw_ema
            metrics[f"stage_{sid}/win_rate_ema"] = sm.win_rate_ema
            metrics[f"stage_{sid}/retention"] = sm.get_retention()
            metrics[f"stage_{sid}/episode_count"] = sm.episode_count
            metrics[f"stage_{sid}/baseline_return"] = bl.episode_return_mean
            metrics[f"stage_{sid}/baseline_win"] = bl.win_rate
            metrics[f"stage_{sid}/win_improvement"] = sm.win_rate_ema - bl.win_rate
            
            if sm.episode_count > 0:
                active_stages.append(sid)
                
        if active_stages:
            retentions = [self.stage_metrics[sid].get_retention() for sid in active_stages]
            normalized_vals = [self.stage_metrics[sid].normalized_ema for sid in active_stages]
            win_rates = [self.stage_metrics[sid].win_rate_ema for sid in active_stages]
            
            metrics["curriculum/min_retention"] = min(retentions)
            metrics["curriculum/mean_retention"] = float(np.mean(retentions))
            metrics["curriculum/mean_normalized"] = float(np.mean(normalized_vals))
            metrics["curriculum/min_normalized"] = min(normalized_vals)
            metrics["curriculum/mean_win_rate"] = float(np.mean(win_rates))
            metrics["curriculum/num_active_stages"] = len(active_stages)
            metrics["curriculum/worst_retention_stage"] = active_stages[int(np.argmin(retentions))]
        else:
            metrics["curriculum/min_retention"] = 1.0
            metrics["curriculum/mean_retention"] = 1.0
            metrics["curriculum/mean_normalized"] = 0.0
            metrics["curriculum/min_normalized"] = 0.0
            metrics["curriculum/mean_win_rate"] = 0.0
            metrics["curriculum/num_active_stages"] = 0
            metrics["curriculum/worst_retention_stage"] = -1
            
        return metrics


class GFootballMultiAgentEnv(MultiAgentEnv):
    EXPECTED_OBS_SIZES = {115, 460}
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        workspace_root = Path(__file__).resolve().parent
        gf_logdir = workspace_root / "gfootball_logs"
        gf_logdir.mkdir(parents=True, exist_ok=True)
        
        self.mode = config.get("mode", "train")
        self.eval_stage_id = config.get("eval_stage_id", None)
        self.curriculum_horizon = config.get("curriculum_horizon", 100000)
        self.sample_method = config.get("sample_method", "retention")
        
        self.baselines = ensure_baseline_objects(config.get("baselines", {}))
        self.metrics_tracker = UnifiedMetricsTracker(self.baselines)
        
        self.stages = STAGE_CONFIGS
        self.episode_idx = 0
        self.current_stage = None
        self.current_stage_id = -1
        
        self.env_base_config = {
            "write_goal_dumps": False,
            "write_full_episode_dumps": False,
            "render": config.get("debug_mode", False),
            "write_video": False,
            "dump_frequency": 0,
            "logdir": str(gf_logdir),
            "other_config_options": {}
        }
            
        self.env = None
        
        self.agent_ids = [f"left_{i}" for i in range(11)] + [f"right_{i}" for i in range(11)]
        self._agent_ids = set(self.agent_ids)
        
        low_f32 = np.float32(-np.inf)
        high_f32 = np.float32(np.inf)
        self._single_observation_space = spaces.Dict({
            "obs": spaces.Box(low=low_f32, high=high_f32, shape=(4, 115), dtype=np.float32),
            "stage_index": spaces.Box(low=0, high=len(self.stages), shape=(1,), dtype=np.int32)
        })
        self._single_action_space = spaces.Discrete(19)

        self.observation_space = spaces.Dict({
            aid: self._single_observation_space for aid in self.agent_ids
        })
        self.action_space = spaces.Dict({
            aid: self._single_action_space for aid in self.agent_ids
        })
        
        self.latest_obs = {}
        self.episode_rewards = []
        self.step_count = 0
        self._create_env_for_stage(self.stages[0])

    def _create_env_for_stage(self, stage: StageConfig):
        if self.env is not None:
            self.env.close()
            self.env = None
            
        kwargs = self.env_base_config.copy()
        kwargs["env_name"] = stage.env_name
        kwargs["representation"] = stage.representation
        kwargs["number_of_left_players_agent_controls"] = stage.left_agents
        kwargs["number_of_right_players_agent_controls"] = stage.right_agents
        
        self.env = football_env.create_environment(**kwargs)
        self.current_stage = stage
        self.current_stage_id = stage.stage_id
        self.active_agents = [f"left_{i}" for i in range(stage.left_agents)] + \
                             [f"right_{i}" for i in range(stage.right_agents)]

    def _sample_stage(self) -> StageConfig:
        if self.curriculum_horizon <= 0:
            max_stage = len(self.stages) - 1
        else:
            progress = min(1.0, self.episode_idx / self.curriculum_horizon)
            max_stage = int(progress * (len(self.stages) - 1))
        available_ids = list(range(max_stage + 1))
        
        if len(available_ids) == 1:
            return self.stages[0]
            
        if self.sample_method == "uniform":
            return self.stages[np.random.choice(available_ids)]
            
        if self.sample_method == "progressive":
            weights = np.zeros(len(available_ids))
            weights[-1] = 0.5
            weights[:-1] = 0.5 / max(len(available_ids) - 1, 1)
            weights /= weights.sum()
            return self.stages[np.random.choice(available_ids, p=weights)]
            
        weights = []
        for sid in available_ids:
            sm = self.metrics_tracker.stage_metrics[sid]
            if sm.episode_count < 10:
                w = 2.0
            else:
                retention = sm.get_retention()
                w = 2.0 - retention
            weights.append(max(w, 0.1))
            
        weights = np.array(weights)
        weights /= weights.sum()
        return self.stages[np.random.choice(available_ids, p=weights)]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if self.mode == "eval" and self.eval_stage_id is not None:
            target_stage = self.stages[self.eval_stage_id]
        else:
            target_stage = self._sample_stage()
                
        if target_stage.stage_id != self.current_stage_id:
            self._create_env_for_stage(target_stage)
            
        self.episode_idx += 1
        self.episode_rewards = []
        self.step_count = 0
        
        raw_obs = self.env.reset()
        self.latest_obs = self._process_obs(raw_obs)
        info = {aid: {"stage_id": self.current_stage_id} for aid in self.active_agents}
        return self.latest_obs, info

    def step(self, action_dict: Dict[str, int]):
        internal_limit = 19
        if hasattr(self.env, 'action_space'):
            if hasattr(self.env.action_space, 'n'):
                internal_limit = self.env.action_space.n
            elif isinstance(self.env.action_space, list) and len(self.env.action_space) > 0:
                if hasattr(self.env.action_space[0], 'n'):
                    internal_limit = self.env.action_space[0].n

        actions_list = []
        for aid in self.active_agents:
            act = action_dict.get(aid, 0)
            if hasattr(act, "item"):
                act = act.item()
            act = int(act)
            if act >= internal_limit:
                act = 0
            actions_list.append(act)
            
        obs, reward, done, info = self.env.step(actions_list)
        self.latest_obs = self._process_obs(obs)
        self.step_count += 1
        
        if isinstance(reward, (list, np.ndarray)):
            raw_step_reward = float(sum(reward))
        else:
            raw_step_reward = float(reward)
            
        self.episode_rewards.append(raw_step_reward)
        
        baseline = self.baselines[self.current_stage_id]
        normalized_step_reward = baseline.normalize_step_reward(raw_step_reward)
        per_agent_reward = normalized_step_reward / len(self.active_agents)
        rewards_dict = {aid: per_agent_reward for aid in self.active_agents}
        
        terminated = bool(done)
        truncated = self.step_count >= self.current_stage.max_steps
        episode_done = terminated or truncated
        
        dones = {aid: episode_done for aid in self.active_agents}
        dones["__all__"] = episode_done
        truncs = {aid: truncated and not terminated for aid in self.active_agents}
        truncs["__all__"] = truncated and not terminated
        
        raw_episode_return = sum(self.episode_rewards)
        normalized_episode_return = baseline.normalize_episode_return(raw_episode_return)
        
        won = False
        if isinstance(info, dict) and "score" in info:
            won = info["score"][0] > info["score"][1]
        else:
            won = raw_episode_return > 0
            
        if episode_done and self.mode == "train":
            self.metrics_tracker.update(self.current_stage_id, raw_episode_return, won)
            
        agent_infos = {}
        for aid in self.active_agents:
            agent_infos[aid] = {
                "stage_id": self.current_stage_id,
                "stage_name": self.current_stage.env_name,
                "num_agents": len(self.active_agents),
                "raw_step_reward": raw_step_reward,
                "normalized_step_reward": normalized_step_reward,
            }
            if episode_done:
                agent_infos[aid]["raw_episode_return"] = raw_episode_return
                agent_infos[aid]["normalized_episode_return"] = normalized_episode_return
                agent_infos[aid]["episode_length"] = self.step_count
                agent_infos[aid]["won"] = won
                agent_infos[aid]["baseline_episode_mean"] = baseline.episode_return_mean
                agent_infos[aid]["baseline_step_mean"] = baseline.step_reward_mean
                
        return self.latest_obs, rewards_dict, dones, truncs, agent_infos

    def _process_obs(self, raw_obs):
        if not isinstance(raw_obs, np.ndarray):
            raw_obs = np.array(raw_obs)
            
        if raw_obs.ndim == 1:
            if raw_obs.size in self.EXPECTED_OBS_SIZES:
                raw_obs = raw_obs.reshape(1, -1)
            else:
                warnings.warn(f"Unexpected 1D obs size: {raw_obs.size}, expected one of {self.EXPECTED_OBS_SIZES}")
                raw_obs = np.zeros((len(self.active_agents), 115), dtype=np.float32)
                
        obs_dict = {}
        for i, aid in enumerate(self.active_agents):
            if i < len(raw_obs):
                data = raw_obs[i]
            else:
                warnings.warn(f"Missing obs for agent {aid}, using zeros")
                data = np.zeros(115, dtype=np.float32)
                
            if data.size == 115:
                data = np.tile(data.reshape(1, 115), (4, 1)).astype(np.float32)
            elif data.size == 460:
                data = data.reshape(4, 115).astype(np.float32)
            else:
                warnings.warn(f"Unexpected obs size {data.size} for agent {aid}, expected 115 or 460")
                data = np.zeros((4, 115), dtype=np.float32)
                
            obs_dict[aid] = {
                "obs": data,
                "stage_index": np.array([self.current_stage_id], dtype=np.int32)
            }
        return obs_dict

    def get_metrics(self) -> Dict[str, float]:
        return self.metrics_tracker.get_all_metrics()

    def close(self):
        if self.env:
            self.env.close()


class CurriculumCallback(RLlibCallback):
    def __init__(self, baselines: Dict[int, StageBaseline]):
        super().__init__()
        self.baselines = ensure_baseline_objects(baselines)
        
    def on_episode_end(self, *, episode, env_runner, metrics_logger, env, env_index, rl_module, **kwargs):
        last_infos = episode.get_infos(-1)
        
        if not last_infos:
            return
        
        last_info = None
        for agent_id, agent_info in last_infos.items():
            if isinstance(agent_info, dict) and "raw_episode_return" in agent_info:
                last_info = agent_info
                break
                        
        if last_info is None:
            return
            
        stage_id = last_info.get("stage_id", -1)
        raw_return = last_info.get("raw_episode_return", 0.0)
        normalized_return = last_info.get("normalized_episode_return", 0.0)
        won = last_info.get("won", False)
        episode_length = last_info.get("episode_length", 0)
        
        metrics_logger.log_value(f"episode/stage_{stage_id}_raw", raw_return, reduce="mean")
        metrics_logger.log_value(f"episode/stage_{stage_id}_normalized", normalized_return, reduce="mean")
        metrics_logger.log_value(f"episode/stage_{stage_id}_won", float(won), reduce="mean")
        metrics_logger.log_value(f"episode/stage_{stage_id}_length", float(episode_length), reduce="mean")
        
        metrics_logger.log_value("episode/all_stages_raw", raw_return, reduce="mean")
        metrics_logger.log_value("episode/all_stages_normalized", normalized_return, reduce="mean")
        metrics_logger.log_value("episode/all_stages_won", float(won), reduce="mean")

    def on_train_result(self, *, algorithm, result, **kwargs):
        env_runner_results = result.get("env_runners", {})
        
        stage_data = {}
        for stage_id in range(len(STAGE_CONFIGS)):
            raw_key = f"episode/stage_{stage_id}_raw"
            norm_key = f"episode/stage_{stage_id}_normalized"
            won_key = f"episode/stage_{stage_id}_won"
            len_key = f"episode/stage_{stage_id}_length"
            
            if raw_key in env_runner_results:
                stage_data[stage_id] = {
                    "raw": env_runner_results.get(raw_key, 0.0),
                    "normalized": env_runner_results.get(norm_key, 0.0),
                    "won": env_runner_results.get(won_key, 0.0),
                    "length": env_runner_results.get(len_key, 0.0),
                }
                result[f"stage_{stage_id}/raw"] = stage_data[stage_id]["raw"]
                result[f"stage_{stage_id}/normalized"] = stage_data[stage_id]["normalized"]
                result[f"stage_{stage_id}/win_rate"] = stage_data[stage_id]["won"]
        
        if stage_data:
            norm_vals = [d["normalized"] for d in stage_data.values()]
            win_vals = [d["won"] for d in stage_data.values()]
            raw_vals = [d["raw"] for d in stage_data.values()]
            
            result["curriculum/mean_normalized"] = float(np.mean(norm_vals))
            result["curriculum/mean_win_rate"] = float(np.mean(win_vals))
            result["curriculum/mean_raw"] = float(np.mean(raw_vals))
            result["curriculum/num_active_stages"] = len(stage_data)
        else:
            raw_mean = env_runner_results.get("episode/all_stages_raw", 0.0)
            norm_mean = env_runner_results.get("episode/all_stages_normalized", 0.0)
            win_mean = env_runner_results.get("episode/all_stages_won", 0.0)
            
            result["curriculum/mean_normalized"] = norm_mean
            result["curriculum/mean_win_rate"] = win_mean
            result["curriculum/mean_raw"] = raw_mean
        
        norm = result.get("curriculum/mean_normalized", 0.0)
        win = result.get("curriculum/mean_win_rate", 0.0)
        result["pbt_metric"] = 0.5 * np.tanh(norm / 2.0) + 0.5 * win


class PeriodicEvalCallback(CurriculumCallback):
    def __init__(self, baselines: Dict[int, StageBaseline], 
                 eval_interval: int = 20, eval_episodes: int = 5):
        super().__init__(baselines)
        self.eval_interval = eval_interval
        self.eval_episodes = eval_episodes
        
    def on_train_result(self, *, algorithm, result, **kwargs):
        super().on_train_result(algorithm=algorithm, result=result, **kwargs)
        
        iteration = result.get("training_iteration", 0)
        
        if iteration > 0 and iteration % self.eval_interval == 0:
            eval_results = self._run_full_evaluation(algorithm)
            for key, value in eval_results.items():
                result[f"eval/{key}"] = value

    def _run_full_evaluation(self, algorithm) -> Dict[str, float]:
        results = {}
        
        for stage_id in range(len(STAGE_CONFIGS)):
            stage_results = self._eval_stage(algorithm, stage_id)
            results[f"stage_{stage_id}_raw"] = stage_results["raw_mean"]
            results[f"stage_{stage_id}_normalized"] = stage_results["normalized_mean"]
            results[f"stage_{stage_id}_win_rate"] = stage_results["win_rate"]
            results[f"stage_{stage_id}_length"] = stage_results["length_mean"]
            
        all_normalized = [results[f"stage_{i}_normalized"] for i in range(len(STAGE_CONFIGS))]
        all_wins = [results[f"stage_{i}_win_rate"] for i in range(len(STAGE_CONFIGS))]
        
        results["overall_normalized_mean"] = float(np.mean(all_normalized))
        results["overall_normalized_min"] = float(np.min(all_normalized))
        results["overall_win_mean"] = float(np.mean(all_wins))
        results["overall_win_min"] = float(np.min(all_wins))
        
        return results

    def _eval_stage(self, algorithm, stage_id: int) -> Dict[str, float]:
        import torch
        from ray.rllib.core.columns import Columns
        
        eval_env = GFootballMultiAgentEnv({
            "mode": "eval",
            "eval_stage_id": stage_id,
            "baselines": {k: v.to_dict() for k, v in self.baselines.items()}
        })
        
        rl_module = algorithm.get_module("policy_left")
        
        raw_returns = []
        normalized_returns = []
        wins = []
        lengths = []
        
        for ep in range(self.eval_episodes):
            obs, info = eval_env.reset()
            done = False
            
            while not done:
                actions = {}
                for agent_id in eval_env.active_agents:
                    if agent_id in obs:
                        agent_obs = obs[agent_id]
                        
                        obs_tensor = {
                            "obs": torch.from_numpy(agent_obs["obs"]).unsqueeze(0).float(),
                            "stage_index": torch.from_numpy(agent_obs["stage_index"]).unsqueeze(0)
                        }
                        
                        with torch.no_grad():
                            fwd_out = rl_module.forward_inference({Columns.OBS: obs_tensor})
                        
                        if Columns.ACTIONS in fwd_out:
                            action = fwd_out[Columns.ACTIONS].squeeze(0).cpu().numpy()
                        else:
                            action_dist_inputs = fwd_out[Columns.ACTION_DIST_INPUTS]
                            action = torch.argmax(action_dist_inputs, dim=-1).squeeze(0).cpu().numpy()
                        
                        if hasattr(action, "item"):
                            action = action.item()
                        actions[agent_id] = int(action)
                        
                obs, rewards, dones, truncs, infos = eval_env.step(actions)
                done = dones.get("__all__", False)
                
            last_info = None
            for agent_info in infos.values():
                if isinstance(agent_info, dict) and "raw_episode_return" in agent_info:
                    last_info = agent_info
                    break
                    
            if last_info is not None:
                raw_returns.append(last_info["raw_episode_return"])
                normalized_returns.append(last_info["normalized_episode_return"])
                wins.append(float(last_info["won"]))
                lengths.append(last_info["episode_length"])
            else:
                warnings.warn(f"No episode info found in eval for stage {stage_id}")
                
        eval_env.close()
        
        return {
            "raw_mean": float(np.mean(raw_returns)) if raw_returns else 0.0,
            "normalized_mean": float(np.mean(normalized_returns)) if normalized_returns else 0.0,
            "win_rate": float(np.mean(wins)) if wins else 0.0,
            "length_mean": float(np.mean(lengths)) if lengths else 0.0
        }


def policy_mapping_fn(agent_id, episode=None, **kwargs):
    return "policy_left" if agent_id.startswith("left") else "policy_right"


def make_callback_class(baselines: Dict[int, StageBaseline]):
    class ConfiguredCallback(PeriodicEvalCallback):
        def __init__(self):
            super().__init__(
                baselines=baselines,
                eval_interval=EVAL_INTERVAL,
                eval_episodes=EVAL_EPISODES_PER_STAGE
            )
    return ConfiguredCallback


def create_config(model_config: Dict, baselines: Dict[int, StageBaseline],
                  num_env_runners: int = 11, train_batch_size: int = 4000):
    
    baselines_serializable = {k: v.to_dict() for k, v in baselines.items()}
    
    dummy = GFootballMultiAgentEnv({"baselines": baselines_serializable})
    single_obs_space = dummy._single_observation_space
    single_act_space = dummy._single_action_space
    dummy.close()
    
    rl_spec = MultiRLModuleSpec(
        rl_module_specs={
            p: RLModuleSpec(
                module_class=GFootballMambaRLModule,
                observation_space=single_obs_space,
                action_space=single_act_space,
                model_config=model_config
            ) for p in ["policy_left", "policy_right"]
        }
    )
    
    config = (
        PPOConfigNoForeach()
        .environment(
            env="gfootball_multi",
            env_config={
                "curriculum_horizon": 500,
                "mode": "train",
                "sample_method": "progressive",
                "baselines": baselines_serializable
            },
            disable_env_checking=True
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        .env_runners(
            num_env_runners=num_env_runners,
            num_envs_per_env_runner=1,
            batch_mode="complete_episodes",
            sample_timeout_s=180.0,
        )
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=train_batch_size // 4,
            num_epochs=5,
            lr=5e-5,
            gamma=0.998,
            lambda_=0.95,
            entropy_coeff=0.01,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            use_gae=True,
            use_critic=True,
        )
        .rl_module(rl_module_spec=rl_spec)
        .multi_agent(
            policies={"policy_left", "policy_right"},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["policy_left"]
        )
        .callbacks(make_callback_class(baselines))
        .resources(num_gpus=1, num_cpus_for_main_process=1)
        .learners(num_learners=0, num_gpus_per_learner=0.5)
    )
    
    return config


def main():
    root = Path(__file__).parent
    res_dir = root / "ray_results"
    tmp_dir = root / "ray_tmp"
    baselines_path = root / "stage_baselines.json"
    
    res_dir.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)
    os.environ["RAY_TMPDIR"] = str(tmp_dir)
    
    calibrator = BaselineCalibrator(baselines_path)
    
    if CALIBRATE_ONLY or not calibrator.load():
        print("Running baseline calibration...")
        ray.init(num_gpus=1, _temp_dir=str(tmp_dir), ignore_reinit_error=True)
        calibrator.calibrate_all_parallel(CALIBRATION_EPISODES, CALIBRATION_WORKERS)
        if CALIBRATE_ONLY:
            print("Calibration complete.")
            ray.shutdown()
            return
    else:
        ray.init(num_gpus=1, _temp_dir=str(tmp_dir), ignore_reinit_error=True)
            
    baselines = calibrator.baselines
    
    print("\nLoaded baselines:")
    for sid, bl in baselines.items():
        print(f"  Stage {sid}: return={bl.episode_return_mean:.3f}, win={bl.win_rate:.1%}")
    print()
    
    baselines_serializable = {k: v.to_dict() for k, v in baselines.items()}
    register_env("gfootball_multi", lambda cfg: GFootballMultiAgentEnv(cfg))
    
    model_config = {
        "d_model": 48,
        "mamba_state": 6,
        "num_mamba_layers": 6,
        "prev_action_emb": 8,
        "gradient_checkpointing": True,
        "mlp_hidden_dims": [256, 128],
        "head_hidden_dims": [128],
        "use_distributional": True,
        "v_min": -10.0,
        "v_max": 10.0,
        "num_atoms": 51,
        "num_stages": len(STAGE_CONFIGS),
        "max_seq_len": 20
    }
    
    config = create_config(model_config, baselines)
    
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="pbt_metric",
        mode="max",
        perturbation_interval=20,
        hyperparam_mutations={
            "lr": tune.uniform(1e-5, 1e-4),
            "entropy_coeff": tune.uniform(0.005, 0.03),
            "lambda_": tune.uniform(0.9, 0.99),
        },
        quantile_fraction=0.25,
        resample_probability=1.0,
    )
    
    tune.run(
        PPONoForeach,
        config=config.to_dict(),
        scheduler=pbt,
        num_samples=NUM_SAMPLES,
        stop={"training_iteration": STOP_ITERATIONS},
        storage_path=str(res_dir),
        name="PPO_GFootball_Curriculum",
        checkpoint_freq=20,
        keep_checkpoints_num=None,
        verbose=1,
        restore=r"C:\clones\rlib_gfootball\PPONoForeach_gfootball_multi_36913_00000_0_2025-11-28_23-11-39\checkpoint_000032",
    )
    
    ray.shutdown()


if __name__ == "__main__":
    main()