"""
PPO Training mit Curriculum Learning für Google Research Football.
Version 4: CANONICAL RLLIB 2.x DESIGN

============================================================
DESIGN-PRINZIP:
============================================================

Dieses Script verwendet die offizielle RLlib 2.x API für MultiAgentEnv
mit variablen Agent-Anzahlen zwischen Episoden:

1. VARIABLE AGENT-ANZAHL PRO STAGE
   - Stage 0-1: 1 Agent (left_0)
   - Stage 2: 2 Agents (left_0, left_1)
   - Stage 3-5: 3 Agents (left_0, left_1, left_2)
   - Stage 6: 5 Agents
   - Stage 7: 11 Agents
   
2. KEINE PADDING / FAKE AGENTS
   - reset() und step() geben nur ECHTE Agents zurück
   - Keine Zero-Observations für nicht-existente Agents
   - Keine dummy Rewards
   
3. RLLIB SPACES KONFIGURATION
   - observation_space / action_space für einzelnen Agent
   - observation_spaces / action_spaces als Dict mit ALLEN möglichen Agent-IDs
   - _agent_ids enthält alle möglichen Agent-IDs (für RLlib)
   
4. CONNECTORV2 OUT-OF-THE-BOX
   - Keine custom Connectors
   - Standard env_to_module / module_to_env Pipelines
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
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
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
import torch

from model_3 import GFootballMambaRLModule

# =============================================================================
# CUSTOM PPO CLASSES (foreach=False fix für Windows/PBT)
# =============================================================================

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


# =============================================================================
# CONFIGURATION
# =============================================================================

CALIBRATE_ONLY = False
CALIBRATION_EPISODES = 100
CALIBRATION_WORKERS = 8
NUM_SAMPLES = 2
STOP_ITERATIONS = 500
EVAL_INTERVAL = 20
EVAL_EPISODES_PER_STAGE = 5

# Maximum Agents über alle Stages (für RLlib space registration)
MAX_POSSIBLE_AGENTS = 11


class StageConfig:
    def __init__(self, stage_id: int, env_name: str, representation: str,
                 left_agents: int = 1, right_agents: int = 0, max_steps: int = 3000):
        self.stage_id = stage_id
        self.env_name = env_name
        self.representation = representation
        self.left_agents = left_agents
        self.right_agents = right_agents
        self.max_steps = max_steps


# Curriculum Stages - variable Agent-Anzahlen!
STAGE_CONFIGS = [
    StageConfig(0, "academy_empty_goal_close", "simple115v2", 1, 0, 400),
    StageConfig(1, "academy_run_to_score_with_keeper", "simple115v2", 1, 0, 400),
    StageConfig(2, "academy_pass_and_shoot_with_keeper", "simple115v2", 2, 0, 400),
    StageConfig(3, "academy_3_vs_1_with_keeper", "simple115v2", 3, 0, 400),
    StageConfig(4, "academy_single_goal_versus_lazy", "simple115v2", 3, 0, 1000),
    StageConfig(5, "11_vs_11_easy_stochastic", "simple115v2", 3, 0, 3000),
    StageConfig(6, "11_vs_11_easy_stochastic", "simple115v2", 5, 0, 3000),
    StageConfig(7, "11_vs_11_stochastic", "simple115v2", 11, 0, 3000),
]


# =============================================================================
# BASELINE CALIBRATION (unverändert)
# =============================================================================

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
            
            saved_stage_ids = set(int(k) for k in data.keys())
            expected_stage_ids = set(range(len(STAGE_CONFIGS)))
            
            if saved_stage_ids != expected_stage_ids:
                print(f"Baseline stage mismatch: saved={sorted(saved_stage_ids)}, expected={sorted(expected_stage_ids)}")
                return False
            
            for k, v in data.items():
                stage_id = int(k)
                if stage_id < len(STAGE_CONFIGS):
                    self.baselines[stage_id] = StageBaseline.from_dict(v)
                    
            return all(b.calibrated for b in self.baselines.values())
        except Exception as e:
            print(f"Failed to load baselines: {e}")
            return False
    
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
            print(f"  Win Rate:       {baseline.win_rate:.2%}")
            
        self.save()
        print(f"Baselines saved to {self.save_path}")
        
        return self.baselines


# =============================================================================
# METRICS TRACKING (vereinfacht)
# =============================================================================

class StageMetrics:
    def __init__(self, stage_id: int, baseline: StageBaseline, ema_alpha: float = 0.05):
        self.stage_id = stage_id
        self.baseline = baseline
        self.ema_alpha = ema_alpha
        self.normalized_ema = 0.0
        self.win_rate_ema = baseline.win_rate
        self.episode_count = 0
        
    def update(self, raw_return: float, won: bool):
        normalized = self.baseline.normalize_episode_return(raw_return)
        win_val = 1.0 if won else 0.0
        
        if self.episode_count == 0:
            self.normalized_ema = normalized
            self.win_rate_ema = win_val
        else:
            self.normalized_ema = (1 - self.ema_alpha) * self.normalized_ema + self.ema_alpha * normalized
            self.win_rate_ema = (1 - self.ema_alpha) * self.win_rate_ema + self.ema_alpha * win_val
            
        self.episode_count += 1


class UnifiedMetricsTracker:
    def __init__(self, baselines: Dict[int, StageBaseline]):
        self.baselines = ensure_baseline_objects(baselines)
        self.stage_metrics = {
            sid: StageMetrics(sid, self.baselines[sid]) 
            for sid in range(len(STAGE_CONFIGS))
        }
        
    def update(self, stage_id: int, raw_return: float, won: bool):
        self.stage_metrics[stage_id].update(raw_return, won)


# =============================================================================
# ENVIRONMENT - CANONICAL RLLIB MULTIAGENTENV
# =============================================================================

class GFootballMultiAgentEnv(MultiAgentEnv):
    """
    Multi-Agent Wrapper für Google Research Football mit Curriculum Learning.
    
    CANONICAL RLLIB 2.x DESIGN:
    - Variable Agent-Anzahl zwischen Episoden
    - Nur echte Agents in reset()/step() Dicts
    - observation_spaces/action_spaces für alle MÖGLICHEN Agents definiert
    - _agent_ids enthält alle möglichen Agent-IDs
    """
    
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
        }
            
        self.env = None
        
        # =====================================================================
        # RLLIB MULTIAGENT SPACE DEFINITION
        # =====================================================================
        
        # Alle MÖGLICHEN Agent-IDs (über alle Stages)
        # RLlib braucht das um zu wissen welche Agents es theoretisch geben kann
        self._all_possible_agent_ids = [f"left_{i}" for i in range(MAX_POSSIBLE_AGENTS)]
        self._agent_ids = set(self._all_possible_agent_ids)
        
        # Per-Agent observation/action space (für einen einzelnen Agent)
        low_f32 = np.float32(-np.inf)
        high_f32 = np.float32(np.inf)
        self._single_observation_space = spaces.Dict({
            "obs": spaces.Box(low=low_f32, high=high_f32, shape=(4, 115), dtype=np.float32),
            "stage_index": spaces.Box(low=0, high=len(self.stages), shape=(1,), dtype=np.int32)
        })
        self._single_action_space = spaces.Discrete(19)
        
        # RLlib "preferred format": Dict mapping Agent-IDs zu ihren Spaces
        # Dies definiert ALLE MÖGLICHEN Agents
        self.observation_spaces = {
            aid: self._single_observation_space for aid in self._all_possible_agent_ids
        }
        self.action_spaces = {
            aid: self._single_action_space for aid in self._all_possible_agent_ids
        }
        
        # Fallback für Code der observation_space/action_space direkt liest
        # (einzelner Agent Space, RLlib inferiert den Rest aus observation_spaces)
        self.observation_space = self._single_observation_space
        self.action_space = self._single_action_space
        
        # =====================================================================
        # EPISODE STATE
        # =====================================================================
        
        # Aktive Agents in DIESER Episode (wird bei reset() gesetzt)
        self._current_agents: List[str] = []
        self.num_active_agents = 0
        
        self.latest_obs = {}
        self.episode_rewards = []
        self.step_count = 0
        
        # Initial environment für erste Stage erstellen
        self._create_env_for_stage(self.stages[0])

    def _create_env_for_stage(self, stage: StageConfig):
        """Erstellt GFootball Env für eine Stage."""
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
        
        # Aktive Agents für diese Stage
        self.num_active_agents = stage.left_agents
        self._current_agents = [f"left_{i}" for i in range(self.num_active_agents)]

    def _sample_stage(self) -> StageConfig:
        """Wählt Stage basierend auf Curriculum Progress."""
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
            
        # Retention-based
        weights = []
        for sid in available_ids:
            sm = self.metrics_tracker.stage_metrics[sid]
            if sm.episode_count < 10:
                w = 2.0
            else:
                w = 2.0 - min(1.0, sm.normalized_ema)
            weights.append(max(w, 0.1))
            
        weights = np.array(weights)
        weights /= weights.sum()
        return self.stages[np.random.choice(available_ids, p=weights)]

    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Reset - gibt nur ECHTE Agents dieser Episode zurück.
        """
        super().reset(seed=seed, options=options)
        
        # Stage auswählen
        if self.mode == "eval" and self.eval_stage_id is not None:
            target_stage = self.stages[self.eval_stage_id]
        else:
            target_stage = self._sample_stage()
                
        # Env für Stage erstellen (setzt auch self._current_agents)
        if target_stage.stage_id != self.current_stage_id:
            self._create_env_for_stage(target_stage)
            
        self.episode_idx += 1
        self.episode_rewards = []
        self.step_count = 0
        
        # GFootball reset
        raw_obs = self.env.reset()
        
        # Observations nur für ECHTE Agents
        self.latest_obs = self._process_obs(raw_obs)
        
        # Info nur für ECHTE Agents
        info = {aid: {"stage_id": self.current_stage_id} for aid in self._current_agents}
        
        return self.latest_obs, info

    def step(
        self, 
        action_dict: Dict[str, int]
    ) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Step - gibt nur ECHTE Agents dieser Episode zurück.
        """
        # Action limit prüfen
        internal_limit = 19
        if hasattr(self.env, 'action_space'):
            if hasattr(self.env.action_space, 'n'):
                internal_limit = self.env.action_space.n
            elif isinstance(self.env.action_space, list) and len(self.env.action_space) > 0:
                if hasattr(self.env.action_space[0], 'n'):
                    internal_limit = self.env.action_space[0].n

        # Actions für alle aktiven Agents sammeln
        actions_list = []
        for aid in self._current_agents:
            act = action_dict.get(aid, 0)
            if hasattr(act, "item"):
                act = act.item()
            act = int(act)
            if act >= internal_limit:
                act = 0
            actions_list.append(act)
            
        # GFootball step
        obs, reward, done, info = self.env.step(actions_list)
        self.latest_obs = self._process_obs(obs)
        self.step_count += 1
        
        # Reward verarbeiten
        if isinstance(reward, (list, np.ndarray)):
            raw_step_reward = float(sum(reward))
        else:
            raw_step_reward = float(reward)
            
        self.episode_rewards.append(raw_step_reward)
        
        # Normalisieren
        baseline = self.baselines[self.current_stage_id]
        normalized_step_reward = baseline.normalize_step_reward(raw_step_reward)
        
        # Rewards nur für ECHTE Agents (gleichmäßig verteilt)
        per_agent_reward = normalized_step_reward / self.num_active_agents
        rewards_dict = {aid: per_agent_reward for aid in self._current_agents}
        
        # Episode Ende prüfen
        terminated = bool(done)
        truncated = self.step_count >= self.current_stage.max_steps
        episode_done = terminated or truncated
        
        # Dones nur für ECHTE Agents
        dones = {aid: episode_done for aid in self._current_agents}
        dones["__all__"] = episode_done
        
        truncs = {aid: truncated and not terminated for aid in self._current_agents}
        truncs["__all__"] = truncated and not terminated
        
        # Episode-Level Stats
        raw_episode_return = sum(self.episode_rewards)
        normalized_episode_return = baseline.normalize_episode_return(raw_episode_return)
        
        won = False
        if isinstance(info, dict) and "score" in info:
            won = info["score"][0] > info["score"][1]
        else:
            won = raw_episode_return > 0
            
        if episode_done and self.mode == "train":
            self.metrics_tracker.update(self.current_stage_id, raw_episode_return, won)
            
        # Infos nur für ECHTE Agents
        agent_infos = {}
        for aid in self._current_agents:
            agent_infos[aid] = {
                "stage_id": self.current_stage_id,
                "stage_name": self.current_stage.env_name,
                "num_agents": self.num_active_agents,
                "raw_step_reward": raw_step_reward,
                "normalized_step_reward": normalized_step_reward,
            }
            if episode_done:
                agent_infos[aid]["raw_episode_return"] = raw_episode_return
                agent_infos[aid]["normalized_episode_return"] = normalized_episode_return
                agent_infos[aid]["episode_length"] = self.step_count
                agent_infos[aid]["won"] = won
                
        return self.latest_obs, rewards_dict, dones, truncs, agent_infos

    def _process_obs(self, raw_obs) -> Dict[str, Dict]:
        """
        Verarbeitet Observations - nur für ECHTE Agents.
        """
        if not isinstance(raw_obs, np.ndarray):
            raw_obs = np.array(raw_obs)
            
        if raw_obs.ndim == 1:
            if raw_obs.size in self.EXPECTED_OBS_SIZES:
                raw_obs = raw_obs.reshape(1, -1)
            else:
                warnings.warn(f"Unexpected 1D obs size: {raw_obs.size}")
                raw_obs = np.zeros((self.num_active_agents, 115), dtype=np.float32)
                
        obs_dict = {}
        for i, aid in enumerate(self._current_agents):
            if i < len(raw_obs):
                data = raw_obs[i]
                
                if data.size == 115:
                    data = np.tile(data.reshape(1, 115), (4, 1)).astype(np.float32)
                elif data.size == 460:
                    data = data.reshape(4, 115).astype(np.float32)
                else:
                    warnings.warn(f"Unexpected obs size {data.size} for agent {aid}")
                    data = np.zeros((4, 115), dtype=np.float32)
                    
                obs_dict[aid] = {
                    "obs": data,
                    "stage_index": np.array([self.current_stage_id], dtype=np.int32)
                }
            else:
                warnings.warn(f"Missing observation for agent {aid}")
                obs_dict[aid] = {
                    "obs": np.zeros((4, 115), dtype=np.float32),
                    "stage_index": np.array([self.current_stage_id], dtype=np.int32)
                }
                
        return obs_dict

    def get_agent_ids(self) -> List[str]:
        """Gibt die aktuell AKTIVEN Agents zurück."""
        return self._current_agents.copy()

    def close(self):
        if self.env:
            self.env.close()


# =============================================================================
# CALLBACKS
# =============================================================================

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
        num_agents = last_info.get("num_agents", 1)
        
        metrics_logger.log_value(f"episode/stage_{stage_id}_raw", raw_return, reduce="mean")
        metrics_logger.log_value(f"episode/stage_{stage_id}_normalized", normalized_return, reduce="mean")
        metrics_logger.log_value(f"episode/stage_{stage_id}_won", float(won), reduce="mean")
        metrics_logger.log_value(f"episode/stage_{stage_id}_num_agents", float(num_agents), reduce="mean")
        
        metrics_logger.log_value("episode/all_stages_raw", raw_return, reduce="mean")
        metrics_logger.log_value("episode/all_stages_normalized", normalized_return, reduce="mean")

    def on_train_result(self, *, algorithm, result, **kwargs):
        env_runner_results = result.get("env_runners", {})
        
        stage_data = {}
        for stage_id in range(len(STAGE_CONFIGS)):
            raw_key = f"episode/stage_{stage_id}_raw"
            
            if raw_key in env_runner_results:
                stage_data[stage_id] = {
                    "raw": env_runner_results.get(raw_key, 0.0),
                    "normalized": env_runner_results.get(f"episode/stage_{stage_id}_normalized", 0.0),
                    "won": env_runner_results.get(f"episode/stage_{stage_id}_won", 0.0),
                }
                result[f"stage_{stage_id}/raw"] = stage_data[stage_id]["raw"]
                result[f"stage_{stage_id}/normalized"] = stage_data[stage_id]["normalized"]
                result[f"stage_{stage_id}/win_rate"] = stage_data[stage_id]["won"]
        
        if stage_data:
            norm_vals = [d["normalized"] for d in stage_data.values()]
            win_vals = [d["won"] for d in stage_data.values()]
            
            result["curriculum/mean_normalized"] = float(np.mean(norm_vals))
            result["curriculum/mean_win_rate"] = float(np.mean(win_vals))
            result["curriculum/num_active_stages"] = len(stage_data)
        
        norm = result.get("curriculum/mean_normalized", 0.0)
        win = result.get("curriculum/mean_win_rate", 0.0)
        result["pbt_metric"] = 0.5 * np.tanh(norm / 2.0) + 0.5 * win


def policy_mapping_fn(agent_id, episode=None, **kwargs):
    """Alle left_X Agents nutzen policy_left."""
    return "policy_left"


def make_callback_class(baselines: Dict[int, StageBaseline]):
    class ConfiguredCallback(CurriculumCallback):
        def __init__(self):
            super().__init__(baselines=baselines)
    return ConfiguredCallback


# =============================================================================
# CONFIG CREATION
# =============================================================================

def create_config(model_config: Dict, baselines: Dict[int, StageBaseline],
                  num_env_runners: int = 11, train_batch_size: int = 4000):
    
    baselines_serializable = {k: v.to_dict() for k, v in baselines.items()}
    
    # Observation/Action spaces aus Dummy-Env holen
    dummy = GFootballMultiAgentEnv({"baselines": baselines_serializable})
    single_obs_space = dummy._single_observation_space
    single_act_space = dummy._single_action_space
    dummy.close()
    
    # RLModule Spec
    rl_spec = MultiRLModuleSpec(
        rl_module_specs={
            "policy_left": RLModuleSpec(
                module_class=GFootballMambaRLModule,
                observation_space=single_obs_space,
                action_space=single_act_space,
                model_config=model_config
            )
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
            policies={"policy_left"},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["policy_left"],
        )
        .callbacks(make_callback_class(baselines))
        .resources(num_gpus=1, num_cpus_for_main_process=1)
        .learners(num_learners=0, num_gpus_per_learner=0.5)
    )
    
    return config


# =============================================================================
# MAIN
# =============================================================================

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
    
    print("=" * 60)
    print("CANONICAL RLLIB 2.x MULTIAGENT DESIGN")
    print("Variable Agent-Anzahl pro Stage (1-11)")
    print("Keine Padding, keine Fake-Agents")
    print(f"Stages: {len(STAGE_CONFIGS)}")
    for i, stage in enumerate(STAGE_CONFIGS):
        print(f"  Stage {i}: {stage.env_name} ({stage.left_agents} agents)")
    print("=" * 60)
    
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
    
    RESTORE_PATH = None
    
    tune.run(
        PPONoForeach,
        config=config.to_dict(),
        scheduler=pbt,
        num_samples=NUM_SAMPLES,
        stop={"training_iteration": STOP_ITERATIONS},
        storage_path=str(res_dir),
        name="PPO_GFootball_Curriculum_v4_canonical",
        checkpoint_freq=20,
        keep_checkpoints_num=None,
        verbose=1,
        restore=RESTORE_PATH,
    )
    
    ray.shutdown()


if __name__ == "__main__":
    main()