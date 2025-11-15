from logging import config
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
from collections import deque

import gfootball.env as football_env
import numpy as np
import ray
from gymnasium import spaces
import torch
from ray import train, tune
from ray.air.config import CheckpointConfig, RunConfig
from ray.rllib.algorithms.impala import Impala, ImpalaConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.train import Checkpoint
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import Trainable

from model_3 import GFootballMamba
from model_2 import GFootballGNN
from policy_pool import EnhancedSelfPlayCallback

import os 

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
    expected_reward_mean: float = 0.0
    expected_reward_std: float = 1.0
    reward_scale: float = 1.0

TRAINING_STAGES = [
    TrainingStage("stage_1_basic", "academy_empty_goal_close", "simple115v2", 1, 0, 0.75, 1_000_000, 
                 "1 attacker, no opponents", expected_reward_mean=3.0, expected_reward_std=2.0, reward_scale=0.3),
    TrainingStage("stage_2_basic", "academy_run_to_score_with_keeper", "simple115v2", 1, 0, 0.75, 200_000_000, 
                 "1 attacker versus goalkeeper", expected_reward_mean=1.0, expected_reward_std=1.5, reward_scale=1.0),
    TrainingStage("stage_3_basic", "academy_pass_and_shoot_with_keeper", "simple115v2", 1, 0, 0.75, 5_000_000,
                 "1 attacker with defender", expected_reward_mean=0.8, expected_reward_std=1.2, reward_scale=1.2),
    TrainingStage("stage_4_1v1", "academy_3_vs_1_with_keeper", "simple115v2", 3, 0, 0.75, 10_000_000,
                 "3 attackers vs 1 defender", expected_reward_mean=1.5, expected_reward_std=1.5, reward_scale=0.8),
    TrainingStage("stage_5_3v0", "academy_single_goal_versus_lazy", "simple115v2", 11, 0, 1.0, 50_000_000,
                 "3 vs 0 full field", expected_reward_mean=2.0, expected_reward_std=2.0, reward_scale=0.5),
    TrainingStage("stage_6_transition", "11_vs_11_easy_stochastic", "simple115v2", 3, 3, 1.0, 100_000_000,
                 "Small team in 11v11", expected_reward_mean=0.5, expected_reward_std=1.0, reward_scale=2.0),
    TrainingStage("stage_7_midgame", "11_vs_11_easy_stochastic", "simple115v2", 5, 5, 1.0, 500_000_000,
                 "3v3 in 11v11", expected_reward_mean=0.3, expected_reward_std=0.8, reward_scale=3.0),
    TrainingStage("stage_8_fullgame", "11_vs_11_stochastic", "simple115v2", 5, 5, 1.0, 1_000_000_000,
                 "Full 11v11", expected_reward_mean=0.1, expected_reward_std=0.5, reward_scale=5.0)
]

class RunningStats:
    def __init__(self, window_size=10000):
        self.values = deque(maxlen=window_size)
        self.mean = 0.0
        self.std = 1.0
        self.count = 0
        
    def update(self, value):
        self.values.append(value)
        self.count += 1
        if len(self.values) >= 100:
            self.mean = np.mean(self.values)
            self.std = np.std(self.values) + 1e-8
    
    def normalize(self, value):
        if self.count < 100:
            return value
        return (value - self.mean) / self.std

class RewardNormalizer:
    def __init__(self):
        self.stage_stats = {}
        self.global_stats = RunningStats(window_size=50000)
        
    def get_stats(self, stage_name):
        if stage_name not in self.stage_stats:
            self.stage_stats[stage_name] = RunningStats()
        return self.stage_stats[stage_name]
    
    def update_and_normalize(self, reward, stage_name, stage_config):
        stats = self.get_stats(stage_name)
        stats.update(reward)
        self.global_stats.update(reward)
        
        if stats.count < 100:
            normalized = reward * stage_config.reward_scale
        else:
            z_score = (reward - stats.mean) / stats.std
            target_reward = z_score * stage_config.expected_reward_std + stage_config.expected_reward_mean
            normalized = target_reward * stage_config.reward_scale
        
        normalized = np.clip(normalized, -10.0, 10.0)
        return normalized

reward_normalizer = RewardNormalizer()

class GFootballMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        ray_temp = os.environ.get('RAY_TEMP_DIR', '/tmp')
        gf_logdir = os.path.join(ray_temp, 'gf')
        
        default_config = {
            "env_name": "11_vs_11_stochastic", "representation": "simple115v2",
            "rewards": "scoring,checkpoints", "number_of_left_players_agent_controls": 1,
            "number_of_right_players_agent_controls": 0,
            "stacked": True,
            "logdir": gf_logdir, "write_goal_dumps": False,
            "write_full_episode_dumps": False, "render": False,
            "write_video": False, "dump_frequency": 1,
        }
        self.env_config = {**default_config, **config}
        self.debug_mode = self.env_config.get("debug_mode", False)
        if self.debug_mode:
            self.env_config.update({"render": True})

        self.left_players = self.env_config["number_of_left_players_agent_controls"]
        self.right_players = self.env_config["number_of_right_players_agent_controls"]
        
        self.stage_config = self.env_config.get("stage_config", None)
        self.stage_name = self.env_config.get("stage_name", "unknown")

        creation_kwargs = self.env_config.copy()
        creation_kwargs.pop("debug_mode", None)
        creation_kwargs.pop("_reset_render_state", None)
        creation_kwargs.pop("stage_config", None)
        creation_kwargs.pop("stage_name", None)

        try:
            self.env = football_env.create_environment(**creation_kwargs)
        except Exception as e:
            raise

        self.agent_ids = [f"left_{i}" for i in range(self.left_players)] + \
                         [f"right_{i}" for i in range(self.right_players)]
        self._agent_ids = set(self.agent_ids)

        try:
            _temp_env = football_env.create_environment(**creation_kwargs)
            _initial_obs_sample = _temp_env.reset()
            _initial_obs_sample = _initial_obs_sample[0] if isinstance(_initial_obs_sample, tuple) else _initial_obs_sample

            if not self.agent_ids:
                single_agent_obs_shape = _initial_obs_sample.shape
            elif _initial_obs_sample.ndim > 0 and _initial_obs_sample.shape[0] == (self.left_players + self.right_players) and (self.left_players + self.right_players) > 0 :
                single_agent_obs_shape = _initial_obs_sample.shape[1:]
            else:
                single_agent_obs_shape = _initial_obs_sample.shape

            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=single_agent_obs_shape, dtype=_initial_obs_sample.dtype)
            if isinstance(self.env.action_space, spaces.Tuple):
                self.action_space = self.env.action_space.spaces[0]
            elif isinstance(self.env.action_space, spaces.Discrete):
                self.action_space = self.env.action_space
            elif hasattr(self.env.action_space, 'nvec'):
                self.action_space = spaces.Discrete(self.env.action_space.nvec[0])
            else:
                self.action_space = spaces.Discrete(19)

            _temp_env.close()

        except Exception as e:
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 115) if self.env_config.get("stacked", True) else (115,), dtype=np.float32)
            self.action_space = spaces.Discrete(19)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        try:
            reset_result = self.env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            return self._split_obs(obs), {aid: {} for aid in self.agent_ids}
        except Exception as e:
            obs_dict = {aid: np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
                        for aid in self.agent_ids}
            return obs_dict, {aid: {} for aid in self.agent_ids}

    def step(self, action_dict):
        actions = [action_dict.get(aid, self.action_space.sample()) for aid in self.agent_ids]
        try:
            step_result = self.env.step(actions)
        except Exception as e:
            obs_dict = {aid: np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
                        for aid in self.agent_ids}
            rewards_dict = {aid: 0.0 for aid in self.agent_ids}
            dones = {aid: True for aid in self.agent_ids}; dones["__all__"] = True
            truncs = {aid: False for aid in self.agent_ids}; truncs["__all__"] = False
            infos = {aid: {"error": str(e)} for aid in self.agent_ids}
            return obs_dict, rewards_dict, dones, truncs, infos

        if len(step_result) == 5:
            obs, rewards, terminated, truncated, info = step_result
        elif len(step_result) == 4:
            obs, rewards, done, info = step_result
            terminated, truncated = done, False
        else:
            raise ValueError(f"Unexpected step result format: {step_result}")

        dones = {aid: terminated for aid in self.agent_ids}; dones["__all__"] = terminated
        truncs = {aid: truncated for aid in self.agent_ids}; truncs["__all__"] = truncated

        if self.debug_mode:
            self.env.render()

        agent_infos = {aid: info for aid in self.agent_ids}
        return self._split_obs(obs), self._split_and_normalize_rewards(rewards), dones, truncs, agent_infos

    def _split_obs(self, obs):
        if not self.agent_ids: return {}
        num_agents = len(self.agent_ids)
        if isinstance(obs, dict): return obs
        if not isinstance(obs, np.ndarray): obs = np.array(obs)

        if obs.ndim > 0 and obs.shape[0] == num_agents:
            if obs.shape[1:] == self.observation_space.shape:
                return {self.agent_ids[i]: obs[i].astype(np.float32) for i in range(num_agents)}
            else:
                return {self.agent_ids[i]: obs[i].astype(np.float32) for i in range(num_agents)}
        elif obs.shape == self.observation_space.shape and num_agents > 0:
            return {aid: obs.astype(np.float32) for aid in self.agent_ids}
        elif obs.ndim == 1 and num_agents > 0 and self.observation_space.shape is not None and np.prod(self.observation_space.shape) > 0 and obs.size == num_agents * int(np.prod(self.observation_space.shape)):
            try:
                obs_reshaped = obs.reshape(num_agents, *self.observation_space.shape)
                return {self.agent_ids[i]: obs_reshaped[i].astype(np.float32) for i in range(num_agents)}
            except ValueError as e:
                pass

        zero_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return {aid: zero_obs.astype(np.float32) for aid in self.agent_ids}

    def _split_and_normalize_rewards(self, rewards):
        global reward_normalizer
        
        if np.isscalar(rewards):
            if self.stage_config:
                normalized = reward_normalizer.update_and_normalize(
                    float(rewards), self.stage_name, self.stage_config
                )
            else:
                normalized = float(rewards)
            return {aid: normalized for aid in self.agent_ids}
        
        if isinstance(rewards, (list, np.ndarray)):
            if len(rewards) == len(self.agent_ids):
                result = {}
                for i in range(len(self.agent_ids)):
                    if self.stage_config:
                        normalized = reward_normalizer.update_and_normalize(
                            float(rewards[i]), self.stage_name, self.stage_config
                        )
                    else:
                        normalized = float(rewards[i])
                    result[self.agent_ids[i]] = normalized
                return result
            else:
                return {aid: 0.0 for aid in self.agent_ids}

        return {aid: 0.0 for aid in self.agent_ids}

    def close(self):
        self.env.close()

def get_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "policy_left" if agent_id.startswith("left_") else "policy_right"

def select_policy_to_train(policy_id, worker, **kwargs):
    return True

def create_impala_config(stage: TrainingStage,
                         tune_config: Dict[str, Any],
                         debug_mode: bool = False,
                         hyperparams: dict = None,
                         policy_pool_dir: str = "policy_pool",
                         max_versions: int = 50,
                         keep_top: int = 10,
                         active_zone: int = 15,
                         auto_prune: bool = True) -> ImpalaConfig:
    config = ImpalaConfig()

    env_config = {
        "debug_mode": debug_mode,
        "representation": stage.representation,
        "env_name": stage.env_name,
        "number_of_left_players_agent_controls": stage.left_players,
        "number_of_right_players_agent_controls": stage.right_players,
        "rewards": "scoring,checkpoints",
        "stacked": True,
        "stage_config": stage,
        "stage_name": stage.name,
    }
    config.environment("gfootball_multi", env_config=env_config)
    config.framework("torch")

    config.env_runners(
        num_env_runners=0 if debug_mode else tune_config["num_env_runners"],
        num_envs_per_env_runner=1,
        num_gpus_per_env_runner=0, 
        num_cpus_per_env_runner=tune_config["cpus_per_runner"],
        rollout_fragment_length=512 if not debug_mode else 8,
        batch_mode="truncate_episodes"
    )

    if hyperparams is None:
        hyperparams = {"lr": 5e-5, "entropy_coeff": 0.008, "vf_loss_coeff": 0.5}

    config.training(
        lr=hyperparams.get("lr", 5e-5),
        gamma=hyperparams.get("gamma", 0.997),
        entropy_coeff=hyperparams.get("entropy_coeff", 0.008),
        vf_loss_coeff=hyperparams.get("vf_loss_coeff", 1.0),
        grad_clip=0.5,
        train_batch_size=128_000,
        learner_queue_size=16,
        num_sgd_iter=1,
        vtrace_clip_rho_threshold=hyperparams.get("vtrace_clip_rho_threshold", 1.0),
        vtrace_clip_pg_rho_threshold=hyperparams.get("vtrace_clip_pg_rho_threshold", 1.0),
    )

    use_custom_model = False
    
    custom_model_config = {
        "custom_model": "GFootballMamba",
        "max_seq_len": 256,
        "custom_model_config": {
            "d_model": 48,
            "mamba_state": 6,
            "num_mamba_layers": 6,

            "prev_action_emb": 8,
            "gradient_checkpointing": True,

            "mlp_hidden_dims": [256, 128],
            "mlp_activation": "silu",
            
            "head_hidden_dims": [128],
            "head_activation": "silu",

            "use_noisy": True,
            "use_distributional": True,
            "v_min": -10.0,
            "v_max": 10.0,
            "num_atoms": 51,

            "pretrained_weights_path": r"/home/john/rlib_gfootball/cold_start/mamba_distillation_training/checkpoint_epoch_40.pth",
        },
    }

    standard_model_config = {
        "fcnet_hiddens": [256, 128], 
        "fcnet_activation": "silu", 
        "use_lstm": True, 
        "lstm_cell_size": 512, 
        "lstm_use_prev_action": True, 
        "lstm_use_prev_reward": False, 
        "vf_share_layers": True, 
        }

    if use_custom_model:
        config.model.update(custom_model_config)
    else:
        config.model.update(standard_model_config)

    config.resources(
        num_cpus_for_main_process=1,
        num_gpus=0
    )
    config.learners(
        num_learners=1,
        num_gpus_per_learner=tune_config["gpu_per_trial"],
        num_cpus_per_learner=1
    )

    policies = {}
    has_left = stage.left_players > 0
    has_right = stage.right_players > 0
    if has_left: policies["policy_left"] = PolicySpec()
    if has_right: policies["policy_right"] = PolicySpec()

    policies_to_train = []
    if has_left: policies_to_train.append("policy_left")
    if has_right: policies_to_train.append("policy_right")

    config.multi_agent(
        policies=policies,
        policy_mapping_fn=get_policy_mapping_fn,
        policies_to_train=policies_to_train
    )

    if has_left or has_right:
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
    try:
        lr = trial.config.get("lr", "na")
        ent = trial.config.get("entropy_coeff", "na")
        vf = trial.config.get("vf_loss_coeff", "na")

        if isinstance(lr, (int, float)):
            lr_str = f"lr={lr:.1e}"
        else:
            lr_str = "lr=na"
            
        if isinstance(ent, (int, float)):
            ent_str = f"ent={ent:.3f}"
        else:
            ent_str = "ent=na"
            
        if isinstance(vf, (int, float)):
            vf_str = f"vf={vf:.2f}"
        else:
            vf_str = "vf=na"
        
        base_name = trial.trainable_name
        trial_id = trial.trial_id[:5] if len(trial.trial_id) >= 5 else trial.trial_id
        
        return f"{base_name}_{trial_id}_{lr_str}_{ent_str}_{vf_str}"
    
    except Exception as e:
        try:
            return f"{trial.trainable_name}_{trial.trial_id[:5]}_HPERROR"
        except:
            return "TRIAL_NAME_ERROR"
    
def mutate_hparams(hp: dict) -> dict:
    import random
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    LR_MIN, LR_MAX = 5e-5, 2e-4
    ENT_MIN, ENT_MAX = 0.003, 0.01
    VF_MIN, VF_MAX = 0.3, 1.2

    if random.random() < 0.20:
        return dict(hp)

    lr_delta = random.uniform(-0.15, 0.15)
    lr = clamp(hp["lr"] * (10 ** lr_delta), LR_MIN, LR_MAX)

    ent_step = random.uniform(-0.0015, 0.0015)
    ent = clamp(hp["entropy_coeff"] + ent_step, ENT_MIN, ENT_MAX)

    vf_step = random.uniform(-0.15, 0.15)
    vf = clamp(hp["vf_loss_coeff"] + vf_step, VF_MIN, VF_MAX)

    lr = float(f"{lr:.8f}")
    ent = float(f"{ent:.6f}")
    vf  = float(f"{vf:.4f}")

    return {"lr": lr, "entropy_coeff": ent, "vf_loss_coeff": vf}

class ImpalaTrainable(Trainable):
    def setup(self, config):
        restore_path = config.pop("_restore_from", None)
        self.stop_timesteps = config.pop("_stop_timesteps", None)
        self.stop_after = config.pop("_stop_after", None)
        self.pbt_interval = config.pop("_pbt_interval", None)

        self.algo = Impala(config=config)
        self.start_timesteps = 0

        if restore_path:
            try:
                self.algo.restore(restore_path)
                if hasattr(self.algo, '_counters'):
                    self.start_timesteps = self.algo._counters.get("num_env_steps_sampled", 0)
                elif self.algo.iteration > 0:
                    dummy_result = self.algo.train()
                    self.start_timesteps = dummy_result.get("timesteps_total", 0) - dummy_result.get("num_env_steps_sampled_this_iter", 0)
            except Exception as e:
                self.start_timesteps = 0

    def step(self):
        result = self.algo.train()
        current_timesteps = result.get("timesteps_total", 0)
        
        if self.stop_after is not None:
            target_timesteps = self.start_timesteps + self.stop_after
            if current_timesteps >= target_timesteps:
                result["done"] = True
        
        if self.stop_timesteps is not None:
            if current_timesteps >= self.stop_timesteps:
                result["done"] = True
        
        return result

    def save_checkpoint(self, checkpoint_dir):
        save_result = self.algo.save(checkpoint_dir)
        chk_path = save_result.checkpoint.path if hasattr(save_result, 'checkpoint') else save_result
        return chk_path

    def load_checkpoint(self, checkpoint_path):
        try:
            self.algo.restore(checkpoint_path)
        except Exception as e:
            pass

    def cleanup(self):
        self.algo.stop()

def train_stage_sequential_pbt(
    stage: TrainingStage,
    tune_config: Dict[str, Any],
    debug_mode: bool,
    start_checkpoint: Optional[str],
    metric_path: str = "env_runners/episode_return_mean",
) -> Tuple[Optional[str], Dict[str, Any]]:
    generations = tune_config.get("generations", 6)
    candidates_per_gen = tune_config.get("candidates_per_gen", 4)
    steps_per_gen = tune_config.get("steps_per_gen", 200_000)

    results_path = Path(__file__).resolve().parent / "training_results_transfer"
    policy_pool_dir = results_path / f"{stage.name}_policy_pool"

    best_hp = {"lr": 0.0001, "entropy_coeff": 0.008, "vf_loss_coeff": 0.5}
    best_ckpt = start_checkpoint

    gen = 0
    while gen < generations:
        candidates = [deepcopy(best_hp)] + [mutate_hparams(best_hp) for _ in range(candidates_per_gen - 1)]

        base_cfg_obj = create_impala_config(
            stage=stage,
            tune_config=tune_config,
            debug_mode=debug_mode,
            hyperparams={},
            policy_pool_dir=str(policy_pool_dir),
            max_versions=25, keep_top=10, active_zone=15, auto_prune=True
        )
        base_cfg = base_cfg_obj.to_dict()

        base_cfg["_restore_from"] = best_ckpt
        base_cfg["_stop_after"] = steps_per_gen
        base_cfg["_pbt_interval"] = steps_per_gen

        base_cfg["_hp_idx"] = tune.grid_search(list(range(len(candidates))))
        base_cfg["lr"]          = tune.sample_from(lambda config: candidates[config["_hp_idx"]]["lr"])
        base_cfg["entropy_coeff"] = tune.sample_from(lambda config: candidates[config["_hp_idx"]]["entropy_coeff"])
        base_cfg["vf_loss_coeff"] = tune.sample_from(lambda config: candidates[config["_hp_idx"]]["vf_loss_coeff"])
        
        num_env_runners = 0 if debug_mode else tune_config["num_env_runners"]
        
        resources_per_trial = tune.PlacementGroupFactory(
            [
                {"CPU": 1},
                {"CPU": 1, "GPU": tune_config["gpu_per_trial"]},
            ] + [
                {"CPU": tune_config["cpus_per_runner"]}
            ] * num_env_runners,
            strategy="SPREAD"
        )

        gen_results_path = results_path / stage.name / f"gen_{gen+1}"

        checkpoint_config = CheckpointConfig(
            num_to_keep=2,
            checkpoint_score_attribute=metric_path,
            checkpoint_score_order="max",
        )

        tuner = tune.Tuner(
            tune.with_resources(ImpalaTrainable, resources=resources_per_trial),
            param_space=base_cfg,
            tune_config=tune.TuneConfig(
                num_samples=1,
                max_concurrent_trials=1,
                trial_dirname_creator=short_trial_name_creator,
            ),
            run_config=RunConfig(
                stop={"training_iteration": 10**9},
                checkpoint_config=checkpoint_config,
                name="run",
                storage_path=str(gen_results_path),
                log_to_file=True,
            ),
        )

        try:
            results = tuner.fit()
        except Exception as e:
            break

        if results.num_errors > 0:
            best_result = results.get_best_result(metric=metric_path, mode="max", filter_nan_and_inf=True)
        else:
            best_result = results.get_best_result(metric=metric_path, mode="max")

        if best_result and best_result.checkpoint:
            cfg = best_result.config
            current_gen_best_hp = {
                "lr": float(cfg["lr"]),
                "entropy_coeff": float(cfg["entropy_coeff"]),
                "vf_loss_coeff": float(cfg["vf_loss_coeff"]),
            }
            current_gen_best_ckpt = str(best_result.checkpoint.path)

            best_hp = current_gen_best_hp
            best_ckpt = current_gen_best_ckpt

        elif not best_result:
            break

        gen += 1

    return best_ckpt, best_hp

def train_single_stage(stage: TrainingStage,
                       stage_index: int,
                       tune_config: Dict[str, Any],
                       scheduler_mode: str,
                       debug_mode: bool,
                       restore_checkpoint: Optional[str]) -> Optional[str]:
    
    results_path = Path(__file__).resolve().parent / "training_results_transfer"
    policy_pool_dir = results_path / f"{stage.name}_policy_pool"
    metric_path = "env_runners/episode_return_mean"

    hyperparams = {
        "lr": tune.loguniform(2e-5, 1e-4),
        "entropy_coeff": tune.uniform(0.006, 0.012),
        "vf_loss_coeff": tune.uniform(0.5, 1.0),
        "gamma": tune.uniform(0.997, 0.9995),
        "vtrace_clip_rho_threshold": tune.uniform(0.95, 1.25),
        "vtrace_clip_pg_rho_threshold": tune.uniform(0.9, 1.1),
    }

    param_space = create_impala_config(
        stage=stage,
        tune_config=tune_config,
        debug_mode=debug_mode,
        hyperparams=hyperparams,
        policy_pool_dir=str(policy_pool_dir),
        max_versions=25, keep_top=10, active_zone=15, auto_prune=True
    ).to_dict()

    param_space["_stop_timesteps"] = stage.max_timesteps
    if restore_checkpoint:
        param_space["_restore_from"] = restore_checkpoint
    
    param_space["_pbt_interval"] = tune_config.get("perturbation_interval", 1_000_000)

    num_runners = 0 if debug_mode else tune_config["num_env_runners"]
    gpus_for_learner = tune_config["gpu_per_trial"]
    
    cpus_for_learner = 5
    cpus_for_driver = 2

    resources_per_trial = tune.PlacementGroupFactory(
        [
            {"CPU": cpus_for_driver, "GPU": 0}, 
            {"CPU": cpus_for_learner, "GPU": gpus_for_learner},
        ] +
        [{"CPU": tune_config["cpus_per_runner"]}] * num_runners,
        strategy="SPREAD"
    )

    stop_criteria = {"timesteps_total": stage.max_timesteps}
    checkpoint_config = CheckpointConfig(
        num_to_keep=None,
        checkpoint_score_attribute=metric_path,
        checkpoint_score_order="max",
    )

    search_alg = None

    scheduler = None
    if scheduler_mode == "pbt_parallel":
        scheduler = PopulationBasedTraining(
            time_attr="timesteps_total",
            metric=metric_path,
            mode="max",
            perturbation_interval=tune_config["perturbation_interval"],
            hyperparam_mutations={
                "lr": tune.loguniform(4e-5, 1e-4),
                "entropy_coeff": tune.uniform(0.006, 0.012),
                "vf_loss_coeff": tune.uniform(0.5, 1.0),
                "gamma": [0.997, 0.9975, 0.998, 0.9985, 0.999],
                "vtrace_clip_rho_threshold": [0.9, 1.0, 1.1],
                "vtrace_clip_pg_rho_threshold": [0.9, 1.0, 1.1],
            },
            quantile_fraction=0.25,
            resample_probability=0.30,
            log_config=True,
        )

    tuner = tune.Tuner(
        tune.with_resources(ImpalaTrainable, resources=resources_per_trial),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            search_alg=search_alg,
            scheduler=scheduler,
            num_samples=tune_config["num_trials"],
            max_concurrent_trials=tune_config["max_concurrent"],
            trial_dirname_creator=short_trial_name_creator,
        ),
        run_config=RunConfig(
            stop=stop_criteria,
            checkpoint_config=checkpoint_config,
            name=f"{stage.name}_{scheduler_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            storage_path=str(results_path),
        ),
    )

    try:
        results = tuner.fit()
    except Exception as e:
        return None

    if results.experiment_path:
        experiment_path = str(results.experiment_path)
        log_file = results_path / "tensorboard_commands.txt"
        tensorboard_command = f"tensorboard --logdir \"{experiment_path}\""
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(f"--- Stage: {stage.name} ({scheduler_mode}) ---\n")
                f.write(f"{tensorboard_command}\n\n")
        except IOError as e:
            pass

    best_result = results.get_best_result(metric=metric_path, mode="max")

    if best_result and best_result.checkpoint:
        checkpoint_path = str(best_result.checkpoint.path)
        return checkpoint_path
    else:
        return None

def main():
    scheduler_mode = "pbt_parallel"
    
    tune_config = None
    if scheduler_mode == "pbt_sequential":
        tune_config = {
            "num_trials": 1,
            "max_concurrent": 1,
            "gpu_per_trial": 1,
            "num_env_runners": 22,
            "cpus_per_runner": 1,
            "candidates_per_gen": 2,
            "steps_per_gen": 1_000_000,
            "generations": 6,
        }

    elif scheduler_mode == "pbt_parallel":
        tune_config = {
            "num_trials": 2,
            "max_concurrent": 2,
            "gpu_per_trial": 0.5,
            "num_env_runners": 120,
            "cpus_per_runner": 1,
            "perturbation_interval": 1_000_000,
        }
    else:
        raise ValueError(f"Unknown SCHEDULER_MODE: {scheduler_mode}")

    start_stage_index = 0
    end_stage_index = len(TRAINING_STAGES) - 1
    
    debug_mode = False
    initial_checkpoint = r"/home/john/rlib_gfootball/training_results_transfer_2/stage_2_basic_pbt_parallel_20251113_110913/train_impala_with_restore_ce0df_lr=2.5e-05_ent=0.009_vf=0.92/checkpoint_001023"
    ray.init(ignore_reinit_error=True, log_to_driver=False, local_mode=debug_mode, address="auto")

    register_env("gfootball_multi", lambda config: GFootballMultiAgentEnv(config))
    ModelCatalog.register_custom_model("GFootballGNN", GFootballGNN)
    ModelCatalog.register_custom_model("GFootballMamba", GFootballMamba)

    current_checkpoint = initial_checkpoint
    final_best_hparams = None
    results_path = Path(__file__).resolve().parent / "training_results_transfer"

    for i in range(start_stage_index, end_stage_index + 1):
        stage = TRAINING_STAGES[i]
        
        stage_checkpoint = None
        stage_hparams = {}

        stage_root = results_path / stage.name
        stage_root.mkdir(parents=True, exist_ok=True)
        stage_tb_cmd = f'tensorboard --logdir "{stage_root}" --reload_multifile true --purge_orphaned_data true --window_title "{stage.name} - All Generations"'
        log_file = results_path / "tensorboard_commands.txt"
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                if log_file.exists() and log_file.stat().st_size > 0:
                        f.write("-" * 60 + "\n")
                f.write(f"--- Stage: {stage.name} (ALL GENERATIONS) ---\n")
                f.write(stage_tb_cmd + "\n\n")
        except IOError as e:
            pass

        if scheduler_mode == "pbt_sequential":
            stage_checkpoint, stage_hparams = train_stage_sequential_pbt(
                stage=stage,
                tune_config=tune_config,
                debug_mode=debug_mode,
                start_checkpoint=current_checkpoint
            )
            final_best_hparams = stage_hparams

        elif scheduler_mode == "pbt_parallel":
            stage_checkpoint = train_single_stage(
                stage=stage,
                stage_index=i,
                tune_config=tune_config,
                scheduler_mode=scheduler_mode,
                debug_mode=debug_mode,
                restore_checkpoint=current_checkpoint
            )
            final_best_hparams = None

        else:
             break

        if stage_checkpoint:
            current_checkpoint = stage_checkpoint
        else:
            break

    ray.shutdown()

if __name__ == "__main__":
    main()