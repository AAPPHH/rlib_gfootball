from logging import config
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple, Optional
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
import pickle
import xgboost as xgb

import sys
sys.path.append("C:/clones/rlib_gfootball/cold_start")
# Dieser Wrapper wird jetzt verwendet
from rllib_wrapper import PretrainedGFootballModel 

from model import GFootballMamba
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

TRAINING_STAGES = [
    # TrainingStage("stage_1_basic", "academy_empty_goal_close", "simple115v2", 1, 0, 0.75, 1_000_000, "1 attacker, no opponents: finishes into an empty goal from close range."),
    TrainingStage("stage_2_basic", "academy_run_to_score_with_keeper", "simple115v2", 1, 0, 0.75, 2_000_000_000, "1 attacker versus a goalkeeper: dribbles towards goal and finishes under light pressure."),
    # TrainingStage("stage_3_basic", "academy_pass_and_shoot_with_keeper", "simple115v2", 1, 0, 0.75, 5_000_000, "1 attacker facing a goalkeeper and nearby defender: focuses on control, positioning, and finishing."),
    # TrainingStage("stage_4_1v1", "academy_3_vs_1_with_keeper", "simple115v2", 3, 0, 0.75, 10_000_000, "3 attackers versus 1 defender and a goalkeeper: encourages passing combinations and shot creation."),
    # TrainingStage("stage_5_3v0", "academy_single_goal_versus_lazy", "simple115v2", 11, 0, 1.0, 500_000_000_000, "3 vs 0 on a full field against static opponents: focuses on offensive buildup and team coordination."),
    # TrainingStage("stage_6_transition", "11_vs_11_easy_stochastic", "simple115v2", 3, 3, 1.0, 100_000_000, "Small-sided (3-player) team in 11v11 environment with easy opponents: transition toward full gameplay."),
    # TrainingStage("stage_7_midgame", "11_vs_11_easy_stochastic", "simple115v2", 5, 5, 1.0, 500_000_000, "3 vs 3 within a full 11v11 match (easy mode): focuses on spacing, positioning, and transitions."),
    # TrainingStage("stage_8_fullgame", "11_vs_11_stochastic", "simple115v2", 5, 5, 1.0, 1_000_000_000, "Full 11v11 stochastic match: standard difficulty with dynamic and realistic gameplay.")
]

class XGBoostTeacher:
    """Lightweight wrapper for XGBoost teacher model"""
    def __init__(self, model_path: str):
        """Load the XGBoost model from disk"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✅ XGBoost teacher loaded from {model_path}")
        except Exception as e:
            print(f"⚠️ Failed to load XGBoost teacher from {model_path}: {e}")
            self.model = None
    
    def predict_action(self, obs: np.ndarray) -> int:
        """Predict action from observation"""
        if self.model is None:
            return np.random.randint(0, 19)  # Fallback to random
        
        try:
            # Ensure obs is 2D for XGBoost
            if obs.ndim == 1:
                obs = obs.reshape(1, -1)
            elif obs.ndim > 2:
                # Flatten if stacked observations
                obs = obs.reshape(1, -1)
            
            # Get prediction
            pred = self.model.predict(obs)
            return int(pred[0])
        except Exception as e:
            print(f"⚠️ Teacher prediction failed: {e}")
            return np.random.randint(0, 19)

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

        # XGBoost Teacher Configuration
        self.use_xgb_teacher = self.env_config.get("use_xgb_teacher", False)
        self.xgb_teacher = None
        if self.use_xgb_teacher:
            xgb_model_path = self.env_config.get("xgb_model_path", "xgboost_coldstart.pkl")
            if os.path.exists(xgb_model_path):
                self.xgb_teacher = XGBoostTeacher(xgb_model_path)
            else:
                print(f"⚠️ XGBoost model not found at {xgb_model_path}, teacher disabled")
                self.use_xgb_teacher = False
        
        # Teacher decay configuration
        self.teacher_mode = self.env_config.get("teacher_mode", "linear")  # linear, exp, adaptive
        self.teacher_start_prob = self.env_config.get("teacher_start_prob", 1.0)
        self.teacher_end_prob = self.env_config.get("teacher_end_prob", 0.0)
        self.teacher_decay_steps = self.env_config.get("teacher_decay_steps", 5_000_000)
        self.current_teacher_prob = self.teacher_start_prob
        self.global_steps = 0
        self.distill_logging = self.env_config.get("distill_logging", False)
        
        # Store last observations for teacher
        self.last_obs = {}

        self.left_players = self.env_config["number_of_left_players_agent_controls"]
        self.right_players = self.env_config["number_of_right_players_agent_controls"]

        creation_kwargs = self.env_config.copy()
        # Remove teacher-specific keys from env creation
        for key in ["debug_mode", "_reset_render_state", "use_xgb_teacher", "xgb_model_path", 
                    "teacher_mode", "teacher_start_prob", "teacher_end_prob", 
                    "teacher_decay_steps", "distill_logging"]:
            creation_kwargs.pop(key, None)

        try:
            self.env = football_env.create_environment(**creation_kwargs)
        except Exception as e:
            print(f"Error creating GFootball environment: {e}")
            print(f"Creation kwargs: {creation_kwargs}")
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
            print(f"CRITICAL ERROR during environment space derivation: {e}")
            self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(4, 115) if self.env_config.get("stacked", True) else (115,), dtype=np.float32)
            self.action_space = spaces.Discrete(19)

    def _compute_teacher_prob(self) -> float:
        """Compute current teacher probability based on decay mode"""
        if not self.use_xgb_teacher:
            return 0.0
        
        progress = min(1.0, self.global_steps / max(1, self.teacher_decay_steps))
        
        if self.teacher_mode == "linear":
            # Linear decay from start_prob to end_prob
            return self.teacher_start_prob + (self.teacher_end_prob - self.teacher_start_prob) * progress
        
        elif self.teacher_mode == "exp":
            # Exponential decay
            decay_rate = -np.log((self.teacher_end_prob + 1e-8) / self.teacher_start_prob)
            return self.teacher_start_prob * np.exp(-decay_rate * progress)
        
        elif self.teacher_mode == "adaptive":
            # For now, use linear decay (could be extended with performance-based adaptation)
            return self.teacher_start_prob + (self.teacher_end_prob - self.teacher_start_prob) * progress
        
        else:
            return self.teacher_start_prob

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        try:
            reset_result = self.env.reset()
            obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
            obs_dict = self._split_obs(obs)
            
            # Store observations for teacher
            self.last_obs = obs_dict.copy()
            
            return obs_dict, {aid: {} for aid in self.agent_ids}
        except Exception as e:
             print(f"ERROR during reset: {e}")
             obs_dict = {aid: np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
                         for aid in self.agent_ids}
             self.last_obs = obs_dict.copy()
             return obs_dict, {aid: {} for aid in self.agent_ids}

    def step(self, action_dict):
        # Update teacher probability
        self.current_teacher_prob = self._compute_teacher_prob()
        self.global_steps += 1
        
        # Potentially override actions with teacher actions
        final_actions = {}
        teacher_used = {}
        
        for aid in self.agent_ids:
            student_action = action_dict.get(aid, self.action_space.sample())
            
            if self.use_xgb_teacher and self.xgb_teacher and aid in self.last_obs:
                # Decide whether to use teacher
                use_teacher = random.random() < self.current_teacher_prob
                
                if use_teacher:
                    # Get teacher action
                    teacher_action = self.xgb_teacher.predict_action(self.last_obs[aid])
                    final_actions[aid] = teacher_action
                    teacher_used[aid] = True
                    
                    if self.distill_logging and self.global_steps % 1000 == 0:
                        print(f"Step {self.global_steps}: Teacher action for {aid}: {teacher_action} (prob={self.current_teacher_prob:.3f})")
                else:
                    final_actions[aid] = student_action
                    teacher_used[aid] = False
            else:
                final_actions[aid] = student_action
                teacher_used[aid] = False
        
        actions = [final_actions.get(aid, self.action_space.sample()) for aid in self.agent_ids]
        
        try:
            step_result = self.env.step(actions)
        except Exception as e:
             print(f"ERROR during step with actions {actions}: {e}")
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

        obs_dict = self._split_obs(obs)
        # Store observations for next step
        self.last_obs = obs_dict.copy()
        
        dones = {aid: terminated for aid in self.agent_ids}; dones["__all__"] = terminated
        truncs = {aid: truncated for aid in self.agent_ids}; truncs["__all__"] = truncated

        if self.debug_mode:
            self.env.render()

        # Add teacher usage info to infos if logging
        agent_infos = {aid: info.copy() if isinstance(info, dict) else info for aid in self.agent_ids}
        if self.distill_logging:
            for aid in self.agent_ids:
                if isinstance(agent_infos[aid], dict):
                    agent_infos[aid]["teacher_used"] = teacher_used.get(aid, False)
                    agent_infos[aid]["teacher_prob"] = self.current_teacher_prob
        
        return obs_dict, self._split_rewards(rewards), dones, truncs, agent_infos

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
                 print(f"ERROR: Could not reshape flattened obs ({obs.shape}) into ({num_agents}, {self.observation_space.shape}). Error: {e}")
        
        zero_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return {aid: zero_obs.astype(np.float32) for aid in self.agent_ids}

    def _split_rewards(self, rewards):
        if np.isscalar(rewards):
            return {aid: float(rewards) for aid in self.agent_ids}
        if isinstance(rewards, (list, np.ndarray)):
            if len(rewards) == len(self.agent_ids):
                return {self.agent_ids[i]: float(rewards[i]) for i in range(len(self.agent_ids))}
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
                         auto_prune: bool = True,
                         use_pretrained: bool = True,
                         load_weights: bool = True,
                         use_xgb_teacher: bool = False,  # New parameter
                         xgb_model_path: str = "xgboost_coldstart.pkl",  # New parameter
                         teacher_config: dict = None  # New parameter
                         ) -> ImpalaConfig:
    config = ImpalaConfig()

    env_config = {
        "debug_mode": debug_mode,
        "representation": stage.representation,
        "env_name": stage.env_name,
        "number_of_left_players_agent_controls": stage.left_players,
        "number_of_right_players_agent_controls": stage.right_players,
        "rewards": "scoring,checkpoints",
        "stacked": True,
    }
    
    # Add XGBoost teacher configuration if enabled
    if use_xgb_teacher:
        teacher_defaults = {
            "use_xgb_teacher": True,
            "xgb_model_path": xgb_model_path,
            "teacher_mode": "linear",
            "teacher_start_prob": 1.0,
            "teacher_end_prob": 0.0,
            "teacher_decay_steps": 5_000_000,
            "distill_logging": False
        }
        if teacher_config:
            teacher_defaults.update(teacher_config)
        env_config.update(teacher_defaults)
    
    config.environment("gfootball_multi", env_config=env_config)
    config.framework("torch")

    config.env_runners(
        num_env_runners=0 if debug_mode else tune_config["num_env_runners"],
        num_envs_per_env_runner=1,
        num_gpus_per_env_runner=0, 
        num_cpus_per_env_runner=tune_config["cpus_per_runner"],
        rollout_fragment_length=128 if not debug_mode else 8,
        batch_mode="truncate_episodes"
    )

    if hyperparams is None:
        hyperparams = {"lr": 5e-5, "entropy_coeff": 0.008, "vf_loss_coeff": 0.5}

    config.training(
        lr=hyperparams.get("lr", 5e-5),
        entropy_coeff=hyperparams.get("entropy_coeff", 0.008),
        vf_loss_coeff=hyperparams.get("vf_loss_coeff", 1.0),
        grad_clip=0.5,
        train_batch_size=128_000,
        learner_queue_size=32,
        num_sgd_iter=1,
        vtrace_clip_rho_threshold=1.0,
        vtrace_clip_pg_rho_threshold=1.0
    )

    if use_pretrained:
        config.model.update({
            "custom_model": "pretrained_model",
            "max_seq_len": 32,
            "custom_model_config": {
                "load_weights": load_weights
            }
        })
    else:
        use_custom_model = False
        
        custom_model_config = {
            "custom_model": "GFootballMamba",
            "max_seq_len": 32,
            "custom_model_config": {
                    "d_model": 48,
                    "mamba_state": 6,
                    "num_mamba_layers": 2,
                    "gnn_type": "sage",
                    "gnn_layers": 2,
                    "gnn_hidden": 24,
                    "gnn_output": 48,
                    "gnn_k_neighbors": 4,
                    "include_ball_node": True,
                    "include_global_node": True,
                    "include_team_nodes": True,
                    "include_possession_node": False,
                    "include_action_node": False,
                    "gnn_dropout": 0.05,
                    "kan_grid": 3,
                    "kan_hidden_dim": 32,
                    "kan_dropout": 0.0,
                    "prev_action_emb": 8,
                    "gradient_checkpointing": True,
                },
            }

        standard_model_config = {
            "fcnet_hiddens": [512, 512], 
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


def train_impala_with_restore(config):
    restore_path = config.pop("_restore_from", None)
    stop_timesteps = config.pop("_stop_timesteps", None)
    stop_after = config.pop("_stop_after", None)
    checkpoint_freq = 50

    algo = Impala(config=config)

    last_reported_checkpoint = None
    start_timesteps = 0

    if restore_path:
        try:
            algo.restore(restore_path)
            start_timesteps = algo._counters.get("num_env_steps_sampled", 0) if hasattr(algo, '_counters') else 0
            if algo.iteration > 0 and start_timesteps == 0:
                dummy_result = algo.train()
                start_timesteps = dummy_result.get("timesteps_total", 0) - dummy_result.get("num_env_steps_sampled_this_iter", 0)

            last_reported_checkpoint = Checkpoint.from_directory(restore_path)
        except Exception as e:
            print(f"⚠️ [Trainer Fn] ERROR restoring {restore_path}: {e}. Training from scratch.")
            save_result = algo.save()
            chk_path = save_result.checkpoint.path if hasattr(save_result, 'checkpoint') else save_result
            last_reported_checkpoint = Checkpoint.from_directory(chk_path)
            start_timesteps = 0
    else:
        save_result = algo.save()
        chk_path = save_result.checkpoint.path if hasattr(save_result, 'checkpoint') else save_result
        last_reported_checkpoint = Checkpoint.from_directory(chk_path)
        start_timesteps = 0

    target_timesteps = float('inf')
    if stop_after is not None:
        target_timesteps = start_timesteps + int(stop_after)
    elif stop_timesteps is not None:
        target_timesteps = int(stop_timesteps)

    timesteps = start_timesteps
    iteration = algo.iteration
    result = {}

    while timesteps < target_timesteps:
        result = algo.train()
        timesteps = result.get("timesteps_total", timesteps)
        iteration += 1

        if iteration % checkpoint_freq == 0:
            save_result = algo.save()
            chk_path = save_result.checkpoint.path if hasattr(save_result, 'checkpoint') else save_result
            if chk_path:
                last_reported_checkpoint = Checkpoint.from_directory(chk_path)
                train.report(metrics=result, checkpoint=last_reported_checkpoint)
            else:
                train.report(metrics=result)
        else:
            train.report(metrics=result)

        if result.get("should_checkpoint", False) and not (iteration % checkpoint_freq == 0):
             pass
        if result.get("done", False):
            break

    final_save_result = algo.save()
    final_chk_path = final_save_result.checkpoint.path if hasattr(final_save_result, 'checkpoint') else final_save_result
    if final_chk_path:
        final_checkpoint = Checkpoint.from_directory(final_chk_path)
        final_metrics = result if result else algo.evaluate() if hasattr(algo, 'evaluate') else {}
        train.report(metrics=final_metrics, checkpoint=final_checkpoint)
    else:
        print("⚠️ [Trainer Fn] Failed to save final checkpoint.")
        if result:
            train.report(metrics=result)

    algo.stop()

def train_stage_sequential_pbt(
    stage: TrainingStage,
    tune_config: Dict[str, Any],
    debug_mode: bool,
    start_checkpoint: Optional[str],
    metric_path: str = "env_runners/episode_return_mean",
    use_pretrained: bool = True,
    load_weights: bool = True,
    use_xgb_teacher: bool = False,  # New parameter
    xgb_model_path: str = "xgboost_coldstart.pkl",  # New parameter
    teacher_config: dict = None  # New parameter
) -> Tuple[Optional[str], Dict[str, Any]]:
    
    candidates_per_gen = tune_config.get("candidates_per_gen", 4)
    steps_per_gen = tune_config.get("steps_per_gen", 200_000)

    results_path = Path(tune_config.get("main_results_path", "training_results_DEFAULT"))
    policy_pool_dir = results_path / f"{stage.name}_policy_pool"

    best_hp = {"lr": 0.0001, "entropy_coeff": 0.008, "vf_loss_coeff": 0.5}
    best_ckpt = start_checkpoint

    gen = 0
    while True: 
        candidates = [deepcopy(best_hp)] + [mutate_hparams(best_hp) for _ in range(candidates_per_gen - 1)]

        base_cfg_obj = create_impala_config(
            stage=stage,
            tune_config=tune_config,
            debug_mode=debug_mode,
            hyperparams={},
            policy_pool_dir=str(policy_pool_dir),
            max_versions=25, keep_top=10, active_zone=15, auto_prune=True,
            use_pretrained=use_pretrained,
            load_weights=load_weights,
            use_xgb_teacher=use_xgb_teacher,  # Pass new parameters
            xgb_model_path=xgb_model_path,
            teacher_config=teacher_config
        )
        base_cfg = base_cfg_obj.to_dict()

        base_cfg["_restore_from"] = best_ckpt
        base_cfg["_stop_after"] = steps_per_gen

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
            tune.with_resources(train_impala_with_restore, resources=resources_per_trial),
            param_space=base_cfg,
            tune_config=tune.TuneConfig(
                num_samples=1, 
                max_concurrent_trials=tune_config["max_concurrent"],
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
            print(f"ERROR during tuner.fit() for Gen {gen+1}: {e}")
            print("Stopping PBT for this stage.")
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

        elif best_result:
            pass
        else:
            break

        gen += 1
        
        break 

    return best_ckpt, best_hp

def main():
    # --- Configuration for XGBoost Teacher Integration ---
    
    # Option A: With XGBoost Teacher
    USE_XGB_TEACHER = True
    XGB_MODEL_PATH = "xgboost_coldstart.pkl"  # Path to your trained XGBoost model
    TEACHER_CONFIG = {
        "teacher_mode": "linear",  # Options: "linear", "exp", "adaptive"
        "teacher_start_prob": 0.8,  # Start with 80% teacher actions
        "teacher_end_prob": 0.0,    # End with 0% teacher actions (pure RL)
        "teacher_decay_steps": 5_000_000,  # Decay over 5M steps
        "distill_logging": True     # Log teacher usage for analysis
    }
    
    # Option B: Without XGBoost Teacher (original behavior)
    # USE_XGB_TEACHER = False
    # XGB_MODEL_PATH = ""
    # TEACHER_CONFIG = {}
    
    # Existing configuration
    USE_PRETRAINED = True         
    LOAD_PRETRAINED_WEIGHTS = True
    results_path_name = "training_results_WITH_XGB_TEACHER" if USE_XGB_TEACHER else "training_results_NO_TEACHER"

    # ----------------------------------------------
    
    scheduler_mode = "pbt_sequential"
    
    tune_config = None
    if scheduler_mode == "pbt_sequential":
        tune_config = {
            "num_trials": 1,
            "max_concurrent": 1,
            "gpu_per_trial": 1,
            "num_env_runners": 22,
            "cpus_per_runner": 1,
            "candidates_per_gen": 3,
            "steps_per_gen": 10_000_000,
        }

    elif scheduler_mode == "pbt_parallel":
        tune_config = {
            "num_trials": 2,
            "max_concurrent": 2,
            "gpu_per_trial": 0.5,
            "num_env_runners": 9,
            "cpus_per_runner": 1,
            "perturbation_interval": 1_000_000,
        }
    else:
        raise ValueError(f"Unknown SCHEDULER_MODE: {scheduler_mode}")

    start_stage_index = 0
    end_stage_index = len(TRAINING_STAGES) - 1
    
    debug_mode = False
    initial_checkpoint = None

    ray.init(ignore_reinit_error=True, log_to_driver=False, local_mode=debug_mode, address="local")

    register_env("gfootball_multi", lambda config: GFootballMultiAgentEnv(config))
    
    if USE_PRETRAINED:
        ModelCatalog.register_custom_model("pretrained_model", PretrainedGFootballModel)
    
    ModelCatalog.register_custom_model("GFootballGNN", GFootballGNN)
    ModelCatalog.register_custom_model("GFootballMamba", GFootballMamba)

    current_checkpoint = initial_checkpoint
    final_best_hparams = None
    
    results_path = Path(__file__).resolve().parent / results_path_name
    tune_config["main_results_path"] = str(results_path) 

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
            print(f"⚠️ [TensorBoard] Could not write command to file: {e}")

        if scheduler_mode == "pbt_sequential":
            stage_checkpoint, stage_hparams = train_stage_sequential_pbt(
                stage=stage,
                tune_config=tune_config,
                debug_mode=debug_mode,
                start_checkpoint=current_checkpoint,
                use_pretrained=USE_PRETRAINED,
                load_weights=LOAD_PRETRAINED_WEIGHTS,
                use_xgb_teacher=USE_XGB_TEACHER,  # Pass XGBoost teacher config
                xgb_model_path=XGB_MODEL_PATH,
                teacher_config=TEACHER_CONFIG
            )
            final_best_hparams = stage_hparams

        else:
             print(f"Scheduler mode '{scheduler_mode}' not configured for stage execution.")
             break

        if stage_checkpoint:
            current_checkpoint = stage_checkpoint
        else:
            print(f"⚠️ Stage {i + 1} ({stage.name}) failed or produced no checkpoint. Stopping progressive training.")
            break

    ray.shutdown()

if __name__ == "__main__":
    main()