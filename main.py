import os
from pathlib import Path
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

import gymnasium as gym
from gymnasium import spaces
import gfootball.env as football_env

import ray
from ray import tune
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import MultiAgentDict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.callback import Callback


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
    TrainingStage("stage_1_basic", "academy_pass_and_shoot_with_keeper", "extracted", 1, 0, 0.75, 10_000_000, "1 Spieler gegen Keeper"),
    TrainingStage("stage_2_1v1", "academy_3_vs_1_with_keeper", "extracted", 1, 1, 0.75, 20_000_000, "1v1 Spiel"),
    TrainingStage("stage_3_3v3", "11_vs_11_easy_stochastic", "extracted", 3, 3, 1.0, 50_000_000, "3v3 Kleinfeld"),
    TrainingStage("stage_4_5v5", "11_vs_11_stochastic", "extracted", 5, 5, 1.0, 100_000_000, "5v5 Mittelfeld"),
    TrainingStage("stage_5_11v11", "11_vs_11_stochastic", "extracted", 11, 11, 1.0, 500_000_000, "11v11 Vollspiel"),
]

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
        
        if not self.agent_ids: single_obs_shape = test_obs.shape
        elif len(test_obs.shape) > 1: single_obs_shape = test_obs[0].shape
        else:
            assert test_obs.shape[0] % len(self.agent_ids) == 0
            single_obs_shape = (test_obs.shape[0] // len(self.agent_ids),)

        self.observation_space = spaces.Dict({
            agent_id: spaces.Box(low=-np.inf, high=np.inf, shape=single_obs_shape, dtype=np.float32)
            for agent_id in self.agent_ids})
        self.action_space = spaces.Dict({
            agent_id: spaces.Discrete(19) for agent_id in self.agent_ids})

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

class RestoreCheckpointCallback(Callback):
    def __init__(self, checkpoint_path: str):
        super().__init__()
        self.checkpoint_path = checkpoint_path

    def on_trial_start(self, iteration: int, trials: list, trial, **info):
        print(f"Trial {trial.trial_id} loading weights from: {self.checkpoint_path}")
        trial.get_trainable().restore(self.checkpoint_path)

def get_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "policy_left" if "left" in agent_id else "policy_right"

def select_policy_to_train(algorithm, train_batch, **kwargs):
    """Dynamically selects which policy to train based on the callback's state."""
    metrics = algorithm.get_local_worker().get_metrics()
    active_policy = metrics.get("custom_metrics", {}).get("active_policy", "policy_left")
    return [active_policy]

class SelfPlayCallback(DefaultCallbacks):
    def __init__(self, update_interval=25, has_left_agents=True, has_right_agents=True):
        super().__init__()
        self.update_interval = update_interval
        self._update_counter = 0
        self.self_play_enabled = has_left_agents and has_right_agents
        self.active_policy_name = "policy_left"

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        if self.self_play_enabled and self._update_counter % self.update_interval == 0:
            self.active_policy_name = "policy_right" if self.active_policy_name == "policy_left" else "policy_left"
            print(f"\n--- Active training policy will switch to: {self.active_policy_name} for the next iterations ---")
        
        result["custom_metrics"]["active_policy"] = self.active_policy_name

        if self.self_play_enabled:
            active_policy_obj = algorithm.get_policy(self.active_policy_name)
            inactive_policy_name = "policy_left" if self.active_policy_name == "policy_right" else "policy_right"
            inactive_policy_obj = algorithm.get_policy(inactive_policy_name)
            
            if active_policy_obj and inactive_policy_obj:
                inactive_policy_obj.set_weights(active_policy_obj.get_weights())
        
        self._update_counter += 1

def create_impala_config(stage: TrainingStage, debug_mode: bool = False, hyperparams: dict = None) -> ImpalaConfig:
    config = ImpalaConfig()
    config.api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
    
    env_config = {
        "debug_mode": debug_mode, "representation": stage.representation, "env_name": stage.env_name,
        "number_of_left_players_agent_controls": stage.left_players,
        "number_of_right_players_agent_controls": stage.right_players,
        "rewards": "scoring,checkpoints",
    }
    config.environment("gfootball_multi", env_config=env_config, disable_env_checking=True)
    config.framework("torch")
    config.env_runners(num_env_runners=0 if debug_mode else 7, num_envs_per_env_runner=2,
                     rollout_fragment_length=10 if debug_mode else 64)
    config.fault_tolerance(restart_failed_env_runners=True)
    
    if hyperparams is None:
        hyperparams = {"lr": 0.0001, "entropy_coeff": 0.008, "vf_loss_coeff": 0.5}

    model_config = {
        "framestack": True,
        "conv_filters": [
            [32, [3, 3], 2], [32, [3, 3], 1],
            [64, [3, 3], 2], [64, [3, 3], 1],
            [128, [3, 3], 2], [128, [3, 3], 1],
        ],
        "conv_activation": "silu",
        "use_attention": True,
        "attention_num_transformer_units": 3,   # +1 Block
        "attention_dim": 256,                   # 4*64
        "attention_num_heads": 4,
        "attention_head_dim": 64,
        "attention_memory_inference": 64,
        "attention_memory_training": 64,
        "attention_position_wise_mlp_dim": 1024,
        "max_seq_len": 64,
        "fcnet_hiddens": [512, 512],
        "fcnet_activation": "silu",
    }



    config.training(
        lr=hyperparams["lr"], 
        entropy_coeff=hyperparams["entropy_coeff"],
        vf_loss_coeff=hyperparams["vf_loss_coeff"], 
        grad_clip=0.5,
        train_batch_size=8192 if not debug_mode else 128,
        minibatch_size=1024 if not debug_mode else 128,
        model=model_config
    )

    config.resources(num_gpus=1/3, num_cpus_for_main_process=1)
    config.learners(num_learners=1, num_gpus_per_learner=1/3, num_cpus_per_learner=1)

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
    config.callbacks(lambda: SelfPlayCallback(25, has_left, has_right))
    config.debugging(seed=42, log_level="WARN")
    return config

def train_single_stage(stage, stage_index, debug_mode, restore_checkpoint):
    print("\n" + "="*80)
    print(f"STARTING STAGE {stage_index + 1}: {stage.name} - {stage.description}")
    if restore_checkpoint:
        print(f"  -> Initializing all trials from: {restore_checkpoint}")
    print("="*80 + "\n")

    results_path = Path(__file__).resolve().parent / "training_results_transfer_pbt"

    param_space = create_impala_config(
        stage=stage,
        debug_mode=debug_mode,
        hyperparams={
            "lr": tune.choice([0.0005, 0.0003, 0.0001]),
            "entropy_coeff": tune.choice([0.006, 0.008, 0.01]),
            "vf_loss_coeff": tune.choice([0.4, 0.5, 0.6]),
        }
    ).to_dict()

    stop_criteria = {
        "env_runners/episode_return_mean": stage.target_reward,
        "timesteps_total": stage.max_timesteps,
    }
    
    checkpoint_config = CheckpointConfig(
        num_to_keep=3, checkpoint_frequency=50,
        checkpoint_score_attribute="env_runners/episode_return_mean",
        checkpoint_score_order="max", checkpoint_at_end=True,
    )
    
    pbt_scheduler = PopulationBasedTraining(
        time_attr="timesteps_total", metric="env_runners/episode_return_mean",
        mode="max", perturbation_interval=128_000,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-5, 1e-3),
            "entropy_coeff": tune.uniform(0.0, 0.02),
            "vf_loss_coeff": tune.uniform(0.1, 1.0),
        }
    )
    
    run_callbacks = []
    if restore_checkpoint:
        run_callbacks.append(RestoreCheckpointCallback(checkpoint_path=restore_checkpoint))

    tuner = tune.Tuner(
        "IMPALA",
        param_space=param_space,
        tune_config=tune.TuneConfig(scheduler=pbt_scheduler, num_samples=3),
        run_config=RunConfig(
            stop=stop_criteria,
            checkpoint_config=checkpoint_config,
            name=f"{stage.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            storage_path=str(results_path),
            callbacks=run_callbacks
        ),
    )
    
    results = tuner.fit()
    best_result = results.get_best_result(metric="env_runners/episode_return_mean", mode="max")
    
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
    print("STARTING PROGRESSIVE TRANSFER LEARNING WITH PBT")
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
    debug_mode = os.environ.get("GFOOTBALL_DEBUG", "").lower() == "true"
    use_transfer = os.environ.get("GFOOTBALL_TRANSFER", "true").lower() == "true"
    start_stage = int(os.environ.get("GFOOTBALL_START_STAGE", "0"))
    end_stage_env = os.environ.get("GFOOTBALL_END_STAGE", "")
    end_stage = int(end_stage_env) if end_stage_env else None
    initial_checkpoint = os.environ.get("GFOOTBALL_CHECKPOINT", None)
    
    ray.init(ignore_reinit_error=True, log_to_driver=False, local_mode=debug_mode)
    register_env("gfootball_multi", lambda config: GFootballMultiAgentEnv(config))
    
    if use_transfer:
        train_progressive(start_stage, end_stage, debug_mode, initial_checkpoint)
    else:
        stage = TRAINING_STAGES[start_stage]
        train_single_stage(stage, start_stage, debug_mode, initial_checkpoint)
    
    ray.shutdown()

if __name__ == "__main__":
    main()