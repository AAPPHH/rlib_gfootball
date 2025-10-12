import os
from pathlib import Path
import sys
import json
import numpy as np
from typing import Dict, Any, Optional, Tuple
from datetime import datetime

import gymnasium as gym
from gymnasium import spaces
import gfootball.env as football_env

import ray
from ray import tune
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.utils.typing import MultiAgentDict
from ray.tune.registry import register_env
from ray.air.config import RunConfig, CheckpointConfig


class GFootballMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        default_env_config = {
            "env_name": "11_vs_11_stochastic",
            "representation": "simple115v2",
            "rewards": "scoring,checkpoints",
            "number_of_left_players_agent_controls": 11,
            "number_of_right_players_agent_controls": 11,
            "stacked": True,
            "logdir": "/tmp/gfootball",
            "write_goal_dumps": False,
            "write_full_episode_dumps": False,
            "render": False,
            "write_video": False,
            "dump_frequency": 1,
        }
        
        self.env_config = {**default_env_config, **config}
        self.debug_mode = self.env_config.get("debug_mode", False)
        if self.debug_mode:
            self._apply_debug_overrides()

        self.left_players = self.env_config["number_of_left_players_agent_controls"]
        self.right_players = self.env_config["number_of_right_players_agent_controls"]
        
        self._create_env()
        
        self.agent_ids = [f"left_{i}" for i in range(self.left_players)] + \
                         [f"right_{i}" for i in range(self.right_players)]
        self._agent_ids = set(self.agent_ids)

        test_obs = self.env.reset()
        if isinstance(test_obs, tuple):
            test_obs = test_obs[0]
        
        single_obs_shape = test_obs[0].shape if len(test_obs.shape) > 1 else (test_obs.shape[0] // len(self.agent_ids),)

        single_obs_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=single_obs_shape,
            dtype=np.float32
        )
        
        single_act_space = spaces.Discrete(19)

        self.observation_space = spaces.Dict({agent_id: single_obs_space for agent_id in self.agent_ids})
        self.action_space = spaces.Dict({agent_id: single_act_space for agent_id in self.agent_ids})

    def _apply_debug_overrides(self):
        self.env_config["render"] = True
        self.env_config["write_video"] = True

    def _create_env(self):
        creation_kwargs = self.env_config.copy()
        creation_kwargs.pop("debug_mode", None)
        self.env = football_env.create_environment(**creation_kwargs)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        super().reset(seed=seed, options=options)
        
        obs = self.env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        
        multi_agent_obs = self._split_obs(obs)
        infos = {agent_id: {} for agent_id in self.agent_ids}
        return multi_agent_obs, infos

    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        actions = [action_dict.get(agent_id, 0) for agent_id in self.agent_ids]
        
        step_result = self.env.step(actions)
        
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, info = step_result
            done = terminated or truncated
        else:
            obs, rewards, done, info = step_result
            terminated = done
            truncated = False

        multi_agent_obs = self._split_obs(obs)
        multi_agent_rewards = self._split_rewards(rewards)
        
        multi_agent_terminateds = {agent_id: done for agent_id in self.agent_ids}
        multi_agent_truncateds = {agent_id: done for agent_id in self.agent_ids}
        multi_agent_terminateds["__all__"] = done
        multi_agent_truncateds["__all__"] = done
        
        multi_agent_infos = {agent_id: info for agent_id in self.agent_ids}
        
        if self.debug_mode:
            import time
            time.sleep(1.0 / 10)

        return (multi_agent_obs, multi_agent_rewards, multi_agent_terminateds, 
                multi_agent_truncateds, multi_agent_infos)

    def _split_obs(self, obs: np.ndarray) -> MultiAgentDict:
        if len(obs.shape) == 1:
            obs_per_agent = obs.shape[0] // len(self.agent_ids)
            obs = obs.reshape(len(self.agent_ids), obs_per_agent)
        
        return {self.agent_ids[i]: obs[i].astype(np.float32) for i in range(len(self.agent_ids))}

    def _split_rewards(self, rewards: np.ndarray) -> MultiAgentDict:
        if isinstance(rewards, (int, float)):
            return {agent_id: float(rewards) for agent_id in self.agent_ids}
        
        return {self.agent_ids[i]: float(rewards[i]) for i in range(len(self.agent_ids))}

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def get_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "left_team" if "left" in agent_id else "right_team"

def get_shared_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "shared_policy"


def create_impala_config(debug_mode: bool = False) -> ImpalaConfig:
    config = ImpalaConfig()

    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )

    config.environment(
        env="gfootball_multi",
        env_config={"debug_mode": debug_mode},
        disable_env_checking=True,
    )
    config.framework(framework="torch")

    config.env_runners(
        num_env_runners=0 if debug_mode else 22,
        num_envs_per_env_runner=2,
        rollout_fragment_length=10 if debug_mode else 64,
        num_cpus_per_env_runner=1,
        num_gpus_per_env_runner=0,
    )

    config.fault_tolerance(
        restart_failed_env_runners=True,
    )
    
    config.training(
        lr_schedule=[
            [0,          0.0001],
            [25_000_000, 0.000001],
        ],
        train_batch_size=50 if debug_mode else 2048,
        grad_clip=0.5,
        vf_loss_coeff=0.5,
        entropy_coeff=0.008,
        model={
            "framestack": False,
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "silu",

            "use_attention": True,
            "attention_num_transformer_units": 2,
            "attention_dim": 192,
            "attention_num_heads": 3, 
            "attention_head_dim": 64,
            "attention_memory_inference": 64,
            "attention_memory_training": 64,
            "attention_position_wise_mlp_dim": 768,
            "max_seq_len": 64,
        }
    )

    config.resources(
        num_gpus=1,
        num_cpus_for_main_process=1,
    )
    
    config.learners(
        num_learners=1,
        num_gpus_per_learner=1,
        num_cpus_per_learner=1,
    )
    
    policy_mapping_mode = "shared"
    
    if policy_mapping_mode == "shared":
        policies = {"shared_policy": PolicySpec()}
        policies_to_train = ["shared_policy"]
        mapping_fn = get_shared_policy_mapping_fn
    else:
        policies = {
            "left_team": PolicySpec(),
            "right_team": PolicySpec(),
        }
        policies_to_train = ["left_team", "right_team"]
        mapping_fn = get_policy_mapping_fn
    
    config.multi_agent(
        policies=policies,
        policy_mapping_fn=mapping_fn,
        policies_to_train=policies_to_train,
    )
    
    if not debug_mode:
        config.evaluation(
            evaluation_interval=50,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=1,
            evaluation_config=config.overrides(explore=False),
        )
        
    config.debugging(
        seed=42,
        log_level="INFO", 
    )

    return config


def train():
    debug_mode = os.environ.get("GFOOTBALL_DEBUG", "").lower() == "true"
    
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=False,
        local_mode=debug_mode,
    )
    
    register_env("gfootball_multi", lambda config: GFootballMultiAgentEnv(config))
    
    impala_config = create_impala_config(debug_mode=debug_mode)

    script_path = Path(__file__).resolve().parent
    results_path = script_path / "training_results"
    print(f"Alle Trainingsergebnisse werden in folgendem Ordner gespeichert: {results_path}")

    resume_from_path = None
    if results_path.exists():
        experiment_folders = [d for d in results_path.iterdir() if d.is_dir()]
        if experiment_folders:
            latest_experiment = max(experiment_folders, key=lambda p: p.name)
            resume_from_path = str(latest_experiment)

    stop_criteria = {
        "episode_reward_mean": 20.0,
        "timesteps_total": 1_000_000_000,
    }
    
    checkpoint_config = CheckpointConfig(
        num_to_keep=5,
        checkpoint_frequency=10,
        checkpoint_score_attribute="episode_reward_mean",
        checkpoint_score_order="max",
        checkpoint_at_end=True,
    )

    if resume_from_path:
        print(f"Neuestes Experiment gefunden. Setze Training fort von: {resume_from_path}")
        tuner = tune.Tuner.restore(
            path=resume_from_path,
            trainable="IMPALA",
            resume_errored=True,
            param_space=impala_config.to_dict(),
        )
    else:
        print("Kein bestehendes Experiment gefunden. Starte ein neues Training.")
        tuner = tune.Tuner(
            "IMPALA",
            param_space=impala_config.to_dict(),
            run_config=RunConfig( 
                stop=stop_criteria,
                checkpoint_config=checkpoint_config,
                name=f"gfootball_impala_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                storage_path=str(results_path),
            ),
        )
    
    results = tuner.fit()
    
    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean", 
        mode="max"
    )
    if best_result and best_result.checkpoint:
        print("Training beendet. Bester Checkpoint gefunden unter:", best_result.checkpoint)
    else:
        print("Training beendet. Es wurde kein bester Checkpoint gefunden.")

    ray.shutdown()

if __name__ == "__main__":
    train()