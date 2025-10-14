import os
from pathlib import Path
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

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.registry import register_env
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune.schedulers import PopulationBasedTraining


class GFootballMultiAgentEnv(MultiAgentEnv):
    """
    RLlib MultiAgentEnv wrapper for the Google Research Football environment.
    """
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

        reset_result = self.env.reset()
        test_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        if len(self.agent_ids) == 0:
            single_obs_shape = test_obs.shape
        elif len(test_obs.shape) > 1:
            single_obs_shape = test_obs[0].shape
        else:
            num_features = test_obs.shape[0]
            num_agents = len(self.agent_ids)
            assert num_features % num_agents == 0, \
                f"Observation size ({num_features}) is not cleanly " \
                f"divisible by the number of agents ({num_agents})."
            single_obs_shape = (num_features // num_agents,)

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
        
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
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
            terminated, truncated = done, False

        multi_agent_obs = self._split_obs(obs)
        multi_agent_rewards = self._split_rewards(rewards)
        
        multi_agent_terminateds = {agent_id: done for agent_id in self.agent_ids}
        multi_agent_truncateds = {agent_id: truncated for agent_id in self.agent_ids}
        multi_agent_terminateds["__all__"] = done
        multi_agent_truncateds["__all__"] = done
        
        multi_agent_infos = {agent_id: info for agent_id in self.agent_ids}
        
        if self.debug_mode:
            import time
            time.sleep(1.0 / 20)

        return (multi_agent_obs, multi_agent_rewards, multi_agent_terminateds, 
                multi_agent_truncateds, multi_agent_infos)

    def _split_obs(self, obs: np.ndarray) -> MultiAgentDict:
        if not self.agent_ids:
            return {}
        if len(obs.shape) == 1:
            obs_per_agent = obs.shape[0] // len(self.agent_ids)
            obs = obs.reshape(len(self.agent_ids), obs_per_agent)
        
        return {self.agent_ids[i]: obs[i].astype(np.float32) for i in range(len(self.agent_ids))}

    def _split_rewards(self, rewards) -> MultiAgentDict:
        if np.isscalar(rewards):
            return {agent_id: float(rewards) for agent_id in self.agent_ids}
        
        return {self.agent_ids[i]: float(rewards[i]) for i in range(len(self.agent_ids))}

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

class SelfPlayCallback(DefaultCallbacks):
    def __init__(self, update_interval: int = 25):
        super().__init__()
        self.update_interval = update_interval
        self._update_counter = 0

    def on_train_result(self, *, algorithm, result: dict, **kwargs):
        self._update_counter += 1
        if self._update_counter % self.update_interval == 0:
            print(f"\n--- [Self-Play] Updating opponent policy with main policy weights (Iteration: {self._update_counter}) ---\n")
            
            main_policy = algorithm.get_policy("main_policy")
            opponent_policy = algorithm.get_policy("opponent_policy")
            
            if main_policy and opponent_policy:
                main_weights = main_policy.get_weights()
                opponent_policy.set_weights(main_weights)
                result["custom_metrics"]["opponent_updated"] = 1.0
            else:
                result["custom_metrics"]["opponent_updated"] = 0.0

def get_self_play_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "main_policy" if "left" in agent_id else "opponent_policy"


def create_impala_config(debug_mode: bool = False, hyperparams: dict = None) -> ImpalaConfig:
    config = ImpalaConfig()

    config.api_stack(
        enable_rl_module_and_learner=False,
        enable_env_runner_and_connector_v2=False
    )

    config.environment(
        env="gfootball_multi",
        env_config={
            "debug_mode": debug_mode,
            "env_name": "academy_pass_and_shoot_with_keeper",
            "number_of_left_players_agent_controls": 1,
            "number_of_right_players_agent_controls": 0,
            "rewards": "scoring,checkpoints",
        },
        disable_env_checking=True,
    )

    config.framework(framework="torch")

    config.env_runners(
        # KORREKTUR: Reduziert, damit 3 Trials auf 24 CPUs passen (3 * (6+2) = 24)
        num_env_runners=0 if debug_mode else 6,
        num_envs_per_env_runner=2,
        gym_env_vectorize_mode="ASYNC",
        rollout_fragment_length=10 if debug_mode else 64,
        num_cpus_per_env_runner=1,
    )

    config.fault_tolerance(restart_failed_env_runners=True)

    if hyperparams is None:
        hyperparams = {
            "lr": 0.0001,
            "entropy_coeff": 0.008,
            "vf_loss_coeff": 0.5
        }

    config.training(
        lr=hyperparams["lr"],
        entropy_coeff=hyperparams["entropy_coeff"],
        vf_loss_coeff=hyperparams["vf_loss_coeff"],
        train_batch_size=50 if debug_mode else 2048,
        grad_clip=0.5,
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
        num_gpus=1/3,
        num_cpus_for_main_process=1,
    )
    
    config.learners(
        num_learners=1,
        num_gpus_per_learner=1/3,
        num_cpus_per_learner=1,
    )
    
    policies = {
        "main_policy": PolicySpec(),
        "opponent_policy": PolicySpec(),
    }
    
    policies_to_train = ["main_policy"]
    
    config.multi_agent(
        policies=policies,
        policy_mapping_fn=get_self_play_policy_mapping_fn,
        policies_to_train=policies_to_train,
    )
    
    config.callbacks(lambda: SelfPlayCallback(update_interval=25))
    
    if not debug_mode:
        config.evaluation(
            evaluation_interval=50,
            evaluation_duration=10,
            evaluation_duration_unit="episodes",
            evaluation_num_env_runners=1,
            evaluation_config=config.overrides(explore=False),
        )
        
    config.debugging(seed=42, log_level="INFO")

    return config

def train():
    debug_mode = os.environ.get("GFOOTBALL_DEBUG", "").lower() == "true"
    
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=False,
        local_mode=debug_mode,
    )
    
    register_env("gfootball_multi", lambda config: GFootballMultiAgentEnv(config))
    
    pbt_scheduler = PopulationBasedTraining(
        time_attr="timesteps_total",
        metric="env_runners/episode_return_mean",
        mode="max",
        perturbation_interval=5_000_000,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-5, 1e-3),
            "entropy_coeff": tune.uniform(0.0, 0.02),
            "vf_loss_coeff": tune.uniform(0.1, 1.0),
        }
    )

    param_space = create_impala_config(
        debug_mode=debug_mode,
        hyperparams={
            "lr": tune.choice([0.0005, 0.0003, 0.0001]),
            "entropy_coeff": tune.choice([0.006, 0.008, 0.01]),
            "vf_loss_coeff": tune.choice([0.4, 0.5, 0.6]),
        }
    ).to_dict()

    script_path = Path(__file__).resolve().parent
    results_path = script_path / "training_results_pbt"
    print(f"All PBT training results will be saved in: {results_path}")

    resume_from_path = None

    stop_criteria = {
        "env_runners/episode_return_mean": 20.0,
        "timesteps_total": 1_000_000_000,
    }
    
    checkpoint_config = CheckpointConfig(
        num_to_keep=5,
        checkpoint_frequency=50,
        checkpoint_score_attribute="env_runners/episode_return_mean",
        checkpoint_score_order="max",
        checkpoint_at_end=True,
    )
    
    print("No existing experiment found. Starting a new PBT training run.")
    tuner = tune.Tuner(
        "IMPALA",
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=pbt_scheduler,
            num_samples=3,
        ),
        run_config=RunConfig( 
            stop=stop_criteria,
            checkpoint_config=checkpoint_config,
            name=f"gfootball_impala_pbt_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            storage_path=str(results_path),
        ),
    )
    
    results = tuner.fit()
    
    best_result = results.get_best_result(
        metric="env_runners/episode_return_mean", 
        mode="max"
    )
    if best_result and best_result.checkpoint:
        print("Training finished. Best checkpoint found at:", best_result.checkpoint)
    else:
        print("Training finished. No best checkpoint was found.")

    ray.shutdown()


if __name__ == "__main__":
    train()