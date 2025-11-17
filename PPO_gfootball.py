import os
import time
from pathlib import Path
import random
from typing import Any, Dict
import numpy as np
from gymnasium import spaces
import gfootball.env as football_env
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining

class GFootballMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        ray_temp = os.environ.get("RAY_TEMP_DIR", "/tmp")
        gf_logdir = os.path.join(ray_temp, "gf")

        default_config = {
            "env_name": "academy_run_to_score_with_keeper",
            "representation": "simple115v2",
            "rewards": "scoring,checkpoints",
            "number_of_left_players_agent_controls": 1,
            "number_of_right_players_agent_controls": 0,
            "stacked": True,
            "logdir": gf_logdir,
            "write_goal_dumps": False,
            "write_full_episode_dumps": False,
            "render": False,
            "write_video": False,
            "dump_frequency": 1,
        }

        self.env_config = {**default_config, **config}
        self.debug_mode = self.env_config.get("debug_mode", False)

        if self.debug_mode:
            self.env_config.update({"render": True})

        self.left_players = self.env_config["number_of_left_players_agent_controls"]
        self.right_players = self.env_config["number_of_right_players_agent_controls"]

        creation_kwargs = self.env_config.copy()
        creation_kwargs.pop("debug_mode", None)
        creation_kwargs.pop("_reset_render_state", None)

        self.env = football_env.create_environment(**creation_kwargs)

        self.agent_ids = [f"left_{i}" for i in range(self.left_players)] + [
            f"right_{i}" for i in range(self.right_players)
        ]
        self._agent_ids = set(self.agent_ids)

        self.agents = list(self.agent_ids)
        self.possible_agents = list(self.agent_ids)

        try:
            _temp_env = football_env.create_environment(**creation_kwargs)
            _initial_obs_sample = _temp_env.reset()
            _initial_obs_sample = (
                _initial_obs_sample[0]
                if isinstance(_initial_obs_sample, tuple)
                else _initial_obs_sample
            )

            if not self.agent_ids:
                single_agent_obs_shape = _initial_obs_sample.shape
            elif (
                _initial_obs_sample.ndim > 0
                and _initial_obs_sample.shape[0]
                == (self.left_players + self.right_players)
                and (self.left_players + self.right_players) > 0
            ):
                single_agent_obs_shape = _initial_obs_sample.shape[1:]
            else:
                single_agent_obs_shape = _initial_obs_sample.shape

            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=single_agent_obs_shape,
                dtype=_initial_obs_sample.dtype,
            )

            if isinstance(self.env.action_space, spaces.Tuple):
                self.action_space = self.env.action_space.spaces[0]
            elif isinstance(self.env.action_space, spaces.Discrete):
                self.action_space = self.env.action_space
            elif hasattr(self.env.action_space, "nvec"):
                self.action_space = spaces.Discrete(self.env.action_space.nvec[0])
            else:
                self.action_space = spaces.Discrete(19)

            _temp_env.close()

        except Exception:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4, 115) if self.env_config.get("stacked", True) else (115,),
                dtype=np.float32,
            )
            self.action_space = spaces.Discrete(19)

        self.latest_obs = {}
        self.latest_info = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        self.latest_info = {
            "score": [0, 0],
            "game_mode": 0,
        }

        self.latest_obs = self._split_obs(obs)
        # agents-Liste für diese Episode (hier statisch)
        self.agents = list(self.agent_ids)

        return self.latest_obs, {aid: {} for aid in self.agent_ids}

    def step(self, action_dict):
        actions = [action_dict.get(aid, self.action_space.sample()) for aid in self.agent_ids]
        step_result = self.env.step(actions)

        # gfootball kann 4- oder 5-Tuple liefern → wie im IMPALA-Env behandeln
        if len(step_result) == 5:
            obs, rewards, terminated, truncated, info = step_result
        elif len(step_result) == 4:
            obs, rewards, done, info = step_result
            terminated, truncated = done, False
        else:
            raise ValueError(f"Unexpected step result format: {step_result}")

        self.latest_info = info
        self.latest_obs = self._split_obs(obs)

        dones = {aid: terminated for aid in self.agent_ids}
        dones["__all__"] = terminated
        truncs = {aid: truncated for aid in self.agent_ids}
        truncs["__all__"] = truncated

        if self.debug_mode:
            self.env.render()

        agent_infos = {aid: info for aid in self.agent_ids}
        return self.latest_obs, self._split_rewards(rewards), dones, truncs, agent_infos

    def _split_obs(self, obs):
        if not self.agent_ids:
            return {}
        num_agents = len(self.agent_ids)

        # IMPALA-Style: dict direkt durchlassen
        if isinstance(obs, dict):
            return obs

        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)

        if obs.ndim > 0 and obs.shape[0] == num_agents:
            return {
                self.agent_ids[i]: obs[i].astype(np.float32)
                for i in range(num_agents)
            }
        elif obs.shape == self.observation_space.shape and num_agents > 0:
            return {aid: obs.astype(np.float32) for aid in self.agent_ids}
        elif (
            obs.ndim == 1
            and num_agents > 0
            and self.observation_space.shape is not None
            and np.prod(self.observation_space.shape) > 0
            and obs.size == num_agents * int(np.prod(self.observation_space.shape))
        ):
            obs_reshaped = obs.reshape(num_agents, *self.observation_space.shape)
            return {
                self.agent_ids[i]: obs_reshaped[i].astype(np.float32)
                for i in range(num_agents)
            }

        zero_obs = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        return {aid: zero_obs.astype(np.float32) for aid in self.agent_ids}

    def _split_rewards(self, rewards):
        if np.isscalar(rewards):
            return {aid: float(rewards) for aid in self.agent_ids}
        if isinstance(rewards, (list, np.ndarray)):
            if len(rewards) == len(self.agent_ids):
                return {
                    self.agent_ids[i]: float(rewards[i])
                    for i in range(len(self.agent_ids))
                }
        return {aid: 0.0 for aid in self.agent_ids}

    def close(self):
        self.env.close()


def policy_mapping_fn(agent_id, episode, worker=None, **kwargs):
    if agent_id.startswith("left"):
        return "policy_left"
    else:
        return "policy_right"

def main():
    ray.init()
    
    env_config = {
        "env_name": "11_vs_11_easy_stochastic",
        "number_of_left_players_agent_controls": 11,
        "number_of_right_players_agent_controls": 0,
    }
    
    register_env("gfootball_multi", lambda cfg: GFootballMultiAgentEnv(cfg))
    
    dummy_env = GFootballMultiAgentEnv(env_config)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    dummy_env.close()

    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="episode_reward_mean",  # <-- GEÄNDERT
        mode="max",
        perturbation_interval=20,
        resample_probability=0.0,
        hyperparam_mutations={
            "lr": [1e-5, 2e-5, 4e-5, 6e-5, 1e-4],
            "num_sgd_iter": [3, 4, 5, 6, 7, 8],
            "gamma": [0.997, 0.9975, 0.998, 0.9985, 0.999],
            "entropy_coeff": [0.006, 0.007, 0.008, 0.009, 0.010, 0.012],
            "vf_loss_coeff": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        },
    )

    config = {
        "env": "gfootball_multi",
        "env_config": env_config,
        "framework": "torch",

        "enable_rl_module_and_learner": False,
        "enable_env_runner_and_connector_v2": False,

        "num_workers": 127,
        "num_envs_per_worker": 1,
        "num_cpus_per_worker": 1,
        
        "train_batch_size": 16384,
        "sgd_minibatch_size": 1024,
        "batch_mode": "complete_episodes",
        
        "num_sgd_iter": 5,
        "gamma": 0.998,
        "lr": 5e-6,
        
        "kl_coeff": 0.2,
        "num_gpus": 0.5,
        "log_level": "WARN",
        "grad_clip": 0.5,
        "model": {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "silu",
            "use_lstm": True,
            "lstm_cell_size": 512,
            "lstm_use_prev_action": True,
            "lstm_use_prev_reward": False,
            "vf_share_layers": True,
        },
        "multiagent": {
            "policies": {
                "policy_left": (None, obs_space, act_space, {}),
                "policy_right": (None, obs_space, act_space, {}),
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["policy_left"],
        },
    }

    tuner = ray.tune.Tuner(
            "PPO",
            param_space=config,
            run_config=ray.train.RunConfig(
                name="PPO_GFootball_PBT",
                stop={
                    "episode_reward_mean": 20.0,  # <-- GEÄNDERT
                    "training_iteration": 2_000_000,
                },
                checkpoint_config=ray.train.CheckpointConfig(
                    checkpoint_frequency=20,
                    checkpoint_at_end=True,
                ),
                storage_path=str(Path("ray_results").absolute()),
            ),
            tune_config=ray.tune.TuneConfig(
                scheduler=pbt_scheduler,
                num_samples=2
            )
        )

    print("Starte Ray Tune Training mit PBT...")
    results = tuner.fit()

    print("Training abgeschlossen.")
    best_result = results.get_best_result(metric="episode_reward_mean", mode="max")
    
    if best_result:
        best_checkpoint = best_result.get_best_checkpoint(metric="episode_reward_mean", mode="max") # <-- GEÄNDERT
        print(f"Bestes Ergebnis erreicht:")
        print(f"  Return Mean: {best_result.metrics['episode_reward_mean']}") # <-- GEÄNDERT
        print(f"  Bester Checkpoint: {best_checkpoint.path}")
    else:
        print("Training beendet, aber kein bestes Ergebnis gefunden.")

    
    ray.shutdown()


if __name__ == "__main__":
    main()