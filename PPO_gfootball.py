import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
from gymnasium import spaces
import gfootball.env as football_env

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig, PPO
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining

from ray.rllib.models import ModelCatalog

from model_3 import GFootballMamba

class GFootballMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        workspace_root = Path(__file__).resolve().parent

        gf_logdir = workspace_root / "gfootball_logs"
        gf_logdir.mkdir(parents=True, exist_ok=True)

        default_config = {
            "env_name": "academy_run_to_score_with_keeper",
            "representation": "simple115v2",
            "rewards": "scoring,checkpoints",
            "number_of_left_players_agent_controls": 1,
            "number_of_right_players_agent_controls": 0,
            "stacked": True,
            "logdir": str(gf_logdir),
            "write_goal_dumps": False,
            "write_full_episode_dumps": False,
            "render": False,
            "write_video": False,
            "dump_frequency": 0,
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
        self.agents = list(self.agent_ids)

        return self.latest_obs, {aid: {} for aid in self.agent_ids}

    def step(self, action_dict):
        actions = [action_dict.get(aid, self.action_space.sample()) for aid in self.agent_ids]
        step_result = self.env.step(actions)

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


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    """Map agent IDs to policy IDs."""
    if agent_id.startswith("left"):
        return "policy_left"
    else:
        return "policy_right"


def main():
    workspace_root = Path(__file__).resolve().parent

    ray_tmp_dir = workspace_root / "ray_tmp"
    ray_tmp_dir.mkdir(parents=True, exist_ok=True)

    os.environ["RAY_TMPDIR"] = str(ray_tmp_dir)
    os.environ["RAY_TEMP_DIR"] = str(ray_tmp_dir)

    ray.init(
        include_dashboard=False,
        ignore_reinit_error=True,
        num_gpus=1,
        _temp_dir=str(ray_tmp_dir),
    )

    env_config = {
        "env_name": "academy_run_to_score_with_keeper",
        "number_of_left_players_agent_controls": 1,
        "number_of_right_players_agent_controls": 0,
    }

    register_env("gfootball_multi", lambda cfg: GFootballMultiAgentEnv(cfg))

    dummy_env = GFootballMultiAgentEnv(env_config)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    dummy_env.close()
    
    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="episode_reward_mean",
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

    config = PPOConfig()
    config = config.environment(
        env="gfootball_multi",
        env_config=env_config,
    )
    config = config.framework("torch")

    config = config.rollouts(
        num_rollout_workers=10,
        num_envs_per_worker=1,
        batch_mode="complete_episodes",
    )

    ModelCatalog.register_custom_model("GFootballMamba", GFootballMamba)
    
    config = config.training(
        train_batch_size=32768,
        sgd_minibatch_size=8192,
        num_sgd_iter=5,
        lr=5e-6,
        gamma=0.998,
        entropy_coeff=0.01,
        vf_loss_coeff=0.5,
        grad_clip=0.5,
        grad_clip_by="global_norm",
        kl_coeff=0.2,
        kl_target=0.01,
        model={
        "custom_model": "GFootballMamba",
        "max_seq_len": 256,
        "custom_model_config": {
            "d_model": 128,
            "mamba_state": 16,
            "num_mamba_layers": 6,
            "prev_action_emb": 16,
            "gradient_checkpointing": False,
            "mlp_hidden_dims": [256, 256],
            "mlp_activation": "silu",
            "head_hidden_dims": [256],
            "head_activation": "silu",
            "use_distributional": True,
            "v_min": -10.0,
            "v_max": 10.0,
            "num_atoms": 51,
        },
    },
    )
    config = config.resources(
        num_cpus_for_local_worker=2,
        num_gpus=0.5,
    )
    config = config.multi_agent(
        policies={
            "policy_left": (None, obs_space, act_space, {}),
            "policy_right": (None, obs_space, act_space, {}),
        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["policy_left"],
    )
    config = config.debugging(
        log_level="WARN",
    )

    config_dict = config.to_dict()

    print("Starte Ray Tune Training mit PBT...")

    results_dir = workspace_root / "ray_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    analysis = tune.run(
        PPO,
        config=config_dict,
        scheduler=pbt_scheduler,
        num_samples=2,
        stop={
            "episode_reward_mean": 20.0,
            "training_iteration": 2000,
        },
        local_dir=str(results_dir),
        name="PPO_GFootball_PBT",
        checkpoint_freq=5,
        checkpoint_at_end=True,
    )

    print("Training abgeschlossen.")

    try:
        best_trial = analysis.get_best_trial(
            metric="episode_reward_mean",
            mode="max",
            scope="all",
        )
        best_result = best_trial.last_result
        best_checkpoint = analysis.get_best_checkpoint(
            best_trial,
            metric="episode_reward_mean",
            mode="max",
        )
        print("Bestes Ergebnis erreicht:")
        print(f"  Return Mean: {best_result['episode_reward_mean']}")
        print(f"  Bester Checkpoint: {best_checkpoint}")
    except Exception as e:
        print("Konnte bestes Ergebnis nicht bestimmen:", e)

    ray.shutdown()


if __name__ == "__main__":
    main()
