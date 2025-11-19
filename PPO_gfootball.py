import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
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
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID

from model_3 import GFootballMamba


class TrainingStage:
    def __init__(self, stage_id: int, env_name: str, representation: str, difficulty: int,
                 left_agents: int = 1, right_agents: int = 0):
        self.stage_id = stage_id
        self.env_name = env_name
        self.representation = representation
        self.difficulty = difficulty
        self.left_agents = left_agents
        self.right_agents = right_agents
        self.ema_return = 0.0
        self.ema_return_prev = 0.0
        self.ema_abs_return = 1.0
        self.learning_progress = 0.0
        self.episode_count = 0
        
TRAINING_STAGES = [
    TrainingStage(0, "academy_empty_goal_close", "simple115v2", 1, left_agents=1, right_agents=0),
    TrainingStage(1, "academy_run_to_score_with_keeper", "simple115v2", 2, left_agents=1, right_agents=0),
    TrainingStage(2, "academy_pass_and_shoot_with_keeper", "simple115v2", 3, left_agents=1, right_agents=0),
    TrainingStage(3, "academy_3_vs_1_with_keeper", "simple115v2", 4, left_agents=3, right_agents=0),
    TrainingStage(4, "academy_single_goal_versus_lazy", "simple115v2", 5, left_agents=3, right_agents=0),
    TrainingStage(5, "11_vs_11_easy_stochastic", "simple115v2", 6, left_agents=3, right_agents=0),
    TrainingStage(6, "11_vs_11_easy_stochastic", "simple115v2", 7, left_agents=3, right_agents=3),
    TrainingStage(7, "11_vs_11_stochastic", "simple115v2", 8, left_agents=11, right_agents=11)
]


class GFootballMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        
        workspace_root = Path(__file__).resolve().parent
        gf_logdir = workspace_root / "gfootball_logs"
        gf_logdir.mkdir(parents=True, exist_ok=True)
        
        self.curriculum_horizon = config.get("curriculum_horizon", 100000)
        self.ema_alpha = config.get("ema_alpha", 0.02)
        self.popart_epsilon = config.get("popart_epsilon", 1e-5)
        
        self.stages = TRAINING_STAGES.copy()
        self.episode_idx = 0
        self.current_stage = None
        self.current_stage_id = None
        
        self.max_left_agents = max(s.left_agents for s in self.stages)
        self.max_right_agents = max(s.right_agents for s in self.stages)
        
        default_config = {
            "representation": "simple115v2",
            "rewards": "scoring,checkpoints",
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
        
        self.env = None
        self._create_env_for_stage(self.stages[0])
        
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(4, 115) if self.env_config.get("stacked", True) else (115,),
                dtype=np.float32
            ),
            "stage_index": spaces.Box(low=0, high=len(self.stages)-1, shape=(1,), dtype=np.int32)
        })
        
        self.action_space = spaces.Discrete(19)
        
        self.latest_obs = {}
        self.latest_info = {}
        self.episode_raw_returns = []
        
    def _create_env_for_stage(self, stage: TrainingStage):
        if self.env is not None:
            self.env.close()
        
        creation_kwargs = self.env_config.copy()
        creation_kwargs["env_name"] = stage.env_name
        creation_kwargs["representation"] = stage.representation
        creation_kwargs["number_of_left_players_agent_controls"] = stage.left_agents
        creation_kwargs["number_of_right_players_agent_controls"] = stage.right_agents
        creation_kwargs.pop("debug_mode", None)
        creation_kwargs.pop("curriculum_horizon", None)
        creation_kwargs.pop("ema_alpha", None)
        creation_kwargs.pop("popart_epsilon", None)
        
        self.env = football_env.create_environment(**creation_kwargs)
        self.current_stage = stage
        self.current_stage_id = stage.stage_id
        
        self.left_players = stage.left_agents
        self.right_players = stage.right_agents
        
        self.agent_ids = [f"left_{i}" for i in range(self.left_players)] + [
            f"right_{i}" for i in range(self.right_players)
        ]
        self._agent_ids = set(self.agent_ids)
        self.agents = list(self.agent_ids)
        self.possible_agents = list(self.agent_ids)
        
    def _sample_stage_alp_gmm(self) -> TrainingStage:
        progress = min(1.0, self.episode_idx / self.curriculum_horizon)
        max_difficulty = 1 + int(progress * (len(self.stages) - 1))
        
        available_stages = [s for s in self.stages if s.difficulty <= max_difficulty]
        
        if len(available_stages) == 1:
            return available_stages[0]
        
        learning_progresses = np.array([s.learning_progress for s in available_stages])
        lp_min = learning_progresses.min()
        lp_max = learning_progresses.max()
        
        if lp_max - lp_min < 1e-6:
            probabilities = np.ones(len(available_stages)) / len(available_stages)
        else:
            normalized_lp = (learning_progresses - lp_min) / (lp_max - lp_min)
            probabilities = normalized_lp + 0.1
            probabilities = probabilities / probabilities.sum()
        
        selected_idx = np.random.choice(len(available_stages), p=probabilities)
        return available_stages[selected_idx]
    
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        selected_stage = self._sample_stage_alp_gmm()
        if selected_stage != self.current_stage:
            self._create_env_for_stage(selected_stage)
        
        self.episode_idx += 1
        self.episode_raw_returns = []
        
        reset_result = self.env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        self.latest_info = {
            "score": [0, 0],
            "game_mode": 0,
            "stage_id": self.current_stage_id,
        }
        
        self.latest_obs = self._split_obs(obs)
        self.agents = list(self.agent_ids)
        
        return self.latest_obs, {aid: {"stage_id": self.current_stage_id} for aid in self.agent_ids}
    
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
        self.latest_info["stage_id"] = self.current_stage_id
        self.latest_obs = self._split_obs(obs)
        
        raw_rewards = self._split_rewards(rewards)
        self.episode_raw_returns.append(sum(raw_rewards.values()) / len(raw_rewards))
        
        scaled_rewards = {}
        for agent_id, raw_reward in raw_rewards.items():
            scaled_reward = raw_reward / max(self.current_stage.ema_abs_return, self.popart_epsilon)
            scaled_rewards[agent_id] = scaled_reward
        
        dones = {aid: terminated for aid in self.agent_ids}
        dones["__all__"] = terminated
        truncs = {aid: truncated for aid in self.agent_ids}
        truncs["__all__"] = truncated
        
        if terminated or truncated:
            self._update_stage_statistics()
        
        if self.debug_mode:
            self.env.render()
        
        agent_infos = {aid: {**info, "stage_id": self.current_stage_id} for aid in self.agent_ids}
        
        return self.latest_obs, scaled_rewards, dones, truncs, agent_infos
    
    def _update_stage_statistics(self):
        if not self.episode_raw_returns:
            return
        
        episode_return = sum(self.episode_raw_returns)
        abs_episode_return = abs(episode_return)
        
        self.current_stage.ema_return_prev = self.current_stage.ema_return
        self.current_stage.ema_return = (
            (1 - self.ema_alpha) * self.current_stage.ema_return +
            self.ema_alpha * episode_return
        )
        
        self.current_stage.ema_abs_return = (
            (1 - self.ema_alpha) * self.current_stage.ema_abs_return +
            self.ema_alpha * abs_episode_return
        )
        
        self.current_stage.learning_progress = abs(
            self.current_stage.ema_return - self.current_stage.ema_return_prev
        )
        
        self.current_stage.episode_count += 1
    
    def _split_obs(self, obs):
        if not self.agent_ids:
            return {}
        
        num_agents = len(self.agent_ids)
        
        if isinstance(obs, dict):
            processed = {}
            for aid in self.agent_ids:
                processed[aid] = {
                    "obs": obs[aid].astype(np.float32),
                    "stage_index": np.array([self.current_stage_id], dtype=np.int32)
                }
            return processed
        
        if not isinstance(obs, np.ndarray):
            obs = np.array(obs)
        
        # Handle different observation shapes
        if obs.ndim == 3 and obs.shape[0] == num_agents:
            # Already in correct shape: (num_agents, 4, 115)
            return {
                self.agent_ids[i]: {
                    "obs": obs[i].astype(np.float32),
                    "stage_index": np.array([self.current_stage_id], dtype=np.int32)
                }
                for i in range(num_agents)
            }
        elif obs.ndim == 2 and obs.shape[0] == num_agents:
            # Shape: (num_agents, 460) - needs reshaping to (num_agents, 4, 115)
            return {
                self.agent_ids[i]: {
                    "obs": obs[i].reshape(4, 115).astype(np.float32),
                    "stage_index": np.array([self.current_stage_id], dtype=np.int32)
                }
                for i in range(num_agents)
            }
        elif obs.ndim == 1 and obs.size == 460:
            # Single agent, flat observation - reshape to (4, 115)
            reshaped_obs = obs.reshape(4, 115).astype(np.float32)
            return {
                aid: {
                    "obs": reshaped_obs,
                    "stage_index": np.array([self.current_stage_id], dtype=np.int32)
                }
                for aid in self.agent_ids
            }
        elif obs.ndim == 2 and obs.shape == (4, 115):
            # Single agent, already in correct shape
            return {
                aid: {
                    "obs": obs.astype(np.float32),
                    "stage_index": np.array([self.current_stage_id], dtype=np.int32)
                }
                for aid in self.agent_ids
            }
        elif obs.ndim == 1 and obs.size == num_agents * 460:
            # Multiple agents, flat - reshape to (num_agents, 4, 115)
            obs_reshaped = obs.reshape(num_agents, 4, 115)
            return {
                self.agent_ids[i]: {
                    "obs": obs_reshaped[i].astype(np.float32),
                    "stage_index": np.array([self.current_stage_id], dtype=np.int32)
                }
                for i in range(num_agents)
            }
        else:
            # Fallback - create zero observation with correct shape
            print(f"Warning: Unexpected observation shape in _split_obs: {obs.shape}. Using zero observation.")
            zero_obs = np.zeros((4, 115), dtype=np.float32)
            return {
                aid: {
                    "obs": zero_obs,
                    "stage_index": np.array([self.current_stage_id], dtype=np.int32)
                }
                for aid in self.agent_ids
            }
    
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
    
    def observation_space_contains(self, x):
        return True
    
    def observation_space_sample(self):
        return {
            aid: {
                "obs": self.observation_space.spaces["obs"].sample(),
                "stage_index": self.observation_space.spaces["stage_index"].sample()
            }
            for aid in self.agent_ids
        }
    
    def action_space_contains(self, x):
        return all(self.action_space.contains(x[aid]) for aid in x if aid in self.agent_ids)
    
    def action_space_sample(self, agent_ids=None):
        if agent_ids is None:
            agent_ids = self.agent_ids
        return {aid: self.action_space.sample() for aid in agent_ids}
    
    def close(self):
        if self.env is not None:
            self.env.close()


class StageReturnCallbacks(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode: EpisodeV2, **kwargs):
        all_agents = episode.get_agents()
        if not all_agents:
            return
            
        first_agent = all_agents[0]
        if hasattr(episode, 'last_info_for'):
            agent_info_data = episode.last_info_for(first_agent)
        elif hasattr(episode, 'last_infos') and first_agent in episode.last_infos:
            agent_info_data = episode.last_infos[first_agent]
        else:
            agent_info_data = None
        stage_id = agent_info_data.get("stage_id", -1) if agent_info_data else -1
        
        if stage_id >= 0:
            total_rewards = {}
            for agent_id in all_agents:
                if agent_id.startswith("left"):
                    policy_id = "policy_left"
                else:
                    policy_id = "policy_right"
                    
                if (policy_id, agent_id) in episode.agent_rewards:
                    agent_rewards = episode.agent_rewards[(policy_id, agent_id)]
                    total_reward = sum(agent_rewards)
                    total_rewards[agent_id] = total_reward
                    episode.custom_metrics[f"return_stage_{stage_id}_agent_{agent_id}"] = total_reward
            
            if total_rewards:
                mean_reward = sum(total_rewards.values()) / len(total_rewards)
                episode.custom_metrics[f"return_stage_{stage_id}_mean_all_agents"] = mean_reward


def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
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
        "curriculum_horizon": 100000,
        "ema_alpha": 0.02,
        "popart_epsilon": 1e-5,
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
        disable_env_checking=True,  # Disable RLlib's environment checker
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
                "num_stages": len(TRAINING_STAGES),
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
    
    config = config.callbacks(StageReturnCallbacks)
    
    config = config.debugging(
        log_level="WARN",
    )
    
    config_dict = config.to_dict()
    
    print("Starting Ray Tune Training with Curriculum Learning...")
    
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
        storage_path=str(results_dir),
        name="PPO_GFootball_Curriculum",
        checkpoint_freq=5,
        checkpoint_at_end=True,
    )
    
    print("Training completed.")
    
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
        print("Best result achieved:")
        print(f"  Return Mean: {best_result['episode_reward_mean']}")
        print(f"  Best Checkpoint: {best_checkpoint}")
    except Exception as e:
        print("Could not determine best result:", e)
    
    ray.shutdown()


if __name__ == "__main__":
    main()