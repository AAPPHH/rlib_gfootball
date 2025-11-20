from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import os
import shutil
import gymnasium as gym
from gymnasium import spaces
import gfootball.env as football_env

import ray
from ray import tune
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.evaluation.episode_v2 import EpisodeV2

from model_3 import GFootballMambaRLModule

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
    TrainingStage(6, "11_vs_11_easy_stochastic", "simple115v2", 7, left_agents=3, right_agents=0),
    TrainingStage(7, "11_vs_11_stochastic", "simple115v2", 8, left_agents=11, right_agents=0)
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
        self.current_stage_id = -1
        self.env_base_config = {
            "write_goal_dumps": False,
            "write_full_episode_dumps": False,
            "render": False,
            "write_video": False,
            "dump_frequency": 0,
            "logdir": str(gf_logdir),
            "other_config_options": {} 
        }
        self.debug_mode = config.get("debug_mode", False)
        if self.debug_mode:
            self.env_base_config["render"] = True
        self.env = None
        self.agent_ids = [f"left_{i}" for i in range(11)] + [f"right_{i}" for i in range(11)]
        self._agent_ids = set(self.agent_ids)
        
        low_f32 = np.float32(-np.inf)
        high_f32 = np.float32(np.inf)
        
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(low=low_f32, high=high_f32, shape=(4, 115), dtype=np.float32),
            "stage_index": spaces.Box(low=0, high=len(self.stages), shape=(1,), dtype=np.int32)
        })
        self.action_space = spaces.Dict({
            aid: spaces.Discrete(19) for aid in self.agent_ids
        })
        self.latest_obs = {}
        self.episode_raw_returns = []
        self._create_env_for_stage(self.stages[0])

    def _create_env_for_stage(self, stage: TrainingStage):
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
        
    def _sample_stage_alp_gmm(self) -> TrainingStage:
        progress = min(1.0, self.episode_idx / self.curriculum_horizon)
        max_difficulty = 1 + int(progress * (len(self.stages) - 1))
        available = [s for s in self.stages if s.difficulty <= max_difficulty]
        if not available: return self.stages[0]
        if len(available) == 1: return available[0]
        lps = np.array([s.learning_progress for s in available])
        probs = (lps - lps.min()) / (lps.max() - lps.min() + 1e-6) + 0.1
        probs /= probs.sum()
        return available[np.random.choice(len(available), p=probs)]

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        target_stage = self._sample_stage_alp_gmm()
        if target_stage.stage_id != self.current_stage_id:
            self._create_env_for_stage(target_stage)
        self.episode_idx += 1
        self.episode_raw_returns = []
        raw_obs = self.env.reset()
        self.latest_obs = self._process_obs(raw_obs)
        info = {aid: {"stage_id": self.current_stage_id} for aid in self.active_agents}
        return self.latest_obs, info

    def step(self, action_dict: Dict[str, int]):
        internal_limit = 19
        if hasattr(self.env, 'action_space') and hasattr(self.env.action_space, 'n'):
             internal_limit = self.env.action_space.n
        elif hasattr(self.env, 'action_space') and isinstance(self.env.action_space, list):
             if len(self.env.action_space) > 0 and hasattr(self.env.action_space[0], 'n'):
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
            
        obs, rewards, done, info = self.env.step(actions_list)
        self.latest_obs = self._process_obs(obs)
        rewards_dict = {}
        raw_rewards_list = rewards if isinstance(rewards, (list, np.ndarray)) else [rewards]
        step_reward_sum = 0.0
        for i, aid in enumerate(self.active_agents):
            val = float(raw_rewards_list[i]) if i < len(raw_rewards_list) else 0.0
            rewards_dict[aid] = val
            step_reward_sum += val
        self.episode_raw_returns.append(step_reward_sum / len(self.active_agents))
        scale = max(self.current_stage.ema_abs_return, self.popart_epsilon)
        scaled_rewards = {k: v / scale for k, v in rewards_dict.items()}
        terminated = bool(done)
        truncated = False
        dones = {aid: terminated for aid in self.active_agents}
        dones["__all__"] = terminated
        truncs = {aid: truncated for aid in self.active_agents}
        truncs["__all__"] = truncated
        if terminated:
            self._update_stage_stats()
        agent_infos = {}
        for aid in self.active_agents:
            agent_infos[aid] = {
                "stage_id": self.current_stage_id,
                "popart_scale": scale,
                "team_raw_reward": step_reward_sum,
                "num_agents": len(self.active_agents)
            }
            if isinstance(info, dict):
                agent_infos[aid].update(info)
        return self.latest_obs, scaled_rewards, dones, truncs, agent_infos

    def _process_obs(self, raw_obs):
        if not isinstance(raw_obs, np.ndarray):
            raw_obs = np.array(raw_obs)
        if raw_obs.shape == (460,):
            raw_obs = raw_obs.reshape(1, 460)
        elif raw_obs.size == 1:
            raw_obs = np.zeros((len(self.active_agents), 460), dtype=np.float32)
        obs_dict = {}
        for i, aid in enumerate(self.active_agents):
            if i < len(raw_obs):
                data = raw_obs[i]
            else:
                data = np.zeros(460, dtype=np.float32)
            if data.size == 460:
                data = data.reshape(4, 115).astype(np.float32)
            else:
                data = np.zeros((4, 115), dtype=np.float32)
            obs_dict[aid] = {
                "obs": data,
                "stage_index": np.array([self.current_stage_id], dtype=np.int32)
            }
        return obs_dict

    def _update_stage_stats(self):
        if not self.episode_raw_returns: return
        ep_ret = sum(self.episode_raw_returns)
        self.current_stage.ema_return = (1-self.ema_alpha)*self.current_stage.ema_return + self.ema_alpha*ep_ret
        self.current_stage.ema_abs_return = (1-self.ema_alpha)*self.current_stage.ema_abs_return + self.ema_alpha*abs(ep_ret)
        self.current_stage.episode_count += 1
        
    def close(self):
        if self.env: self.env.close()

class StageReturnCallbacks(RLlibCallback):
    def on_episode_end(self, *, episode, **kwargs):
        agent_ids = list(episode.agent_ids)
        if not agent_ids:
            return
        first_agent = agent_ids[0]
        agent_info = None
        if hasattr(episode, "get_infos"):
            try:
                infos = episode.get_infos(first_agent)
                if infos:
                    agent_info = infos[-1]
            except Exception:
                pass 
        if agent_info is None:
            if hasattr(episode, "_agent_to_last_info"):
                agent_info = episode._agent_to_last_info.get(first_agent)
        if agent_info is None:
            return
        stage_id = agent_info.get("stage_id", -1)
        scale = agent_info.get("popart_scale", 1.0)
        raw_ret = agent_info.get("team_raw_reward", 0.0)
        norm_ret = raw_ret / max(scale, 1e-5)
        metrics = {
            "popart_normalized_return_mean": norm_ret,
            f"stage_{stage_id}_return": raw_ret,
            "current_stage": stage_id
        }
        if hasattr(episode, "add_custom_metrics"):
            episode.add_custom_metrics(metrics)
        elif hasattr(episode, "custom_metrics") and isinstance(episode.custom_metrics, dict):
            episode.custom_metrics.update(metrics)

def policy_mapping_fn(agent_id, episode=None, **kwargs):
    return "policy_left" if agent_id.startswith("left") else "policy_right"

def main():
    root = Path(__file__).parent
    res_dir = root / "ray_results"
    tmp_dir = root / "ray_tmp"
    res_dir.mkdir(exist_ok=True)
    tmp_dir.mkdir(exist_ok=True)
    os.environ["RAY_TMPDIR"] = str(tmp_dir)
    ray.init(num_gpus=1, _temp_dir=str(tmp_dir), ignore_reinit_error=True)
    register_env("gfootball_multi", lambda cfg: GFootballMultiAgentEnv(cfg))
    dummy = GFootballMultiAgentEnv({})
    obs_space = dummy.observation_space
    act_space = dummy.action_space["left_0"]
    dummy.close()
    del dummy
    
    model_config = {
        "d_model": 48, "mamba_state": 6, "num_mamba_layers": 6,
        "prev_action_emb": 8, "gradient_checkpointing": True,
        "mlp_hidden_dims": [256, 128], "head_hidden_dims": [128],
        "use_distributional": True, "v_min": -10.0, "v_max": 10.0,
        "num_atoms": 51, "num_stages": 8,
        "max_seq_len": 20
    }
    
    rl_spec = MultiRLModuleSpec(
        rl_module_specs={
            p: RLModuleSpec(
                module_class=GFootballMambaRLModule,
                observation_space=obs_space,
                action_space=act_space,
                model_config=model_config
            ) for p in ["policy_left", "policy_right"]
        }
    )
    
    config = (
        PPOConfig()
        .environment(
            env="gfootball_multi",
            env_config={"curriculum_horizon": 100000},
            disable_env_checking=True
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True
        )
        # HIER: Default Connectors explizit aktivieren
        .env_runners(
            num_env_runners=11,
            num_envs_per_env_runner=1,
            batch_mode="complete_episodes",
            sample_timeout_s=120.0,
            add_default_connectors_to_env_to_module_pipeline=True,
            add_default_connectors_to_module_to_env_pipeline=True,
        )
        # HIER: GAE und Critic explizit aktivieren
        .training(
            train_batch_size=4000, 
            minibatch_size=1000,
            num_epochs=5,
            lr=5e-5,
            gamma=0.998,
            entropy_coeff=0.01,
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
        .callbacks(StageReturnCallbacks)
        .resources(num_gpus=1, num_cpus_for_main_process=1)
        .learners(num_learners=0, num_gpus_per_learner=0.5)
    )
    
    pbt = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="env_runners/episode_return_mean",
        mode="max",
        perturbation_interval=20,
        hyperparam_mutations={
            "lr": [1e-5, 5e-5, 1e-4],
            "entropy_coeff": [0.005, 0.01, 0.02]
        }
    )
    
    print("Starting Robust Training...")
    tune.run(
        "PPO",
        config=config.to_dict(),
        scheduler=pbt,
        num_samples=2,
        stop={"training_iteration": 100},
        storage_path=str(res_dir),
        name="PPO_GFootball_Robust",
        checkpoint_freq=10
    )
    ray.shutdown()

if __name__ == "__main__":
    main()