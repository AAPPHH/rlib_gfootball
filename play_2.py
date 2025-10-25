import ray
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
import gfootball.env as football_env
import numpy as np
import time
from gymnasium import spaces
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass
import os
from pathlib import Path
from datetime import datetime

try:
    from model import GFootballTCN
except ImportError as e:
    print(f"❌ model.py nicht gefunden: {e}")

class DualEnvironmentRecorder:
    def __init__(self, env_name: str, left_players: int = 1, right_players: int = 0,
                 video_dir: str = "./presentation_videos"):
        self.env_name = env_name
        self.left_players = left_players
        self.right_players = right_players
        self.video_dir = Path(video_dir)
        self.video_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_env = football_env.create_environment(
            env_name=env_name,
            representation="simple115v2",
            number_of_left_players_agent_controls=left_players,
            number_of_right_players_agent_controls=right_players,
            rewards="scoring,checkpoints",
            stacked=True,
            render=False,
            write_video=False
        )
        self.video_env = football_env.create_environment(
            env_name=env_name,
            representation="pixels",
            number_of_left_players_agent_controls=left_players,
            number_of_right_players_agent_controls=0, 
            rewards="scoring,checkpoints",
            stacked=False,
            render=True,
            write_video=True,
            write_full_episode_dumps=True,
            logdir=str(self.video_dir),
            dump_frequency=1
        )
        
    def reset(self):
        obs_train = self.train_env.reset()
        obs_video = self.video_env.reset()
        return obs_train, obs_video
    
    def step(self, actions):
        train_result = self.train_env.step(actions)

        if len(train_result) == 5:
            obs_train, reward, terminated, truncated, info = train_result
            done = terminated or truncated
        else:
            obs_train, reward, done, info = train_result

        if not done:
            video_actions = actions[:self.left_players]
            try:
                 video_result = self.video_env.step(video_actions)
            except AssertionError as e:
                 if "Cant call step() once episode finished" in str(e):
                     pass 
                 else:
                     raise 
                     
        return obs_train, reward, done, info
    
    def close(self):
        self.train_env.close()
        self.video_env.close()

class GFootballMultiAgentEnv(MultiAgentEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        default_config = {
            "env_name": "academy_pass_and_shoot_with_keeper",
            "representation": "simple115v2",
            "rewards": "scoring,checkpoints",
            "number_of_left_players_agent_controls": 1,
            "number_of_right_players_agent_controls": 0,
            "stacked": True,
            "render": False,
        }
        self.env_config = {**default_config, **config}
        
        self.left_players = self.env_config["number_of_left_players_agent_controls"]
        self.right_players = self.env_config["number_of_right_players_agent_controls"]
        
        creation_kwargs = self.env_config.copy()
        self.env = football_env.create_environment(**creation_kwargs)
        
        self.agent_ids = [f"left_{i}" for i in range(self.left_players)] + \
                         [f"right_{i}" for i in range(self.right_players)]
        self._agent_ids = set(self.agent_ids)
        
        reset_result = self.env.reset()
        test_obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        if not self.agent_ids:
            single_obs_shape = test_obs.shape
        elif len(test_obs.shape) > 1:
            single_obs_shape = test_obs[0].shape
        else:
            if len(self.agent_ids) == 0:
                 single_obs_shape = test_obs.shape 
            else:
                 assert test_obs.shape[0] % len(self.agent_ids) == 0
                 single_obs_shape = (test_obs.shape[0] // len(self.agent_ids),)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=single_obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Discrete(19)
    
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
        
        dones = {aid: done for aid in self.agent_ids}
        dones["__all__"] = done
        truncs = {aid: truncated for aid in self.agent_ids}
        truncs["__all__"] = truncated
        
        return (self._split_obs(obs), self._split_rewards(rewards), dones, 
                truncs, {aid: info for aid in self.agent_ids})
    
    def _split_obs(self, obs):
        if not self.agent_ids:
            return {}
        if len(obs.shape) == 1 and len(self.agent_ids) > 0:
             if len(self.agent_ids) > 0 and obs.shape[0] % len(self.agent_ids) == 0:
                  obs = obs.reshape(len(self.agent_ids), -1)
             elif len(self.agent_ids) == 1:
                  pass 
             else:
                  pass 
        elif len(obs.shape) == 1 and len(self.agent_ids) == 0:
             return {} 

        if len(obs.shape) > 1 and obs.shape[0] == len(self.agent_ids):
            return {self.agent_ids[i]: obs[i].astype(np.float32) for i in range(len(self.agent_ids))}
        elif len(obs.shape) == 1 and len(self.agent_ids) == 1:
             return {self.agent_ids[0]: obs.astype(np.float32)}
        else:
             if len(self.agent_ids) > 0:
                  return {self.agent_ids[0]: obs.astype(np.float32)}
             else:
                  return {} 

    def _split_rewards(self, rewards):
        if np.isscalar(rewards):
            return {aid: float(rewards) for aid in self.agent_ids}
        if len(rewards) == len(self.agent_ids):
             return {self.agent_ids[i]: float(rewards[i]) for i in range(len(self.agent_ids))}
        elif len(self.agent_ids) > 0:
             avg_reward = np.mean(rewards) if len(rewards) > 0 else 0.0
             return {aid: float(avg_reward) for aid in self.agent_ids}
        else:
             return {} 

    
    def close(self):
        self.env.close()

DEMO_STAGES = {
    "s1_empty_goal": {"name": "Empty Goal Close", "env": "academy_empty_goal_close", "left": 1, "right": 0, "desc": "1 Spieler vor Tor"},
    "s2_run_to_score": {"name": "Run to Score w/ Keeper", "env": "academy_run_to_score_with_keeper", "left": 1, "right": 0, "desc": "1 Spieler rennt zum Tor"},
    "s3_pass_shoot": {"name": "Pass and Shoot", "env": "academy_pass_and_shoot_with_keeper", "left": 1, "right": 0, "desc": "1 Spieler gegen Keeper"},
    "s4_3v1": {"name": "3 vs 1", "env": "academy_3_vs_1_with_keeper", "left": 3, "right": 0, "desc": "3 vs 1 (Agent steuert 3)"},
    "s5_11v11_easy_3v0": {"name": "11_vs_11_easy (3v0)", "env": "11_vs_11_easy_stochastic", "left": 3, "right": 0, "desc": "3 Agenten (links) auf 11v11 Feld"},
    "s6_11v11_easy_3v3": {"name": "11_vs_11_easy (3v3)", "env": "11_vs_11_easy_stochastic", "left": 3, "right": 3, "desc": "3v3 Agenten auf 11v11 Feld"},
    "s7_11v11_stoch_3v3": {"name": "11_vs_11_stochastic (3v3)", "env": "11_vs_11_stochastic", "left": 3, "right": 3, "desc": "3v3 Agenten (schwer)"},
}

def get_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "policy_left" if "left" in agent_id else "policy_right"

def create_video_demo(checkpoint_path: str, stage_key: str = "1v0", num_episodes: int = 3):
    if stage_key not in DEMO_STAGES:
        print(f"❌ Stage '{stage_key}' nicht gefunden! Verfügbar: {list(DEMO_STAGES.keys())}")
        return
    
    stage = DEMO_STAGES[stage_key]
    
    if checkpoint_path:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint nicht gefunden: {checkpoint_path}")
            return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo_type = "trained" if checkpoint_path else "random_per_episode"
    output_dir = Path(f"presentation_demo_{stage_key}_{demo_type}_{timestamp}")
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    ray.init(ignore_reinit_error=True, local_mode=True, log_to_driver=False)
    
    register_env("gfootball_multi", lambda config: GFootballMultiAgentEnv(config))
    ModelCatalog.register_custom_model("GFootballTCN", GFootballTCN)
    
    config = (
        ImpalaConfig()
        .environment("gfootball_multi", env_config={
            "env_name": stage["env"],
            "number_of_left_players_agent_controls": stage["left"],
            "number_of_right_players_agent_controls": stage["right"],
        })
        .framework("torch")
        .env_runners(num_env_runners=0)
        .training(
            model = {
                "custom_model": "GFootballTCN",
                "max_seq_len": 32,
                "lstm_use_prev_action": True, 
                "lstm_use_prev_reward": False,

                "custom_model_config": {
                    "d_model": 96,
                    "gru_hidden": 320,
                    "prev_action_emb": 16,
                    "tcn_kernel": 3,
                    "tcn_dilations": [1, 2],
                    "dropout": 0.05,
                    "gradient_checkpointing": True
                }
            }
        )
        .resources(num_gpus=0) 
        .debugging(log_level="ERROR") 
    )
    
    policies = {}
    if stage["left"] > 0:
        policies["policy_left"] = PolicySpec()
    if stage["right"] > 0:
        policies["policy_right"] = PolicySpec()
    
    config.multi_agent(policies=policies, policy_mapping_fn=get_policy_mapping_fn)
    
    algo = None 
    try:
        if checkpoint_path:
            algo = config.build() 
            algo.restore(checkpoint_path)
        
    except Exception as e:
        print(f"❌ Fehler beim Bauen/Laden des Agenten: {e}")
        if algo: algo.stop()
        ray.shutdown()
        return
    
    dual_env = DualEnvironmentRecorder(
        env_name=stage["env"],
        left_players=stage["left"],
        right_players=stage["right"],
        video_dir=str(video_dir)
    )
    
    print(f"🎮 Starte Aufnahme für {num_episodes} Episoden ({'Zufällig pro Episode' if not checkpoint_path else 'Trainiert'})...")
    
    try:
        for ep in range(num_episodes):
            print(f"   Episode {ep + 1}/{num_episodes}...")
            if not checkpoint_path:
                if algo: algo.stop() 
                algo = config.build() 
            
            policy_left = algo.get_policy("policy_left") if stage["left"] > 0 else None
            policy_right = algo.get_policy("policy_right") if stage["right"] > 0 else None

            states_left = {}
            prev_actions_left = {}
            if policy_left:
                for i in range(stage["left"]):
                    agent_id = f"left_{i}"
                    states_left[agent_id] = policy_left.get_initial_state()
                    prev_actions_left[agent_id] = 0

            states_right = {}
            prev_actions_right = {}
            if policy_right:
                 for i in range(stage["right"]):
                    agent_id = f"right_{i}"
                    states_right[agent_id] = policy_right.get_initial_state()
                    prev_actions_right[agent_id] = 0

            obs_train, obs_video = dual_env.reset()
            
            obs_dict = {}
            num_expected_obs = stage["left"] + stage["right"]
            if num_expected_obs == 0:
                obs_dict = {} 
            elif num_expected_obs == 1 and stage["left"] == 1:
                obs_dict = {"left_0": obs_train[0] if isinstance(obs_train, list) and len(obs_train) > 0 and isinstance(obs_train[0], np.ndarray) and len(obs_train[0].shape) > 1 else obs_train}
            elif isinstance(obs_train, (list, np.ndarray)) and len(obs_train) >= num_expected_obs:
                obs_dict = {f"left_{i}": obs_train[i] for i in range(stage["left"])}
                if stage["right"] > 0:
                    obs_dict.update({f"right_{i}": obs_train[stage["left"] + i] for i in range(stage["right"])})
            elif isinstance(obs_train, np.ndarray) and len(obs_train.shape) > 0 and obs_train.shape[0] >= num_expected_obs:
                obs_reshaped = obs_train.reshape(num_expected_obs, -1)
                obs_dict = {f"left_{i}": obs_reshaped[i] for i in range(stage["left"])}
                if stage["right"] > 0:
                    obs_dict.update({f"right_{i}": obs_reshaped[stage["left"] + i] for i in range(stage["right"])})
            else:
                print(f"Warnung: Unerwartetes Observationsformat oder Länge. Überspringe Episode {ep + 1}.")
                continue 

            done = False
            step_count = 0
            
            while not done:
                actions = []
                
                for i in range(stage["left"]):
                    agent_id = f"left_{i}"
                    if agent_id in obs_dict:
                        action, new_state, _ = algo.compute_single_action(
                            observation=obs_dict[agent_id],
                            state=states_left[agent_id],
                            prev_action=prev_actions_left[agent_id],
                            policy_id="policy_left",
                            explore=False
                        )
                        actions.append(action)
                        states_left[agent_id] = new_state
                        prev_actions_left[agent_id] = action
                    elif stage["left"] > 0: actions.append(0) 

                for i in range(stage["right"]):
                    agent_id = f"right_{i}"
                    if agent_id in obs_dict:
                        action, new_state, _ = algo.compute_single_action(
                            observation=obs_dict[agent_id],
                            state=states_right[agent_id],
                            prev_action=prev_actions_right[agent_id],
                            policy_id="policy_right",
                            explore=False
                        )
                        actions.append(action)
                        states_right[agent_id] = new_state
                        prev_actions_right[agent_id] = action
                    elif stage["right"] > 0: actions.append(0)

                if num_expected_obs == 0:
                    num_players_total = 1 
                    if hasattr(dual_env.train_env, 'action_space') and hasattr(dual_env.train_env.action_space, 'n'):
                        num_players_total = dual_env.train_env.action_space.n 
                    actions = [0] * num_players_total 
                
                obs_train, reward, done, info = dual_env.step(actions)

                obs_dict = {}
                if num_expected_obs == 0:
                    obs_dict = {}
                elif num_expected_obs == 1 and stage["left"] == 1:
                    obs_dict = {"left_0": obs_train[0] if isinstance(obs_train, list) and len(obs_train) > 0 and isinstance(obs_train[0], np.ndarray) and len(obs_train[0].shape) > 1 else obs_train}
                elif isinstance(obs_train, (list, np.ndarray)) and len(obs_train) >= num_expected_obs:
                    obs_dict = {f"left_{i}": obs_train[i] for i in range(stage["left"])}
                    if stage["right"] > 0:
                        obs_dict.update({f"right_{i}": obs_train[stage["left"] + i] for i in range(stage["right"])})
                elif isinstance(obs_train, np.ndarray) and len(obs_train.shape) > 0 and obs_train.shape[0] >= num_expected_obs:
                        obs_reshaped = obs_train.reshape(num_expected_obs, -1)
                        obs_dict = {f"left_{i}": obs_reshaped[i] for i in range(stage["left"])}
                        if stage["right"] > 0:
                            obs_dict.update({f"right_{i}": obs_reshaped[stage["left"] + i] for i in range(stage["right"])})
                else:
                    done = True 

                step_count += 1
            
    except KeyboardInterrupt:
        print("\n⚠️ Aufnahme abgebrochen.")
    
    finally:
        dual_env.close()
        if algo: algo.stop()
        ray.shutdown()
        
        print(f"\n✅ Aufnahme beendet. Videos gespeichert in:")
        print(f"   {video_dir.absolute()}")

if __name__ == "__main__":
    
    USE_RANDOM_WEIGHTS = False
    CHECKPOINT_PFAD = r"C:\clones\rlib_gfootball\training_results_transfer_night_2\stage_2_basic\gen_52\run\train_impala_with_restore_20f03_lr=na_ent=na_vf=na\checkpoint_000000"
    STAGE_KEY = "s2_run_to_score"
    NUM_EPISODEN = 5

    if USE_RANDOM_WEIGHTS:
        create_video_demo(None, STAGE_KEY, NUM_EPISODEN)
    elif not CHECKPOINT_PFAD or not os.path.exists(CHECKPOINT_PFAD):
        print(f"❌ Ungültiger Checkpoint-Pfad: {CHECKPOINT_PFAD}")
    else:
        create_video_demo(CHECKPOINT_PFAD, STAGE_KEY, NUM_EPISODEN)