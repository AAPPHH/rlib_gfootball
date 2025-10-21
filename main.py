from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import gfootball.env as football_env
import numpy as np
import ray
from gymnasium import spaces
from ray import train, tune
from ray.air.config import CheckpointConfig, RunConfig
from ray.rllib.algorithms.impala import Impala, ImpalaConfig
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.policy import PolicySpec
from ray.train import Checkpoint
from ray.tune.registry import register_env
from ray.tune.schedulers import PopulationBasedTraining

from model import GFootballMamba
from policy_pool import EnhancedSelfPlayCallback

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
    TrainingStage("stage_1_basic", "academy_empty_goal_close", "simple115v2", 1, 0, 0.75, 1_000_000, "1 attacker, no opponents: finishes into an empty goal from close range."),
    # TrainingStage("stage_2_basic", "academy_run_to_score_with_keeper", "simple115v2", 1, 0, 0.75, 200_000_000, "1 attacker versus a goalkeeper: dribbles towards goal and finishes under light pressure."),
    # TrainingStage("stage_3_basic", "academy_pass_and_shoot_with_keeper", "simple115v2", 1, 0, 0.75, 5_000_000, "1 attacker facing a goalkeeper and nearby defender: focuses on control, positioning, and finishing."),
    # TrainingStage("stage_4_1v1", "academy_3_vs_1_with_keeper", "simple115v2", 3, 0, 0.75, 10_000_000, "3 attackers versus 1 defender and a goalkeeper: encourages passing combinations and shot creation."),
    # TrainingStage("stage_5_3v3", "academy_single_goal_versus_lazy", "simple115v2", 3, 0, 1.0, 50_000_000, "3 vs 3 on a full field against static opponents: focuses on offensive buildup and team coordination."),
    # TrainingStage("stage_6_transition", "11_vs_11_easy_stochastic", "simple115v2", 3, 3, 1.0, 100_000_000, "Small-sided (3-player) team in 11v11 environment with easy opponents: transition toward full gameplay."),
    # TrainingStage("stage_7_midgame", "11_vs_11_easy_stochastic", "simple115v2", 5, 5, 1.0, 500_000_000, "3 vs 3 within a full 11v11 match (easy mode): focuses on spacing, positioning, and transitions."),
    # TrainingStage("stage_8_fullgame", "11_vs_11_stochastic", "simple115v2", 5, 5, 1.0, 1_000_000_000, "Full 11v11 stochastic match: standard difficulty with dynamic and realistic gameplay.")
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
        
        if not self.agent_ids:
            single_obs_shape = test_obs.shape
        elif len(test_obs.shape) > 1:
            single_obs_shape = test_obs[0].shape
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


def get_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return "policy_left" if "left" in agent_id else "policy_right"


def select_policy_to_train(policy_id, worker, **kwargs):
    if hasattr(worker, 'get_metrics'):
        metrics = worker.get_metrics()
        active_policy_value = metrics.get("custom_metrics", {}).get("active_policy", 1.0)
        active_policy = "policy_left" if active_policy_value >= 0.5 else "policy_right"
        return policy_id == active_policy
    return True


def create_impala_config(stage: TrainingStage, debug_mode: bool = False, 
                         hyperparams: dict = None, 
                         policy_pool_dir: str = "policy_pool",
                         max_versions: int = 50,
                         keep_top: int = 10,
                         active_zone: int = 15,
                         auto_prune: bool = True) -> ImpalaConfig:
    config = ImpalaConfig()
    
    env_config = {
        "debug_mode": debug_mode, "representation": stage.representation, "env_name": stage.env_name,
        "number_of_left_players_agent_controls": stage.left_players,
        "number_of_right_players_agent_controls": stage.right_players,
        "rewards": "scoring,checkpoints",
    }
    config.environment("gfootball_multi", env_config=env_config)
    config.framework("torch")
    config.env_runners(num_env_runners=0 if debug_mode else 5, num_envs_per_env_runner=1,
                       num_cpus_per_env_runner=2,
                       rollout_fragment_length=8 if debug_mode else 8)
    
    if hyperparams is None:
        hyperparams = {"lr": 0.0001, "entropy_coeff": 0.008, "vf_loss_coeff": 0.5}

    config.training(
            lr=hyperparams["lr"],
            entropy_coeff=hyperparams["entropy_coeff"],
            vf_loss_coeff=hyperparams["vf_loss_coeff"],
            grad_clip=0.5,
            train_batch_size=1024 if not debug_mode else 8,
            learner_queue_size=1,
        )
    
    use_custom_model = True
    custom_model_config = {
            "custom_model": "GFootballMamba",
            "custom_model_config": {
            "d_model": 128,
            "num_layers": 4,
            "d_state": 16,
            "d_conv": 3,
            "expand": 2,
            "dropout": 0.03,

        }
    }
    
    standard_model_config = {
        "fcnet_hiddens": [256, 256],
        "lstm_cell_size": 256,
    }

    if use_custom_model:
        print("--- Using CUSTOM model: GFootballMamba ---")
        config.model.update(custom_model_config)
    else:
        print("--- Using STANDARD model: IMPALA CNN+LSTM ---")
        config.model.update(standard_model_config)

    config.resources(num_gpus=1/2, num_cpus_for_main_process=2)
    config.learners(num_learners=1, num_gpus_per_learner=1/2, num_cpus_per_learner=1)

    policies = {}
    has_left, has_right = stage.left_players > 0, stage.right_players > 0

    if has_left: policies["policy_left"] = PolicySpec()
    if has_right: policies["policy_right"] = PolicySpec()
        
    policies_to_train_fn = None
    if has_left and has_right:
        policies_to_train_fn = select_policy_to_train
    elif has_left:
        policies_to_train_fn = ["policy_left"]
    elif has_right:
        policies_to_train_fn = ["policy_right"]
    else: 
        policies_to_train_fn = []
    
    config.multi_agent(
        policies=policies, 
        policy_mapping_fn=get_policy_mapping_fn, 
        policies_to_train=policies_to_train_fn
    )

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
    return trial.trial_id

def train_impala_with_restore(config):
    restore_path = config.pop("_restore_from", None)
    stop_timesteps = config.pop("_stop_timesteps", 1_000_000)
    checkpoint_freq = 50

    algo = Impala(config=config)

    start_timesteps = 0
    if restore_path:
        print(f"🔄 [Trainer Fn] Restoring from: {restore_path}")
        try:
            algo.restore(restore_path)
            start_timesteps = algo._counters.get("num_env_steps_sampled", 0)
            print(f"📊 [Trainer Fn] Successfully restored. Starting from timestep: {start_timesteps}")
        except Exception as e:
            print(f"⚠️ [Trainer Fn] ERROR restoring {restore_path}: {e}. Starting fresh.")
    
    timesteps = start_timesteps
    iteration = 0

    while timesteps < stop_timesteps:
        result = algo.train()
        timesteps = result.get("timesteps_total", timesteps)
        iteration += 1

        if iteration % checkpoint_freq == 0:
            save_result = algo.save()
            chk_path = save_result.checkpoint.path if hasattr(save_result, 'checkpoint') else save_result
            checkpoint = Checkpoint.from_directory(chk_path)
            train.report(metrics=result, checkpoint=checkpoint)
        else:
            train.report(metrics=result)

    save_result = algo.save()
    chk_path = save_result.checkpoint.path if hasattr(save_result, 'checkpoint') else save_result
    checkpoint = Checkpoint.from_directory(chk_path)
    print(f"✅ [Trainer Fn] Complete: {start_timesteps} → {timesteps} timesteps")
    train.report(metrics=result, checkpoint=checkpoint)
    
    algo.stop()

def train_single_stage(stage, stage_index, debug_mode, restore_checkpoint):
    print("\n" + "="*80)
    print(f"STARTING STAGE {stage_index + 1}: {stage.name} - {stage.description}")
    if restore_checkpoint:
        print(f"  -> Providing restore checkpoint: {restore_checkpoint}")
    print("="*80 + "\n")

    results_path = Path(__file__).resolve().parent / "training_results_transfer_pbt"
    policy_pool_dir = results_path / f"{stage.name}_policy_pool"

    param_space = create_impala_config(
        stage=stage,
        debug_mode=debug_mode,
        hyperparams = {
            "lr": tune.choice([0.0001, 0.0003, 0.0005]),
            "entropy_coeff": tune.choice([0.004, 0.006, 0.01]),
            "vf_loss_coeff": tune.choice([0.6, 0.8, 1.0]),
        },
        policy_pool_dir=str(policy_pool_dir),
        max_versions=25,
        keep_top=10,
        active_zone=15,
        auto_prune=True
    ).to_dict()

    param_space["_stop_timesteps"] = stage.max_timesteps
    if restore_checkpoint:
        param_space["_restore_from"] = restore_checkpoint 

    metric_path = "env_runners/episode_return_mean"
    

    num_runners = 0 if debug_mode else 5
    cpus_per_runner = 2
    gpus_for_learner = 0.5
    cpus_for_learner_and_driver = 2

    resources = tune.PlacementGroupFactory(
        [{"CPU": cpus_for_learner_and_driver, "GPU": gpus_for_learner}] +
        [{"CPU": cpus_per_runner}] * num_runners
    )

    stop_criteria = {
        "timesteps_total": stage.max_timesteps,
    }
    
    checkpoint_config = CheckpointConfig(
        num_to_keep=None, 
        checkpoint_frequency=0,
        checkpoint_score_attribute=metric_path,
        checkpoint_score_order="max",
    )
    
    pbt_scheduler = PopulationBasedTraining(
        time_attr="timesteps_total", metric=metric_path,
        mode="max", perturbation_interval=64_000,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-5, 1e-3),
            "entropy_coeff": tune.uniform(0.0, 0.02),
            "vf_loss_coeff": tune.uniform(0.1, 1.0),
        }
    )

    tuner = tune.Tuner(
        tune.with_resources(
            train_impala_with_restore,
            resources=resources
        ),
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=pbt_scheduler, 
            num_samples=2,
            max_concurrent_trials=2,
            trial_dirname_creator=short_trial_name_creator
        ),
        run_config=RunConfig(
            stop=stop_criteria,
            checkpoint_config=checkpoint_config,
            name=f"{stage.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            storage_path=str(results_path),
        ),
    )
    
    results = tuner.fit()

    if results.experiment_path:
        experiment_path = str(results.experiment_path)
        log_file = results_path / "tensorboard_commands.txt"
        tensorboard_command = f"tensorboard --logdir \"{experiment_path}\""
        
        with open(log_file, "a") as f:
            f.write(f"--- Stage: {stage.name} ---\n")
            f.write(f"{tensorboard_command}\n\n")
            
    best_result = results.get_best_result(metric=metric_path, mode="max")
    
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
    print("PROGRESSIVE TRAINING WITH AUTO-PRUNING + CHAMPION TRACKING")
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
    debug_mode = False
    use_transfer = True
    start_stage = 0
    end_stage = 8
    initial_checkpoint = r"C:\clones\rlib_gfootball\training_results_transfer_pbt_21\stage_1_basic_20251020_183644\f71c1_00000\checkpoint_000003"
    
    ray.init(ignore_reinit_error=True, log_to_driver=False, local_mode=debug_mode)
    print(ray.cluster_resources())
    
    register_env("gfootball_multi", lambda config: GFootballMultiAgentEnv(config))
    ModelCatalog.register_custom_model("GFootballMamba", GFootballMamba)

    if use_transfer:
        train_progressive(start_stage, end_stage, debug_mode, initial_checkpoint)
    else:
        stage = TRAINING_STAGES[start_stage]
        train_single_stage(stage, start_stage, debug_mode, initial_checkpoint)
    
    ray.shutdown()

if __name__ == "__main__":
    main() 