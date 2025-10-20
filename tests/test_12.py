import ray
from ray import tune
from ray.rllib.algorithms.impala import ImpalaConfig
from ray.air.config import RunConfig, CheckpointConfig
from ray.tune.schedulers import PopulationBasedTraining

def train_stage_1_create_checkpoint():
    config = ImpalaConfig()
    config.environment("CartPole-v1")
    config.framework("torch")
    config.env_runners(num_env_runners=20, num_envs_per_env_runner=1)
    config.resources(num_gpus=0)
    config.training(
        train_batch_size=500,
        lr=0.0005,
    )
    config.debugging(seed=42)
    
    param_space = config.to_dict()
    
    metric_path = "env_runners/episode_reward_mean"
    
    tuner = tune.Tuner(
        "IMPALA",
        param_space=param_space,
        run_config=RunConfig(
            name="stage_1_create_checkpoint",
            stop={"training_iteration": 5},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=5,
                num_to_keep=1,
                checkpoint_at_end=True,
            ),
            verbose=0,
        ),
    )
    
    results = tuner.fit()

    best_result = results.get_best_result(
        metric=metric_path,
        mode="max"
    )
    
    if best_result and best_result.checkpoint:
        checkpoint_path = str(best_result.checkpoint.path)
        final_reward = best_result.metrics['env_runners']['episode_reward_mean']
        return checkpoint_path, final_reward
    else:
        raise Exception("Failed to create checkpoint in Stage 1!")


def train_stage_2_with_transfer(checkpoint_path):
    metric_path = "env_runners/episode_reward_mean"
    
    config = ImpalaConfig()
    config.environment("CartPole-v1")
    config.framework("torch")
    config.env_runners(num_env_runners=20, num_envs_per_env_runner=1)
    config.resources()
    config.training(
        train_batch_size=500,
        lr=tune.choice([0.0001, 0.0003, 0.0005]),
    )
    config.debugging(seed=123)
    
    param_space = config.to_dict()
    param_space["_restore_checkpoint_path"] = checkpoint_path
    
    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric=metric_path,
        mode="max",
        perturbation_interval=5,
        hyperparam_mutations={
            "lr": tune.loguniform(1e-5, 1e-3),
        }
    )
    
    tuner = tune.Tuner(
        "IMPALA",
        param_space=param_space,
        tune_config=tune.TuneConfig(
            scheduler=pbt_scheduler,
            num_samples=3,
        ),
        run_config=RunConfig(
            name="stage_2_transfer_learning",
            stop={"training_iteration": 10},
            checkpoint_config=CheckpointConfig(
                checkpoint_frequency=5,
                num_to_keep=1,
            ),
            verbose=0,
        ),
    )
    
    results = tuner.fit()
    best_result = results.get_best_result(
        metric=metric_path,
        mode="max"
    )
    return best_result


def train_stage_2_WITHOUT_transfer():
    metric_path = "env_runners/episode_reward_mean"
    
    config = ImpalaConfig()
    config.environment("CartPole-v1")
    config.framework("torch")
    config.env_runners(num_env_runners=2, num_envs_per_env_runner=1)
    config.resources(num_gpus=0)
    config.training(
        train_batch_size=500,
        lr=0.0003,
    )
    config.debugging(seed=123)
    
    param_space = config.to_dict()
    
    tuner = tune.Tuner(
        "IMPALA",
        param_space=param_space,
        run_config=RunConfig(
            name="stage_2_no_transfer_control",
            stop={"training_iteration": 10},
            verbose=0,
        ),
    )
    
    results = tuner.fit()
    best_result = results.get_best_result(
        metric=metric_path,
        mode="max"
    )
    
    initial_reward = best_result.metrics_dataframe.iloc[0][metric_path]
    final_reward = best_result.metrics['env_runners']['episode_reward_mean']
    
    return initial_reward, final_reward


def verify_checkpoint_loading():
    ray.init(ignore_reinit_error=True)
    
    try:
        metric_path = "env_runners/episode_reward_mean"
        
        checkpoint_path, stage1_reward = train_stage_1_create_checkpoint()
        
        stage2_result = train_stage_2_with_transfer(checkpoint_path)
        
        stage2_initial = stage2_result.metrics_dataframe.iloc[0][metric_path]
        stage2_final = stage2_result.metrics['env_runners']['episode_reward_mean']
        
        control_initial, control_final = train_stage_2_WITHOUT_transfer()
        
        print("\n" + "╔" + "="*78 + "╗")
        print("║" + " "*30 + "FINAL ANALYSIS" + " "*34 + "║")
        print("╚" + "="*78 + "╝\n")
        
        print("Stage 1 (Create Checkpoint):")
        print(f"   Final reward after 20 iterations: {stage1_reward:.2f}")
        print()
        
        print("Stage 2 WITH Transfer Learning (Best Trial):")
        print(f"   Initial reward (should ≈ {stage1_reward:.2f}): {stage2_initial:.2f}")
        print(f"   Final reward after 10 more iter:      {stage2_final:.2f}")
        print()
        
        print("Stage 2 WITHOUT Transfer (Control):")
        print(f"   Initial reward (random weights):    {control_initial:.2f}")
        print(f"   Final reward after 10 iterations: {control_final:.2f}")
        print()
        
        print("─"*80)
        print("VERIFICATION:")
        print("─"*80)
        
        reward_diff = abs(stage2_initial - stage1_reward)
        if reward_diff < 50:
            print(f"✅ PASSED: Checkpoint was loaded successfully!")
            print(f"   Stage 2 initial reward ({stage2_initial:.2f}) is close to Stage 1 final ({stage1_reward:.2f})")
        else:
            print(f"❌ FAILED: Checkpoint may not have been loaded!")
            print(f"   Stage 2 initial reward ({stage2_initial:.2f}) differs from Stage 1 final ({stage1_reward:.2f})")
        
        print()
        
        if stage2_initial > control_initial + 20:
            print(f"✅ PASSED: Transfer learning starts with much better performance!")
            print(f"   With transfer: {stage2_initial:.2f} vs Without: {control_initial:.2f}")
        else:
            print(f"⚠️  WARNING: Transfer learning advantage not clear")
            print(f"   With transfer: {stage2_initial:.2f} vs Without: {control_initial:.2f}")
            
    finally:
        ray.shutdown()


if __name__ == "__main__":
    verify_checkpoint_loading()