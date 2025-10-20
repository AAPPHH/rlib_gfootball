import ray
from ray import tune, train
from ray.rllib.algorithms.impala import Impala, ImpalaConfig
from ray.air.config import RunConfig, CheckpointConfig
from ray.train import Checkpoint 

def train_impala_fn(config):
    restore_path = config.pop("_restore_from", None)
    stop_timesteps = config.pop("_stop_timesteps", 1_000_000)
    checkpoint_freq = config.pop("_checkpoint_freq", 10)
    
    algo = Impala(config=config)
    
    start_timesteps = 0
    if restore_path:
        print(f"🔄 Restoring from: {restore_path}")
        algo.restore(restore_path)
        start_timesteps = algo._counters.get("num_env_steps_sampled", 0)
        print(f"📊 Starting from timestep: {start_timesteps}")
    
    timesteps = start_timesteps
    iteration = 0
    
    while timesteps < stop_timesteps:
        result = algo.train()
        timesteps = result.get("timesteps_total", timesteps)
        iteration += 1
        
        reward_mean = result.get("env_runners", {}).get("episode_reward_mean", 0)
        print(f"Iteration {iteration}: timesteps={timesteps}/{stop_timesteps}, reward={reward_mean:.2f}")
        
        if iteration % checkpoint_freq == 0:
            save_result = algo.save()
            checkpoint_path = save_result.checkpoint.path if hasattr(save_result, 'checkpoint') else save_result
            checkpoint = Checkpoint.from_directory(checkpoint_path)
            print(f"💾 Checkpoint saved: {checkpoint_path}")
            train.report(metrics=result, checkpoint=checkpoint)
        else:
            train.report(metrics=result)

    save_result = algo.save()
    checkpoint_path = save_result.checkpoint.path if hasattr(save_result, 'checkpoint') else save_result
    checkpoint = Checkpoint.from_directory(checkpoint_path)
    print(f"💾 Final checkpoint saved: {checkpoint_path}")
    print(f"✅ Stage completed: {start_timesteps} → {timesteps} timesteps ({timesteps - start_timesteps} new)")
    train.report(metrics=result, checkpoint=checkpoint)
    
    algo.stop()

def train_stage(stage_name, max_timesteps, restore_checkpoint=None):
    print(f"\n{'='*60}")
    print(f"Training Stage: {stage_name}")
    if restore_checkpoint:
        print(f"Restore from: {restore_checkpoint}")
    print(f"{'='*60}\n")
    
    config = ImpalaConfig()
    config.environment("CartPole-v1")
    config.framework("torch")
    config.rollouts(num_rollout_workers=2)
    config.training(
        lr=0.0003, 
        train_batch_size=512,
    )
    config.resources(num_gpus=0)
    
    param_space = config.to_dict()
    
    param_space["_stop_timesteps"] = max_timesteps
    param_space["_checkpoint_freq"] = 1
    if restore_checkpoint:
        param_space["_restore_from"] = restore_checkpoint
    
    checkpoint_config = CheckpointConfig(
        num_to_keep=2,
        checkpoint_score_attribute="env_runners/episode_reward_mean",
        checkpoint_score_order="max"
    )
    
    run_config = RunConfig(
        name=stage_name,
        stop={"timesteps_total": max_timesteps},
        checkpoint_config=checkpoint_config,
        verbose=1
    )

    resources = tune.PlacementGroupFactory(
        [{"CPU": 5.0}] + [{"CPU": 15.0}] * 2
    )


    tuner = tune.Tuner(
        tune.with_resources(
            train_impala_fn, 
            resources=resources
        ),
        param_space=param_space,
        run_config=run_config
    )
    
    results = tuner.fit()

    best_result = results.get_best_result(
        metric="env_runners/episode_reward_mean",
        mode="max"
    )
    
    if best_result and best_result.checkpoint:
        checkpoint_path = best_result.checkpoint.path
        reward = best_result.metrics.get("env_runners", {}).get("episode_reward_mean", 0)
        print(f"✅ Best checkpoint: {checkpoint_path}")
        print(f"   Reward: {reward:.2f}\n")
        return checkpoint_path
    
    print("⚠️ No checkpoint found!")
    print(f"   Best result exists: {best_result is not None}")
    if best_result:
        print(f"   Checkpoint exists: {best_result.checkpoint is not None}\n")
    return None

def train_progressive():
    stages = [
        ("stage_1", 50_000),
        ("stage_2", 100_000),
        ("stage_3", 150_000),
    ]
    
    current_checkpoint = None
    
    for stage_name, max_timesteps in stages:
        checkpoint = train_stage(
            stage_name, 
            max_timesteps, 
            restore_checkpoint=current_checkpoint
        )
        
        if not checkpoint:
            print(f"❌ Stage {stage_name} failed - stopping!")
            break
        
        current_checkpoint = checkpoint
    
    print(f"\n{'='*60}")
    print("✅ Progressive Training Complete!")
    print(f"Final Checkpoint: {current_checkpoint}")
    print(f"Total Timesteps: 150,000")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    train_progressive()
    ray.shutdown()