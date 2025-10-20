import torch
from ray.rllib.algorithms.ppo import PPOConfig

config = (
    PPOConfig()
    .environment("CartPole-v1")
    .framework("torch")
    .env_runners(num_env_runners=0) 
    .resources(num_gpus=1)
    .training(
        amp_backend="torch",
        torch_amp_float16_dtype="fp16"
    )
)

algo = config.build()

for i in range(10):
    result = algo.train()
    print(f"Iter {i} | Reward mean: {result['episode_reward_mean']:.2f}")

algo.stop()