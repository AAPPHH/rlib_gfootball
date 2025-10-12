import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import time

from main import GFootballMultiAgentEnv

CHECKPOINT_PATH = "C:/Users/jfham/OneDrive/Desktop/rlib_gfootball/training_results/gfootball_impala_20251009_123108/IMPALA_gfootball_multi_11b7b_00000_0_2025-10-09_12-31-08/checkpoint_000054"

def play():
    ray.init(ignore_reinit_error=True)

    register_env("gfootball_multi", lambda config: GFootballMultiAgentEnv(config))
    # ---------------------------------------------------

    print(f"Lade Agent vom Checkpoint: {CHECKPOINT_PATH}")
    algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
    
    env_config = {
        "render": True,
        "number_of_left_players_agent_controls": 2,
        "number_of_right_players_agent_controls": 2,
    }
    env = GFootballMultiAgentEnv(config=env_config)

    print("Starte eine Episode...")
    
    terminateds = {"__all__": False}
    obs, info = env.reset(seed=42) 
    
    total_reward = 0.0
    episode_steps = 0

    while not terminateds["__all__"]:
        actions = algo.compute_actions(
            obs, 
            policy_id="shared_policy",
            explore=False
        )
        
        obs, reward, terminateds, truncateds, info = env.step(actions)
        
        total_reward += sum(reward.values())
        episode_steps += 1
        
        time.sleep(0.05)

    print("-" * 30)
    print(f"Episode beendet nach {episode_steps} Schritten.")
    print(f"Gesamtbelohnung in dieser Episode: {total_reward:.2f}")
    print("-" * 30)

    env.close()
    ray.shutdown()


if __name__ == "__main__":
    play()