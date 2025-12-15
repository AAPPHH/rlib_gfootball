"""
Check: Gehen checkpoint rewards nur an bestimmte Agenten?
"""

import numpy as np
import gfootball.env as football_env


def test_per_agent_rewards():
    print(">>> Multi-Agent Checkpoint Rewards per Agent")
    print("=" * 60)
    
    env = football_env.create_environment(
        env_name="11_vs_11_easy_stochastic",
        representation="simple115v2",
        number_of_left_players_agent_controls=11,
        rewards="scoring,checkpoints",
        render=False
    )
    
    obs = env.reset()
    
    for step in range(500):
        actions = [np.random.randint(0, 19) for _ in range(11)]
        obs, rew, done, info = env.step(actions)
        
        rew = np.array(rew)
        
        # Zeige JEDEN non-zero reward, auch wenn nur ein Agent ihn bekommt
        if np.any(rew != 0):
            nonzero_agents = np.where(rew != 0)[0]
            print(f"Step {step:3d}:")
            print(f"  Full reward array: {rew}")
            print(f"  Non-zero agents: {nonzero_agents}")
            print(f"  Mean: {np.mean(rew):.4f}, Sum: {np.sum(rew):.4f}")
            print()
        
        if done:
            print(f"Episode done at step {step}")
            obs = env.reset()
    
    env.close()


if __name__ == "__main__":
    test_per_agent_rewards()