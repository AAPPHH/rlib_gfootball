import numpy as np
import gfootball.env as football_env

def test_rewards_mode(rewards_mode="scoring", episodes=10, end_on_score=True):
    env = football_env.create_environment(
        env_name="academy_empty_goal_close",
        representation="simple115v2",
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        rewards=rewards_mode,
        render=False,
        other_config_options={"end_episode_on_score": end_on_score}
    )
    totals = []
    for ep in range(episodes):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            a = np.random.randint(19)
            step = env.step([a])
            if len(step) == 5:
                obs, rew, terminated, truncated, info = step
                done = terminated or truncated
            else:
                obs, rew, done, info = step
            total += float(rew if np.isscalar(rew) else np.sum(rew))
        print(f"[{rewards_mode}] ep{ep:02d} total={total:.3f}")
        totals.append(total)
    print(f"[{rewards_mode}] mean={np.mean(totals):.3f}, max={np.max(totals):.3f}")
    env.close()

# Erwartung:
test_rewards_mode("scoring", episodes=20, end_on_score=True)               # <= 1.0
test_rewards_mode("checkpoints", episodes=20, end_on_score=True)           # meist < 1.0, kann variieren
test_rewards_mode("scoring,checkpoints", episodes=20, end_on_score=True)   # oft > 1.0

