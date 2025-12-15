import numpy as np
import gfootball.env as football_env

print("=== Test 1: 1 Agent Control ===")
env1 = football_env.create_environment(
    env_name="11_vs_11_easy_stochastic",
    representation="simple115v2",
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=0,
    rewards="scoring",
    render=False
)

obs = env1.reset()
print(f"Obs shape: {np.array(obs).shape}")

active_players = []
for step in range(500):
    action = [np.random.randint(0, 19)]
    obs, rew, done, info = env1.step(action)
    
    raw_obs = env1.unwrapped.observation()
    if raw_obs and len(raw_obs) > 0:
        active = raw_obs[0].get('active', -1)
        active_players.append(active)
    
    if done:
        break

env1.close()

unique = set(active_players)
print(f"Active players seen: {unique}")
print(f"Switched {len(unique)} times across {len(active_players)} steps")
print(f"First 50 active: {active_players[:50]}")

print("\n=== Test 2: 11 Agent Control ===")
env11 = football_env.create_environment(
    env_name="11_vs_11_easy_stochastic",
    representation="simple115v2",
    number_of_left_players_agent_controls=11,
    number_of_right_players_agent_controls=0,
    rewards="scoring",
    render=False
)

obs = env11.reset()
print(f"Obs shape: {np.array(obs).shape}")

active_players_11 = []
for step in range(500):
    action = [np.random.randint(0, 19) for _ in range(11)]
    obs, rew, done, info = env11.step(action)
    
    raw_obs = env11.unwrapped.observation()
    if raw_obs and len(raw_obs) > 0:
        actives = [raw_obs[i].get('active', -1) for i in range(min(11, len(raw_obs)))]
        active_players_11.append(actives)
    
    if done:
        break

env11.close()

print(f"First 10 steps active arrays:")
for i, a in enumerate(active_players_11[:10]):
    print(f"  Step {i}: {a}")

print("\n=== Summary ===")
print(f"1-Agent: {len(unique)} unique active players")
if len(unique) > 1:
    print("-> Active player DOES switch automatically")
else:
    print("-> Active player does NOT switch")