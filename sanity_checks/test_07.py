"""Sanity check: Was gibt info bei Episode-Ende zurück?"""

import gfootball.env as football_env
import numpy as np

env = football_env.create_environment(
    env_name="11_vs_11_easy_stochastic",
    representation="simple115v2",
    number_of_left_players_agent_controls=1,
    number_of_right_players_agent_controls=0,
    rewards="scoring",
    render=False
)

print("Running episodes to check info dict...")
print("=" * 60)

for ep in range(3):
    obs = env.reset()
    ep_ret = 0
    step = 0
    
    while True:
        action = env.action_space.sample()
        obs, rew, done, info = env.step([action])
        ep_ret += rew
        step += 1
        
        # Bei Tor oder Episode-Ende
        if rew != 0 or done:
            print(f"\nEp {ep+1}, Step {step}:")
            print(f"  reward = {rew}")
            print(f"  done = {done}")
            print(f"  ep_ret = {ep_ret}")
            print(f"  info type = {type(info)}")
            print(f"  info = {info}")
            
            if isinstance(info, dict):
                print(f"  'score' in info: {'score' in info}")
                if 'score' in info:
                    print(f"  info['score'] = {info['score']}")
                    print(f"  score[0] > score[1]: {info['score'][0] > info['score'][1]}")
                
                # Alle keys auflisten
                print(f"  info keys: {list(info.keys())}")
        
        if done:
            print(f"\n--- Episode {ep+1} ended: return={ep_ret:.1f} ---")
            
            # Test der aktuellen Logik
            won_current = info.get("score", [0, 1])[0] > info.get("score", [0, 1])[1] if isinstance(info, dict) else ep_ret > 0
            won_simple = ep_ret > 0
            
            print(f"  won (current logic): {won_current}")
            print(f"  won (simple ep_ret>0): {won_simple}")
            print("=" * 60)
            break
        
        if step >= 3000:
            print(f"\n--- Episode {ep+1} timeout ---")
            break

env.close()
print("\nDone!")