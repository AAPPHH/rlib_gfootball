"""
Test: Welche Agenten werden bei number_of_left_players_agent_controls kontrolliert?
Feste Indizes oder dynamisch nach Ballnähe?
"""

import numpy as np
import gfootball.env as football_env


def test_agent_selection(num_agents=2):
    env = football_env.create_environment(
        env_name="11_vs_11_easy_stochastic",
        representation="simple115v2",
        number_of_left_players_agent_controls=num_agents,
        rewards="scoring,checkpoints",
        render=False
    )
    
    obs = env.reset()
    obs = np.array(obs)
    
    print(f"num_agents={num_agents}")
    print(f"obs.shape: {obs.shape}")
    print()
    
    # Extrahiere Positionen aus simple115v2
    # left_team: 0:22 (11 Spieler x 2 coords)
    # ball: 88:91
    
    for step in range(50):
        left_pos = obs[0, 0:22].reshape(11, 2)  # Erste Observation reicht
        ball_pos = obs[0, 88:90]
        
        # Distanz aller Spieler zum Ball
        distances = np.linalg.norm(left_pos - ball_pos, axis=1)
        closest_idx = np.argsort(distances)[:3]
        
        # Active player Index ist bei simple115v2 an Position 97
        # Jede Observation könnte unterschiedliche active_idx haben
        active_indices = [int(o[97]) for o in obs]
        
        # Zeige auch Positionen der kontrollierten Spieler
        controlled_positions = [left_pos[idx] for idx in active_indices]
        
        print(f"Step {step:2d} | Ball: ({ball_pos[0]:+.2f}, {ball_pos[1]:+.2f})")
        print(f"         | Kontrollierte Spieler idx: {active_indices}")
        print(f"         | Positionen: {[f'({p[0]:+.2f},{p[1]:+.2f})' for p in controlled_positions]}")
        print(f"         | Ballnächste Spieler: {closest_idx.tolist()} (dist: {distances[closest_idx].round(2)})")
        print(f"         | Wechsel? {set(active_indices) == set(closest_idx[:num_agents])}")
        print()
        
        # Random actions - muss Liste von ints sein
        actions = [np.random.randint(0, 19) for _ in range(num_agents)]
        obs, rew, done, info = env.step(actions)
        obs = np.array(obs)
        
        if done:
            print("Episode done, reset...")
            obs = env.reset()
            obs = np.array(obs)
    
    env.close()


if __name__ == "__main__":
    print("=" * 60)
    print("TEST: Sind kontrollierte Agenten fix oder ballnah?")
    print("=" * 60)
    print()
    
    test_agent_selection(num_agents=2)