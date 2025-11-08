import numpy as np
import gfootball.env as football_env

def test_rewards_mode(rewards_mode="scoring,checkpoints", 
                      episodes=10, 
                      end_on_score=True, 
                      num_agents=1): # <-- Neuer Parameter
    """
    Testet eine gfootball-Umgebung mit einer variablen Anzahl von Agenten.
    """
    
    print(f"\n--- Starte Test: rewards='{rewards_mode}', agents={num_agents}, episodes={episodes} ---")
    
    env = football_env.create_environment(
        env_name="academy_single_goal_versus_lazy",
        representation="simple115v2",
        # HIER: Die Anzahl der Agenten wird jetzt durch den Parameter gesteuert
        number_of_left_players_agent_controls=num_agents, 
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
            # KORREKTUR: Generiere 'num_agents' Aktionen (eine pro Agent)
            # np.random.randint(19, size=num_agents) erstellt ein Array mit 'num_agents' Zufallszahlen.
            actions = np.random.randint(19, size=num_agents)
            
            # Sende das Array der Aktionen an die Umgebung
            step = env.step(actions)
            
            # Dein Code zur Verarbeitung der Rückgabewerte (ist korrekt)
            if len(step) == 5:
                obs, rew, terminated, truncated, info = step
                done = terminated or truncated
            else:
                obs, rew, done, info = step
            
            # Belohnung aufsummieren
            # Wenn num_agents > 1, ist 'rew' ein Array. np.sum() ist hier korrekt.
            total += float(rew if np.isscalar(rew) else np.sum(rew))
            
        print(f"[{rewards_mode}, {num_agents} agents] ep{ep:02d} total={total:.3f}")
        totals.append(total)
        
    print(f"[{rewards_mode}, {num_agents} agents] mean={np.mean(totals):.3f}, max={np.max(totals):.3f}")
    env.close()

# --- Beispielaufrufe ---

# # Test mit 1 Agenten
# test_rewards_mode("scoring,checkpoints", episodes=10, end_on_score=True, num_agents=1)

# Test mit 11 Agenten (so wie in deinem ursprünglichen Versuch)
test_rewards_mode("scoring,checkpoints", episodes=10, end_on_score=True, num_agents=11)

# # Test mit 5 Agenten
# test_rewards_mode("scoring", episodes=10, end_on_score=True, num_agents=5)