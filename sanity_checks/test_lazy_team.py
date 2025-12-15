"""
Sanity Check: Welche Spieler sind in welchen obs rows?
Überprüft ob obs[0:11] wirklich das linke Team ist.
"""

import numpy as np
import gfootball.env as football_env


def test_player_mapping():
    print("=" * 60)
    print("TEST: Player Mapping in Observations")
    print("=" * 60)
    
    env = football_env.create_environment(
        env_name="11_vs_11_easy_stochastic",
        representation="simple115v2",
        number_of_left_players_agent_controls=11,
        number_of_right_players_agent_controls=11,
        rewards="scoring",
        render=False
    )
    
    obs = env.reset()
    obs = np.array(obs)
    
    print(f"\nobs.shape: {obs.shape}")
    print("\nsimple115v2 Format:")
    print("  [0:22]   = left team positions (11*2)")
    print("  [22:44]  = left team directions (11*2)")
    print("  [44:66]  = right team positions (11*2)")
    print("  [66:88]  = right team directions (11*2)")
    print("  [88:91]  = ball position (x,y,z)")
    print("  [91:94]  = ball direction")
    print("  [94:97]  = ball ownership one-hot")
    print("  [97]     = active player index")
    print("  [98:108] = game mode + sticky actions")
    
    print("\n" + "-" * 60)
    print("Analyse jeder Observation Row:")
    print("-" * 60)
    
    # Extrahiere Positionen aller Spieler (gleich für alle obs)
    left_pos = obs[0, 0:22].reshape(11, 2)
    right_pos = obs[0, 44:66].reshape(11, 2)
    
    print(f"\nLeft Team Positionen (x,y):")
    for i, pos in enumerate(left_pos):
        print(f"  Player {i}: ({pos[0]:+.3f}, {pos[1]:+.3f})")
    
    print(f"\nRight Team Positionen (x,y):")
    for i, pos in enumerate(right_pos):
        print(f"  Player {i}: ({pos[0]:+.3f}, {pos[1]:+.3f})")
    
    print("\n" + "-" * 60)
    print("Active Player Index pro Observation Row:")
    print("-" * 60)
    
    for i in range(22):
        active_idx = int(obs[i, 97])
        team = "LEFT" if i < 11 else "RIGHT"
        row_in_team = i if i < 11 else i - 11
        
        # Position des aktiven Spielers
        if i < 11:
            active_pos = left_pos[active_idx]
        else:
            active_pos = right_pos[active_idx]
        
        print(f"  Row {i:2d} ({team:5s} agent {row_in_team:2d}): active_idx={active_idx}, pos=({active_pos[0]:+.3f}, {active_pos[1]:+.3f})")
    
    print("\n" + "-" * 60)
    print("Verifikation:")
    print("-" * 60)
    
    # Check: Sind die ersten 11 rows für Spieler 0-10 des linken Teams?
    left_active_indices = [int(obs[i, 97]) for i in range(11)]
    right_active_indices = [int(obs[i, 97]) for i in range(11, 22)]
    
    print(f"\n  Left team (rows 0-10) active indices:  {left_active_indices}")
    print(f"  Right team (rows 11-21) active indices: {right_active_indices}")
    
    left_correct = left_active_indices == list(range(11))
    right_correct = right_active_indices == list(range(11))
    
    print(f"\n  Left team indices correct (0-10)?  {left_correct}")
    print(f"  Right team indices correct (0-10)? {right_correct}")
    
    # Step und nochmal prüfen
    print("\n" + "-" * 60)
    print("Nach einem Step:")
    print("-" * 60)
    
    actions = list(range(11)) + [0] * 11  # Verschiedene Aktionen links, idle rechts
    obs, rew, done, info = env.step(actions)
    obs = np.array(obs)
    
    left_active_indices = [int(obs[i, 97]) for i in range(11)]
    right_active_indices = [int(obs[i, 97]) for i in range(11, 22)]
    
    print(f"  Left team active indices:  {left_active_indices}")
    print(f"  Right team active indices: {right_active_indices}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("FAZIT:")
    print("=" * 60)
    if left_correct:
        print("  ✓ obs[0:11] = Left Team Agents 0-10")
        print("  ✓ obs[11:22] = Right Team Agents 0-10")
        print("  → Wir können sicher obs[:11] für unser Team nehmen")
    else:
        print("  ✗ WARNUNG: Mapping ist anders als erwartet!")
        print("  → Code muss angepasst werden")


if __name__ == "__main__":
    test_player_mapping()