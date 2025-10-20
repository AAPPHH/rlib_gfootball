import gfootball.env as football_env
import numpy as np

def run_random_agent_test():
    """
    Initialisiert eine GFootball-Umgebung und führt für einige Episoden
    zufällige Aktionen aus, um den Belohnungsfluss zu überprüfen.
    """
    env_config = {
        "env_name": "11_vs_11_stochastic",
        "representation": "simple115v2",
        "rewards": "scoring,checkpoints",
        "number_of_left_players_agent_controls": 11,
        "number_of_right_players_agent_controls": 11,
        "render": False,
        "write_video": False,
    }

    env = football_env.create_environment(**env_config)

    num_agents = (
        env_config["number_of_left_players_agent_controls"] +
        env_config["number_of_right_players_agent_controls"]
    )
    print(f"Umgebung '{env_config['env_name']}' mit {num_agents} Agenten erstellt.")
    print("-" * 40)

    num_episodes = 10
    for i in range(num_episodes):
        print(f"Starte Episode {i + 1}/{num_episodes}...")
        
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]
        else:
            obs = reset_result
            
        done = False
        step_count = 0
        total_episode_reward = np.zeros(num_agents)

        while not done:
            actions = env.action_space.sample()
            
            obs, rewards, done, info = env.step(actions)

            if np.any(rewards != 0):
                print(f"\n🔥🔥🔥 BELOHNUNG IN SCHRITT {step_count} ERHALTEN! 🔥🔥🔥")
                print(f"  -> Rohes Belohnungs-Array: {rewards}")
                print(f"  -> Info-Dictionary: {info}")
                print("-" * 20)
            
            total_episode_reward += rewards
            step_count += 1

        print(f"Episode beendet nach {step_count} Schritten.")
        print(f"Gesamtbelohnung der Episode: {total_episode_reward}")
        print("-" * 40 + "\n")

    env.close()
    print("Test beendet.")

if __name__ == "__main__":
    run_random_agent_test()