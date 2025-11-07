import gfootball.env as football_env
from gfootball.env import football_action_set as fas
from pathlib import Path

def run_dump_collection():
    logs_dir = Path.cwd() / "gfootball_dumps"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = football_env.create_environment(
        env_name="11_vs_11_stochastic",
        representation="simple115v2",          # gibt bereits 115er-Arrays zurück
        number_of_left_players_agent_controls=1,  # 1 Agent links
        number_of_right_players_agent_controls=1, # 1 Agent rechts
        render=False,
        write_full_episode_dumps=True,
        write_video=False,
        logdir=str(logs_dir),
    )

    # num_agents wird nicht mehr benötigt, da env.action_space.sample()
    # automatisch die korrekte Anzahl von Aktionen generiert (in diesem Fall 2).
    num_episodes = 1

    for _ in range(num_episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out  # Gym alte/new API abfangen
        done = False
        
        while not done:
            # === ÄNDERUNG HIER ===
            # Wählt eine zufällige Aktion für JEDEN Agenten (1 links, 1 rechts).
            # env.action_space ist ein Tuple(Discrete(19), Discrete(19))
            # .sample() gibt daher ein Tupel mit zwei zufälligen Aktionen zurück, z.B. (5, 12)
            actions = env.action_space.sample() 
            
            step_out = env.step(actions)

            # === ÄNDERUNG HIER ===
            # Robuste Handhabung für alte Gym (4 Rückgabewerte) und neue Gymnasium API (5 Rückgabewerte)
            if len(step_out) == 5:
                # Neue gymnasium API: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated
            else:
                # Alte gym API: obs, reward, done, info
                obs, reward, done, info = step_out
                
            # obs, reward, done, info sind jetzt korrekt zugewiesen
            # (obs wird eine Liste von 2 Beobachtungen sein)

    env.close()
    print(f"Dumps erfolgreich gesammelt in: {logs_dir}")

if __name__ == "__main__":
    run_dump_collection()