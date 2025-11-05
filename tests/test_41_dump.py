import gfootball.env as football_env
from gfootball.env import football_action_set as fas
from pathlib import Path

def run_dump_collection():
    logs_dir = Path.cwd() / "gfootball_dumps"
    logs_dir.mkdir(parents=True, exist_ok=True)

    env = football_env.create_environment(
        env_name="11_vs_11_stochastic",
        representation="simple115v2",          # gibt bereits 115er-Arrays zurück
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        render=False,
        write_full_episode_dumps=True,
        write_video=False,
        logdir=str(logs_dir),
    )

    num_agents = 1
    num_episodes = 1

    for _ in range(num_episodes):
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out  # Gym alte/new API abfangen
        done = False
        while not done:
            actions = [fas.action_idle] * num_agents  # exakt 1 Aktion
            step_out = env.step(actions)
            # alte Gym-API: (obs, reward, done, info)
            obs, reward, done, info = step_out

    env.close()

if __name__ == "__main__":
    run_dump_collection()
