import gfootball.env as football_env

szenario_namen = [
    "11_vs_11_stochastic",
    "academy_empty_goal",
    "academy_run_to_score",
    "academy_pass_and_shoot_with_keeper",
    "academy_3_vs_1_with_keeper",
    "academy_corner",
    "academy_counterattack_easy",
]

print(f"{'Szenario':<40} | {'Spieler Links':<15} | {'Spieler Rechts':<15}")
print("-" * 75)

for szenario in szenario_namen:
    try:
        env = football_env.create_environment(
            env_name=szenario,
            representation='simple115v2',
            render=False
        )

        if hasattr(env.unwrapped, '_env'):
            game_env = env.unwrapped._env
            if hasattr(game_env, '_config'):
                config = game_env._config
            else:
                config = env.unwrapped._config
        else:
            config = env.unwrapped._config
        

        spieler_links = config.get('players', 'N/A')
        
        spieler_rechts = config.get('number_of_right_players_agent_controls', 0)
        
        if spieler_links == 'N/A':
            spieler_links = config.get('number_of_left_players_agent_controls', 1)
        
        print(f"{szenario:<40} | {spieler_links:<15} | {spieler_rechts:<15}")
        
        env.close()
    except Exception as e:
        print(f"Konnte Szenario '{szenario}' nicht laden: {e}")