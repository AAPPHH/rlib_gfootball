import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import sys

OBS_DIM = 115


def raw_obs_to_simple115(obs):
    left_team = np.array(obs['left_team']).flatten()
    left_team_direction = np.array(obs['left_team_direction']).flatten()
    right_team = np.array(obs['right_team']).flatten()
    right_team_direction = np.array(obs['right_team_direction']).flatten()
    
    ball = np.array(obs['ball'])
    ball_direction = np.array(obs['ball_direction'])
    
    ball_owned_team = obs['ball_owned_team']
    ball_owned = np.zeros(3, dtype=np.float32)
    if ball_owned_team == -1:
        ball_owned[0] = 1.0
    elif ball_owned_team == 0:
        ball_owned[1] = 1.0
    else:
        ball_owned[2] = 1.0
    
    active = obs['active']
    active_onehot = np.zeros(11, dtype=np.float32)
    active_onehot[active] = 1.0
    
    game_mode = obs['game_mode']
    game_mode_onehot = np.zeros(7, dtype=np.float32)
    game_mode_onehot[min(game_mode, 6)] = 1.0
    
    sticky = np.array(obs['sticky_actions'], dtype=np.float32)
    
    simple115 = np.concatenate([
        left_team,
        left_team_direction,
        right_team,
        right_team_direction,
        ball,
        ball_direction,
        ball_owned,
        active_onehot,
        game_mode_onehot,
        sticky,
    ]).astype(np.float32)
    
    return simple115[:OBS_DIM], active


def parse_replays(replay_dir, output_path):
    replay_dir = Path(replay_dir)
    json_files = list(replay_dir.glob("*.json"))
    print(f"Found {len(json_files)} replay files")
    
    all_obs = []
    all_act = []
    all_rew = []
    all_active = []
    all_episode = []
    all_step = []
    all_score = []
    
    episode_id = 0
    
    for json_path in json_files:
        try:
            with open(json_path) as f:
                data = json.load(f)
        except:
            print(f"Failed: {json_path}")
            continue
        
        rewards = data.get('rewards', [0, 0])
        steps = data['steps']
        
        for team_idx in [0, 1]:
            won = rewards[team_idx] > rewards[1 - team_idx]
            final_score = rewards[team_idx]
            
            if not won or final_score <= 0:
                continue
            
            prev_score = 0
            step_num = 0
            
            for step in steps[1:]:
                agent_data = step[team_idx]
                
                if 'observation' not in agent_data:
                    continue
                if 'players_raw' not in agent_data['observation']:
                    continue
                if len(agent_data['observation']['players_raw']) == 0:
                    continue
                
                raw_obs = agent_data['observation']['players_raw'][0]
                action = agent_data.get('action', [0])
                if isinstance(action, list):
                    action = action[0] if action else 0
                
                obs, active = raw_obs_to_simple115(raw_obs)
                
                current_score = raw_obs.get('score', [0, 0])
                my_score = current_score[team_idx]
                rew = my_score - prev_score
                prev_score = my_score
                
                all_obs.append(obs)
                all_act.append(action)
                all_rew.append(rew)
                all_active.append(active)
                all_episode.append(episode_id)
                all_step.append(step_num)
                all_score.append(final_score)
                
                step_num += 1
            
            if step_num > 0:
                episode_id += 1
        
        if (json_files.index(json_path) + 1) % 20 == 0:
            print(f"Processed {json_files.index(json_path) + 1}/{len(json_files)} | Episodes: {episode_id} | Samples: {len(all_obs)}")
    
    print(f"\nTotal: {episode_id} episodes, {len(all_obs)} samples")
    
    obs_bytes = [o.tobytes() for o in all_obs]
    
    table = pa.table({
        'obs': obs_bytes,
        'action': all_act,
        'reward': all_rew,
        'active': all_active,
        'episode_id': all_episode,
        'step': all_step,
        'score': all_score,
    })
    
    pq.write_table(table, output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    replay_dir = sys.argv[1] if len(sys.argv) > 1 else r"C:\clones\rlib_gfootball\cold_start\archive (1)"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "./expert.parquet"
    parse_replays(replay_dir, output_path)