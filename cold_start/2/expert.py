import numpy as np
import gfootball.env as football_env
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

OBS_DIM = 115
NUM_AGENTS = 11
NUM_ACTIONS = 19

ROLE_ENCODING = np.array([
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0],
    [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0],
    [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1],
], dtype=np.float32)

ACTION_IDLE = 0
ACTION_LEFT = 1
ACTION_TOP_LEFT = 2
ACTION_TOP = 3
ACTION_TOP_RIGHT = 4
ACTION_RIGHT = 5
ACTION_BOTTOM_RIGHT = 6
ACTION_BOTTOM = 7
ACTION_BOTTOM_LEFT = 8
ACTION_LONG_PASS = 9
ACTION_HIGH_PASS = 10
ACTION_SHORT_PASS = 11
ACTION_SHOT = 12
ACTION_SPRINT = 13
ACTION_RELEASE_DIRECTION = 14
ACTION_RELEASE_SPRINT = 15
ACTION_SLIDING = 16
ACTION_DRIBBLE = 17
ACTION_RELEASE_DRIBBLE = 18


def angle_to_action(dx, dy):
    if abs(dx) < 0.01 and abs(dy) < 0.01:
        return ACTION_IDLE
    angle = np.arctan2(dy, dx)
    idx = int(np.round(angle / (np.pi / 4))) % 8
    return [ACTION_RIGHT, ACTION_TOP_RIGHT, ACTION_TOP, ACTION_TOP_LEFT, 
            ACTION_LEFT, ACTION_BOTTOM_LEFT, ACTION_BOTTOM, ACTION_BOTTOM_RIGHT][idx]


def expert_action(obs, agent_id):
    left_pos = obs[0:22].reshape(11, 2)
    right_pos = obs[44:66].reshape(11, 2)
    ball_pos = obs[88:90]
    ball_owned_team = np.argmax(obs[94:97]) - 1
    active_player = int(obs[97]) if len(obs) > 97 else agent_id
    
    my_pos = left_pos[agent_id]
    goal = np.array([1.0, 0.0])
    own_goal = np.array([-1.0, 0.0])
    
    dist_to_ball = np.linalg.norm(ball_pos - my_pos)
    dist_to_goal = np.linalg.norm(goal - ball_pos)
    
    am_closest = True
    for i in range(11):
        if i != agent_id:
            if np.linalg.norm(ball_pos - left_pos[i]) < dist_to_ball - 0.05:
                am_closest = False
                break
    
    if ball_owned_team == 0:
        if am_closest or dist_to_ball < 0.03:
            if dist_to_goal < 0.3:
                return ACTION_SHOT
            if ball_pos[0] > 0.7 and abs(ball_pos[1]) < 0.2:
                return ACTION_SHOT
            
            to_goal = goal - ball_pos
            
            keeper_pos = right_pos[0]
            keeper_dist = np.linalg.norm(keeper_pos - ball_pos)
            if keeper_dist < 0.2 and dist_to_goal < 0.4:
                return ACTION_SHOT
            
            opp_near = False
            for i in range(11):
                if np.linalg.norm(right_pos[i] - my_pos) < 0.1:
                    opp_near = True
                    break
            
            if opp_near and dist_to_goal > 0.3:
                for i in range(11):
                    if i != agent_id:
                        teammate = left_pos[i]
                        if teammate[0] > my_pos[0] and np.linalg.norm(teammate - my_pos) < 0.4:
                            return ACTION_SHORT_PASS
            
            return angle_to_action(to_goal[0], to_goal[1])
        else:
            support_pos = ball_pos + np.array([0.1, 0.15 if agent_id % 2 == 0 else -0.15])
            diff = support_pos - my_pos
            return angle_to_action(diff[0], diff[1])
    
    elif ball_owned_team == 1:
        if am_closest:
            diff = ball_pos - my_pos
            if dist_to_ball < 0.05:
                return ACTION_SLIDING
            return angle_to_action(diff[0], diff[1])
        else:
            defend_pos = own_goal * 0.3 + ball_pos * 0.7
            if agent_id < 5:
                defend_pos = own_goal * 0.5 + ball_pos * 0.5
            diff = defend_pos - my_pos
            return angle_to_action(diff[0], diff[1])
    
    else:
        diff = ball_pos - my_pos
        return angle_to_action(diff[0], diff[1])


def collect_expert_data(num_episodes=100, save_path="expert_data.parquet"):
    env = football_env.create_environment(
        env_name="11_vs_11_easy_stochastic",
        representation="simple115v2",
        number_of_left_players_agent_controls=NUM_AGENTS,
        number_of_right_players_agent_controls=NUM_AGENTS,
        rewards="scoring",
        render=False
    )
    
    all_obs = []
    all_actions = []
    all_agent_ids = []
    all_episode_ids = []
    all_step_ids = []
    all_rewards = []
    all_dones = []
    
    wins = 0
    goals_for = 0
    goals_against = 0
    
    for ep in range(num_episodes):
        obs = env.reset()
        obs = np.array(obs)[:NUM_AGENTS]
        
        episode_obs = []
        episode_actions = []
        episode_agent_ids = []
        episode_steps = []
        episode_rewards = []
        episode_dones = []
        
        done = False
        step = 0
        ep_return = 0
        
        while not done and step < 3000:
            actions_left = []
            for agent_id in range(NUM_AGENTS):
                action = expert_action(obs[agent_id], agent_id)
                actions_left.append(action)
                
                episode_obs.append(obs[agent_id].copy())
                episode_actions.append(action)
                episode_agent_ids.append(agent_id)
                episode_steps.append(step)
            
            actions_right = [ACTION_IDLE] * NUM_AGENTS
            
            obs, rew, done, info = env.step(actions_left + actions_right)
            obs = np.array(obs)[:NUM_AGENTS]
            
            rew_scalar = float(rew[0]) if hasattr(rew, '__len__') else float(rew)
            ep_return += rew_scalar
            
            for agent_id in range(NUM_AGENTS):
                episode_rewards.append(rew_scalar)
                episode_dones.append(float(done))
            
            if rew_scalar > 0:
                goals_for += 1
            elif rew_scalar < 0:
                goals_against += 1
            
            step += 1
        
        if ep_return > 0:
            wins += 1
            all_obs.extend(episode_obs)
            all_actions.extend(episode_actions)
            all_agent_ids.extend(episode_agent_ids)
            all_episode_ids.extend([ep] * len(episode_obs))
            all_step_ids.extend(episode_steps)
            all_rewards.extend(episode_rewards)
            all_dones.extend(episode_dones)
        
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1}/{num_episodes} | Wins: {wins} | GF: {goals_for} GA: {goals_against} | Samples: {len(all_obs)}")
    
    env.close()
    
    if len(all_obs) == 0:
        print("No winning episodes collected!")
        return None
    
    obs_array = np.array(all_obs, dtype=np.float32)
    
    table = pa.table({
        'obs': [obs_array[i].tobytes() for i in range(len(obs_array))],
        'action': all_actions,
        'agent_id': all_agent_ids,
        'episode_id': all_episode_ids,
        'step_id': all_step_ids,
        'reward': all_rewards,
        'done': all_dones,
    })
    
    pq.write_table(table, save_path)
    
    print(f"\nSaved {len(all_obs)} samples from {wins} winning episodes to {save_path}")
    print(f"Win rate: {wins/num_episodes*100:.1f}%")
    print(f"Goals: {goals_for} for, {goals_against} against")
    
    return save_path


def load_expert_data(path="expert_data.parquet"):
    table = pq.read_table(path)
    df = table.to_pandas()
    
    obs = np.array([np.frombuffer(b, dtype=np.float32) for b in df['obs']])
    actions = df['action'].values
    agent_ids = df['agent_id'].values
    
    print(f"Loaded {len(obs)} samples")
    print(f"Obs shape: {obs.shape}")
    print(f"Action distribution: {np.bincount(actions, minlength=NUM_ACTIONS)}")
    
    return obs, actions, agent_ids


if __name__ == "__main__":
    path = collect_expert_data(num_episodes=1, save_path="expert_data.parquet")
    
    if path:
        obs, actions, agent_ids = load_expert_data(path)