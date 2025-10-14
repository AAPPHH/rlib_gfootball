import numpy as np
import gymnasium as gym
import gfootball.env as football_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
 
 
class GFootballGymnasiumWrapper(gym.Env):
    """Wrapper um altes Gym-Environment für Gymnasium kompatibel zu machen"""
   
    def __init__(self, env_name="academy_empty_goal_close"):
        super().__init__()
       
        self.env = football_env.create_environment(
            env_name=env_name,
            representation="simple115v2",
            rewards="scoring",
            number_of_left_players_agent_controls=1,
            number_of_right_players_agent_controls=0,
        )
       
        self.action_space = gym.spaces.Discrete(19)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(115,),
            dtype=np.float32
        )
       
    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs = self.env.reset()
        return obs[0].astype(np.float32), {}
   
    def step(self, action):
        obs, reward, done, info = self.env.step([action])
        terminated = done
        truncated = False
        return obs[0].astype(np.float32), float(reward), terminated, truncated, info
   
    def render(self):
        return self.env.render()
   
    def close(self):
        self.env.close()
 
 
def make_env():
    """Factory Funktion für die Environment"""
    def _init():
        env = GFootballGymnasiumWrapper("academy_empty_goal_close")
        env = Monitor(env)
        return env
    return _init
 
 
def train_agent():
    print(">>> Erstelle Google Football Environment...")
   
    env = DummyVecEnv([make_env()])
   
    print(">>> Environment erstellt.")
    print(">>> Starte Training mit PPO...")
    print(">>> Ziel: Lerne Torschuss im leeren Tor (academy_empty_goal_close)")
   
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./gfootball_tensorboard/"
    )
   
    print("\nTraining startet - das dauert ca. 5-10 Minuten...\n")
    model.learn(total_timesteps=50000, progress_bar=True)
   
    print("\n>>> Training abgeschlossen!")
    model.save("gfootball_ppo_model")
    print(">>> Model gespeichert als 'gfootball_ppo_model.zip'")
   
    env.close()
    return model
 
 
def evaluate_agent(model_path="gfootball_ppo_model", n_episodes=5):
    print("\n>>> Evaluiere trainierten Agent...")
   
    env = DummyVecEnv([make_env()])
    model = PPO.load(model_path)
   
    total_goals = 0
    for ep in range(n_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
       
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward[0]
            steps += 1
           
            if done[0]:
                break
       
        goals = 1 if episode_reward > 0 else 0
        total_goals += goals
        print(f"Episode {ep + 1}: Reward={episode_reward[0]:.2f}, Steps={steps}, Tor={'JA ⚽' if goals else 'NEIN'}")
   
    print(f"\n>>> Tore geschossen: {total_goals}/{n_episodes}")
    print(f">>> Erfolgsrate: {total_goals/n_episodes*100:.1f}%")
   
    env.close()
 
 
if __name__ == '__main__':
    # Training
    print("="*60)
    print("GOOGLE FOOTBALL - PPO TRAINING")
    print("Szenario: academy_empty_goal_close")
    print("="*60)
   
    model = train_agent()
   
    # Evaluation
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    evaluate_agent()