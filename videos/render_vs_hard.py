"""
Render trained agent vs hard bot to MP4
"""
import numpy as np
import torch
import gfootball.env as football_env
from pathlib import Path

# Import from your training script
FEATURE_DIM = 93
OBS_DIM = 115
NUM_ACTIONS = 19


class FeatureEngineer:
    GOAL = np.array([1.0, 0.0], dtype=np.float32)
    OWN_GOAL = np.array([-1.0, 0.0], dtype=np.float32)
    
    @staticmethod
    def extract(obs: np.ndarray, active_idx: int = None) -> np.ndarray:
        squeeze = obs.ndim == 1
        if squeeze:
            obs = obs.reshape(1, -1)
        B = obs.shape[0]
        obs = obs[:, :115] if obs.shape[1] >= 115 else np.pad(obs, ((0, 0), (0, 115 - obs.shape[1])))
        feat = np.zeros((B, FEATURE_DIM), dtype=np.float32)
        
        left_pos = obs[:, 0:22].reshape(B, 11, 2)
        left_dir = obs[:, 22:44].reshape(B, 11, 2)
        right_pos = obs[:, 44:66].reshape(B, 11, 2)
        right_dir = obs[:, 66:88].reshape(B, 11, 2)
        ball_pos, ball_z, ball_dir = obs[:, 88:90], obs[:, 90:91], obs[:, 91:94]
        ball_owned_team = np.argmax(obs[:, 94:97], axis=1) - 1
        game_mode, sticky = obs[:, 98:105], obs[:, 105:115]
        
        if active_idx is None:
            active_idx = np.argmax(obs[:, 97:108], axis=1)
        elif isinstance(active_idx, int):
            active_idx = np.full(B, active_idx)
        
        bi = np.arange(B)
        active_pos = left_pos[bi, active_idx]
        ball_speed = np.linalg.norm(ball_dir[:, :2], axis=1)
        
        feat[:, 0:2] = ball_pos
        feat[:, 2] = np.clip(ball_z[:, 0], 0, 1)
        feat[:, 3] = np.clip(ball_speed, 0, 2)
        feat[:, 4:6] = ball_dir[:, :2]
        feat[:, 6] = ball_owned_team / 2.0
        
        rel_ball = ball_pos - active_pos
        feat[:, 7:9] = rel_ball
        feat[:, 9] = np.clip(np.linalg.norm(rel_ball, axis=1), 0, 2)
        feat[:, 10] = np.arctan2(rel_ball[:, 1], rel_ball[:, 0]) / np.pi
        
        goal_vec = FeatureEngineer.GOAL - ball_pos
        dist_goal = np.linalg.norm(goal_vec, axis=1)
        feat[:, 11] = np.clip(dist_goal, 0, 2)
        feat[:, 12] = np.abs(np.arctan2(goal_vec[:, 1], goal_vec[:, 0])) / np.pi
        feat[:, 13] = np.clip(0.088 / (dist_goal + 0.01), 0, 1)
        feat[:, 14] = (dist_goal < 0.35).astype(np.float32)
        feat[:, 15] = np.clip(np.linalg.norm(FeatureEngineer.OWN_GOAL - ball_pos, axis=1), 0, 2)
        
        right_x = right_pos[:, :, 0]
        keeper_idx = np.argmax(right_x, axis=1)
        keeper_pos = right_pos[bi, keeper_idx]
        keeper_dist = np.linalg.norm(ball_pos - keeper_pos, axis=1)
        feat[:, 16] = np.clip(keeper_dist, 0, 2)
        feat[:, 17] = np.arctan2((ball_pos - keeper_pos)[:, 1], (ball_pos - keeper_pos)[:, 0]) / np.pi
        
        left_active = np.abs(left_pos[:, :, 0]) > 0.01
        tm_dist = np.linalg.norm(left_pos - active_pos[:, None, :], axis=2)
        tm_dist[bi, active_idx] = 999.0
        tm_dist = np.where(left_active, tm_dist, 999.0)
        tm_idx = np.argsort(tm_dist, axis=1)
        
        for i in range(5):
            idx = tm_idx[:, i]
            valid = tm_dist[bi, idx] < 100
            feat[:, 18+i*4:20+i*4] = np.where(valid[:, None], left_pos[bi, idx] - active_pos, 0)
            feat[:, 20+i*4:22+i*4] = np.where(valid[:, None], left_dir[bi, idx], 0)
        
        right_active = np.abs(right_pos[:, :, 0]) > 0.01
        op_dist = np.linalg.norm(right_pos - active_pos[:, None, :], axis=2)
        op_dist = np.where(right_active, op_dist, 999.0)
        op_idx = np.argsort(op_dist, axis=1)
        
        for i in range(5):
            idx = op_idx[:, i]
            valid = op_dist[bi, idx] < 100
            feat[:, 38+i*4:40+i*4] = np.where(valid[:, None], right_pos[bi, idx] - active_pos, 0)
            feat[:, 40+i*4:42+i*4] = np.where(valid[:, None], right_dir[bi, idx], 0)
        
        ball_x = ball_pos[:, 0]
        left_x = left_pos[:, :, 0]
        feat[:, 58] = np.sum((left_x > ball_x[:, None]) & left_active, axis=1) / 11.0
        feat[:, 59] = np.sum((right_x > ball_x[:, None]) & right_active, axis=1) / 11.0
        feat[:, 60] = np.clip((feat[:, 58] - feat[:, 59]) * 2, -1, 1)
        
        opp_dist_active = np.where(right_active, np.linalg.norm(right_pos - active_pos[:, None, :], axis=2), 10.0)
        feat[:, 63] = np.clip(np.min(opp_dist_active, axis=1), 0, 1)
        
        sorted_rx = np.sort(right_x, axis=1)
        offside_line = np.maximum(ball_x, sorted_rx[:, 1])
        feat[:, 67] = offside_line
        feat[:, 68] = ((active_pos[:, 0] > offside_line) & (ball_owned_team == 0)).astype(np.float32)
        
        feat[:, 69:72] = np.column_stack([(ball_x > 0.33), ((ball_x >= -0.33) & (ball_x <= 0.33)), (ball_x < -0.33)])
        feat[:, 72:74] = np.column_stack([(ball_pos[:, 1] > 0.2), (ball_pos[:, 1] < -0.2)])
        feat[:, 74:76] = sticky[:, 8:10]
        
        sticky_dir = sticky[:, :8]
        sticky_active = np.any(sticky_dir > 0, axis=1)
        angle = np.argmax(sticky_dir, axis=1) * (2 * np.pi / 8)
        feat[:, 76:78] = np.column_stack([np.where(sticky_active, np.cos(angle), 0), np.where(sticky_active, np.sin(angle), 0)])
        
        feat[:, 78:85] = game_mode
        feat[:, 85:93] = 0
        
        return feat[0] if squeeze else feat


class PopArtValueHead(torch.nn.Module):
    def __init__(self, input_dim: int, beta: float = 1e-3):
        super().__init__()
        self.beta = beta
        self.linear = torch.nn.Linear(input_dim, 1)
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sigma', torch.ones(1))
        self.register_buffer('nu', torch.ones(1))
    
    def forward(self, x):
        return self.linear(x).squeeze(-1)
    
    def denormalize(self, normalized):
        return normalized * self.sigma + self.mu


class Net(torch.nn.Module):
    def __init__(self, d_model: int = 128, lstm_hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.action_emb = torch.nn.Embedding(NUM_ACTIONS + 1, 16)
        input_dim = OBS_DIM + FEATURE_DIM + 16
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, d_model),
            torch.nn.LayerNorm(d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.LayerNorm(d_model),
            torch.nn.ReLU()
        )
        self.lstm = torch.nn.LSTM(d_model, lstm_hidden, num_layers=1, batch_first=True)
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(lstm_hidden, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, NUM_ACTIONS)
        )
        self.value = PopArtValueHead(lstm_hidden)
    
    def init_hidden(self, batch_size, device):
        return (
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        )
    
    def forward(self, obs, feat, prev_actions=None, hidden=None):
        squeeze = obs.dim() == 2
        if squeeze:
            obs, feat = obs.unsqueeze(1), feat.unsqueeze(1)
        B, L, _ = obs.shape
        
        if prev_actions is None:
            prev_actions = torch.full((B,), NUM_ACTIONS, dtype=torch.long, device=obs.device)
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.unsqueeze(1).expand(-1, L)
        
        x = torch.cat([obs, feat, self.action_emb(prev_actions)], dim=-1)
        x = self.encoder(x)
        
        if hidden is None:
            hidden = self.init_hidden(B, obs.device)
        x, hidden = self.lstm(x, hidden)
        
        logits = self.policy(x)
        values_norm = self.value(x)
        
        if squeeze:
            logits, values_norm = logits.squeeze(1), values_norm.squeeze(1)
        
        return logits, values_norm, hidden
    
    def get_action(self, obs, feat, prev_actions, hidden=None, deterministic=False):
        logits, values_norm, hidden = self.forward(obs, feat, prev_actions, hidden)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
        return actions, hidden


def render_episode(checkpoint_path: str, output_dir: str, d_model: int = 512, 
                   lstm_hidden: int = 512, deterministic: bool = True,
                   max_steps: int = 3000, render_live: bool = True):
    """Render one episode against hard bot"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = Net(d_model=d_model, lstm_hidden=lstm_hidden).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'], strict=False)
    model.eval()
    
    if 'rating_mu' in ckpt:
        print(f"Checkpoint rating: μ={ckpt['rating_mu']:.1f}, σ={ckpt.get('rating_sigma', 0):.2f}")
    if 'updates' in ckpt:
        print(f"Updates: {ckpt['updates']}, Steps: {ckpt.get('total_steps', 0):,}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create environment
    print(f"Creating environment (11_vs_11_hard_stochastic)...")
    print(f"Live render: {render_live}")
    print(f"Dump dir: {output_dir}")
    
    env = football_env.create_environment(
        env_name="11_vs_11_hard_stochastic",
        representation="simple115v2",
        number_of_left_players_agent_controls=1,
        number_of_right_players_agent_controls=0,
        rewards="scoring",
        render=render_live,  # Live Fenster anzeigen
        write_video=render_live,  # Video nur wenn render=True
        write_full_episode_dumps=True,  # Dump für spätere Konvertierung
        logdir=str(output_dir),
    )
    
    feat_eng = FeatureEngineer()
    
    # Run episode
    print("Starting episode..." + (" (Fenster sollte aufgehen)" if render_live else ""))
    raw_obs = env.reset()
    obs = np.array(raw_obs).flatten()[:OBS_DIM].astype(np.float32)
    feat = feat_eng.extract(obs)
    
    prev_act = torch.tensor([NUM_ACTIONS], dtype=torch.long, device=device)
    hidden = model.init_hidden(1, device)
    
    total_reward = 0
    step = 0
    
    while step < max_steps:
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            action, hidden = model.get_action(obs_t, feat_t, prev_act, hidden, 
                                              deterministic=deterministic)
        
        prev_act = action.clone()
        raw_obs, reward, done, info = env.step([action.item()])
        
        total_reward += reward
        step += 1
        
        if step % 500 == 0:
            print(f"  Step {step}, reward so far: {total_reward}")
        
        if done:
            break
        
        obs = np.array(raw_obs).flatten()[:OBS_DIM].astype(np.float32)
        feat = feat_eng.extract(obs)
    
    # Get final score
    score = info.get('score', [0, 0]) if isinstance(info, dict) else [0, 0]
    print(f"\nEpisode finished!")
    print(f"  Steps: {step}")
    print(f"  Total reward: {total_reward}")
    print(f"  Score: {score[0]} - {score[1]} (Agent - Hard Bot)")
    print(f"  Result: {'WIN' if score[0] > score[1] else 'LOSS' if score[0] < score[1] else 'DRAW'}")
    
    env.close()
    
    # Find output files
    print(f"\nOutput directory: {output_dir}")
    
    video_files = list(output_dir.glob("*.avi")) + list(output_dir.glob("*.webm"))
    if video_files:
        print(f"Video: {video_files[-1]}")
    
    dump_files = list(output_dir.glob("*.dump"))
    if dump_files:
        print(f"Dump: {dump_files[-1]}")
        print(f"\nZum späteren Konvertieren (Linux/WSL):")
        print(f"  python -m gfootball.replay --trace {dump_files[-1]}")
    
    return total_reward, score


if __name__ == "__main__":
    # ============ KONFIGURATION ============
    CHECKPOINT = r"C:\clones\rlib_gfootball\checkpoints_league\snap_u99_r105.pt"
    OUTPUT_DIR = r"./videos"
    D_MODEL = 512
    LSTM_HIDDEN = 512
    NUM_EPISODES = 1
    DETERMINISTIC = True
    RENDER_LIVE = True

    
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    for i in range(NUM_EPISODES):
        ep_output_dir = output_dir / f"episode_{i+1}"
        
        print(f"\n{'='*60}")
        print(f"Episode {i+1}/{NUM_EPISODES}")
        print(f"{'='*60}")
        
        reward, score = render_episode(
            checkpoint_path=CHECKPOINT,
            output_dir=str(ep_output_dir),
            d_model=D_MODEL,
            lstm_hidden=LSTM_HIDDEN,
            deterministic=DETERMINISTIC,
            render_live=RENDER_LIVE,
        )
        results.append({'reward': reward, 'score': score})
    
    if NUM_EPISODES > 1:
        wins = sum(1 for r in results if r['score'][0] > r['score'][1])
        draws = sum(1 for r in results if r['score'][0] == r['score'][1])
        losses = sum(1 for r in results if r['score'][0] < r['score'][1])
        avg_reward = np.mean([r['reward'] for r in results])
        
        print(f"\n{'='*60}")
        print(f"Summary ({NUM_EPISODES} episodes)")
        print(f"{'='*60}")
        print(f"  Wins: {wins}, Draws: {draws}, Losses: {losses}")
        print(f"  Win rate: {wins/NUM_EPISODES*100:.1f}%")
        print(f"  Average reward: {avg_reward:.2f}")