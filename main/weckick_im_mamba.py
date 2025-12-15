"""Behavioral Cloning warmstart with Mamba architecture (stable version)."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import pyarrow.parquet as pq
import sys

OBS_DIM = 115
FEATURE_DIM = 93
NUM_ACTIONS = 19


class FeatureEngineer:
    GOAL = np.array([1.0, 0.0], dtype=np.float32)
    OWN_GOAL = np.array([-1.0, 0.0], dtype=np.float32)

    @staticmethod
    def extract(obs: np.ndarray, active_idx: int = 0) -> np.ndarray:
        obs = obs.flatten()[:115]
        feat = np.zeros(FEATURE_DIM, dtype=np.float32)
        
        left_pos = obs[0:22].reshape(11, 2)
        left_dir = obs[22:44].reshape(11, 2)
        right_pos = obs[44:66].reshape(11, 2)
        ball_pos = obs[88:90]
        ball_z = obs[90]
        ball_dir = obs[91:94]
        ball_owned_team = np.argmax(obs[94:97]) - 1
        game_mode = obs[98:105]
        sticky = obs[105:115]
        
        active_pos = left_pos[active_idx]
        ball_speed = np.linalg.norm(ball_dir[:2])
        
        feat[0:2] = ball_pos
        feat[2] = np.clip(ball_z, 0, 1)
        feat[3] = np.clip(ball_speed, 0, 2)
        feat[4:6] = ball_dir[:2]
        feat[6] = ball_owned_team / 2.0
        rel_ball = ball_pos - active_pos
        feat[7:9] = rel_ball
        feat[9] = np.clip(np.linalg.norm(rel_ball), 0, 2)
        feat[10] = np.arctan2(rel_ball[1], rel_ball[0]) / np.pi
        
        goal_vec = FeatureEngineer.GOAL - ball_pos
        dist_goal = np.linalg.norm(goal_vec)
        feat[11] = np.clip(dist_goal, 0, 2)
        feat[12] = np.abs(np.arctan2(goal_vec[1], goal_vec[0])) / np.pi
        feat[13] = np.clip(0.088 / (dist_goal + 0.01), 0, 1)
        feat[14] = float(dist_goal < 0.35)
        feat[15] = np.clip(np.linalg.norm(FeatureEngineer.OWN_GOAL - ball_pos), 0, 2)
        
        right_x = right_pos[:, 0]
        keeper_idx = np.argmax(right_x)
        keeper_pos = right_pos[keeper_idx]
        keeper_dist = np.linalg.norm(ball_pos - keeper_pos)
        feat[16] = np.clip(keeper_dist, 0, 2)
        feat[17] = np.arctan2((ball_pos - keeper_pos)[1], (ball_pos - keeper_pos)[0]) / np.pi
        
        left_active = np.abs(left_pos[:, 0]) > 0.01
        tm_dist = np.linalg.norm(left_pos - active_pos, axis=1)
        tm_dist[active_idx] = 999.0
        tm_dist = np.where(left_active, tm_dist, 999.0)
        tm_idx = np.argsort(tm_dist)
        for i in range(5):
            idx = tm_idx[i]
            if tm_dist[idx] < 100:
                feat[18+i*4:20+i*4] = left_pos[idx] - active_pos
                feat[20+i*4:22+i*4] = left_dir[idx]
        
        right_active = np.abs(right_pos[:, 0]) > 0.01
        op_dist = np.linalg.norm(right_pos - active_pos, axis=1)
        op_dist = np.where(right_active, op_dist, 999.0)
        op_idx = np.argsort(op_dist)
        for i in range(5):
            idx = op_idx[i]
            if op_dist[idx] < 100:
                feat[38+i*4:40+i*4] = right_pos[idx] - active_pos
                feat[40+i*4:42+i*4] = np.zeros(2)
        
        ball_x = ball_pos[0]
        left_x = left_pos[:, 0]
        feat[58] = np.sum((left_x > ball_x) & left_active) / 11.0
        feat[59] = np.sum((right_x > ball_x) & right_active) / 11.0
        feat[60] = np.clip((feat[58] - feat[59]) * 2, -1, 1)
        
        opp_dist = np.where(right_active, np.linalg.norm(right_pos - active_pos, axis=1), 10.0)
        feat[63] = np.clip(np.min(opp_dist), 0, 1)
        
        sorted_rx = np.sort(right_x)
        offside_line = max(ball_x, sorted_rx[1])
        feat[67] = offside_line
        feat[68] = float((active_pos[0] > offside_line) and (ball_owned_team == 0))
        feat[69] = float(ball_x > 0.33)
        feat[70] = float(-0.33 <= ball_x <= 0.33)
        feat[71] = float(ball_x < -0.33)
        feat[72] = float(ball_pos[1] > 0.2)
        feat[73] = float(ball_pos[1] < -0.2)
        
        feat[74:76] = sticky[8:10]
        sticky_dir = sticky[:8]
        if np.any(sticky_dir > 0):
            angle = np.argmax(sticky_dir) * (2 * np.pi / 8)
            feat[76] = np.cos(angle)
            feat[77] = np.sin(angle)
        feat[78:85] = game_mode
        
        return feat


class ExpertDataset(Dataset):
    def __init__(self, parquet_path):
        print(f"Loading {parquet_path}...")
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        
        self.obs = np.array([np.frombuffer(b, dtype=np.float32) for b in df['obs']])
        self.actions = df['action'].values.astype(np.int64)
        self.active = df['active'].values.astype(np.int64)
        
        print(f"Computing features for {len(self.obs)} samples...")
        feat_eng = FeatureEngineer()
        self.feat = np.array([feat_eng.extract(o, a) for o, a in zip(self.obs, self.active)])
        
        print(f"Dataset ready: {len(self.obs)} samples")
        print(f"Action distribution:")
        counts = np.bincount(self.actions, minlength=NUM_ACTIONS)
        for i, c in enumerate(counts):
            if c > 0:
                print(f"  {i:2d}: {c:6d} ({c/len(self.actions)*100:5.1f}%)")
    
    def __len__(self):
        return len(self.obs)
    
    def __getitem__(self, idx):
        return {
            'obs': torch.from_numpy(self.obs[idx]).float(),
            'feat': torch.from_numpy(self.feat[idx]).float(),
            'action': torch.tensor(self.actions[idx]).long(),
        }


class SoftClamp(nn.Module):
    """Soft Clipping with preserved gradients using tanh."""
    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.range = (max_val - min_val) / 2
        self.center = (max_val + min_val) / 2
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.center + self.range * torch.tanh((x - self.center) / self.range)


class MambaBlock(nn.Module):
    """Mamba selective state space block with numerical stability."""
    def __init__(self, d_model: int, d_state: int = 16, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.dt_proj = nn.Linear(d_model, d_model, bias=True)
        
        # A_log with bounded initialization
        self.A_log_diag = nn.Parameter(torch.log(torch.linspace(1.0, d_state, d_state)).clamp(max=2.0))
        
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)
        self.D = nn.Parameter(torch.ones(d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        # Soft clamps for numerical stability
        self.dt_clamp = SoftClamp(1e-4, 0.5)
        self.h_clamp = SoftClamp(-10.0, 10.0)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.in_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.B_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.C_proj.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
        
        # dt_proj: small weights and bias so softplus outputs small values
        nn.init.normal_(self.dt_proj.weight, std=0.001)
        nn.init.constant_(self.dt_proj.bias, -4.6)  # softplus(-4.6) ≈ 0.01

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, D = x.shape
        
        x_norm = self.norm(x)
        xz = self.in_proj(x_norm)
        x_in, z = xz.chunk(2, dim=-1)
        
        # dt with soft clamp for stability
        dt = self.dt_clamp(F.softplus(self.dt_proj(x_in)))
        
        B_t = self.B_proj(x_in)
        C_t = self.C_proj(x_in)
        
        # A with bounded range
        A_diag = -torch.exp(self.A_log_diag.clamp(max=2.0))
        
        outputs = []
        for t in range(L):
            dt_t = dt[:, t, :].unsqueeze(-1)
            dA = torch.exp(dt_t * A_diag.unsqueeze(0).unsqueeze(0))
            dB = dt_t * B_t[:, t, :].unsqueeze(1)
            
            h = h * dA + x_in[:, t, :].unsqueeze(-1) * dB
            h = self.h_clamp(h)
            
            y_t = (h * C_t[:, t, :].unsqueeze(1)).sum(dim=-1)
            outputs.append(y_t)
        
        y = torch.stack(outputs, dim=1)
        out = y * F.silu(z) + x_in * self.D
        out = x + self.dropout(self.out_proj(out))
        
        return out, h


class PopArtValueHead(nn.Module):
    """PopArt value head - included for checkpoint compatibility."""
    def __init__(self, input_dim: int, beta: float = 1e-3):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(input_dim, 1)
        self.register_buffer('mu', torch.zeros(1))
        self.register_buffer('sigma', torch.ones(1))
        self.register_buffer('nu', torch.ones(1))
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(x).squeeze(-1)
    
    def denormalize(self, normalized):
        return normalized * self.sigma + self.mu
    
    def normalize_target(self, targets):
        return (targets - self.mu) / self.sigma


class Net(nn.Module):
    def __init__(self, d_model: int = 128, d_state: int = 32, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.num_layers = num_layers
        
        self.action_emb = nn.Embedding(NUM_ACTIONS + 1, 16)
        
        input_dim = OBS_DIM + FEATURE_DIM + 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        
        # Mamba layers
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout) for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)
        
        self.policy = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )
        
        # Value head for RL compatibility
        self.value = PopArtValueHead(d_model)
        
        self._init()
        print(f"Net (Mamba): {sum(p.numel() for p in self.parameters()):,} params")

    def _init(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)

    def init_hidden(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        return [torch.zeros(batch_size, self.d_model, self.d_state, device=device) 
                for _ in range(self.num_layers)]

    def forward(self, obs, feat, prev_action=None):
        """Forward for BC training (single timestep)."""
        B = obs.shape[0]
        device = obs.device
        
        if prev_action is None:
            prev_action = torch.full((B,), NUM_ACTIONS, dtype=torch.long, device=device)
        
        obs = obs.unsqueeze(1)
        feat = feat.unsqueeze(1)
        prev_action = prev_action.unsqueeze(1)
        
        x = torch.cat([obs, feat, self.action_emb(prev_action)], dim=-1)
        x = self.encoder(x)
        
        hidden = self.init_hidden(B, device)
        
        for i, layer in enumerate(self.mamba_layers):
            x, _ = layer(x, hidden[i])
        
        x = self.final_norm(x)
        logits = self.policy(x).squeeze(1)
        return logits


def train_bc(parquet_path, epochs=20, batch_size=512, lr=1e-3, save_path="bc_mamba_warmstart.pt",
             d_model=128, d_state=32, num_layers=2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    dataset = ExpertDataset(parquet_path)
    
    n_train = int(len(dataset) * 0.9)
    n_val = len(dataset) - n_train
    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"Train: {n_train} | Val: {n_val}")
    print(f"Model: d_model={d_model}, d_state={d_state}, layers={num_layers}\n")
    
    model = Net(d_model=d_model, d_state=d_state, num_layers=num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9}")
    print("-" * 60)
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        
        for batch in train_loader:
            obs = batch['obs'].to(device)
            feat = batch['feat'].to(device)
            actions = batch['action'].to(device)
            
            logits = model(obs, feat)
            loss = F.cross_entropy(logits, actions)
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(obs)
            train_correct += (logits.argmax(dim=-1) == actions).sum().item()
            train_total += len(obs)
        
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        
        with torch.no_grad():
            for batch in val_loader:
                obs = batch['obs'].to(device)
                feat = batch['feat'].to(device)
                actions = batch['action'].to(device)
                
                logits = model(obs, feat)
                loss = F.cross_entropy(logits, actions)
                
                val_loss += loss.item() * len(obs)
                val_correct += (logits.argmax(dim=-1) == actions).sum().item()
                val_total += len(obs)
        
        train_loss /= train_total
        train_acc = train_correct / train_total * 100
        val_loss /= val_total
        val_acc = val_correct / val_total * 100
        
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:8.1f}% | {val_loss:10.4f} | {val_acc:8.1f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch + 1,
                'config': {
                    'd_model': d_model,
                    'd_state': d_state,
                    'num_layers': num_layers,
                }
            }, save_path)
    
    print(f"\nFinal: Train {train_acc:.1f}% | Val {val_acc:.1f}%")
    print(f"Best model saved to {save_path} (Val Acc: {best_val_acc:.1f}%)")
    
    return model, val_acc


if __name__ == "__main__":
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\clones\rlib_gfootball\main\expert.parquet"
    save_path = sys.argv[2] if len(sys.argv) > 2 else "./bc_mamba_warmstart.pt"
    
    train_bc(
        parquet_path, 
        epochs=20, 
        batch_size=512, 
        lr=1e-3, 
        save_path=save_path,
        d_model=128,
        d_state=32,
        num_layers=2,
    )