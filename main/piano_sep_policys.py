"""
IMPALA 11-Agent with PIANO-style Communication vs Idle Opponents
Each agent has its own policy network with message passing.
Team reward: all agents get +1/-1 on goals.
"""
import time
from pathlib import Path
from collections import deque
from typing import List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.amp import autocast, GradScaler
import ray
import gfootball.env as football_env

FEATURE_DIM = 93
OBS_DIM = 115
NUM_ACTIONS = 19
NUM_AGENTS = 11
MSG_DIM = 32  # Communication message dimension


class FeatureEngineer:
    GOAL = np.array([1.0, 0.0], dtype=np.float32)
    OWN_GOAL = np.array([-1.0, 0.0], dtype=np.float32)

    @staticmethod
    def extract(obs: np.ndarray, agent_idx: int) -> np.ndarray:
        """Extract features for a single agent given full obs."""
        if obs.ndim == 1:
            obs = obs.reshape(1, -1)
        obs = obs[:, :115] if obs.shape[1] >= 115 else np.pad(obs, ((0, 0), (0, 115 - obs.shape[1])))
        B = obs.shape[0]
        feat = np.zeros((B, FEATURE_DIM), dtype=np.float32)
        
        left_pos = obs[:, 0:22].reshape(B, 11, 2)
        left_dir = obs[:, 22:44].reshape(B, 11, 2)
        right_pos = obs[:, 44:66].reshape(B, 11, 2)
        right_dir = obs[:, 66:88].reshape(B, 11, 2)
        ball_pos, ball_z, ball_dir = obs[:, 88:90], obs[:, 90:91], obs[:, 91:94]
        ball_owned_team = np.argmax(obs[:, 94:97], axis=1) - 1
        game_mode, sticky = obs[:, 98:105], obs[:, 105:115]
        
        bi = np.arange(B)
        active_pos = left_pos[:, agent_idx]
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
        
        # Teammates relative to this agent
        left_active = np.abs(left_pos[:, :, 0]) > 0.01
        tm_dist = np.linalg.norm(left_pos - active_pos[:, None, :], axis=2)
        tm_dist[:, agent_idx] = 999.0
        tm_dist = np.where(left_active, tm_dist, 999.0)
        tm_idx = np.argsort(tm_dist, axis=1)
        for i in range(5):
            idx = tm_idx[:, i]
            valid = tm_dist[bi, idx] < 100
            feat[:, 18+i*4:20+i*4] = np.where(valid[:, None], left_pos[bi, idx] - active_pos, 0)
            feat[:, 20+i*4:22+i*4] = np.where(valid[:, None], left_dir[bi, idx], 0)
        
        # Opponents relative to this agent
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
        
        # Agent role encoding
        feat[:, 85] = agent_idx / 10.0
        feat[:, 86] = 1.0 if agent_idx == 0 else 0.0  # is_keeper
        feat[:, 87] = 1.0 if agent_idx <= 4 else 0.0  # is_defender
        feat[:, 88] = 1.0 if 5 <= agent_idx <= 7 else 0.0  # is_midfielder
        feat[:, 89] = 1.0 if agent_idx >= 8 else 0.0  # is_forward
        
        return feat[0] if feat.shape[0] == 1 else feat


class PopArtValueHead(nn.Module):
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

    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor):
        old_mu, old_sigma = self.mu.clone(), self.sigma.clone()
        self.mu.mul_(1 - self.beta).add_(self.beta * targets.mean())
        self.nu.mul_(1 - self.beta).add_(self.beta * (targets ** 2).mean())
        self.sigma.copy_(torch.sqrt(torch.clamp(self.nu - self.mu ** 2, min=1e-4)))
        self.linear.weight.data.mul_(old_sigma / self.sigma)
        self.linear.bias.data.copy_((self.linear.bias.data * old_sigma + old_mu - self.mu) / self.sigma)


class AgentNet(nn.Module):
    """Single agent network with PIANO-style communication."""
    def __init__(self, agent_idx: int, d_model: int = 128, lstm_hidden: int = 128, msg_dim: int = MSG_DIM):
        super().__init__()
        self.agent_idx = agent_idx
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
        self.msg_dim = msg_dim
        
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
        
        # Message output head: what I broadcast to teammates
        self.msg_out = nn.Sequential(
            nn.Linear(d_model, msg_dim),
            nn.Tanh()  # Bounded messages
        )
        
        # Message aggregation: attention over incoming messages
        self.msg_query = nn.Linear(d_model, msg_dim)
        self.msg_key = nn.Linear(msg_dim, msg_dim)
        self.msg_value = nn.Linear(msg_dim, msg_dim)
        
        # Combine encoded state with aggregated messages
        self.combine = nn.Sequential(
            nn.Linear(d_model + msg_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )
        
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=1, batch_first=True)
        
        self.policy = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )
        self.value = PopArtValueHead(lstm_hidden)
        
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
        # Small init for message heads to start with weak communication
        nn.init.orthogonal_(self.msg_out[0].weight, gain=0.1)

    def init_hidden(self, batch_size, device):
        return (torch.zeros(1, batch_size, self.lstm_hidden, device=device),
                torch.zeros(1, batch_size, self.lstm_hidden, device=device))

    def encode(self, obs, feat, prev_actions=None):
        """First pass: encode observation and generate outgoing message."""
        squeeze = obs.dim() == 2
        if squeeze:
            obs, feat = obs.unsqueeze(1), feat.unsqueeze(1)
        
        B, L, _ = obs.shape
        
        if prev_actions is None:
            prev_actions = torch.full((B,), NUM_ACTIONS, dtype=torch.long, device=obs.device)
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.unsqueeze(1).expand(-1, L)
        
        x = torch.cat([obs, feat, self.action_emb(prev_actions)], dim=-1)
        h = self.encoder(x)  # [B, L, d_model]
        msg = self.msg_out(h)  # [B, L, msg_dim]
        
        if squeeze:
            h, msg = h.squeeze(1), msg.squeeze(1)
        return h, msg

    def forward_with_messages(self, h, incoming_msgs, hidden=None):
        """Second pass: aggregate messages and compute action."""
        squeeze = h.dim() == 2
        if squeeze:
            h = h.unsqueeze(1)
            incoming_msgs = incoming_msgs.unsqueeze(1)
        
        B, L, _ = h.shape
        
        # Attention over incoming messages
        # incoming_msgs: [B, L, num_teammates, msg_dim]
        query = self.msg_query(h)  # [B, L, msg_dim]
        keys = self.msg_key(incoming_msgs)  # [B, L, 10, msg_dim]
        values = self.msg_value(incoming_msgs)  # [B, L, 10, msg_dim]
        
        # Scaled dot-product attention
        attn_scores = torch.einsum('bld,blnd->bln', query, keys) / (self.msg_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, L, 10]
        msg_agg = torch.einsum('bln,blnd->bld', attn_weights, values)  # [B, L, msg_dim]
        
        # Combine
        h_combined = self.combine(torch.cat([h, msg_agg], dim=-1))
        
        if hidden is None:
            hidden = self.init_hidden(B, h.device)
        x, hidden = self.lstm(h_combined, hidden)
        
        logits = self.policy(x)
        values_norm = self.value(x)
        
        if squeeze:
            logits, values_norm = logits.squeeze(1), values_norm.squeeze(1)
        return logits, values_norm, hidden, attn_weights.squeeze(1) if squeeze else attn_weights

    def forward(self, obs, feat, prev_actions=None, hidden=None, incoming_msgs=None):
        """Full forward pass."""
        h, msg = self.encode(obs, feat, prev_actions)
        
        if incoming_msgs is None:
            # No messages: use zeros
            if h.dim() == 2:
                incoming_msgs = torch.zeros(h.shape[0], NUM_AGENTS - 1, self.msg_dim, device=h.device)
            else:
                incoming_msgs = torch.zeros(h.shape[0], h.shape[1], NUM_AGENTS - 1, self.msg_dim, device=h.device)
        
        logits, values_norm, hidden, attn = self.forward_with_messages(h, incoming_msgs, hidden)
        return logits, values_norm, hidden, msg

    def get_action(self, obs, feat, prev_actions, hidden=None, incoming_msgs=None):
        logits, values_norm, hidden, msg = self.forward(obs, feat, prev_actions, hidden, incoming_msgs)
        dist = Categorical(logits=logits)
        actions = dist.sample()
        values = self.value.denormalize(values_norm)
        return actions, dist.log_prob(actions), values, hidden, msg


class MultiAgentNet(nn.Module):
    """Container for all 11 agent networks with communication."""
    def __init__(self, d_model: int = 128, lstm_hidden: int = 128, msg_dim: int = MSG_DIM):
        super().__init__()
        self.msg_dim = msg_dim
        self.agents = nn.ModuleList([
            AgentNet(i, d_model, lstm_hidden, msg_dim) for i in range(NUM_AGENTS)
        ])
        total_params = sum(p.numel() for p in self.parameters())
        print(f"MultiAgentNet: {NUM_AGENTS} agents x ~{total_params//NUM_AGENTS:,} = {total_params:,} params (msg_dim={msg_dim})")
    
    def get_agent(self, idx: int) -> AgentNet:
        return self.agents[idx]
    
    def communicate(self, obs_list, feat_list, prev_acts_list, hiddens, device):
        """Two-pass communication: encode -> exchange -> act."""
        # Pass 1: All agents encode and generate messages
        encodings = []
        messages = []
        for i in range(NUM_AGENTS):
            agent = self.agents[i]
            h, msg = agent.encode(
                torch.from_numpy(obs_list[i]).float().unsqueeze(0).to(device),
                torch.from_numpy(feat_list[i]).float().unsqueeze(0).to(device),
                prev_acts_list[i].to(device)
            )
            encodings.append(h)
            messages.append(msg)
        
        # Stack messages for exchange
        all_msgs = torch.stack(messages, dim=1)  # [1, 11, msg_dim]
        
        # Pass 2: Each agent receives others' messages and acts
        actions = []
        log_probs = []
        values = []
        new_hiddens = []
        out_messages = []
        
        for i in range(NUM_AGENTS):
            # Get messages from other agents (exclude self)
            other_msgs = torch.cat([all_msgs[:, :i], all_msgs[:, i+1:]], dim=1)  # [1, 10, msg_dim]
            
            agent = self.agents[i]
            logits, values_norm, new_hidden, attn = agent.forward_with_messages(
                encodings[i], other_msgs, hiddens[i]
            )
            
            dist = Categorical(logits=logits)
            act = dist.sample()
            lp = dist.log_prob(act)
            val = agent.value.denormalize(values_norm)
            
            actions.append(act.item())
            log_probs.append(lp.item())
            values.append(val.item())
            new_hiddens.append(new_hidden)
            out_messages.append(messages[i].detach().cpu().numpy())
        
        return actions, log_probs, values, new_hiddens, out_messages


def vtrace(behavior_lp, target_lp, rewards, values, bootstrap, dones, gamma=0.99, rho_bar=1.0, c_bar=1.0):
    B, T = rewards.shape
    rhos = torch.exp((target_lp - behavior_lp).clamp(-20, 20))
    clipped_rhos = torch.clamp(rhos, max=rho_bar)
    cs = torch.clamp(rhos, max=c_bar)
    not_done = 1.0 - dones
    values_tp1 = torch.cat([values[:, 1:], bootstrap.unsqueeze(1)], dim=1)
    deltas = clipped_rhos * (rewards + gamma * values_tp1 * not_done - values)
    
    vs_minus_v = torch.zeros_like(rewards)
    acc = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        acc = deltas[:, t] + gamma * cs[:, t] * acc * not_done[:, t]
        vs_minus_v[:, t] = acc
    
    vs = values + vs_minus_v
    vs_tp1 = torch.cat([vs[:, 1:], bootstrap.unsqueeze(1)], dim=1)
    advantages = clipped_rhos * (rewards + gamma * vs_tp1 * not_done - values)
    return vs, advantages, clipped_rhos


@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, wid: int, d_model: int, lstm_hidden: int, rollout_len: int, env_name: str, msg_dim: int = MSG_DIM):
        self.wid = wid
        self.rollout_len = rollout_len
        self.feat_eng = FeatureEngineer()
        
        self.env = football_env.create_environment(
            env_name=env_name,
            representation="simple115v2",
            number_of_left_players_agent_controls=NUM_AGENTS,
            number_of_right_players_agent_controls=NUM_AGENTS,
            rewards="scoring",
            render=False
        )
        
        self.model = MultiAgentNet(d_model, lstm_hidden, msg_dim)
        self.model.eval()
        self._reset()

    def set_weights(self, weights: Dict):
        self.model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in weights.items()})

    def _reset(self):
        obs = self.env.reset()
        self.obs = [np.array(obs[i]).flatten()[:OBS_DIM].astype(np.float32) for i in range(NUM_AGENTS)]
        self.feat = [self.feat_eng.extract(self.obs[i].reshape(1, -1), i) for i in range(NUM_AGENTS)]
        self.ep_ret, self.ep_len = 0.0, 0
        self.prev_acts = [torch.tensor([NUM_ACTIONS], dtype=torch.long) for _ in range(NUM_AGENTS)]
        self.hiddens = [self.model.get_agent(i).init_hidden(1, torch.device('cpu')) for i in range(NUM_AGENTS)]
        self.last_messages = [np.zeros(MSG_DIM, dtype=np.float32) for _ in range(NUM_AGENTS)]

    def collect(self):
        # Data per agent
        data = {i: {k: [] for k in ['obs', 'feat', 'prev_act', 'act', 'lp', 'rew', 'done', 'msg_in']} 
                for i in range(NUM_AGENTS)}
        episodes = []
        max_rew, min_rew = -999, 999
        
        for _ in range(self.rollout_len):
            # Two-pass communication
            with torch.no_grad():
                actions, log_probs, values, self.hiddens, new_messages = self.model.communicate(
                    self.obs, self.feat, self.prev_acts, self.hiddens, torch.device('cpu')
                )
            
            # Store data for each agent
            for i in range(NUM_AGENTS):
                # Get incoming messages (from others)
                other_msgs = np.stack([self.last_messages[j] for j in range(NUM_AGENTS) if j != i])
                
                data[i]['obs'].append(self.obs[i].copy())
                data[i]['feat'].append(self.feat[i].copy())
                data[i]['prev_act'].append(self.prev_acts[i].item())
                data[i]['act'].append(actions[i])
                data[i]['lp'].append(log_probs[i])
                data[i]['msg_in'].append(other_msgs.copy())
                self.prev_acts[i] = torch.tensor([actions[i]], dtype=torch.long)
            
            # Update messages for next step
            self.last_messages = [m.flatten() for m in new_messages]
            
            # Right team: idle
            right_actions = [0] * NUM_AGENTS
            
            obs_new, rew, done, info = self.env.step(actions + right_actions)
            
            # Fix: reward is array for multi-agent
            rew = float(np.sum(rew)) if hasattr(rew, '__len__') else float(rew)
            
            self.ep_ret += rew
            self.ep_len += 1
            max_rew, min_rew = max(max_rew, rew), min(min_rew, rew)
            
            ep_done = bool(done) or self.ep_len >= 3000
            
            # All agents get same team reward
            for i in range(NUM_AGENTS):
                data[i]['rew'].append(rew)
                data[i]['done'].append(float(ep_done))
            
            if ep_done:
                score = info.get("score", [0, 0]) if isinstance(info, dict) else [0, 0]
                won = score[0] > score[1]
                episodes.append({'return': self.ep_ret, 'won': won, 'length': self.ep_len, 'score': score})
                self._reset()
            else:
                for i in range(NUM_AGENTS):
                    self.obs[i] = np.array(obs_new[i]).flatten()[:OBS_DIM].astype(np.float32)
                    self.feat[i] = self.feat_eng.extract(self.obs[i].reshape(1, -1), i)
        
        # Compute bootstraps with communication
        bootstraps = []
        with torch.no_grad():
            # Encode all agents
            encodings = []
            messages = []
            for i in range(NUM_AGENTS):
                agent = self.model.get_agent(i)
                h, msg = agent.encode(
                    torch.from_numpy(self.obs[i]).float().unsqueeze(0),
                    torch.from_numpy(self.feat[i]).float().unsqueeze(0),
                    self.prev_acts[i]
                )
                encodings.append(h)
                messages.append(msg)
            
            all_msgs = torch.stack(messages, dim=1)
            
            for i in range(NUM_AGENTS):
                other_msgs = torch.cat([all_msgs[:, :i], all_msgs[:, i+1:]], dim=1)
                agent = self.model.get_agent(i)
                _, values_norm, _, _ = agent.forward_with_messages(encodings[i], other_msgs, self.hiddens[i])
                bootstrap = agent.value.denormalize(values_norm)
                bootstraps.append(bootstrap.item())
        
        # Package rollouts per agent
        rollouts = {}
        for i in range(NUM_AGENTS):
            rollouts[i] = {
                'obs': np.array(data[i]['obs'], dtype=np.float32),
                'feat': np.array(data[i]['feat'], dtype=np.float32),
                'prev_act': np.array(data[i]['prev_act'], dtype=np.int64),
                'act': np.array(data[i]['act'], dtype=np.int64),
                'lp': np.array(data[i]['lp'], dtype=np.float32),
                'rew': np.array(data[i]['rew'], dtype=np.float32),
                'done': np.array(data[i]['done'], dtype=np.float32),
                'msg_in': np.array(data[i]['msg_in'], dtype=np.float32),  # [T, 10, msg_dim]
                'bootstrap': bootstraps[i],
            }
        
        return {
            'rollouts': rollouts,
            'episodes': episodes,
            'max_rew': max_rew,
            'min_rew': min_rew
        }

    def close(self):
        self.env.close()


class Learner:
    def __init__(self, num_workers=16, rollout_len=256, batch_size=32,
                 lr=5e-4, gamma=0.997, entropy_coeff=0.01, value_coeff=0.5,
                 d_model=128, lstm_hidden=128, msg_dim=MSG_DIM, env_name="11_vs_11_stochastic",
                 checkpoint_dir="./checkpoints_11agent"):
        
        self.num_workers = num_workers
        self.rollout_len, self.batch_size = rollout_len, batch_size
        self.gamma, self.entropy_coeff, self.value_coeff = gamma, entropy_coeff, value_coeff
        self.msg_dim = msg_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.model = MultiAgentNet(d_model, lstm_hidden, msg_dim).to(self.device)
        
        # Separate optimizer per agent for cleaner updates
        self.optimizers = [
            torch.optim.Adam(self.model.get_agent(i).parameters(), lr=lr, eps=1e-5)
            for i in range(NUM_AGENTS)
        ]
        self.scaler = GradScaler('cuda')
        
        ray.init(ignore_reinit_error=True, num_cpus=num_workers + 4)
        self.workers = [Worker.remote(i, d_model, lstm_hidden, rollout_len, env_name, msg_dim)
                        for i in range(num_workers)]
        
        self.total_steps, self.updates, self.start = 0, 0, None
        self.returns, self.wins, self.lengths = deque(maxlen=100), deque(maxlen=100), deque(maxlen=100)
        self.max_rew, self.min_rew = -999, 999
        self.pending, self.queue = {}, []
        
        # Per-agent stats
        self.agent_stats = {i: {'entropy': deque(maxlen=50), 'grad': deque(maxlen=50)} 
                           for i in range(NUM_AGENTS)}
        
        print(f"\n{'='*70}")
        print(f"IMPALA 11-Agent PIANO Communication | {self.device} | {num_workers}W")
        print(f"Batch: {batch_size} x {rollout_len} = {batch_size * rollout_len:,} samples/update/agent")
        print(f"LR: {lr} | γ: {gamma} | Ent: {entropy_coeff} | Val: {value_coeff}")
        print(f"Message dim: {msg_dim} | Attention-based aggregation")
        print(f"Opponent: IDLE (right team does nothing)")
        print(f"{'='*70}\n")

    def _weights(self):
        return {k: v.cpu().numpy() for k, v in self.model.state_dict().items()}

    def _broadcast(self):
        ray.get([w.set_weights.remote(self._weights()) for w in self.workers])

    def _prepare_agent_batch(self, rollouts: List[Dict], agent_idx: int):
        """Prepare batch for a single agent from multiple worker rollouts."""
        agent_rollouts = [r['rollouts'][agent_idx] for r in rollouts]
        B, T = len(agent_rollouts), self.rollout_len
        
        obs = torch.from_numpy(np.stack([r['obs'] for r in agent_rollouts])).float().to(self.device)
        feat = torch.from_numpy(np.stack([r['feat'] for r in agent_rollouts])).float().to(self.device)
        prev_act = torch.from_numpy(np.stack([r['prev_act'] for r in agent_rollouts])).long().to(self.device)
        act = torch.from_numpy(np.stack([r['act'] for r in agent_rollouts])).long().to(self.device)
        beh_lp = torch.from_numpy(np.stack([r['lp'] for r in agent_rollouts])).float().to(self.device)
        rew = torch.from_numpy(np.stack([r['rew'] for r in agent_rollouts])).float().to(self.device)
        done = torch.from_numpy(np.stack([r['done'] for r in agent_rollouts])).float().to(self.device)
        msg_in = torch.from_numpy(np.stack([r['msg_in'] for r in agent_rollouts])).float().to(self.device)
        bootstrap = torch.tensor([r['bootstrap'] for r in agent_rollouts], dtype=torch.float32, device=self.device)
        
        return obs, feat, prev_act, act, beh_lp, rew, done, msg_in, bootstrap, T

    def _update_agent(self, rollouts: List[Dict], agent_idx: int) -> Dict:
        """Update a single agent's policy with communication."""
        obs, feat, prev_act, act, beh_lp, rew, done, msg_in, bootstrap, T = self._prepare_agent_batch(rollouts, agent_idx)
        agent = self.model.get_agent(agent_idx)
        optimizer = self.optimizers[agent_idx]
        
        B = obs.shape[0]
        
        with autocast('cuda'):
            # Encode
            h, _ = agent.encode(obs, feat, prev_act)
            # Forward with messages
            logits, values_norm, _, _ = agent.forward_with_messages(h, msg_in, None)
            
            dist = Categorical(logits=logits)
            target_lp = dist.log_prob(act)
            entropy = dist.entropy()
            values = agent.value.denormalize(values_norm)
        
        with torch.no_grad():
            vs, adv, rhos = vtrace(beh_lp, target_lp.float().detach(), rew,
                                   values.float().detach(), bootstrap.float(), done, self.gamma)
            agent.value.update_stats(vs)
            vs_norm = agent.value.normalize_target(vs)
            mean_rho = rhos.mean().item()
        
        with autocast('cuda'):
            policy_loss = -(target_lp * adv.detach()).mean()
            value_loss = F.mse_loss(values_norm, vs_norm.detach())
            ent_loss = -entropy.mean()
            loss = policy_loss + self.value_coeff * value_loss + self.entropy_coeff * ent_loss
        
        optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(optimizer)
        grad_norm = sum(p.grad.norm(2).item() ** 2 for p in agent.parameters() if p.grad is not None) ** 0.5
        nn.utils.clip_grad_norm_(agent.parameters(), 40.0)
        self.scaler.step(optimizer)
        self.scaler.update()
        
        return {
            'loss': loss.item(), 'pi': policy_loss.item(), 'v': value_loss.item(),
            'ent': entropy.mean().item(), 'rho': mean_rho, 'grad': grad_norm,
            'adv_mean': adv.mean().item()
        }

    def train(self, max_time=3600, target_wr=95):
        print(f"Training for {max_time}s or {target_wr}% WR...\n")
        self.start = time.time()
        self._broadcast()
        
        for w in self.workers:
            self.pending[w.collect.remote()] = w
        
        while time.time() - self.start < max_time:
            while len(self.queue) < self.batch_size:
                done_refs, _ = ray.wait(list(self.pending.keys()), num_returns=1)
                for ref in done_refs:
                    w = self.pending.pop(ref)
                    r = ray.get(ref)
                    self.queue.append(r)
                    self.total_steps += self.rollout_len * NUM_AGENTS
                    self.max_rew = max(self.max_rew, r['max_rew'])
                    self.min_rew = min(self.min_rew, r['min_rew'])
                    for ep in r['episodes']:
                        self.returns.append(ep['return'])
                        self.wins.append(float(ep['won']))
                        self.lengths.append(ep['length'])
                    
                    w.set_weights.remote(self._weights())
                    self.pending[w.collect.remote()] = w
            
            batch = self.queue[:self.batch_size]
            self.queue = self.queue[self.batch_size:]
            
            # Update each agent
            all_stats = []
            for i in range(NUM_AGENTS):
                stats = self._update_agent(batch, i)
                all_stats.append(stats)
                self.agent_stats[i]['entropy'].append(stats['ent'])
                self.agent_stats[i]['grad'].append(stats['grad'])
            
            self.updates += 1
            
            if self.updates % 10 == 0:
                elapsed = time.time() - self.start
                sps = self.total_steps / elapsed
                wr = np.mean(self.wins) * 100 if self.wins else 0
                ret = np.mean(self.returns) if self.returns else 0
                ret_max = np.max(self.returns) if self.returns else 0
                
                # Aggregate stats
                avg_ent = np.mean([s['ent'] for s in all_stats])
                avg_grad = np.mean([s['grad'] for s in all_stats])
                avg_loss = np.mean([s['loss'] for s in all_stats])
                avg_pi = np.mean([s['pi'] for s in all_stats])
                avg_v = np.mean([s['v'] for s in all_stats])
                avg_adv = np.mean([s['adv_mean'] for s in all_stats])
                avg_rho = np.mean([s['rho'] for s in all_stats])
                
                # Per-role entropy
                ent_keeper = all_stats[0]['ent']
                ent_def = np.mean([all_stats[i]['ent'] for i in [1, 2, 3, 4]])
                ent_mid = np.mean([all_stats[i]['ent'] for i in [5, 6, 7]])
                ent_fwd = np.mean([all_stats[i]['ent'] for i in [8, 9, 10]])
                
                # Per-role grad
                grad_keeper = all_stats[0]['grad']
                grad_fwd = np.mean([all_stats[i]['grad'] for i in [8, 9, 10]])
                
                # Per-role adv
                adv_keeper = all_stats[0]['adv_mean']
                adv_fwd = np.mean([all_stats[i]['adv_mean'] for i in [8, 9, 10]])
                
                # PopArt stats (from agent 0 as reference)
                mu = self.model.get_agent(0).value.mu.item()
                sig = self.model.get_agent(0).value.sigma.item()
                
                print(f"[{self.updates:4d}] {self.total_steps/1e6:.1f}M {sps/1e3:.0f}k/s {elapsed/60:.0f}m | W:{wr:4.0f}% R:{ret:+.1f}({ret_max:+.0f}) rw:[{self.min_rew:.1f},{self.max_rew:.1f}] | L:{avg_loss:.2f} p:{avg_pi:+.2f} v:{avg_v:.2f} rho:{avg_rho:.1f} | H:{avg_ent:.2f}(K:{ent_keeper:.2f} D:{ent_def:.2f} M:{ent_mid:.2f} F:{ent_fwd:.2f}) | adv:{avg_adv:.3f}(K:{adv_keeper:.3f} F:{adv_fwd:.3f}) | grad:{avg_grad:.1f}(K:{grad_keeper:.1f} F:{grad_fwd:.1f}) | mu:{mu:.2f} sig:{sig:.2f} scale:{self.scaler.get_scale():.0f}")
            
            if self.updates % 100 == 0:
                self._save()
            
            wr = np.mean(self.wins) * 100 if self.wins else 0
            if wr >= target_wr and len(self.wins) >= 50:
                print(f"\n{wr:.0f}% WR reached in {(time.time()-self.start)/60:.1f}m!")
                self._save("winner")
                return True
        
        print(f"\nTime limit. Final WR: {np.mean(self.wins)*100:.0f}%")
        self._save("final")
        return False

    def _save(self, name=None):
        name = name or f"u{self.updates}"
        path = self.checkpoint_dir / f"ckpt_{name}.pt"
        torch.save({
            'model': self.model.state_dict(),
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'steps': self.total_steps,
            'updates': self.updates,
            'wr': np.mean(self.wins) if self.wins else 0
        }, path)
        print(f"  Saved {path}")

    def close(self):
        for ref in self.pending:
            try:
                ray.cancel(ref)
            except:
                pass
        for w in self.workers:
            try:
                ray.get(w.close.remote(), timeout=2)
            except:
                pass
        ray.shutdown()


if __name__ == "__main__":
    learner = Learner(
        num_workers=16,
        env_name="11_vs_11_stochastic",
        
        rollout_len=256,
        batch_size=32,
        lr=0.0005,
        gamma=0.997,
        entropy_coeff=0.01,
        value_coeff=0.5,
        
        d_model=128,
        lstm_hidden=128,
        msg_dim=32,  # PIANO communication
        
        checkpoint_dir="./checkpoints_11agent_comm",
    )
    
    try:
        learner.train(max_time=7200, target_wr=95)
    except KeyboardInterrupt:
        print("\nStopped!")
    finally:
        learner.close()