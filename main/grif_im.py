"""
Griffin Policy for Google Research Football
Optimized for set pieces (throw-ins, offsides, free kicks, etc.)

Key improvements over LSTM:
1. Local Attention can retrieve relevant context for rare game modes
2. RG-LRU has more stable gradients than LSTM forget gate
3. Auxiliary game mode prediction loss
4. Explicit game mode embedding
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict

# GFootball constants
OBS_DIM = 115
FEATURE_DIM = 93
NUM_ACTIONS = 19

# Game modes from GFootball
GAME_MODES = {
    0: "Normal",
    1: "KickOff", 
    2: "GoalKick",
    3: "FreeKick",
    4: "Corner",
    5: "ThrowIn",
    6: "Penalty",
}
NUM_GAME_MODES = 7


# =============================================================================
# Griffin Components (same as before, but with some optimizations)
# =============================================================================

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RGLRU(nn.Module):
    """
    Real-Gated Linear Recurrent Unit with improved initialization
    for handling rare events (set pieces).
    """
    def __init__(self, d_rnn: int, c: float = 8.0):
        super().__init__()
        self.d_rnn = d_rnn
        self.c = c
        
        self.W_r = nn.Linear(d_rnn, d_rnn, bias=True)
        self.W_i = nn.Linear(d_rnn, d_rnn, bias=True)
        self.W_a = nn.Linear(d_rnn, d_rnn, bias=True)
        
        self._init_weights()
    
    def _init_weights(self):
        # Balanced initialization for rare event handling
        nn.init.zeros_(self.W_r.bias)
        nn.init.zeros_(self.W_i.bias)
        # Slightly higher decay to preserve more history
        nn.init.uniform_(self.W_a.bias, -3.0, -1.5)
        
        for w in [self.W_r, self.W_i, self.W_a]:
            nn.init.xavier_uniform_(w.weight, gain=0.5)

    def forward(
        self, 
        x: torch.Tensor, 
        h: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        
        if h is None:
            h = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        
        r = torch.sigmoid(self.W_r(x))
        i = torch.sigmoid(self.W_i(x))
        a = torch.sigmoid(self.W_a(x)) ** self.c
        norm = torch.sqrt(torch.clamp(1 - a ** 2, min=1e-6))
        
        outputs = []
        for t in range(T):
            h = a[:, t] * h + norm[:, t] * (r[:, t] * x[:, t])
            outputs.append(i[:, t] * h)
        
        return torch.stack(outputs, dim=1), h


class RecurrentBlock(nn.Module):
    """Recurrent block with Conv1D + RG-LRU."""
    def __init__(self, d_model: int, d_rnn: int = None, conv_kernel: int = 4):
        super().__init__()
        self.d_model = d_model
        self.d_rnn = d_rnn or int(d_model * 4 / 3)
        
        self.linear_x = nn.Linear(d_model, self.d_rnn, bias=False)
        self.linear_y = nn.Linear(d_model, self.d_rnn, bias=False)
        
        self.conv = nn.Conv1d(
            self.d_rnn, self.d_rnn, 
            kernel_size=conv_kernel, 
            padding=conv_kernel - 1,
            groups=self.d_rnn
        )
        self.rglru = RGLRU(self.d_rnn)
        self.linear_out = nn.Linear(self.d_rnn, d_model, bias=False)
    
    def forward(self, x: torch.Tensor, h: Optional[torch.Tensor] = None):
        B, T, _ = x.shape
        
        x_branch = self.linear_x(x)
        x_conv = self.conv(x_branch.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_rnn, h_new = self.rglru(x_conv, h)
        
        y_branch = F.gelu(self.linear_y(x))
        out = self.linear_out(x_rnn * y_branch)
        
        return out, h_new


class LocalMQA(nn.Module):
    """Local Multi-Query Attention with larger window for set pieces."""
    def __init__(self, d_model: int, n_heads: int = 8, window_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window_size = window_size
        
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, self.head_dim, bias=False)
        self.W_v = nn.Linear(d_model, self.head_dim, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        
        q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, 1, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal_mask, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        
        return self.W_o(out)


class GatedMLP(nn.Module):
    def __init__(self, d_model: int, expand_factor: float = 3.0):
        super().__init__()
        d_ff = int(d_model * expand_factor)
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.gate(x)) * self.up(x))


# =============================================================================
# Griffin Backbone with Game Mode Conditioning
# =============================================================================

class GriffinBackbone(nn.Module):
    """
    Griffin with explicit game mode conditioning.
    
    The idea: inject game mode information at multiple points so the network
    can learn mode-specific behaviors (different strategies for throw-ins vs normal play).
    """
    def __init__(
        self, 
        d_model: int = 256,
        d_rnn: int = None,
        n_layers: int = 6,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_rnn = d_rnn or int(d_model * 4 / 3)
        self.n_layers = n_layers
        
        # Game mode embedding - added to residual stream
        self.game_mode_emb = nn.Embedding(NUM_GAME_MODES, d_model)
        
        # Build layers with 2:1 RNN:Attention ratio
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.layer_types = []
        
        for i in range(n_layers):
            self.norms.append(RMSNorm(d_model))
            
            if (i + 1) % 3 == 0:
                self.layers.append(LocalMQA(d_model, n_heads))
                self.layer_types.append('attention')
            else:
                self.layers.append(RecurrentBlock(d_model, self.d_rnn))
                self.layer_types.append('recurrent')
        
        self.mlp_norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.mlps = nn.ModuleList([GatedMLP(d_model) for _ in range(n_layers)])
        
        self.final_norm = RMSNorm(d_model)
        self.n_recurrent = sum(1 for t in self.layer_types if t == 'recurrent')
    
    def forward(
        self, 
        x: torch.Tensor,
        game_mode: torch.Tensor,
        h_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            x: (B, T, d_model) - encoded input
            game_mode: (B, T) - game mode indices
            h_states: list of hidden states for recurrent layers
        """
        if h_states is None:
            h_states = [None] * self.n_recurrent
        
        # Add game mode embedding to input
        mode_emb = self.game_mode_emb(game_mode)  # (B, T, d_model)
        x = x + mode_emb
        
        h_new = []
        h_idx = 0
        
        for i, (layer, layer_type) in enumerate(zip(self.layers, self.layer_types)):
            # Temporal mixing
            normed = self.norms[i](x)
            if layer_type == 'recurrent':
                out, h = layer(normed, h_states[h_idx])
                h_new.append(h)
                h_idx += 1
            else:
                out = layer(normed)
            x = x + out
            
            # Channel mixing (MLP)
            x = x + self.mlps[i](self.mlp_norms[i](x))
        
        return self.final_norm(x), h_new


# =============================================================================
# Full Policy Network
# =============================================================================

class GriffinPolicy(nn.Module):
    """
    Griffin-based policy for GFootball with:
    1. Explicit game mode conditioning
    2. Auxiliary game mode prediction (helps learn mode-specific features)
    3. Dual value heads (sparse reward + dense reward)
    4. Offside-aware features
    """
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input processing
        self.action_emb = nn.Embedding(NUM_ACTIONS + 1, 32)
        self.game_mode_input_emb = nn.Embedding(NUM_GAME_MODES, 32)
        
        # Separate embedding for offside situation
        self.offside_emb = nn.Linear(3, 32)  # offside_line, is_offside, ball_x
        
        input_dim = OBS_DIM + FEATURE_DIM + 32 + 32 + 32  # obs + feat + action + mode + offside
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            RMSNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        
        # Griffin backbone
        self.griffin = GriffinBackbone(d_model, n_layers=n_layers, n_heads=n_heads)
        
        # Policy head
        self.policy = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS)
        )
        
        # Value heads (dual: sparse + dense rewards)
        self.value_sparse = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.value_dense = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Auxiliary: predict game mode from features
        # This forces the network to learn mode-discriminative representations
        self.aux_game_mode = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, NUM_GAME_MODES)
        )
        
        self._init_heads()
        
        n_params = sum(p.numel() for p in self.parameters())
        print(f"GriffinPolicy: {n_params:,} params")

    def _init_heads(self):
        # Small init for policy (exploration)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy[-1].bias)
        
        # Standard init for value heads
        for head in [self.value_sparse, self.value_dense]:
            nn.init.orthogonal_(head[-1].weight, gain=1.0)
            nn.init.zeros_(head[-1].bias)

    def _extract_game_mode(self, feat: torch.Tensor) -> torch.Tensor:
        """Extract game mode index from feature vector."""
        # feat[78:85] contains one-hot game mode
        game_mode_onehot = feat[..., 78:85]
        # Add small epsilon to avoid argmax ties
        return game_mode_onehot.argmax(dim=-1)
    
    def _extract_offside_features(self, feat: torch.Tensor) -> torch.Tensor:
        """Extract offside-related features."""
        # feat[67] = offside_line, feat[68] = is_offside, feat[0] = ball_x
        offside_line = feat[..., 67:68]
        is_offside = feat[..., 68:69]
        ball_x = feat[..., 0:1]
        return torch.cat([offside_line, is_offside, ball_x], dim=-1)

    def forward(
        self,
        obs: torch.Tensor,
        feat: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        h_states: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            obs: (B,) or (B, T, obs_dim)
            feat: (B,) or (B, T, feat_dim)
            prev_action: (B,) or (B, T)
            h_states: hidden states for recurrent layers
        
        Returns:
            dict with: logits, value_sparse, value_dense, aux_game_mode, h_states
        """
        # Handle both (B, dim) and (B, T, dim) inputs
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            feat = feat.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        B, T, _ = obs.shape
        device = obs.device
        
        if prev_action is None:
            prev_action = torch.full((B, T), NUM_ACTIONS, dtype=torch.long, device=device)
        elif prev_action.dim() == 1:
            prev_action = prev_action.unsqueeze(1)
        
        # Extract game mode and offside features
        game_mode = self._extract_game_mode(feat)
        offside_feat = self._extract_offside_features(feat)
        
        # Encode inputs
        action_enc = self.action_emb(prev_action)
        mode_enc = self.game_mode_input_emb(game_mode)
        offside_enc = self.offside_emb(offside_feat)
        
        x = torch.cat([obs, feat, action_enc, mode_enc, offside_enc], dim=-1)
        x = self.input_proj(x)
        
        # Griffin forward
        x, h_new = self.griffin(x, game_mode, h_states)
        
        # Heads
        logits = self.policy(x)
        value_s = self.value_sparse(x)
        value_d = self.value_dense(x)
        aux_mode = self.aux_game_mode(x)
        
        if squeeze_output:
            logits = logits.squeeze(1)
            value_s = value_s.squeeze(1)
            value_d = value_d.squeeze(1)
            aux_mode = aux_mode.squeeze(1)
        
        return {
            'logits': logits,
            'value_sparse': value_s,
            'value_dense': value_d,
            'aux_game_mode': aux_mode,
            'h_states': h_new,
        }
    
    def get_action(
        self,
        obs: torch.Tensor,
        feat: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        h_states: Optional[List[torch.Tensor]] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Sample action for environment interaction.
        
        Returns:
            action: (B,) sampled or argmax action
            log_prob: (B,) log probability of action
            h_states: updated hidden states
        """
        out = self.forward(obs, feat, prev_action, h_states)
        logits = out['logits']
        
        if deterministic:
            action = logits.argmax(dim=-1)
            log_prob = F.log_softmax(logits, dim=-1).gather(-1, action.unsqueeze(-1)).squeeze(-1)
        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        
        return action, log_prob, out['h_states']
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        feat: torch.Tensor,
        actions: torch.Tensor,
        prev_action: Optional[torch.Tensor] = None,
        h_states: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Evaluate actions for PPO/IMPALA loss computation.
        
        Returns:
            dict with: log_prob, entropy, value_sparse, value_dense, aux_game_mode_logits
        """
        out = self.forward(obs, feat, prev_action, h_states)
        logits = out['logits']
        
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return {
            'log_prob': log_prob,
            'entropy': entropy,
            'value_sparse': out['value_sparse'].squeeze(-1),
            'value_dense': out['value_dense'].squeeze(-1),
            'aux_game_mode_logits': out['aux_game_mode'],
            'h_states': out['h_states'],
        }


# =============================================================================
# Loss computation with auxiliary game mode prediction
# =============================================================================

def compute_griffin_loss(
    policy: GriffinPolicy,
    batch: Dict[str, torch.Tensor],
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    aux_coef: float = 0.1,  # weight for game mode prediction loss
) -> Dict[str, torch.Tensor]:
    """
    Compute PPO-style loss with auxiliary game mode prediction.
    
    The auxiliary loss forces the network to learn features that distinguish
    between game modes, which helps with set piece handling.
    """
    obs = batch['obs']
    feat = batch['feat']
    actions = batch['action']
    old_log_prob = batch['old_log_prob']
    advantages = batch['advantages']
    returns_sparse = batch['returns_sparse']
    returns_dense = batch['returns_dense']
    game_modes = batch['game_mode']  # ground truth game modes
    
    # Forward pass
    eval_out = policy.evaluate_actions(obs, feat, actions)
    
    # Policy loss (PPO clipped)
    ratio = torch.exp(eval_out['log_prob'] - old_log_prob)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages
    policy_loss = -torch.min(surr1, surr2).mean()
    
    # Value losses (dual heads)
    value_loss_sparse = F.mse_loss(eval_out['value_sparse'], returns_sparse)
    value_loss_dense = F.mse_loss(eval_out['value_dense'], returns_dense)
    value_loss = value_loss_sparse + value_loss_dense
    
    # Entropy bonus
    entropy_loss = -eval_out['entropy'].mean()
    
    # Auxiliary game mode prediction loss
    aux_loss = F.cross_entropy(eval_out['aux_game_mode_logits'], game_modes)
    
    # Total loss
    total_loss = (
        policy_loss 
        + value_coef * value_loss 
        + entropy_coef * entropy_loss
        + aux_coef * aux_loss
    )
    
    return {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'value_loss_sparse': value_loss_sparse,
        'value_loss_dense': value_loss_dense,
        'entropy': -entropy_loss,
        'aux_game_mode_loss': aux_loss,
        'clip_fraction': ((ratio - 1).abs() > clip_eps).float().mean(),
    }


# =============================================================================
# Example usage
# =============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    # Create policy
    policy = GriffinPolicy(
        d_model=256,
        n_layers=6,
        n_heads=8,
    ).to(device)
    
    # Test forward pass
    B, T = 4, 16
    obs = torch.randn(B, T, OBS_DIM, device=device)
    feat = torch.randn(B, T, FEATURE_DIM, device=device)
    
    # Set some game mode values in feat for testing
    feat[..., 78:85] = 0
    feat[0, :, 78] = 1  # Normal
    feat[1, :, 83] = 1  # ThrowIn
    feat[2, :, 81] = 1  # FreeKick
    feat[3, :, 82] = 1  # Corner
    
    out = policy(obs, feat)
    
    print("Output shapes:")
    print(f"  logits: {out['logits'].shape}")
    print(f"  value_sparse: {out['value_sparse'].shape}")
    print(f"  value_dense: {out['value_dense'].shape}")
    print(f"  aux_game_mode: {out['aux_game_mode'].shape}")
    print(f"  h_states: {len(out['h_states'])} tensors")
    
    # Test action sampling
    action, log_prob, h_new = policy.get_action(
        obs[:, 0], feat[:, 0], deterministic=False
    )
    print(f"\nSampled actions: {action}")
    print(f"Log probs: {log_prob}")
    
    # Test with sequential steps (simulating rollout)
    print("\nSequential rollout test:")
    h_states = None
    for t in range(5):
        action, log_prob, h_states = policy.get_action(
            obs[:, t], feat[:, t], 
            prev_action=action if t > 0 else None,
            h_states=h_states
        )
        print(f"  Step {t}: action={action.tolist()}")