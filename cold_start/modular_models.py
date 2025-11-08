import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from gymnasium.spaces import Space

MODEL_SCALES = {
    's': 0.5,
    'm': 1.0,
    'l': 1.5,
    'xl': 2.0,
    'xxl': 3.0,
}

def scale_dim(base_dim: int, scale: float) -> int:
    """Scale a dimension by the given factor and round to nearest int."""
    return max(1, int(base_dim * scale))

def count_parameters(model: nn.Module, detailed: bool = False) -> int:
    """Count total parameters in model. If detailed=True, print breakdown."""
    total = 0
    if detailed:
        print("\nParameter Count Breakdown:")
        print("-" * 60)
    
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        if detailed:
            print(f"{name:45s} {n:10,}")
    
    if detailed:
        print("-" * 60)
        print(f"{'Total parameters:':45s} {total:10,}")
        print(f"{'Size in MB (float32):':45s} {total * 4 / 1024 / 1024:10.2f}")
    
    return total

class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size

        self.in_norm = nn.LayerNorm(in_dim)
        self.spline_weight = nn.Parameter(torch.randn(out_dim, in_dim, grid_size) * 0.1)
        self.scale_base = nn.Parameter(torch.ones(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.rbf_beta = nn.Parameter(torch.ones(1) * 2.0)
        self.register_buffer('grid', torch.linspace(-1, 1, grid_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_norm(x)
        grid = self.grid.to(x.dtype).to(x.device)
        beta = self.rbf_beta.clamp(0.5, 6.0)
        basis = torch.exp(-((x.unsqueeze(-1) - grid.view(1, 1, -1)) ** 2) * beta)
        spline = torch.einsum('big,oig->bo', basis, self.spline_weight)
        base = torch.einsum('bi,oi->bo', x, self.scale_base)
        return base + spline + self.bias


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, model_scale='m', **kwargs):
        super().__init__()
        scale = MODEL_SCALES.get(model_scale, 1.0)
        hidden_dim = scale_dim(hidden_dim, scale)

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.norm(x)


class CNNEncoder(nn.Module):
    """Fixed CNN encoder using adaptive pooling to avoid parameter explosion."""
    def __init__(self, input_dim, output_dim, hidden_dim=128,
                 cnn_channels=None, cnn_kernels=None, model_scale='m', **kwargs):
        super().__init__()
        scale = MODEL_SCALES.get(model_scale, 1.0)

        if cnn_channels is None:
            base_channels = [32, 64]
            cnn_channels = [scale_dim(c, scale) for c in base_channels]
        if cnn_kernels is None:
            cnn_kernels = [8, 4]

        layers = []
        in_channels = 1
        for channels, kernel in zip(cnn_channels, cnn_kernels):
            layers.append(nn.Conv1d(in_channels, channels, kernel, padding=kernel // 2))
            layers.append(nn.ReLU())
            in_channels = channels
        self.conv = nn.Sequential(*layers)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        final_channels = cnn_channels[-1] if cnn_channels else 64
        self.fc = nn.Linear(final_channels, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        B = x.shape[0]
        x = x.view(B, 1, -1)
        x = self.conv(x)
        x = self.adaptive_pool(x)
        x = x.view(B, -1)
        x = self.fc(x)
        return self.norm(x)


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256, gnn_hidden=24, gnn_layers=2,
                 gnn_k_neighbors=6, gnn_dropout=0.1, gnn_type='sage', **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, gnn_hidden)
        self.gnn_layers = nn.ModuleList()
        for _ in range(gnn_layers):
            self.gnn_layers.append(nn.Linear(gnn_hidden, gnn_hidden))

        self.dropout = nn.Dropout(gnn_dropout)
        self.output_proj = nn.Linear(gnn_hidden, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = F.relu(self.input_proj(x))
        for layer in self.gnn_layers:
            identity = x
            x = F.relu(layer(x))
            x = self.dropout(x)
            x = x + identity
        x = self.output_proj(x)
        return self.norm(x)


class ResMLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim, width=512, depth=3, dropout=0.1, **kwargs):
        super().__init__()
        layers = [nn.Linear(input_dim, width), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout)]
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(width, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        x = self.mlp(x)
        x = self.proj(x)
        return self.norm(x)


class GRUSequence(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, **kwargs):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B = x.size(0)

        if hidden is None:
            hidden = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        else:
            if hidden.dim() == 2:
                if hidden.shape[0] == 1:
                    hidden = hidden.expand(B, -1)
                hidden = hidden.reshape(self.num_layers, B, self.hidden_dim)

        out, hidden = self.gru(x, hidden)
        return out[:, -1, :], hidden

    def get_initial_state(self, batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


class LSTMSequence(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, **kwargs):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B = x.size(0)

        if hidden is None:
            h = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
            c = torch.zeros(self.num_layers, B, self.hidden_dim, device=x.device)
        else:
            h, c = hidden
            if h.dim() == 2 and h.shape[0] == 1:
                h = h.expand(B, -1).reshape(self.num_layers, B, self.hidden_dim)
                c = c.expand(B, -1).reshape(self.num_layers, B, self.hidden_dim)

        out, (h, c) = self.lstm(x, (h, c))
        return out[:, -1, :], (h, c)

    def get_initial_state(self, batch_size: int, device: torch.device):
        h = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h, c)


class TCNSequence(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2,
                 tcn_channels=None, tcn_kernel_size=3, **kwargs):
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [48, 48]

        self.input_proj = nn.Linear(input_dim, tcn_channels[0])
        self.tcn_layers = nn.ModuleList()

        for i in range(len(tcn_channels)):
            in_ch = tcn_channels[i - 1] if i > 0 else tcn_channels[0]
            out_ch = tcn_channels[i]
            dilation = 2 ** i
            padding = (tcn_kernel_size - 1) * dilation // 2
            self.tcn_layers.append(
                nn.Conv1d(in_ch, out_ch, tcn_kernel_size,
                          padding=padding, dilation=dilation)
            )

        self.output_proj = nn.Linear(tcn_channels[-1], hidden_dim)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, T, D = x.shape
        x = self.input_proj(x)
        x = x.transpose(1, 2)

        for layer in self.tcn_layers:
            identity = x
            x = F.relu(layer(x))
            if x.shape == identity.shape:
                x = x + identity

        x = x.transpose(1, 2)
        x = self.output_proj(x)
        return x.squeeze(1), None

    def get_initial_state(self, batch_size: int, device: torch.device):
        return None


class MambaSequence(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2,
                 mamba_state_dim=6, **kwargs):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.state_dim = mamba_state_dim
        self.num_layers = num_layers

        self.mamba_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.mamba_layers.append(self._create_mamba_layer(hidden_dim, mamba_state_dim))

    def _create_mamba_layer(self, d_model, d_state):
        layer = nn.Module()
        layer.norm = nn.LayerNorm(d_model)
        layer.in_proj = nn.Linear(d_model, d_model * 2)
        layer.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3,
                                 padding=1, groups=d_model)
        layer.dt_proj = nn.Linear(d_model, d_model)
        layer.A_log = nn.Parameter(torch.randn(d_model, d_state))
        layer.B_proj = nn.Linear(d_model, d_state, bias=False)
        layer.C_proj = nn.Linear(d_model, d_state, bias=False)
        layer.D = nn.Parameter(torch.ones(d_model))
        layer.out_proj = nn.Linear(d_model, d_model)
        return layer

    def forward(self, x, hidden=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.input_proj(x)

        for layer in self.mamba_layers:
            identity = x
            x = layer.norm(x)
            x_proj = layer.in_proj(x)
            x_conv, z = x_proj.chunk(2, dim=-1)

            B, T, D = x_conv.shape
            x_conv = x_conv.transpose(1, 2)
            x_conv = layer.conv1d(x_conv)
            x_conv = x_conv.transpose(1, 2)

            x_conv = F.silu(x_conv)
            dt = F.softplus(layer.dt_proj(x_conv))  # currently unused
            y = x_conv * layer.D

            out = layer.out_proj(y * F.silu(z))
            x = identity + out

        return x.squeeze(1), None

    def get_initial_state(self, batch_size: int, device: torch.device):
        return None


class MLPHead(nn.Module):
    def __init__(self, input_dim, output_dim,
                 policy_hidden_dims=None,
                 value_hidden_dims=None,
                 head_dropout=0.1,
                 **kwargs):
        super().__init__()
        if policy_hidden_dims is None:
            policy_hidden_dims = [32, 16]
        if value_hidden_dims is None:
            value_hidden_dims = [24, 12]

        policy_layers = []
        prev_dim = input_dim
        for hidden_dim in policy_hidden_dims:
            policy_layers.append(nn.Linear(prev_dim, hidden_dim))
            policy_layers.append(nn.ReLU())
            policy_layers.append(nn.Dropout(head_dropout))
            prev_dim = hidden_dim
        policy_layers.append(nn.Linear(prev_dim, output_dim))
        self.policy_head = nn.Sequential(*policy_layers)

        value_layers = []
        prev_dim = input_dim
        for hidden_dim in value_hidden_dims:
            value_layers.append(nn.Linear(prev_dim, hidden_dim))
            value_layers.append(nn.ReLU())
            value_layers.append(nn.Dropout(head_dropout))
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_head = nn.Sequential(*value_layers)

    def forward(self, x):
        return self.policy_head(x), self.value_head(x)



class KANHead(nn.Module):
    def __init__(self, input_dim, output_dim,
                 policy_hidden_dims=None,
                 value_hidden_dims=None,
                 head_dropout=0.1,
                 kan_grid=3,
                 **kwargs):
        super().__init__()
        if policy_hidden_dims is None:
            policy_hidden_dims = [32, 16]
        if value_hidden_dims is None:
            value_hidden_dims = [24, 12]

        # Policy / logits stream
        policy_layers = []
        prev_dim = input_dim
        for hidden_dim in policy_hidden_dims:
            policy_layers.append(KANLayer(prev_dim, hidden_dim, grid_size=kan_grid))
            policy_layers.append(nn.ReLU())
            policy_layers.append(nn.Dropout(head_dropout))
            prev_dim = hidden_dim
        policy_layers.append(nn.Linear(prev_dim, output_dim))
        self.policy_head = nn.Sequential(*policy_layers)

        # Value stream
        value_layers = []
        prev_dim = input_dim
        for hidden_dim in value_hidden_dims:
            value_layers.append(KANLayer(prev_dim, hidden_dim, grid_size=kan_grid))
            value_layers.append(nn.ReLU())
            value_layers.append(nn.Dropout(head_dropout))
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_head = nn.Sequential(*value_layers)

    def forward(self, x):
        return self.policy_head(x), self.value_head(x)


class DuelingMLPHead(nn.Module):
    """
    Dueling Network Head:
    Q(s, a) = V(s) + (A(s, a) - mean_A)
    Uses 'policy_hidden_dims' for the advantage stream
    and 'value_hidden_dims' for the value stream.
    """
    def __init__(self, input_dim, output_dim,
                 policy_hidden_dims=None,
                 value_hidden_dims=None,
                 head_dropout=0.1,
                 **kwargs):
        super().__init__()
        if policy_hidden_dims is None:
            policy_hidden_dims = [32, 16]
        if value_hidden_dims is None:
            value_hidden_dims = [24, 12]

        # Advantage stream
        advantage_layers = []
        prev_dim = input_dim
        for hidden_dim in policy_hidden_dims:
            advantage_layers.append(nn.Linear(prev_dim, hidden_dim))
            advantage_layers.append(nn.ReLU())
            advantage_layers.append(nn.Dropout(head_dropout))
            prev_dim = hidden_dim
        advantage_layers.append(nn.Linear(prev_dim, output_dim))
        self.advantage_head = nn.Sequential(*advantage_layers)

        # Value stream
        value_layers = []
        prev_dim = input_dim
        for hidden_dim in value_hidden_dims:
            value_layers.append(nn.Linear(prev_dim, hidden_dim))
            value_layers.append(nn.ReLU())
            value_layers.append(nn.Dropout(head_dropout))
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_head = nn.Sequential(*value_layers)

    def forward(self, x):
        advantages = self.advantage_head(x)
        value = self.value_head(x)
        adv_mean = advantages.mean(dim=-1, keepdim=True)
        logits = value + advantages - adv_mean
        return logits, value


class MoELayer(nn.Module):
    """
    Simple dense Mixture-of-Experts layer.
    Weights outputs of several expert MLPs via a gating network.
    """
    def __init__(self, input_dim: int, output_dim: int,
                 num_experts: int, expert_hidden_dim: int):
        super().__init__()
        self.num_experts = num_experts

        # Gating network produces weights over experts
        self.gating_net = nn.Linear(input_dim, num_experts)

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, expert_hidden_dim),
                nn.ReLU(),
                nn.Linear(expert_hidden_dim, output_dim),
            )
            for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate weights: [B, num_experts]
        gate_weights = F.softmax(self.gating_net(x), dim=-1)

        # Expert outputs: [B, output_dim, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)

        # Expand gate weights: [B, 1, num_experts]
        gate_weights = gate_weights.unsqueeze(1)

        # Weighted sum over experts -> [B, output_dim]
        weighted_sum = (expert_outputs * gate_weights).sum(dim=2)
        return weighted_sum


class MoEHead(nn.Module):
    """
    Head using MoELayer instead of plain Linear for hidden layers.
    Requires 'moe_num_experts' and 'moe_expert_hidden_dim' in config (via kwargs),
    otherwise uses defaults.
    """
    def __init__(self, input_dim, output_dim,
                 policy_hidden_dims=None,
                 value_hidden_dims=None,
                 head_dropout=0.1,
                 **kwargs):
        super().__init__()

        num_experts = kwargs.get('moe_num_experts', 4)
        expert_hidden_dim = kwargs.get('moe_expert_hidden_dim', 32)

        if policy_hidden_dims is None:
            policy_hidden_dims = [32, 16]
        if value_hidden_dims is None:
            value_hidden_dims = [24, 12]

        # Policy / logits stream with MoE layers
        policy_layers = []
        prev_dim = input_dim
        for hidden_dim in policy_hidden_dims:
            policy_layers.append(MoELayer(prev_dim, hidden_dim,
                                          num_experts, expert_hidden_dim))
            policy_layers.append(nn.ReLU())
            policy_layers.append(nn.Dropout(head_dropout))
            prev_dim = hidden_dim
        policy_layers.append(nn.Linear(prev_dim, output_dim))
        self.policy_head = nn.Sequential(*policy_layers)

        # Value stream with MoE layers
        value_layers = []
        prev_dim = input_dim
        for hidden_dim in value_hidden_dims:
            value_layers.append(MoELayer(prev_dim, hidden_dim,
                                         num_experts, expert_hidden_dim))
            value_layers.append(nn.ReLU())
            value_layers.append(nn.Dropout(head_dropout))
            prev_dim = hidden_dim
        value_layers.append(nn.Linear(prev_dim, 1))
        self.value_head = nn.Sequential(*value_layers)

    def forward(self, x):
        return self.policy_head(x), self.value_head(x)


class ModularStudentModel(nn.Module):
    def __init__(self,
                 obs_space: Space,
                 action_space: Space,
                 num_outputs: int,
                 config: Dict):
        super().__init__()

        self.config = config
        self.num_frames = config.get('num_frames', 4)
        self.stack_frames = config.get('stack_frames', 4)

        total_obs_dim = int(np.prod(obs_space.shape))
        input_dim = total_obs_dim  # kept simple; can customize for stacking

        encoder_output_dim = config.get('encoder_output_dim', 48)

        # Encoder selection
        if config['encoder_type'] == 'linear':
            self.encoder = LinearEncoder(
                input_dim,
                encoder_output_dim,
                hidden_dim=config.get('encoder_hidden_dim', 256),
                **config
            )
        elif config['encoder_type'] == 'cnn':
            self.encoder = CNNEncoder(input_dim, encoder_output_dim, **config)
        elif config['encoder_type'] == 'gnn':
            self.encoder = GNNEncoder(input_dim, encoder_output_dim, **config)
        elif config['encoder_type'] == 'resmlp':
            self.encoder = ResMLPEncoder(input_dim, encoder_output_dim, **config)
        else:
            raise ValueError(f"Unknown encoder_type: {config['encoder_type']}")

        # Sequence module
        sequence_hidden_dim = config.get('sequence_hidden_dim', 48)
        prev_action_dim = config.get('prev_action_emb_dim', 8)
        sequence_input_dim = encoder_output_dim + prev_action_dim

        if config['sequence_type'] == 'gru':
            self.sequence = GRUSequence(sequence_input_dim, sequence_hidden_dim, **config)
        elif config['sequence_type'] == 'lstm':
            self.sequence = LSTMSequence(sequence_input_dim, sequence_hidden_dim, **config)
        elif config['sequence_type'] == 'tcn':
            self.sequence = TCNSequence(sequence_input_dim, sequence_hidden_dim, **config)
        elif config['sequence_type'] == 'mamba':
            self.sequence = MambaSequence(sequence_input_dim, sequence_hidden_dim, **config)
        else:
            raise ValueError(f"Unknown sequence_type: {config['sequence_type']}")

        # Head selection
        if config['head_type'] == 'mlp':
            self.head = MLPHead(sequence_hidden_dim, num_outputs, **config)
        elif config['head_type'] == 'kan':
            self.head = KANHead(sequence_hidden_dim, num_outputs, **config)
        elif config['head_type'] == 'dueling_mlp':
            self.head = DuelingMLPHead(sequence_hidden_dim, num_outputs, **config)
        elif config['head_type'] == 'moe':
            self.head = MoEHead(sequence_hidden_dim, num_outputs, **config)
        else:
            raise ValueError(f"Unknown head_type: {config['head_type']}")

        self.prev_action_embed = nn.Embedding(num_outputs + 1, prev_action_dim)
        self._value_out = None

    def forward(self, input_dict: Dict, state: List, seq_lens):
        # Expect obs_flat or obs from RLlib-style input_dict
        obs = input_dict.get('obs_flat', input_dict.get('obs'))
        if obs is None:
            raise ValueError("Input dict must contain 'obs' or 'obs_flat'.")

        if 'prev_actions' in input_dict:
            prev_actions = input_dict['prev_actions']
        else:
            prev_actions = torch.zeros(
                obs.size(0),
                dtype=torch.long,
                device=obs.device
            )

        encoded = self.encoder(obs)
        pa_emb = self.prev_action_embed(prev_actions)
        seq_in = torch.cat([encoded, pa_emb], dim=-1)

        hidden_in = state[0] if state else None
        seq_out, hidden_out = self.sequence(seq_in, hidden_in)

        logits, value = self.head(seq_out)
        self._value_out = value.squeeze(-1)

        new_state = [hidden_out] if hidden_out is not None else []
        return logits, new_state

    def value_function(self):
        return self._value_out

    def get_initial_state(self, batch_size: int, device: torch.device):
        if hasattr(self.sequence, 'get_initial_state'):
            state = self.sequence.get_initial_state(batch_size, device)
            return [state] if state is not None else [None]
        return [None]
