"""Neural network architecture for IMPALA-based football agent."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from feature_engineer import OBS_DIM, FEATURE_DIM

NUM_ACTIONS = 19


class PopArtValueHead(nn.Module):
    """Adaptive value head using PopArt normalization.

    Maintains running statistics of value targets and rescales the output
    layer weights whenever the statistics are updated, keeping the
    represented value function stable across non-stationary targets.

    Args:
        input_dim: Dimensionality of the input feature vector.
        beta: Exponential moving average decay rate for statistics.
    """

    def __init__(self, input_dim: int, beta: float = 1e-3):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(input_dim, 1)
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))
        self.register_buffer("nu", torch.ones(1))
        nn.init.orthogonal_(self.linear.weight, gain=1.0)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized value prediction.

        Args:
            x: Input tensor of shape ``(B, input_dim)`` or ``(B, T, input_dim)``.

        Returns:
            Normalized value predictions with the last dimension squeezed.
        """
        return self.linear(x).squeeze(-1)

    def denormalize(self, normalized: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to the original scale.

        Args:
            normalized: Normalized value predictions.

        Returns:
            Denormalized values.
        """
        return normalized * self.sigma + self.mu

    def normalize_target(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize raw value targets for loss computation.

        Args:
            targets: Raw value targets.

        Returns:
            Normalized targets.
        """
        return (targets - self.mu) / self.sigma

    @torch.no_grad()
    def update_stats(self, targets: torch.Tensor) -> None:
        """Update running statistics and correct output layer weights.

        Uses an exponential moving average to track the mean and variance of
        incoming targets, then rescales the linear layer so that its output
        remains consistent with the updated normalization.

        Args:
            targets: Batch of raw value targets.
        """
        old_mu, old_sigma = self.mu.clone(), self.sigma.clone()
        t_mean, t_sq_mean = targets.mean(), (targets**2).mean()
        self.mu.mul_(1 - self.beta).add_(self.beta * t_mean)
        self.nu.mul_(1 - self.beta).add_(self.beta * t_sq_mean)
        var = torch.clamp(self.nu - self.mu**2, min=1e-4)
        self.sigma.copy_(torch.sqrt(var))
        self.linear.weight.data.mul_(old_sigma / self.sigma)
        self.linear.bias.data.copy_(
            (self.linear.bias.data * old_sigma + old_mu - self.mu) / self.sigma
        )


class Net(nn.Module):
    """Encoder-LSTM-policy/value network for IMPALA.

    Architecture:
        ``[obs | feat | action_emb] -> 2-layer MLP -> LSTM -> policy + value``

    Args:
        d_model: Hidden dimension of the MLP encoder.
        lstm_hidden: Hidden dimension of the LSTM.
    """

    def __init__(self, d_model: int = 128, lstm_hidden: int = 128):
        super().__init__()
        self.d_model = d_model
        self.lstm_hidden = lstm_hidden
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
        self.lstm = nn.LSTM(d_model, lstm_hidden, num_layers=1, batch_first=True)
        self.policy = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Linear(128, NUM_ACTIONS),
        )
        self.value = PopArtValueHead(lstm_hidden)
        self._init()
        print(f"Net: {sum(p.numel() for p in self.parameters()):,} params")

    def _init(self) -> None:
        """Apply orthogonal initialization to all linear layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.policy[-1].weight, gain=0.01)

    def init_hidden(
        self, batch_size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create zero-initialized LSTM hidden state.

        Args:
            batch_size: Number of sequences in the batch.
            device: Device to place the tensors on.

        Returns:
            Tuple of ``(h_0, c_0)`` each with shape ``(1, batch_size, lstm_hidden)``.
        """
        return (
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
            torch.zeros(1, batch_size, self.lstm_hidden, device=device),
        )

    def forward(
        self,
        obs: torch.Tensor,
        feat: torch.Tensor,
        prev_actions: torch.Tensor = None,
        hidden: tuple[torch.Tensor, torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the full network.

        Supports both single-step ``(B, dim)`` and sequential ``(B, T, dim)``
        inputs.  Single-step inputs are automatically unsqueezed/squeezed.

        Args:
            obs: Raw observations.
            feat: Engineered features.
            prev_actions: Previous actions for embedding lookup. Defaults to
                a padding index when ``None``.
            hidden: LSTM hidden state ``(h, c)``. Initialized to zeros when
                ``None``.

        Returns:
            Tuple of ``(logits, normalized_values, hidden_state)``.
        """
        squeeze = obs.dim() == 2
        if squeeze:
            obs, feat = obs.unsqueeze(1), feat.unsqueeze(1)
        B, L, _ = obs.shape
        if prev_actions is None:
            prev_actions = torch.full(
                (B,), NUM_ACTIONS, dtype=torch.long, device=obs.device
            )
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

    def get_action(
        self,
        obs: torch.Tensor,
        feat: torch.Tensor,
        prev_actions: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
        """Select an action and return auxiliary outputs.

        Args:
            obs: Raw observations.
            feat: Engineered features.
            prev_actions: Previous actions.
            hidden: LSTM hidden state.
            deterministic: If ``True``, pick the argmax action instead of
                sampling.

        Returns:
            Tuple of ``(actions, log_probs, denormalized_values, hidden)``.
        """
        logits, values_norm, hidden = self.forward(obs, feat, prev_actions, hidden)
        dist = Categorical(logits=logits)
        if deterministic:
            actions = logits.argmax(dim=-1)
        else:
            actions = dist.sample()
        log_probs = dist.log_prob(actions)
        values = self.value.denormalize(values_norm)
        return actions, log_probs, values, hidden
