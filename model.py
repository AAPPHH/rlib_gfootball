import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from gymnasium.spaces import Space
from typing import Tuple, Optional, List, Dict, Union

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.policy.view_requirement import ViewRequirement

class DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1, bias: bool = True):
        super().__init__()
        padding = (kernel - 1) * dilation // 2
        self.depthwise = nn.Conv1d(in_ch, in_ch, kernel_size=kernel, padding=padding,
                                   dilation=dilation, groups=in_ch, bias=False)
        self.pointwise = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=bias)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))

class EfficientGroupNorm(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-5):
        super().__init__()
        num_groups = min(32, num_channels)
        while num_channels % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        self.norm = nn.GroupNorm(max(1, num_groups), num_channels, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)

class EfficientTCNBlock(nn.Module):
    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float = 0.0,
                 use_checkpoint: bool = False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        expand = 2
        inner_dim = dim * expand
        self.norm = EfficientGroupNorm(dim, eps=1e-5)
        self.conv = DepthwiseSeparableConv1d(dim, inner_dim * 2, kernel, dilation)
        self.proj = nn.Conv1d(inner_dim, dim, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.layer_scale = nn.Parameter(torch.ones(1, 1, dim) * 0.1)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm(x.transpose(1, 2))
        conv_out = self.conv(x_norm)
        gate, value = conv_out.chunk(2, dim=1)
        out = value * torch.sigmoid(gate)
        out = self.proj(out).transpose(1, 2)
        out = self.dropout(out)
        return out * self.layer_scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            out = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            out = self._forward_impl(x)
        return x + out

class GFootballTCN(TorchRNN, nn.Module):
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int,
                 model_config: ModelConfigDict, name: str):
            
        nn.Module.__init__(self)

        cfg = model_config.get("custom_model_config", {})
        self.gru_hidden = cfg.get("gru_hidden", 128)
        
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.view_requirements["prev_actions"] = ViewRequirement(
            data_col="actions", shift=-1, space=self.action_space
        )

        self.d_model = cfg.get("d_model", 160)
        self.prev_action_emb_dim = cfg.get("prev_action_emb", 16)

        self.tcn_kernel = cfg.get("tcn_kernel", 3)
        self.tcn_dilations = cfg.get("tcn_dilations", [1, 2, 4])
        self.use_checkpoint = cfg.get("gradient_checkpointing", True)
        self.dropout = cfg.get("dropout", 0.05)
        
        self.num_frames = 4
        total_obs_dim = int(np.prod(obs_space.shape))
        if total_obs_dim % self.num_frames != 0:
            raise ValueError(
                f"Gesamte Obs-Dimension ({total_obs_dim}) nicht teilbar "
                f"durch num_frames ({self.num_frames}). Stellen Sie sicher, dass 'stacked=True' (ergibt 460) "
                "und 'representation=simple115v2' verwendet werden."
            )
        self.frame_dim = total_obs_dim // self.num_frames
        
        if self.frame_dim != 115:
             print(f"WARN: frame_dim wurde als {self.frame_dim} berechnet. "
                   "Erwartet wurde 115 für 'simple115v2'.")

        self.frame_encoder = nn.Linear(self.frame_dim, self.d_model)
        self.frame_norm = nn.LayerNorm(self.d_model)
        
        self.tcn_blocks = nn.ModuleList([
            EfficientTCNBlock(self.d_model, self.tcn_kernel, dil, self.dropout, self.use_checkpoint)
            for dil in self.tcn_dilations
        ])

        self.prev_action_embed = nn.Embedding(
            self.action_space.n, self.prev_action_emb_dim
        )
        
        self.gru_input_size = self.d_model + self.prev_action_emb_dim
        
        self.gru = nn.GRU(
            self.gru_input_size, 
            self.gru_hidden, 
            batch_first=True
        )

        self.policy_head = nn.Sequential(
            nn.Linear(self.gru_hidden, max(64, self.gru_hidden // 2)),
            nn.ReLU(),
            nn.Linear(max(64, self.gru_hidden // 2), num_outputs)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(self.gru_hidden, max(32, self.gru_hidden // 4)),
            nn.ReLU(),
            nn.Linear(max(32, self.gru_hidden // 4), 1)
        )
        
        with torch.no_grad():
            for name, param in self.gru.named_parameters():
                if "bias_ih" in name:
                    nn.init.constant_(param, 0.0)
                if "bias_hh" in name:
                    nn.init.constant_(param, 0.0)

        self._value_out: Optional[torch.Tensor] = None
        self._features: Optional[torch.Tensor] = None
            
    @override(TorchRNN)
    def get_initial_state(self) -> List[torch.Tensor]:
        return [torch.zeros(self.gru_hidden, dtype=torch.float32)]

    
    def _encode_frames(self, obs_flat: TensorType) -> TensorType:
        x = self.frame_encoder(obs_flat)
        x = F.relu(self.frame_norm(x))
        
        for block in self.tcn_blocks:
            x = block(x)

        z_t = 0.6 * x.mean(dim=1) + 0.4 * x.max(dim=1)[0]
        return z_t

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: Dict[str, TensorType],
        state: List[TensorType],
        seq_lens: TensorType
    ) -> Tuple[TensorType, List[TensorType]]:

        obs_flat = input_dict.get("obs_flat", input_dict.get("obs"))
        if obs_flat is None:
            raise ValueError("Missing 'obs_flat' or 'obs' in input_dict.")

        BT, obs_dim = obs_flat.shape
        device = obs_flat.device
        
        if seq_lens is None:
            T_max = 1
        else:
            T_max = int(seq_lens.max().item())

        B = int(BT // T_max)
        
        if B * T_max != BT:
             obs_unpadded = torch.split(obs_flat, seq_lens.int().tolist())
             obs_padded = nn.utils.rnn.pad_sequence(obs_unpadded, batch_first=True, padding_value=0.0)
             B, T_max, _ = obs_padded.shape
             obs_flat = obs_padded.reshape(B * T_max, obs_dim)
             BT = B * T_max

        prev_actions = input_dict["prev_actions"].long().view(B, T_max)

        obs_frames = obs_flat.view(BT, self.num_frames, self.frame_dim)

        z_t = self._encode_frames(obs_frames)


        pa_emb = self.prev_action_embed(prev_actions)
        pa_emb_flat = pa_emb.view(BT, self.prev_action_emb_dim)

        gru_in_flat = torch.cat([z_t, pa_emb_flat], dim=-1)
        gru_in = gru_in_flat.view(B, T_max, self.gru_input_size)

        h_in = state[0].to(device).unsqueeze(0)
        packed_gru_in = nn.utils.rnn.pack_padded_sequence(
            gru_in, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        
        packed_gru_out, h_out = self.gru(packed_gru_in, h_in)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_gru_out, batch_first=True, total_length=T_max
        )

        gru_out_flat = gru_out.reshape(BT, self.gru_hidden)
        self._features = gru_out_flat
        logits = self.policy_head(gru_out_flat)
        self._value_out = self.value_head(gru_out_flat).squeeze(-1)

        new_state = [h_out.squeeze(0)]                   
        return logits, new_state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        assert self._value_out is not None, "value_function() aufgerufen vor forward()"
        return self._value_out