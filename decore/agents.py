"""DECORE policy agents: one learnable weight per channel."""
from __future__ import annotations

import torch
import torch.nn as nn


class Agent(nn.Module):
    """A layer's policy: one scalar weight per channel.

    ``sigmoid(weight)`` is the probability of KEEPING each channel. Weights are
    initialised high (6.9 => ~0.99) so the pretrained network is left intact at
    the start of policy learning.
    """

    def __init__(self, num_channels: int, init_weight: float = 6.9):
        super().__init__()
        self.weights = nn.Parameter(torch.full((num_channels,), float(init_weight)))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.weights)


def build_agents(model, device, init_weight: float = 6.9):
    """Create one Agent per prunable block in the model.

    Returns (agents, block_indices) where block_indices are the positions of
    the prunable blocks inside ``model.features``.
    """
    from .models import PrunableConvBNReLU

    block_indices = [i for i, layer in enumerate(model.features)
                     if isinstance(layer, PrunableConvBNReLU)]
    agents = [Agent(model.features[i].out_channels, init_weight).to(device)
              for i in block_indices]
    return agents, block_indices
