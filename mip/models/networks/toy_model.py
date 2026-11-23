"""Tiny proof-of-concept action network."""

import torch
import torch.nn as nn

from mip.models.networks.base import BaseNetwork


class ToyModel(BaseNetwork):
    """A minimal MLP that matches the framework network interface."""

    def __init__(
        self,
        act_dim: int,
        Ta: int,
        obs_dim: int,
        To: int,
        emb_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__(act_dim, Ta, obs_dim, To, emb_dim, n_layers)

        input_dim = act_dim * Ta + obs_dim * To + 2
        output_dim = act_dim * Ta

        layers = [nn.Linear(input_dim, emb_dim), nn.GELU()]
        for _ in range(n_layers):
            layers.extend(
                [
                    nn.Linear(emb_dim, emb_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )

        self.net = nn.Sequential(*layers)
        self.action_head = nn.Linear(emb_dim, output_dim)
        self.scalar_head = nn.Linear(emb_dim, 1)

    def forward(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        t: torch.Tensor,
        condition: torch.Tensor | None = None,
    ):
        batch_size = x.shape[0]
        x_flat = x.flatten(1)

        if condition is None:
            condition_flat = torch.zeros(
                batch_size,
                self.To * self.obs_dim,
                device=x.device,
                dtype=x.dtype,
            )
        else:
            condition_flat = condition.flatten(1)

        h = torch.cat([x_flat, condition_flat, s[:, None], t[:, None]], dim=-1)
        h = self.net(h)

        action = self.action_head(h).view(batch_size, self.Ta, self.act_dim)
        scalar = self.scalar_head(h)
        return action, scalar
