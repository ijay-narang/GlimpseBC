"""Identity and MLP encoders for state observations."""

import torch
import torch.nn as nn

from mip.models.encoders.base import BaseEncoder, get_mask
from mip.models.layers import Mlp
from mip.torch_utils import at_least_ndim


class IdentityEncoder(BaseEncoder):
    """Identity encoder does not change the input condition.

    Input:
        - condition: (b, *cond_in_shape) or dict of tensors
        - mask :     (b, ) or None, None means no mask

    Output:
        - condition: (b, *cond_in_shape)
    """

    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.dropout = dropout

    def forward(self, condition: torch.Tensor | dict, mask: torch.Tensor = None):
        # Handle dict input by concatenating all tensors
        if isinstance(condition, dict):
            # Sort keys for consistent ordering and concatenate all values
            keys = sorted(condition.keys())
            tensors = [condition[k] for k in keys]
            # Flatten each tensor to (batch, -1) and concatenate
            flattened = [t.reshape(t.shape[0], -1) for t in tensors]
            condition = torch.cat(flattened, dim=-1)

        mask = at_least_ndim(
            get_mask(
                mask,
                (condition.shape[0],),
                self.dropout,
                self.training,
                condition.device,
            ),
            condition.dim(),
        )
        return condition * mask


class MLPEncoder(BaseEncoder):
    """A simple MLP encoder.

    Use a simple MLP to project the input condition to the desired dimension.

    Args:
        in_dim: int,
            The input dimension of the condition
        out_dim: int,
            The output dimension of the condition
        hidden_dims: List[int],
            The hidden dimensions of the MLP
        act: nn.Module,
            The activation function of the MLP
        dropout: float,
            The label dropout rate

    Examples:
        >>> encoder = MLPEncoder(in_dim=5, out_dim=10)
        >>> condition = torch.randn(2, 5)
        >>> encoder(condition).shape
        torch.Size([2, 10])
        >>> condition = torch.randn(2, 20, 5)
        >>> encoder(condition).shape
        torch.Size([2, 20, 10])
    """

    def __init__(
        self,
        obs_dim: int,
        emb_dim: int,
        To: int,
        hidden_dims: list[int],
        act=nn.LeakyReLU(),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.dropout = dropout
        self.To = To
        self.emb_dim = emb_dim
        hidden_dims = (
            [
                hidden_dims,
            ]
            if isinstance(hidden_dims, int)
            else hidden_dims
        )
        self.mlp = Mlp(obs_dim * To, hidden_dims, emb_dim * To, act)

    def forward(self, obs: torch.Tensor | dict, mask: torch.Tensor = None):
        # Handle dict input by concatenating all tensors
        if isinstance(obs, dict):
            # Sort keys for consistent ordering and concatenate all values
            keys = sorted(obs.keys())
            obs_list = [obs[k] for k in keys]
        else:
            obs_list = [obs]
        # Flatten each tensor to (batch, -1) and concatenate
        flattened = [t.reshape(t.shape[0], -1) for t in obs_list]
        obs = torch.cat(flattened, dim=-1)

        mask = at_least_ndim(
            get_mask(
                mask,
                (obs.shape[0],),
                self.dropout,
                self.training,
                obs.device,
            ),
            obs.dim(),
        )
        emb_features = self.mlp(obs) * mask
        return emb_features.reshape(obs.shape[0], self.To, self.emb_dim)


class PerStepMLPEncoder(BaseEncoder):
    """Apply an MLP to each timestep independently (no time mixing)."""

    def __init__(
        self,
        obs_dim: int,
        emb_dim: int,
        hidden_dims: list[int],
        act=nn.LeakyReLU(),
        dropout: float = 0.25,
    ):
        super().__init__()
        self.dropout = dropout
        self.emb_dim = emb_dim
        hidden_dims = (
            [
                hidden_dims,
            ]
            if isinstance(hidden_dims, int)
            else hidden_dims
        )

        layers = []
        in_dim = obs_dim
        if len(hidden_dims) > 0:
            for dim in hidden_dims:
                layers.append(nn.Linear(in_dim, dim))
                layers.append(act)
                in_dim = dim
        layers.append(nn.Linear(in_dim, emb_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor | dict, mask: torch.Tensor = None):
        if isinstance(obs, dict):
            keys = sorted(obs.keys())
            obs_list = [obs[k] for k in keys]
            obs = torch.cat(obs_list, dim=-1)

        # obs: (B, To, obs_dim) -> (B*To, obs_dim)
        b, t, d = obs.shape
        obs_flat = obs.reshape(b * t, d)
        emb_flat = self.mlp(obs_flat)
        emb = emb_flat.reshape(b, t, self.emb_dim)

        mask = at_least_ndim(
            get_mask(
                mask,
                (obs.shape[0],),
                self.dropout,
                self.training,
                obs.device,
            ),
            emb.dim(),
        )
        return emb * mask
