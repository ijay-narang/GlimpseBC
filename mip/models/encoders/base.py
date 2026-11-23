"""Shared observation encoder interface and masking."""

import torch
import torch.nn as nn


def get_mask(
    mask: torch.Tensor,
    mask_shape: tuple,
    dropout: float,
    train: bool,
    device: torch.device,
):
    if train:
        mask = (torch.rand(mask_shape, device=device) > dropout).float()
    else:
        mask = 1.0 if mask is None else mask
    return mask


class BaseEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, condition: torch.Tensor, mask: torch.Tensor = None):
        raise NotImplementedError
