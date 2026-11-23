"""Dataset loading and nested-batch helpers.

Author: Chaoyi Pan
Date: 2025-10-03
"""

from collections.abc import Callable

import torch

from mip.config import TaskConfig
from mip.data.datasets.base import BaseDataset


def get_dataset(config: TaskConfig) -> BaseDataset:
    raise NotImplementedError("Dataset not implemented.")


def dict_apply(
    x: dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]
) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        elif value is None:
            result[key] = None
        else:
            result[key] = func(value)
    return result


def loop_dataloader(dl):
    while True:
        yield from dl


def loop_two_dataloaders(dl1, dl2):
    while True:
        yield from zip(dl1, dl2, strict=False)


# Preserve legacy pickle paths while implementations live in the data package.
