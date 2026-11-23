"""Rotation representation transforms.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import functools

import numpy as np
import torch

from mip.data import rotation_conversion as rc


class RotationTransformer:
    valid_reps = ["axis_angle", "euler_angles", "quaternion", "rotation_6d", "matrix"]

    def __init__(
        self,
        from_rep="axis_angle",
        to_rep="rotation_6d",
        from_convention=None,
        to_convention=None,
    ):
        """Valid representations.

        Always use matrix as intermediate representation.
        """
        assert from_rep != to_rep
        assert from_rep in self.valid_reps
        assert to_rep in self.valid_reps
        if from_rep == "euler_angles":
            assert from_convention is not None
        if to_rep == "euler_angles":
            assert to_convention is not None

        forward_funcs = []
        inverse_funcs = []

        if from_rep != "matrix":
            funcs = [
                getattr(rc, f"{from_rep}_to_matrix"),
                getattr(rc, f"matrix_to_{from_rep}"),
            ]
            if from_convention is not None:
                funcs = [
                    functools.partial(func, convention=from_convention)
                    for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        if to_rep != "matrix":
            funcs = [
                getattr(rc, f"matrix_to_{to_rep}"),
                getattr(rc, f"{to_rep}_to_matrix"),
            ]
            if to_convention is not None:
                funcs = [
                    functools.partial(func, convention=to_convention) for func in funcs
                ]
            forward_funcs.append(funcs[0])
            inverse_funcs.append(funcs[1])

        inverse_funcs = inverse_funcs[::-1]

        self.forward_funcs = forward_funcs
        self.inverse_funcs = inverse_funcs

    @staticmethod
    def _apply_funcs(
        x: np.ndarray | torch.Tensor, funcs: list
    ) -> np.ndarray | torch.Tensor:
        x_ = x
        if isinstance(x, np.ndarray):
            x_ = torch.tensor(x)
        x_: torch.Tensor
        for func in funcs:
            x_ = func(x_)
        y = x_
        if isinstance(x, np.ndarray):
            y = x_.numpy()
        return y

    def forward(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self._apply_funcs(x, self.forward_funcs)

    def inverse(self, x: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        return self._apply_funcs(x, self.inverse_funcs)


# Preserve legacy pickle paths while implementations live in the data package.
