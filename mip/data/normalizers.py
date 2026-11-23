"""Dataset normalization utilities.

Author: Chaoyi Pan
Date: 2025-10-03
"""

import numpy as np
from loguru import logger
from scipy import interpolate


def empirical_cdf(sample):
    """https://stackoverflow.com/a/33346366."""
    # find the unique values and their corresponding counts
    quantiles, counts = np.unique(sample, return_counts=True)

    # take the cumulative sum of the counts and divide by the sample size to
    # get the cumulative probabilities between 0 and 1
    cumprob = np.cumsum(counts).astype(np.double) / sample.size

    return quantiles, cumprob


class CDFNormalizer1d:
    """CDF normalizer for a single dimension."""

    def __init__(self, X):
        assert X.ndim == 1
        self.X = X.astype(np.float32)
        quantiles, cumprob = empirical_cdf(self.X)
        self.fn = interpolate.interp1d(quantiles, cumprob)
        self.inv = interpolate.interp1d(cumprob, quantiles)
        self.xmin, self.xmax = quantiles.min(), quantiles.max()
        self.ymin, self.ymax = cumprob.min(), cumprob.max()

    def normalize(self, x):
        x = np.clip(x, self.xmin, self.xmax)
        y = self.fn(x)
        y = 2 * y - 1
        return y

    def unnormalize(self, x, eps=1e-4):
        x = (x + 1) / 2.0
        if (x < self.ymin - eps).any() or (x > self.ymax + eps).any():
            logger.warning(
                f"""[ dataset/normalization ] Warning: out of range in unnormalize: """
                f"""[{x.min()}, {x.max()}] | """
                f"""x : [{self.xmin}, {self.xmax}] | """
                f"""y: [{self.ymin}, {self.ymax}]"""
            )
        x = np.clip(x, self.ymin, self.ymax)
        y = self.inv(x)
        return y


class CDFNormalizer:
    """makes training data uniform (over each dimension) by transforming it with marginal CDFs."""

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.mins, self.maxs = X.min(0), X.max(0)
        self.dim = X.shape[-1]
        self.cdfs = [CDFNormalizer1d(self.X[:, i]) for i in range(self.dim)]

    def wrap(self, fn_name, x):
        shape = x.shape
        x = x.reshape(-1, self.dim)
        out = np.zeros_like(x)
        for i, cdf in enumerate(self.cdfs):
            fn = getattr(cdf, fn_name)
            out[:, i] = fn(x[:, i])
        return out.reshape(shape)

    def normalize(self, x):
        return self.wrap("normalize", x)

    def unnormalize(self, x):
        return self.wrap("unnormalize", x)


class GaussianNormalizer:
    """normalizes data to have zero mean and unit variance."""

    def __init__(self, X):
        self.X = X.astype(np.float32)
        self.means, self.stds = X.mean(0), X.std(0)
        self.stds[self.stds == 0] = 1.0

    def normalize(self, x):
        return (x - self.means[None,]) / self.stds[None,]

    def unnormalize(self, x):
        return x * self.stds[None,] + self.means[None,]


class ImageNormalizer:
    """normalizes image data from range [0, 1] to [-1, 1]."""

    def __init__(self):
        pass

    def normalize(self, x):
        return x * 2.0 - 1.0

    def unnormalize(self, x):
        return (x + 1.0) / 2.0


class MinMaxNormalizer:
    """normalizes data through maximum and minimum expansion."""

    def __init__(self, X):
        X = X.reshape(-1, X.shape[-1]).astype(np.float32)
        self.min, self.max = np.min(X, axis=0), np.max(X, axis=0)
        self.range = self.max - self.min
        if np.any(self.range == 0):
            self.range = self.max - self.min
            logger.warning(
                "Warning: Some features have the same min and max value. These will be set to 0."
            )
            self.range[self.range == 0] = 1

    def normalize(self, x):
        x = x.astype(np.float32)
        # nomalize to [0,1]
        nx = (x - self.min) / self.range
        # normalize to [-1, 1]
        nx = nx * 2 - 1
        return nx

    def unnormalize(self, x):
        x = x.astype(np.float32)
        nx = (x + 1) / 2
        x = nx * self.range + self.min
        return x


class EmptyNormalizer:
    """do nothing and change nothing."""

    def __init__(self):
        pass

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x


class IdentityNormalizer:
    """Pass-through normalizer that leaves data unchanged."""

    def normalize(self, x):
        return x

    def unnormalize(self, x):
        return x


class CompositeNormalizer:
    """Applies different normalizers to different dimension slices.

    Example: for 10D actions [pos(3) + rot6d(6) + grip(1)]:
        CompositeNormalizer(
            normalizers=[MinMaxNormalizer(pos_data), IdentityNormalizer(), MinMaxNormalizer(grip_data)],
            dim_slices=[(0, 3), (3, 9), (9, 10)],
        )
    """

    def __init__(self, normalizers: list, dim_slices: list[tuple[int, int]]):
        assert len(normalizers) == len(dim_slices)
        self.normalizers = normalizers
        self.dim_slices = dim_slices

    def normalize(self, x):
        out = np.empty_like(x, dtype=np.float32)
        for norm, (start, end) in zip(self.normalizers, self.dim_slices, strict=True):
            out[..., start:end] = norm.normalize(x[..., start:end])
        return out

    def unnormalize(self, x):
        out = np.empty_like(x, dtype=np.float32)
        for norm, (start, end) in zip(self.normalizers, self.dim_slices, strict=True):
            out[..., start:end] = norm.unnormalize(x[..., start:end])
        return out


# Preserve legacy pickle paths while implementations live in the data package.
