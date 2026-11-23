"""Episode-bounded sequence sampling and padding.

Original implementation: https://github.com/real-stanford/diffusion_policy

Author: Chaoyi Pan
Date: 2025-10-03
"""

import numba
import numpy as np

from mip.data.replay_buffer import ReplayBuffer


@numba.jit(nopython=True)
def create_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
    debug: bool = True,
) -> np.ndarray:
    pad_before = min(max(pad_before, 0), sequence_length - 1)
    pad_after = min(max(pad_after, 0), sequence_length - 1)

    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0  # episode start index
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]  # episode end index
        episode_length = end_idx - start_idx  # episode length

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            if debug:
                assert start_offset >= 0
                assert end_offset >= 0
                assert (sample_end_idx - sample_start_idx) == (
                    buffer_end_idx - buffer_start_idx
                )
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


class SequenceSampler:
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0,
        keys=None,
        key_first_k=None,
        zero_padding: bool = False,
    ):
        """key_first_k: dict str: int
        Only take first k data from these keys (to improve perf).
        """
        if key_first_k is None:
            key_first_k = {}
        super().__init__()
        assert sequence_length >= 1

        # all keys
        if keys is None:
            keys = list(replay_buffer.keys())

        episode_ends = replay_buffer.episode_ends[:]

        # create indices
        # indices (buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx)
        # buffer_start_idx and buffer_end_idx define the actual start and end positions of the sample sequence within the original dataset.
        # sample_start_idx and sample_end_idx define the relative start and end positions within the sample sequence,
        # which is particularly useful when dealing with padding as it can affect the actual length of the sequence.
        indices = create_indices(
            episode_ends=episode_ends,
            sequence_length=sequence_length,
            pad_before=pad_before,
            pad_after=pad_after,
        )

        self.indices = indices
        self.keys = list(keys)  # prevent OmegaConf list performance problem
        self.sequence_length = sequence_length
        self.replay_buffer = replay_buffer
        self.zero_padding = zero_padding
        self.key_first_k = key_first_k

    def __len__(self):
        return len(self.indices)

    def sample_sequence(self, idx):
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = (
            self.indices[idx]
        )
        result = {}
        for key in self.keys:
            input_arr = self.replay_buffer[key]
            # performance optimization, avoid small allocation if possible
            if key not in self.key_first_k:
                sample = input_arr[buffer_start_idx:buffer_end_idx]
            else:
                # performance optimization, only load used obs steps
                n_data = buffer_end_idx - buffer_start_idx
                k_data = min(self.key_first_k[key], n_data)
                # fill value with Nan to catch bugs
                # the non-loaded region should never be used
                sample = np.full(
                    (n_data,) + input_arr.shape[1:],
                    fill_value=np.nan,
                    dtype=input_arr.dtype,
                )
                sample[:k_data] = input_arr[
                    buffer_start_idx : buffer_start_idx + k_data
                ]
            data = sample
            if (sample_start_idx > 0) or (sample_end_idx < self.sequence_length):
                data = np.zeros(
                    shape=(self.sequence_length,) + input_arr.shape[1:],
                    dtype=input_arr.dtype,
                )
                if not self.zero_padding:
                    if sample_start_idx > 0:
                        data[:sample_start_idx] = sample[0]
                    if sample_end_idx < self.sequence_length:
                        data[sample_end_idx:] = sample[-1]
                data[sample_start_idx:sample_end_idx] = sample
            result[key] = data
        return result


# Preserve legacy pickle paths while implementations live in the data package.
