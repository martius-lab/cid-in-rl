import collections
from typing import Dict, Iterable, Mapping

import gin
import numpy as np

from cid.memory.base import BaseReplayMemory
from cid.memory.buffer import buffer_from_example, get_leading_dims


@gin.configurable(blacklist=['example', 'episode_len'])
class EpisodicReplayMemory(BaseReplayMemory):
    """Memory that stores transitions episode-wise"""
    def __init__(self, example, size, episode_len, goal_based=False):
        self._size = size
        # Plus one because the buffer needs to fit one more observation
        # than the number of steps in the episode
        self._episode_len = episode_len + 1
        self._buffer = buffer_from_example(example, (size, self._episode_len))
        self._goal_based = goal_based

        self._next_idx = 0
        self._current_size = 0
        self._total_episodes = 0

    @property
    def current_size(self) -> int:
        return self._current_size

    @property
    def episode_len(self) -> int:
        return self._episode_len

    def get_labels(self, key):
        labels = self._buffer[key][:self._current_size, :-1]
        return labels.astype(np.bool)

    def store_episodes(self, episodes):
        batch_size, episode_len = get_leading_dims(episodes, n_dims=2)
        assert episode_len == self._episode_len, \
            (f'Memory was constructed for {self._episode_len} observations, '
             f'but number of observations to store is {episode_len}.')

        end_idx = self._next_idx + batch_size
        if end_idx > self._size:
            idxs = np.arange(self._next_idx, end_idx) % self._size
        else:
            idxs = slice(self._next_idx, end_idx)

        if self._current_size < self._size:
            self._current_size = min(self._size, end_idx)

        self._next_idx = end_idx % self._size

        for key, value in self._buffer.items():
            value[idxs] = episodes[key]

        self._total_episodes += batch_size

    def sample_batch(self, batch_size: int):
        assert self._current_size > 0, 'Trying to sample from empty buffer'

        E_idxs = np.random.randint(low=0,
                                   high=self._current_size,
                                   size=batch_size)
        T_idxs = np.random.randint(low=0,
                                   high=self._episode_len - 1,
                                   size=batch_size)
        batch = {
            's0': self._buffer['s'][E_idxs, T_idxs],
            's1': self._buffer['s'][E_idxs, T_idxs + 1],
            'a': self._buffer['a'][E_idxs, T_idxs],
            'r': self._buffer['r'][E_idxs, T_idxs].squeeze(),
            'd': self._buffer['d'][E_idxs, T_idxs + 1].squeeze()
        }

        if self._goal_based:
            batch['g'] = self._buffer['g'][E_idxs, T_idxs]

        return batch

    def report_statistics(self):
        return {}

    def get_first_done_indices(self, episode_indices):
        dones = self._buffer['d'][episode_indices].squeeze()
        done_idxs = np.argmax(dones, axis=1)
        done_idxs[(done_idxs == 0) & (dones[:, 0] == 0)] = self._episode_len

        return done_idxs

    def to_transition_sequence(self,
                               keys: Iterable[str] = None,
                               filter_fn=None,
                               split_ratios=None,
                               split_idx=0):
        if keys is None:
            keys = {'s0', 's1', 'a', 'r', 'd'}

        keys_current_state = {}
        keys_next_state = {}

        for key in keys:
            if key.endswith('0'):
                keys_current_state[key] = key[:-1]
            elif key.endswith('1'):
                keys_next_state[key] = key[:-1]
            else:
                keys_current_state[key] = key

        buffer_valid = {key: arr[:self._current_size]
                        for key, arr in self._buffer.items()}
        if filter_fn:
            entries_to_include = filter_fn(buffer_valid)
        else:
            entries_to_include = np.ones((self._current_size,
                                          self._episode_len)).astype(np.bool)

        assert entries_to_include.ndim == 2
        entries_to_include = entries_to_include[:, :self._episode_len - 1]

        mapping_E, mapping_T = np.nonzero(entries_to_include)

        if split_ratios is not None:
            split_indices = _get_split_indices(len(mapping_E),
                                               split_ratios,
                                               split_idx)
            mapping_E = mapping_E[split_indices]
            mapping_T = mapping_T[split_indices]

        if self._size == self._current_size and self._next_idx > 0:
            # Sequence always begins with the oldest episode in the memory
            n_rolls = np.argmax(mapping_E >= self._next_idx)
            mapping_E = np.roll(mapping_E, -n_rolls)
            mapping_T = np.roll(mapping_T, -n_rolls)

        return _EpisodicMemorySequenceAdapter(buffer_valid,
                                              mapping_E,
                                              mapping_T,
                                              keys_current_state,
                                              keys_next_state)

    def __iter__(self):
        for ep in range(self._current_size):
            yield {key: buf[ep] for key, buf in self._buffer.items()}

    def save(self, path: str):
        np.save(path,
                [self._buffer, self._current_size, self._next_idx],
                allow_pickle=True)

    def load(self, path: str):
        data = np.load(path, allow_pickle=True)
        self._buffer = data[0]
        shape = next(iter(self._buffer.values())).shape
        self._size = shape[0]
        self._episode_len = shape[1]
        self._current_size = data[1]
        self._next_idx = data[2]


class _EpisodicMemorySequenceAdapter(collections.abc.Sequence):
    def __init__(self,
                 buffers: Dict[str, np.ndarray],
                 mapping_E: np.ndarray,
                 mapping_T: np.ndarray,
                 keys_current_state: Mapping[str, str],
                 keys_next_state: Mapping[str, str]):
        self._buffer = buffers
        self._mapping_E = mapping_E
        self._mapping_T = mapping_T
        self._keys_current_state = keys_current_state
        self._keys_next_state = keys_next_state

    @property
    def n_episodes(self):
        return np.max(self._mapping_E) + 1

    def get_start_index_of_n_latest_episodes(self, n_latest):
        assert n_latest >= 1

        latest_episodes = set()
        count = 0
        for ep in self._mapping_E[::-1]:
            latest_episodes.add(ep)
            if len(latest_episodes) > n_latest:
                break
            count += 1

        return len(self._mapping_E) - count

    def __len__(self) -> int:
        return len(self._mapping_E)

    def __getitem__(self, idx) -> Dict[str, np.ndarray]:
        E_idx = self._mapping_E[idx]
        T_idx = self._mapping_T[idx]

        data = {}
        for data_key, source_key in self._keys_current_state.items():
            data[data_key] = self._buffer[source_key][E_idx, T_idx]
        for data_key, source_key in self._keys_next_state.items():
            data[data_key] = self._buffer[source_key][E_idx, T_idx + 1]

        return data


def _get_split_indices(n_indices, split_ratios, split_idx):
    indices = np.arange(n_indices)
    rng = np.random.RandomState(59942)
    rng.shuffle(indices)

    cum_split_ratios = np.cumsum(np.array([0] + split_ratios))
    assert cum_split_ratios[-1] == 1, 'Split ratios have to sum to 1'
    split_ends = np.floor(cum_split_ratios *
                          np.array([len(indices)] * len(cum_split_ratios)))
    split_ends = split_ends.astype(np.int64)
    split_indices = indices[split_ends[split_idx]:split_ends[split_idx + 1]]
    split_indices = np.sort(split_indices)

    return split_indices
