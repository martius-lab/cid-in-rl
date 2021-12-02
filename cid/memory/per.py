from typing import Callable

import gin
import numpy as np

from cid.memory import EpisodicReplayMemory, HERReplayMemory


class _PrioritizedReplayMixin:
    def store_episodes(self, episodes):
        first_elem = next(iter(episodes.values()))
        batch_size = first_elem.shape[0]
        episode_len = first_elem.shape[1]

        episodes['td_error'] = np.full((batch_size, episode_len, 1),
                                       self._max_error,
                                       dtype=np.float64)
        super().store_episodes(episodes)

    def _sample_idxs(self, batch_size):
        td_errors = self._buffer['td_error'][:self._current_size, :-1]
        td_errors = td_errors.reshape(-1)
        td_errors = td_errors**self._alpha

        p = td_errors / np.sum(td_errors)
        min_p = np.min(p)

        idxs = np.random.choice(len(p), size=batch_size, replace=True, p=p)
        E_idxs = idxs // (self.episode_len - 1)
        T_idxs = idxs % (self.episode_len - 1)

        return E_idxs, T_idxs, p[idxs], min_p

    def update_td_errors(self, batch, td_errors):
        E_idxs = batch['E_idxs']
        T_idxs = batch['T_idxs']

        td_errors = np.maximum(np.abs(td_errors), self._min_error)
        self._buffer['td_error'][E_idxs, T_idxs, 0] = np.abs(td_errors)

        self._max_error = max(self._max_error, np.max(td_errors))

    def _compute_is_weights(self, p_transitions, min_p):
        exp = self._is_weight_exponent_fn(self._total_episodes)
        n_transitions = self._current_size * (self.episode_len - 1)
        weights = (n_transitions * p_transitions)**(-exp)
        max_weight = (n_transitions * min_p)**(-exp)

        return weights / max_weight


@gin.configurable(blacklist=['example', 'episode_len'])
class PERReplayMemory(_PrioritizedReplayMixin, EpisodicReplayMemory):
    """Memory that implements prioritized experience replay"""
    def __init__(self, example, size, episode_len,
                 goal_based=False, min_error=1e-6, alpha=1.0, beta=None):
        """
        :param min_error: TD errors are clipped to this minimum value
        :param alpha: Exponent for weighting TD errors. `alpha=0` corresponds
            to uniform sampling
        :param beta: Exponent for importance sampling. Value can be linearly
            annealed by specifying a tuple of `(start_value, end_value,
            period)`, where `period` specifies the total number of episodes
            stored in the buffer upon reaching the end of annealing. If `None`,
            no importance sampling is used.
        """
        example['td_error'] = np.empty((1,), dtype=np.float64)
        super().__init__(example, size, episode_len, goal_based=goal_based)

        self._alpha = alpha
        self._min_error = min_error
        self._max_error = 1.0  # As in the OpenAI baselines implementation

        self._is_weight_exponent_fn = _make_is_weight_exponent_fn(beta)

    def sample_batch(self, batch_size: int):
        E_idxs, T_idxs, p_transitions, min_p = self._sample_idxs(batch_size)

        rewards = self._buffer['r'][E_idxs, T_idxs].squeeze()

        if self._is_weight_exponent_fn is not None:
            weights = self._compute_is_weights(p_transitions, min_p)
            # Add weights to rewards in a separate dimension
            rewards = np.stack((rewards, weights), axis=0)

        batch = {
            's0': self._buffer['s'][E_idxs, T_idxs],
            's1': self._buffer['s'][E_idxs, T_idxs + 1],
            'a': self._buffer['a'][E_idxs, T_idxs],
            'r': rewards,
            'd': self._buffer['d'][E_idxs, T_idxs + 1].squeeze(),
            'E_idxs': E_idxs,
            'T_idxs': T_idxs
        }

        if self._goal_based:
            batch['g'] = self._buffer['g'][E_idxs, T_idxs]

        return batch


@gin.configurable(blacklist=['example', 'episode_len'])
class HERPERReplayMemory(_PrioritizedReplayMixin, HERReplayMemory):
    """Memory that implements prioritized experience replay with HER"""
    def __init__(self, example, size, episode_len,
                 reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 p_replay=0.8, min_error=1e-6, alpha=1.0, beta=None):
        """
        :param reward_fn: Function taking (achieved_goals, goals) and returning
            the reward
        :param p_replay: For each item of the batch, probability of
            replaying a goal for that item
            :param min_error: TD errors are clipped to this minimum value
        :param alpha: Exponent for weighting TD errors. `alpha=0` corresponds
            to uniform sampling
        :param beta: Exponent for importance sampling. Value can be linearly
            annealed by specifying a tuple of `(start_value, end_value,
            period)`, where `period` specifies the total number of episodes
            stored in the buffer upon reaching the end of annealing. If `None`,
            no importance sampling is used.
        """
        example['td_error'] = np.empty((1,), dtype=np.float64)
        super().__init__(example, size, episode_len, reward_fn, p_replay)

        self._alpha = alpha
        self._min_error = min_error
        self._max_error = 1.0  # As in the OpenAI baselines implementation

        self._is_weight_exponent_fn = _make_is_weight_exponent_fn(beta)

    def sample_batch(self, batch_size: int):
        E_idxs, T_idxs, p_transitions, min_p = self._sample_idxs(batch_size)

        goals, rewards = self.resample_goals(E_idxs, T_idxs)

        if self._is_weight_exponent_fn is not None:
            weights = self._compute_is_weights(p_transitions, min_p)
            # Add weights to rewards in a separate dimension
            rewards = np.stack((rewards, weights), axis=0)

        batch = {
            's0': self._buffer['s'][E_idxs, T_idxs],
            's1': self._buffer['s'][E_idxs, T_idxs + 1],
            'a': self._buffer['a'][E_idxs, T_idxs],
            'r': rewards,
            'd': self._buffer['d'][E_idxs, T_idxs + 1].squeeze(),
            'g': goals,
            'E_idxs': E_idxs,
            'T_idxs': T_idxs
        }

        return batch


def _make_is_weight_exponent_fn(beta):
    if beta is None:
        fn = None
    elif isinstance(beta, tuple):
        start, end, period = beta

        def fn(n):
            return np.clip(start + n * (end - start) / period,
                           start, end)
    else:
        def fn(n):
            return beta

    return fn
