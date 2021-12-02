from typing import Callable

import gin
import numpy as np

from cid.memory import EpisodicReplayMemory


@gin.configurable(blacklist=['example', 'episode_len'])
class HERReplayMemory(EpisodicReplayMemory):
    """Memory that implements hindsight experience replay"""
    def __init__(self, example, size, episode_len,
                 reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 p_replay=0.8):
        """
        :param reward_fn: Function taking (achieved_goals, goals) and returning
            the reward
        :param p_replay: For each item of the batch, probability of
            replaying a goal for that item
        """
        super().__init__(example, size, episode_len, goal_based=True)
        self._reward_fn = reward_fn
        self._p_replay = p_replay

    def resample_goals(self, E_idxs, T_idxs, done_idxs=None):
        # Sample on how many states the goal is replaced with one from the
        # episode. This many of the first samples of the batch have their
        # goal replaced then.
        n_her_samples = np.random.binomial(len(E_idxs), self._p_replay)
        her_selection = slice(0, n_her_samples)

        # Sample a goal from the future of the episode
        if done_idxs is not None:
            goal_idxs = np.random.randint(low=T_idxs[her_selection] + 1,
                                          high=done_idxs[her_selection])
        else:
            goal_idxs = np.random.randint(low=T_idxs[her_selection] + 1,
                                          high=self._episode_len)

        # Replace goals
        goals = self._buffer['g'][E_idxs, T_idxs]
        if len(goal_idxs) > 0:
            goals[her_selection] = self._buffer['ag'][E_idxs[her_selection],
                                                      goal_idxs]

        achieved_goals_next = self._buffer['ag'][E_idxs, T_idxs + 1]

        rewards = self._reward_fn(achieved_goals_next, goals).squeeze()

        return goals, rewards

    def sample_batch(self, batch_size: int):
        assert self._current_size > 0, 'Trying to sample from empty buffer'

        # Sample states for the batch
        E_idxs = np.random.randint(low=0,
                                   high=self._current_size,
                                   size=batch_size)
        done_idxs = self.get_first_done_indices(E_idxs)
        T_idxs = np.random.randint(low=0,
                                   high=done_idxs - 1,
                                   size=batch_size)

        goals, rewards = self.resample_goals(E_idxs, T_idxs, done_idxs)

        batch = {
            's0': self._buffer['s'][E_idxs, T_idxs],
            's1': self._buffer['s'][E_idxs, T_idxs + 1],
            'a': self._buffer['a'][E_idxs, T_idxs],
            'r': rewards,
            'd': self._buffer['d'][E_idxs, T_idxs + 1].squeeze(),
            'g': goals
        }

        return batch
