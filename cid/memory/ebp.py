from typing import Callable

import gin
import numpy as np

from cid.memory.episodic import EpisodicReplayMemory
from cid.memory.score_based import ScoreBasedReplayMemory

_FETCH_ENVS = {'FetchPickAndPlace', 'FetchSlide', 'FetchPush'}

# Constants used for energy computation
_G = 9.81
_M = 1
_DELTA_T = 0.04


@gin.configurable(blacklist=['example', 'episode_len'])
class EBPReplayMemory(EpisodicReplayMemory):
    """Memory implementing energy-based priorization with HER

    See Zhao et al, Energy-Based Hindsight Experience Prioritization, 2018
    http://proceedings.mlr.press/v87/zhao18a.html

    Adapted from https://github.com/ruizhaogit/EnergyBasedPrioritization (MIT)
    """
    def __init__(self, example, size, episode_len,
                 env_name,
                 reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 p_replay=0.8):
        example['energy'] = np.empty((1,), dtype=np.float64)
        super().__init__(example, size, episode_len, goal_based=True)

        self._reward_fn = reward_fn
        self._p_replay = p_replay

        if any(env_name.startswith(name) for name in _FETCH_ENVS):
            self._traj_energy_fn = _compute_traj_energy_fetch_envs
            self._clip_energy = 0.5
        else:
            raise ValueError(f'Environment {env_name} not supported with EBP')

    def store_episodes(self, episodes):
        traj_energy = self._traj_energy_fn(episodes['ag'],
                                           clip_energy=self._clip_energy)
        traj_energy = traj_energy.reshape(-1, 1, 1)
        episodes['energy'] = np.repeat(traj_energy, self.episode_len, axis=1)
        super().store_episodes(episodes)

    def sample_batch(self, batch_size: int):
        assert self._current_size > 0, 'Trying to sample from empty buffer'

        # Sample episodes according to priority
        p_episodes = self._buffer['energy'][:self._current_size, 0, 0]
        p_episodes = p_episodes / p_episodes.sum()
        E_idxs = np.random.choice(self._current_size,
                                  size=batch_size,
                                  replace=True,
                                  p=p_episodes)

        # Sample states for the batch
        T_idxs = np.random.randint(low=0,
                                   high=self._episode_len - 1,
                                   size=batch_size)

        # Sample on how many states the goal is replaced with one from the
        # episode. This many of the first samples of the batch have their
        # goal replaced then.
        n_her_samples = np.random.binomial(batch_size, self._p_replay)
        her_selection = slice(0, n_her_samples)

        # Sample a goal in the future of the episode
        goal_idxs = np.random.randint(low=T_idxs[her_selection] + 1,
                                      high=self.episode_len)

        # Replace goals
        goals = self._buffer['g'][E_idxs, T_idxs]
        if len(goal_idxs) > 0:
            goals[her_selection] = self._buffer['ag'][E_idxs[her_selection],
                                                      goal_idxs]

        achieved_goals_next = self._buffer['ag'][E_idxs, T_idxs + 1]

        batch = {
            's0': self._buffer['s'][E_idxs, T_idxs],
            's1': self._buffer['s'][E_idxs, T_idxs + 1],
            'a': self._buffer['a'][E_idxs, T_idxs],
            'r': self._reward_fn(achieved_goals_next, goals).squeeze(),
            'd': self._buffer['d'][E_idxs, T_idxs + 1].squeeze(),
            'g': goals
        }

        return batch


def _compute_traj_energy_fetch_envs(episode_achieved_goals,
                                    w_potential=1.0,
                                    w_linear=1.0,
                                    clip_energy=999):
    # Compute potential energy
    height = episode_achieved_goals[:, :, 2]
    base_height = np.repeat(height[:, 0].reshape(-1, 1),
                            height[:, 1:].shape[1],
                            axis=1)
    height = height[:, 1:] - base_height
    potential_energy = _G * _M * height

    # Compute kinetic energy
    diff = np.diff(episode_achieved_goals, axis=1)
    velocity = diff / _DELTA_T
    kinetic_energy = 0.5 * _M * np.power(velocity, 2)
    kinetic_energy = np.sum(kinetic_energy, axis=2)

    # Compute total energy difference over episode
    energy_total = w_potential * potential_energy + w_linear * kinetic_energy
    energy_diff = np.diff(energy_total, axis=1)

    energy_transition = energy_total
    energy_transition[:, 1:] = energy_diff
    energy_transition = np.clip(energy_transition, 0, clip_energy)

    energy_transition_total = np.sum(energy_transition, axis=1)

    return energy_transition_total


@gin.configurable(blacklist=['example', 'episode_len'])
class EBPReplayMemoryWithScores(ScoreBasedReplayMemory):
    """Memory implementing energy-based priorization with HER

    Utility class that does EBP but also stores scores from a model
    """
    def __init__(self, example, size, episode_len, env_name, model,
                 reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 p_replay=0.8):
        example['energy'] = np.empty((1,), dtype=np.float64)
        super().__init__(example, size, episode_len, model)

        self._reward_fn = reward_fn
        self._p_replay = p_replay

        if any(env_name.startswith(name) for name in _FETCH_ENVS):
            self._traj_energy_fn = _compute_traj_energy_fetch_envs
            self._clip_energy = 0.5
        else:
            raise ValueError(f'Environment {env_name} not supported with EBP')

    def store_episodes(self, episodes):
        traj_energy = self._traj_energy_fn(episodes['ag'],
                                           clip_energy=self._clip_energy)
        traj_energy = traj_energy.reshape(-1, 1, 1)
        episodes['energy'] = np.repeat(traj_energy, self.episode_len, axis=1)
        super().store_episodes(episodes)

    def sample_batch(self, batch_size: int):
        assert self._current_size > 0, 'Trying to sample from empty buffer'

        # Sample episodes according to priority
        p_episodes = self._buffer['energy'][:self._current_size, 0, 0]
        p_episodes = p_episodes / p_episodes.sum()
        E_idxs = np.random.choice(self._current_size,
                                  size=batch_size,
                                  replace=True,
                                  p=p_episodes)

        # Sample states for the batch
        T_idxs = np.random.randint(low=0,
                                   high=self._episode_len - 1,
                                   size=batch_size)

        # Sample on how many states the goal is replaced with one from the
        # episode. This many of the first samples of the batch have their
        # goal replaced then.
        n_her_samples = np.random.binomial(batch_size, self._p_replay)
        her_selection = slice(0, n_her_samples)

        # Sample a goal in the future of the episode
        goal_idxs = np.random.randint(low=T_idxs[her_selection] + 1,
                                      high=self.episode_len)

        # Replace goals
        goals = self._buffer['g'][E_idxs, T_idxs]
        if len(goal_idxs) > 0:
            goals[her_selection] = self._buffer['ag'][E_idxs[her_selection],
                                                      goal_idxs]

        achieved_goals_next = self._buffer['ag'][E_idxs, T_idxs + 1]

        batch = {
            's0': self._buffer['s'][E_idxs, T_idxs],
            's1': self._buffer['s'][E_idxs, T_idxs + 1],
            'a': self._buffer['a'][E_idxs, T_idxs],
            'r': self._reward_fn(achieved_goals_next, goals).squeeze(),
            'd': self._buffer['d'][E_idxs, T_idxs + 1].squeeze(),
            'g': goals
        }

        return batch
