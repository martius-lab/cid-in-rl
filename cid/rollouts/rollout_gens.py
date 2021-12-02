import collections
import time
from typing import TYPE_CHECKING, Any, Dict, Tuple

import gin
import numpy as np

from cid.rollouts.base import BaseRolloutGenerator

if TYPE_CHECKING:
    import gym


@gin.configurable(blacklist=['env'])
class RolloutGenerator(BaseRolloutGenerator):
    def __init__(self, env, fps=None):
        self._env = env
        self._first_obs = None
        self._fps = fps
        self.reset()

    def reset(self):
        self._first_obs = self._env.reset()
        return self._first_obs

    def close(self):
        self._env.close()

    def rollout(self,
                agent,
                n_steps: int,
                evaluate: bool = False,
                render: bool = False) -> Tuple[Dict[str, np.ndarray],
                                               Dict[str, Any]]:
        """Perform a single rollout in the environment

        :param agent: Agent performing the rollout
        :param n_steps: Number of steps to perform, i.e. the number of actions
            that are taken
        :param evaluate: If `True`, run the agent in evaluation mode
        :param render: If `True`, render the environment before each step

        :return: Tuple. First entry is a dictionary containing observed the
            transitions. Shape of arrays is `(1, n_steps + 1, ...)`. Second
            entry is a dictionary with statistics about the rollout.
        """
        states = [self._first_obs]
        actions = []
        is_randoms = []
        dones = [False]
        rewards = []

        obs = self._first_obs
        for step in range(n_steps):
            if render:
                self._env.render()
                if self._fps is not None:
                    time.sleep(1 / self._fps)

            action, is_random = agent.get_action(obs, evaluate=evaluate)

            next_obs, reward, done, info = self._env.step(action)

            # We only record the done signal if it resulted from genuinely
            # being in a terminal state, and not from running into the time
            # limit. Gym's `TimeLimit` env wrapper class indicates this by
            # setting `info['TimeLimit.truncated'] == True`.
            if done and info.get('TimeLimit.truncated', False):
                done = False

            states.append(next_obs)
            actions.append(action)
            is_randoms.append(is_random)
            dones.append(done)
            rewards.append(reward)

            obs = next_obs

        # Store last observation for non-resettable environments
        self._first_obs = obs

        # Add invalid action and reward for last state to pad arrays to same
        # length
        actions.append(np.zeros_like(action))
        is_randoms.append(False)
        rewards.append(np.zeros_like(reward))

        episode = dict(s=np.expand_dims(np.array(states, dtype=np.float32),
                                        axis=0),
                       a=np.expand_dims(np.array(actions, dtype=np.float32),
                                        axis=0),
                       rand_a=np.array(is_randoms,
                                       dtype=np.float32).reshape(1, -1, 1),
                       d=np.array(dones, dtype=np.float32).reshape(1, -1, 1),
                       r=np.array(rewards, dtype=np.float32).reshape(1, -1, 1))
        stats = {
            'Reward': episode['r'][0, :-1].mean(),
            'Return': episode['r'][0, :-1].sum()
        }

        return episode, stats

    @property
    def example_transition(self) -> Dict[str, np.ndarray]:
        return {
            's': self._first_obs.astype(np.float32),
            'a': self._env.action_space.sample().astype(np.float32),
            'rand_a': np.zeros((1,), dtype=np.float32),
            'd': np.zeros((1,), dtype=np.float32),
            'r': np.zeros((1,), dtype=np.float32)
        }

    @property
    def transition_help(self) -> Dict[str, str]:
        return {
            's': 'Observation from environment',
            'a': 'Action performed by agent in reaction to observation',
            'rand_a': 'Flag indicating whether action was random',
            'd': 'Flag indicating whether current state is a terminal state',
            'r': 'Reward from environment for action'
        }

    @property
    def observation_space(self) -> 'gym.Space':
        return self._env.observation_space

    @property
    def action_space(self) -> 'gym.Space':
        return self._env.action_space


@gin.configurable(blacklist=['env'])
class GoalBasedRolloutGenerator(BaseRolloutGenerator):
    """RolloutGenerator for goal-based environments"""
    def __init__(self, env, store_contact_info=False, store_control_info=False,
                 store_noise=False, fps=None, store_from_info=None):
        self._env = env
        self._store_contact_info = store_contact_info
        self._store_control_info = store_control_info
        self._store_noise = store_noise
        self._action_dim = env.action_space.shape[0]
        self._first_obs = None
        self._fps = fps

        info_keys = store_from_info if store_from_info else {}
        if store_contact_info:
            info_keys['is_contact'] = 'contact'
        if store_control_info:
            info_keys['has_control'] = 'control'
        self._info_keys = info_keys

        self.reset()

    def reset(self):
        self._first_obs = self._env.reset()
        return self._first_obs

    def close(self):
        self._env.close()

    def rollout(self,
                agent,
                n_steps: int,
                evaluate: bool = False,
                render: bool = False) -> Tuple[Dict[str, np.ndarray],
                                               Dict[str, Any]]:
        """Perform a single rollout in the environment

        :param agent: Agent performing the rollout
        :param n_steps: Number of steps to perform, i.e. the number of actions
            that are taken
        :param evaluate: If `True`, run the agent in evaluation mode
        :param render: If `True`, render the environment before each step

        :return: Tuple. First entry is a dictionary containing observed the
            transitions. Shape of arrays is `(1, n_steps + 1, ...)`. Second
            entry is a dictionary with statistics about the rollout.
        """
        states = [self._first_obs['observation']]
        actions = []
        is_randoms = []
        dones = [False]
        rewards = []
        achieved_goals = [self._first_obs['achieved_goal']]
        desired_goal = [self._first_obs['desired_goal']]
        infos = collections.defaultdict(list)
        action_noises = []

        obs = self._first_obs
        for step in range(n_steps):
            if render:
                self._env.render()
                if self._fps is not None:
                    time.sleep(1 / self._fps)

            action, is_random = agent.get_action((states[-1], desired_goal[0]),
                                                 evaluate=evaluate)

            if self._store_noise:
                if len(action) > self._action_dim:
                    action_noises.append(action[self._action_dim:])
                    action = action[:self._action_dim]
                else:
                    action_noises.append(np.zeros_like(action))

            next_obs, reward, done, info = self._env.step(action)

            # We only record the done signal if it resulted from genuinely
            # being in a terminal state, and not from running into the time
            # limit. Gym's `TimeLimit` env wrapper class indicates this by
            # setting `info['TimeLimit.truncated'] == True`.
            if done and info.get('TimeLimit.truncated', False):
                done = False

            states.append(next_obs['observation'])
            actions.append(action)
            is_randoms.append(is_random)
            dones.append(done)
            rewards.append(reward)
            achieved_goals.append(next_obs['achieved_goal'])

            for key in self._info_keys:
                value = info.get(key)
                assert value is not None, \
                    (f'Storing `{key}` from info was requested, but there is '
                     f'no `{key}` in the environment\'s info dict')
                infos[key].append(value)

            obs = next_obs

        # Store last observation for non-resettable environments
        self._first_obs = obs

        # Add invalid action and reward for last state to pad arrays to same
        # length
        actions.append(np.zeros_like(action))
        action_noises.append(np.zeros_like(action))
        is_randoms.append(False)
        rewards.append(np.zeros_like(reward))
        for values in infos.values():
            values.append(False)

        episode = dict(s=np.expand_dims(np.array(states, dtype=np.float32),
                                        axis=0),
                       a=np.expand_dims(np.array(actions, dtype=np.float32),
                                        axis=0),
                       rand_a=np.array(is_randoms,
                                       dtype=np.float32).reshape(1, -1, 1),
                       d=np.array(dones, dtype=np.float32).reshape(1, -1, 1),
                       r=np.array(rewards, dtype=np.float32).reshape(1, -1, 1),
                       ag=np.expand_dims(np.array(achieved_goals,
                                                  dtype=np.float32), axis=0),
                       g=np.expand_dims(np.array(desired_goal * (n_steps + 1),
                                                 dtype=np.float32),
                                        axis=0))
        stats = {
            'Reward': episode['r'][0, :-1].mean(),
            'Return': episode['r'][0, :-1].sum(),
            'Success': info.get('is_success', False)
        }

        for key, mem_key in self._info_keys.items():
            values = infos[key]
            episode[mem_key] = np.array(infos[key],
                                        dtype=np.float32).reshape(1, -1, 1)
            stats[mem_key.capitalize()] = episode[mem_key][0, :-1].sum()

        if self._store_noise:
            episode['action_noise'] = np.array(action_noises,
                                               dtype=np.float32)[np.newaxis]

        return episode, stats

    @property
    def example_transition(self) -> Dict[str, np.ndarray]:
        ex = {
            's': self._first_obs['observation'].astype(np.float32),
            'a': self._env.action_space.sample().astype(np.float32),
            'rand_a': np.zeros((1,), dtype=np.float32),
            'd': np.zeros((1,), dtype=np.float32),
            'r': np.zeros((1,), dtype=np.float32),
            'ag': self._first_obs['achieved_goal'].astype(np.float32),
            'g': self._first_obs['desired_goal'].astype(np.float32)
        }

        for value in self._info_keys.values():
            ex[value] = np.zeros((1,), dtype=np.float32)

        if self._store_noise:
            ex['action_noise'] = ex['a']

        return ex

    @property
    def transition_help(self) -> Dict[str, str]:
        return {
            's': 'Observation from environment',
            'a': 'Action performed by agent in reaction to observation',
            'rand_a': 'Flag indicating whether action was random',
            'd': 'Flag indicating whether current state is a terminal state',
            'r': 'Reward from environment for action',
            'ag': 'Goal achieved in the current state',
            'g': 'Goal desired to achieve',
            'contact': 'Flag indicating whether agent had contact',
            'control': 'Flag indicating whether agent had control',
            'action_noise': 'Noise added to original action'
        }

    @property
    def observation_space(self) -> 'gym.Space':
        return self._env.observation_space

    @property
    def action_space(self) -> 'gym.Space':
        return self._env.action_space
