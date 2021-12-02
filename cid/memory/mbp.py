import enum
from typing import Callable, Tuple, Union

import gin
import numpy as np
import torch
from scipy.special import softmax

from cid.memory.score_based import (ScoreBasedReplayMemory,
                                    TransitionScorer)

FloatTupleT = Tuple[float, float]


@gin.configurable(blacklist=['example', 'episode_len'])
class MBPReplayMemory(ScoreBasedReplayMemory):
    """Memory that implements model-based prioritization with HER"""
    def __init__(self, example, size, episode_len,
                 model: TransitionScorer,
                 reward_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 p_replay=0.8,
                 sampling_coldness=1,
                 prioritize_episodes=False,
                 prioritize_transitions=True,
                 episode_score_is_max=False,
                 p_forward_relabeling=0.0,
                 only_bonus_reward=False,
                 bonus_type='additive',
                 influence_bonus=0.0,
                 score_threshold=0.0,
                 score_epsilon=0.0,
                 bonus_score_max=10,
                 reward_max=None,
                 softmax_dist=False,
                 ranking_dist=False,
                 importance_weights=False,
                 importance_weights_exponent: Union[float, FloatTupleT] = 1.0):
        """
        :param model: Model that given a transition, produces a score
        :param reward_fn: Function taking (achieved_goals, goals) and returning
            the reward
        :param p_replay: For each item of the batch, probability of
            replaying a goal for that item
        :param sampling_coldness: Inverse temperature of distribution used for
            sampling. Higher temperatures put less weight on the individual
            scores, lower temperatures put more weight on the scores. Coldness
            zero corresponds to a uniform distribution.
        :param prioritize_episodes: If `True`, prioritize which episodes
            are sampled from this memory. Episodes are sampled proportionally
            according to the sum of scores
        :param prioritize_transitions: If `True`, prioritize which transitions
            are sampled for replaying. Transitions are sampled proportionally
            according to their score.
        :param p_forward_relabeling: For each relabeled sample, probability
            to choose a state based on the score compared to choosing a goal
            based on the score
        :param episode_score_is_max: If `True`, the total score of an episode
            is computed based on the maximum transition score. If `False`, the
            score is the sum of transition scores.
        :param only_bonus_reward: If `True`, the reward consists only of the
            bonus reward. No task rewards are given.
        :param bonus_type: Type of bonus to give. `flat` for discrete valued
            bonuses for influence states, `additive` for continuous valued
            ones based on the score that get added to the reward,
            `multiplicative_neg` for continuous valued ones that get multiplied
            to negative rewards
        :param influence_bonus: If bonus type is `flat`, then if this value
            is greater zero, each sampled transition receives a bonus reward
            equal to this argument's height, if the transition is an influence
            transition according to `score_threshold`. If bonus type is
            `additive`, each sampled transition receives a bonus reward of its
            score weighted by this argument.
        :param score_epsilon: Value which is added to scores to prevent
            starvation of states with zero score
        :param bonus_score_max: Maximum value score can have for bonus
            computation
        :param reward_max: Maximum reward to return. If `None`, no clipping
            is performed
        :param softmax_dist: Whether to use a softmax distribution for sampling
        :param ranking_dist: Whether to use ranking for the sampling
            distribution
        :param importance_weights: Whether to compute and return importance
            sampling weights which can be used to correct for the changed
            transition sampling distribution
        :param importance_weight_exponent: Exponent to raise importance weights
            by. Value can be linearly annealed by specifying a tuple of
            `(start_value, end_value, period)`, where `period` specifies the
            total number of episodes stored in the buffer upon reaching the
            end of annealing
        """
        super().__init__(example, size, episode_len, model, score_threshold)
        self._reward_fn = reward_fn
        self._p_replay = p_replay
        self._sampling_coldness = sampling_coldness
        self._prioritize_episodes = prioritize_episodes
        self._prioritize_transitions = prioritize_transitions
        self._p_forward_relabeling = p_forward_relabeling
        self._episode_score_is_max = episode_score_is_max
        self._only_bonus_reward = only_bonus_reward
        self._score_eps = score_epsilon
        self._bonus_score_max = bonus_score_max
        self._reward_max = reward_max
        self._softmax_dist = softmax_dist
        self._ranking_dist = ranking_dist
        assert not (softmax_dist and ranking_dist), \
            'Either `softmax_dist` or `ranking_dist` can be selected'
        self._is_weights = importance_weights
        if importance_weights and self._prioritize_transitions:
            raise NotImplementedError(('Importance weights only implemented '
                                       'for episode prioritization'))
        if isinstance(importance_weights_exponent, tuple):
            start, end, period = importance_weights_exponent
            self._is_weight_exponent = \
                lambda n: np.clip(start + n * (end - start) / period,
                                  start, end)
        else:
            self._is_weight_exponent = lambda n: importance_weights_exponent

        if bonus_type == 'none' or influence_bonus == 0:
            self._bonus_type = _BonusType.NONE
        elif bonus_type == 'flat':
            self._bonus_type = _BonusType.FLAT
        elif bonus_type == 'additive':
            self._bonus_type = _BonusType.ADDITIVE
        elif bonus_type == 'multiplicative_neg':
            self._bonus_type = _BonusType.MULTIPLICATIVE_NEG
        else:
            raise ValueError(f'Unknown bonus type `{bonus_type}`')

        if isinstance(influence_bonus, tuple):
            start, end, period = influence_bonus
            min_val = min(start, end)
            max_val = max(start, end)
            self._influence_bonus = \
                lambda n: np.clip(start + n * (end - start) / period,
                                  min_val, max_val)
        else:
            self._influence_bonus = lambda n: influence_bonus

        # For reporting statistics
        self._last_report = 0
        self._bonus_sum = 0
        self._n_samples = 0

    def sample_batch(self, batch_size: int):
        assert self._current_size > 0, 'Trying to sample from empty buffer'

        # Sample on how many states the goal is replaced with one from the
        # episode. This many of the first samples of the batch have their
        # goal replaced then.
        n_her_samples = np.random.binomial(batch_size, self._p_replay)
        her_selection = slice(0, n_her_samples)

        scores = self._buffer['score'][:self._current_size].squeeze()

        if self._prioritize_episodes:
            p_episodes = self._get_episode_weights()
            # Sample episodes proportional to sum of scores on episode
            E_idxs = np.random.choice(self._current_size,
                                      size=batch_size,
                                      replace=True,
                                      p=p_episodes)
        else:
            p_episodes = None
            E_idxs = np.random.randint(low=0,
                                       high=self._current_size,
                                       size=batch_size)

        if self._prioritize_transitions:
            if p_episodes is None:
                p_episodes = self._get_episode_weights()

            T_idxs = np.random.randint(low=0,
                                       high=self._episode_len - 1,
                                       size=batch_size)

            n_forw_samples = np.random.binomial(n_her_samples,
                                                self._p_forward_relabeling)
            forw_selection = slice(0, n_forw_samples)
            res = self._sample_states_by_score(scores,
                                               n_forw_samples,
                                               p_episodes)
            her_E_idxs, her_T_idxs, forw_goal_idxs = res
            E_idxs[forw_selection] = her_E_idxs
            T_idxs[forw_selection] = her_T_idxs

            n_backw_samples = n_her_samples - n_forw_samples
            backw_selection = slice(n_forw_samples, n_her_samples)
            res = self._sample_goals_by_score(scores,
                                              n_backw_samples,
                                              p_episodes)
            her_E_idxs, her_T_idxs, backw_goal_idxs = res
            E_idxs[backw_selection] = her_E_idxs
            T_idxs[backw_selection] = her_T_idxs

            goal_idxs = np.concatenate((forw_goal_idxs, backw_goal_idxs))
        else:
            T_idxs = np.random.randint(low=0,
                                       high=self._episode_len - 1,
                                       size=batch_size)

            # Sample a goal in the future of the episode
            goal_idxs = np.random.randint(low=T_idxs[her_selection] + 1,
                                          high=self._episode_len)

        # Replace goals
        goals = self._buffer['g'][E_idxs, T_idxs]
        if len(goal_idxs) > 0:
            goals[her_selection] = self._buffer['ag'][E_idxs[her_selection],
                                                      goal_idxs]

        achieved_goals_next = self._buffer['ag'][E_idxs, T_idxs + 1]

        rewards = self._reward_fn(achieved_goals_next, goals).squeeze()

        if self._bonus_type != _BonusType.NONE:
            influence_bonus = self._influence_bonus(self._total_episodes)
            T_scores = scores[E_idxs, T_idxs + 1]
            if self._bonus_type == _BonusType.FLAT:
                T_influence = (T_scores > self._score_threshold)
                bonus = influence_bonus * T_influence.astype(np.float32)
            elif self._bonus_type == _BonusType.ADDITIVE:
                clipped_scores = np.clip(T_scores, 0, self._bonus_score_max)
                bonus = influence_bonus * clipped_scores
            elif self._bonus_type == _BonusType.MULTIPLICATIVE_NEG:
                T_p = np.clip(T_scores / self._episode_scores[E_idxs],
                              0, self._bonus_score_max)
                bonus_factor = influence_bonus * T_p
                bonus = -rewards * bonus_factor

            if self._only_bonus_reward:
                rewards = bonus
            else:
                rewards += bonus
            self._bonus_sum += bonus.sum()
            self._n_samples += len(rewards)

        if self._reward_max is not None:
            rewards = np.clip(rewards, a_min=None, a_max=self._reward_max)

        if self._is_weights:
            exp = self._is_weight_exponent(self._total_episodes)
            weights = (1 / (self._current_size * p_episodes))**exp
            batch_weights = weights[E_idxs] / np.max(weights)
            # Add weights to rewards in a separate dimension
            rewards = np.stack((rewards, batch_weights), axis=0)

        batch = {
            's0': self._buffer['s'][E_idxs, T_idxs],
            's1': self._buffer['s'][E_idxs, T_idxs + 1],
            'a': self._buffer['a'][E_idxs, T_idxs],
            'r': rewards,
            'd': self._buffer['d'][E_idxs, T_idxs + 1].squeeze(),
            'g': goals
        }

        return batch

    def _get_episode_weights(self):
        if self._episode_score_is_max:
            scores = self._buffer['score'][:self._current_size].squeeze()
            scores = np.max(scores, axis=1)
        else:
            scores = self._episode_scores

        if self._score_eps > 0:
            scores = self._episode_scores + self._score_eps

        if self._softmax_dist:
            p = softmax(scores * self._sampling_coldness)
        elif self._ranking_dist:
            order = scores.argsort()[::-1]
            inverse_ranks = 1 / (order.argsort() + 1)
            p_unnormalized = inverse_ranks**self._sampling_coldness
            p = p_unnormalized / np.sum(p_unnormalized)
        else:
            p_unnormalized = scores**self._sampling_coldness
            p_sum = np.sum(p_unnormalized)
            if p_sum == 0:
                p = p_unnormalized + 1 / len(scores)
            else:
                p = p_unnormalized / p_sum

        return p

    def _sample_states_by_score(self, scores, n_samples, p_episodes):
        E_idxs = np.random.choice(self._current_size,
                                  size=n_samples,
                                  replace=True,
                                  p=p_episodes)

        # Shift scores backwards by 1 along transition axis to align scores
        # with states and remove the last state as it should not be sampled
        scores = scores[E_idxs, 1:]
        if self._softmax_dist:
            p_states = softmax(scores * self._sampling_coldness)
        else:
            p_states = scores / self._episode_scores[E_idxs, np.newaxis]

        dist = torch.distributions.Categorical(torch.from_numpy(p_states))
        T_idxs = dist.sample().numpy()

        # Sample a goal from the future of the episode
        goal_idxs = np.random.randint(low=T_idxs + 1,
                                      high=self._episode_len)

        return E_idxs, T_idxs, goal_idxs

    def _sample_goals_by_score(self, scores, n_samples, p_episodes):
        E_idxs = np.random.choice(self._current_size,
                                  size=n_samples,
                                  replace=True,
                                  p=p_episodes)

        # Shift scores backwards by 1 along transition axis to remove first
        # state as it should not be sampled as a goal
        scores = scores[E_idxs, 1:]
        if self._softmax_dist:
            p_states = softmax(scores * self._sampling_coldness)
        else:
            p_states = scores / self._episode_scores[E_idxs, np.newaxis]

        dist = torch.distributions.Categorical(torch.from_numpy(p_states))
        goal_idxs = dist.sample().numpy() + 1

        # Sample a state before the goal on the episode
        T_idxs = np.random.randint(low=0, high=goal_idxs)

        return E_idxs, T_idxs, goal_idxs

    def report_statistics(self):
        stats = super().report_statistics()

        if self._prioritize_episodes:
            p_episodes = self._get_episode_weights()
            p50, p99, p100 = np.percentile(p_episodes, (50, 99, 100))
            stats['Memory/p_eps'] = np.mean(p_episodes)
            stats['Memory/p_eps_median'] = p50
            stats['Memory/p_eps_p99'] = p99
            stats['Memory/p_eps_max'] = p100

        if self._bonus_type != _BonusType.NONE:
            if self._bonus_type == _BonusType.MULTIPLICATIVE_NEG:
                scores = self._buffer['score'][:self._current_size, 1:, 0]
                T_p = scores / self._episode_scores[:, np.newaxis]
                p50, p99, p100 = np.percentile(T_p, (50, 99, 100))
                stats['Memory/p_ts'] = np.mean(T_p)
                stats['Memory/p_ts_median'] = p50
                stats['Memory/p_ts_p99'] = p99
                stats['Memory/p_ts_max'] = p100
            elif self._bonus_type == _BonusType.ADDITIVE:
                # Compute average bonus on new episodes since last reporting
                influence_bonus = self._influence_bonus(self._total_episodes)
                start = self._last_report % self._size
                end = self._total_episodes % self._size
                scores = self._buffer['score'][start:end, 1:, 0]
                clipped_scores = np.clip(scores, 0, self._bonus_score_max)
                bonus = influence_bonus * clipped_scores
                bonus_per_episode = np.sum(bonus, axis=-1)
                stats['Memory/bonus'] = np.mean(bonus)
                stats['Memory/bonus_per_episode'] = np.mean(bonus_per_episode)

            n_samples = max(self._n_samples, 1)
            stats['Memory/bonus_given'] = self._bonus_sum / n_samples

            self._bonus_sum = 0
            self._n_samples = 0

        self._last_report = self._total_episodes

        return stats


class _BonusType(enum.Enum):
    NONE = 0
    FLAT = 1
    ADDITIVE = 2
    MULTIPLICATIVE_NEG = 3
