import abc
from typing import Dict

import numpy as np
import sklearn.metrics

from cid.memory.episodic import EpisodicReplayMemory


class TransitionScorer(abc.ABC):
    """Interface for models that compute scores for `ScoreBasedReplayMemory`"""
    _default_memory_to_scorer_keys = {'s': 'states',
                                      'a': 'actions',
                                      'ag': 'achieved_goals',
                                      'ag1': 'achieved_goals_next'}

    @property
    def memory_to_scorer_keys(self) -> Dict[str, str]:
        return TransitionScorer._default_memory_to_scorer_keys

    @abc.abstractmethod
    def __call__(self, **kwargs) -> np.ndarray:
        ...


class ScoreBasedReplayMemory(EpisodicReplayMemory):
    """Base class for memories that use a model to score transitions"""
    def __init__(self, example, size, episode_len, model: TransitionScorer,
                 score_threshold=0.0):
        """
        :param model: Model that given a transition, produces a score
        :param score_threshold: Threshold above which a transition is
            considered to be an influence transition
        """
        example['score'] = np.empty((1,), dtype=np.float64)
        super().__init__(example, size, episode_len, goal_based=True)
        self._scorer = model
        self._score_threshold = score_threshold
        self._mem_to_scorer_keys = model.memory_to_scorer_keys

        self._episode_scores = None
        self._total_score = 0

    def store_episodes(self, episodes):
        transitions = {key: episodes[key]
                       for key in self._mem_to_scorer_keys
                       if key in episodes}
        scores = score_transitions(transitions,
                                   self._scorer,
                                   self._mem_to_scorer_keys)
        episodes['score'] = scores
        super().store_episodes(episodes)

        scores = self._buffer['score'][:self._current_size]
        self._episode_scores = np.sum(scores, axis=1).squeeze(-1)
        self._total_score = np.sum(self._episode_scores)

    def rescore_transitions(self, batch_size=128, n_latest_episodes=None,
                            verbose=False):
        """Recompute scores for transitions in buffer

        :param batch_size: Number of episodes to run through scorer at once
        :param n_latest_episodes: If not `None`, only recompute scores for
            this many newest episodes
        :param verbose: Print a progress bar
        """
        self._total_score = 0

        if n_latest_episodes is None:
            n_latest_episodes = self._current_size

        n_latest_episodes = min(n_latest_episodes, self._current_size)
        if (self._current_size == self._size
                and self._next_idx - n_latest_episodes < 0):
            # Buffer is fully filled and wrap-around will happen
            start = self._size - (n_latest_episodes - self._next_idx)
        else:
            start = max(0, self._next_idx - n_latest_episodes)

        batch_size = min(batch_size, n_latest_episodes)

        if verbose:
            import tqdm
            indices = tqdm.trange(0, n_latest_episodes, batch_size,
                                  desc='Rescoring transitions')
        else:
            indices = range(0, n_latest_episodes, batch_size)

        for idx in indices:
            bs = min(idx + batch_size, n_latest_episodes) - idx
            pos = (start + idx) % self._size
            if pos + bs > self._size:
                # Wrap-around case
                remaining = pos + bs - self._size
                transitions = {key: np.concatenate(
                                (self._buffer[key][pos:pos + bs],
                                 self._buffer[key][:remaining]))
                               for key in self._mem_to_scorer_keys
                               if key in self._buffer}
                scores = score_transitions(transitions,
                                           self._scorer,
                                           self._mem_to_scorer_keys)
                self._buffer['score'][pos:pos + bs] = scores[:-remaining]
                self._buffer['score'][:remaining] = scores[-remaining:]
            else:
                transitions = {key: self._buffer[key][pos:pos + bs]
                               for key in self._mem_to_scorer_keys
                               if key in self._buffer}
                scores = score_transitions(transitions,
                                           self._scorer,
                                           self._mem_to_scorer_keys)
                self._buffer['score'][pos:pos + bs] = scores

        scores = self._buffer['score'][:self._current_size]
        self._episode_scores = np.sum(scores, axis=1).squeeze(-1)
        self._total_score = np.sum(self._episode_scores)

    @property
    def scores(self):
        return self._buffer['score'][:self._current_size, 1:].copy()

    def report_statistics(self):
        scores = self._buffer['score'][:self._current_size, 1:].reshape(-1)
        p50, p95, p99, p100 = np.percentile(scores, (50, 95, 99, 100))

        stats = super().report_statistics()
        stats['Memory/score_median'] = p50
        stats['Memory/score_p95'] = p95
        stats['Memory/score_p99'] = p99
        stats['Memory/score_max'] = p100

        scores = self._buffer['score'][:self._current_size, 1:].reshape(-1)
        over_threshold = scores > self._score_threshold

        if self._score_threshold > 0:
            n_over_threshold = np.sum(over_threshold)
            pos_predictions = over_threshold
            stats['Memory/over_threshold'] = n_over_threshold / len(scores)
        else:
            pos_predictions = None

        if 'contact' in self._buffer:
            contacts = self._buffer['contact'][:self._current_size, :-1]
            contacts = contacts.astype(np.bool).reshape(-1)
            _add_classification_stats(stats, scores, contacts,
                                      'Memory/contact', pos_predictions)
        if 'control' in self._buffer:
            controls = self._buffer['control'][:self._current_size, :-1]
            controls = controls.astype(np.bool).reshape(-1)
            _add_classification_stats(stats, scores, controls,
                                      'Memory/control', pos_predictions)

        return stats

    def load(self, path: str, rescore_batch_size=128,
             rescore=False, verbose=False):
        super().load(path)

        if 'score' not in self._buffer or rescore:
            buf = np.zeros((self._size, self.episode_len, 1), dtype=np.float64)
            self._buffer['score'] = buf
            self.rescore_transitions(rescore_batch_size, verbose=verbose)


def score_transitions(transitions: Dict[str, np.ndarray],
                      scorer: TransitionScorer,
                      mem_to_scorer_keys: Dict[str, str]):
    first_elem = next(iter(transitions.values()))
    batch_size = first_elem.shape[0]
    episode_len = first_elem.shape[1]

    if 's1' in mem_to_scorer_keys:
        transitions['s1'] = np.roll(transitions['s'], shift=-1, axis=1)

    if 'ag1' in mem_to_scorer_keys:
        transitions['ag1'] = np.roll(transitions['ag'], shift=-1, axis=1)

    transitions = {
        mapped_key: transitions[key].reshape(batch_size * episode_len, -1)
        for key, mapped_key in mem_to_scorer_keys.items()
    }

    scores = scorer(**transitions)
    # Note that we also computed a score for the last state, which does not
    # consist of a full transition. We ignore this score here.
    scores = np.roll(scores.reshape(-1, episode_len, 1), shift=1, axis=1)
    scores[:, 0] = 0.0

    return scores


def _add_classification_stats(stats, scores, labels, prefix,
                              pos_predictions=None):
    n_positives = np.sum(labels)
    if n_positives == 0:
        roc_auc = float('nan')
        ap = float('nan')
    elif n_positives == len(labels):
        roc_auc = float('nan')
        ap = 1.0
    else:
        roc_auc = sklearn.metrics.roc_auc_score(labels, scores)
        ap = sklearn.metrics.average_precision_score(labels, scores)
    stats['{}_roc'.format(prefix)] = roc_auc
    stats['{}_ap'.format(prefix)] = ap

    if pos_predictions is not None:
        n_pred_pos = np.sum(pos_predictions)
        tp = pos_predictions & labels
        prec = np.sum(tp) / n_pred_pos if n_pred_pos > 0 else float('nan')
        rec = np.sum(tp) / n_positives if n_positives > 0 else float('nan')
        stats['{}_prec'.format(prefix)] = prec
        stats['{}_rec'.format(prefix)] = rec
