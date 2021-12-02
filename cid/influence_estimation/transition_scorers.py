import collections
import enum
from typing import Callable, Dict

import gin
import numpy as np
import torch

import cid.memory.score_based
from cid import models as models_
from cid.influence_estimation.kl_torch import (gaussian_entropy,
                                               kl_div,
                                               kl_div_mixture_app)


class _KLType(enum.Enum):
    MEAN_APPROX = 1
    VAR_PROD_APPROX = 2
    VAR_PROD_APPROX_SANDWICH = 3


@gin.configurable
class CMIScorer(cid.memory.score_based.TransitionScorer):
    def __init__(self,
                 full_model: torch.nn.Module,
                 n_expectation_samples=64,
                 n_mixture_samples=64,
                 reuse_action_samples=False,
                 kl_type='var_prod_approx',
                 threshold_zero=False,
                 **kwargs):
        self._model = full_model
        self._n_expectation_samples = n_expectation_samples
        self._n_mixture_samples = n_mixture_samples
        self._reuse_action_samples = reuse_action_samples
        if reuse_action_samples:
            assert n_expectation_samples == n_mixture_samples
        self._threshold_zero = threshold_zero

        if kl_type == 'mean_approx':
            self._kl_type = _KLType.MEAN_APPROX
        elif kl_type == 'var_prod_approx':
            self._kl_type = _KLType.VAR_PROD_APPROX
        elif kl_type == 'var_prod_approx_sandwich':
            self._kl_type = _KLType.VAR_PROD_APPROX_SANDWICH
        else:
            raise ValueError(f'Unknown KL type {kl_type}')

    @property
    def memory_to_scorer_keys(self) -> Dict[str, str]:
        return {'s': 'states', 'a': 'actions'}

    def _eval_model(self, states, actions):
        bs = len(states)
        n_actions = actions.shape[1]
        states = (states.unsqueeze(1)
                        .repeat(1, n_actions, 1)
                        .view(-1, states.shape[-1]))
        actions = actions.view(bs * n_actions, -1)

        states_and_actions = torch.cat((states, actions), dim=-1)
        res = self._model(states_and_actions)

        means = res[0].view(bs, n_actions, -1)
        variances = res[1].view(bs, n_actions, -1)

        return means, variances

    @staticmethod
    def _sample_actions(bs, n_actions, dim_actions):
        actions = np.random.rand(bs, n_actions, dim_actions) * 2 - 1
        return actions.astype(np.float32)

    @torch.no_grad()
    def action_scores(self, states: np.ndarray, actions: np.ndarray) \
            -> np.ndarray:
        """Get scores for each action

        Shape `states`: Batch x DimState
        Shape `actions`: Batch x Actions x DimAction
        """
        if self._model.training:
            self._model.eval()

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)

        means_full, vars_full = self._eval_model(states, actions)

        if self._reuse_action_samples:
            means_capped = means_full
            vars_capped = vars_full
        else:
            actions = self._sample_actions(len(states),
                                           self._n_mixture_samples,
                                           actions.shape[-1])
            actions = torch.from_numpy(actions)
            means_capped, vars_capped = self._eval_model(states, actions)

        if self._kl_type == _KLType.MEAN_APPROX:
            means_capped = torch.mean(means_capped, dim=1)
            vars_capped = torch.mean(vars_capped, dim=1)
            kls = kl_div(means_full, vars_full,
                         means_capped[:, None], vars_capped[:, None])
        elif self._kl_type == _KLType.VAR_PROD_APPROX:
            kls = kl_div_mixture_app(means_full,
                                     vars_full,
                                     means_capped[:, None],
                                     vars_capped[:, None])
        elif self._kl_type == _KLType.VAR_PROD_APPROX_SANDWICH:
            kls, kls_upper = kl_div_mixture_app(means_full,
                                                vars_full,
                                                means_capped[:, None],
                                                vars_capped[:, None],
                                                return_upper_bound=True)
            kls = 0.5 * (kls + kls_upper)

        scores = kls

        if self._threshold_zero:
            scores = np.clip(scores, a_min=0, a_max=None)

        return scores.numpy()

    def __call__(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        actions = self._sample_actions(len(states),
                                       self._n_expectation_samples,
                                       actions.shape[-1])

        kls = self.action_scores(states, actions)

        cmi = np.mean(kls, axis=1)

        return cmi


@gin.configurable
class EntropyScorer(cid.memory.score_based.TransitionScorer):
    def __init__(self,
                 full_model: torch.nn.Module,
                 n_expectation_samples=64,
                 threshold_zero=False,
                 **kwargs):
        self._model = full_model
        self._n_expectation_samples = n_expectation_samples
        self._threshold_zero = threshold_zero

    @property
    def memory_to_scorer_keys(self) -> Dict[str, str]:
        return {'s': 'states', 'a': 'actions'}

    def _eval_model(self, states, actions):
        bs = len(states)
        n_actions = actions.shape[1]
        states = (states.unsqueeze(1)
                        .repeat(1, n_actions, 1)
                        .view(-1, states.shape[-1]))
        actions = actions.view(bs * n_actions, -1)

        states_and_actions = torch.cat((states, actions), dim=-1)
        res = self._model(states_and_actions)

        means = res[0].view(bs, n_actions, -1)
        variances = res[1].view(bs, n_actions, -1)

        return means, variances

    @staticmethod
    def _sample_actions(bs, n_actions, dim_actions):
        return torch.rand(bs, n_actions, dim_actions) * 2 - 1

    def __call__(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        if self._model.training:
            self._model.eval()

        bs = len(states)
        dim_actions = actions.shape[-1]
        states = torch.from_numpy(states)

        with torch.no_grad():
            actions = self._sample_actions(bs, self._n_expectation_samples,
                                           dim_actions)
            means, variances = self._eval_model(states, actions)

            entropy = gaussian_entropy(means, variances).numpy()

        scores = np.mean(entropy, axis=1)

        if self._threshold_zero:
            scores = np.clip(scores, a_min=0, a_max=None)

        return scores


@gin.configurable
class MaskScorer(cid.memory.score_based.TransitionScorer):
    def __init__(self,
                 full_model: torch.nn.Module,
                 state_factorizer: Callable[[torch.Tensor],
                                            Dict[str, torch.Tensor]],
                 key_source: str = 'a',
                 key_target: str = 'o',
                 **kwargs):
        assert hasattr(full_model, 'get_mask'), \
            'MaskScorer needs a model supporting getting masks with `get_mask`'
        assert hasattr(full_model, 'get_input_index'), \
            'MaskScorer needs a model with `get_input_index`'
        assert hasattr(full_model, 'get_output_index'), \
            'MaskScorer needs a model with `get_output_index`'
        self._model = full_model
        self._factorizer = state_factorizer
        self._idx_source = full_model.get_input_index(key_source)
        self._idx_target = full_model.get_output_index(key_target)

    def __call__(self,
                 states: np.ndarray,
                 actions: np.ndarray,
                 achieved_goals: np.ndarray,
                 achieved_goals_next: np.ndarray):
        if self._model.training:
            self._model.eval()

        inp = self._factorizer(torch.from_numpy(states))
        inp['a'] = torch.from_numpy(actions)

        with torch.no_grad():
            mask = self._model.get_mask(inp).numpy()

        score = mask[:, self._idx_source, self._idx_target]

        return score


@gin.configurable
class ContactScorer(cid.memory.score_based.TransitionScorer):
    def __init__(self, contact_score=1, **kwargs):
        self._contact_score = contact_score

    @property
    def memory_to_scorer_keys(self) -> Dict[str, str]:
        return {'contact': 'contacts'}

    def __call__(self, contacts: np.ndarray) -> np.ndarray:
        return contacts * self._contact_score


@gin.configurable
class ControlScorer(cid.memory.score_based.TransitionScorer):
    def __init__(self, control_score=1, **kwargs):
        self._control_score = control_score

    @property
    def memory_to_scorer_keys(self) -> Dict[str, str]:
        return {'control': 'controls'}

    def __call__(self, controls: np.ndarray) -> np.ndarray:
        return controls * self._control_score


@gin.configurable
class VIMEScorer(cid.memory.score_based.TransitionScorer):
    def __init__(self, model: torch.nn.Module,
                 target_transform: Callable[[torch.Tensor],
                                            torch.Tensor] = None,
                 target_is_state_diff=True,
                 n_info_gain_update_steps=10,
                 info_gain_batch_size=10,
                 max_inf_gain=1000,
                 normalize_scores=False,
                 normalize_q_size=100,
                 normalize_n_rollouts=10,
                 **kwargs):
        self._model = model
        self._target_transform = target_transform
        self._target_is_state_diff = target_is_state_diff
        self._n_info_gain_update_steps = n_info_gain_update_steps
        self._batch_size = info_gain_batch_size
        self._max_inf_gain = max_inf_gain
        self._normalize_scores = normalize_scores
        if normalize_scores:
            self._median_queue = collections.deque(maxlen=normalize_q_size)
            self._normalize_n_rollouts = normalize_n_rollouts
            self._last_scores = collections.deque(maxlen=normalize_n_rollouts)

    @property
    def memory_to_scorer_keys(self) -> Dict[str, str]:
        return {'s': 'states', 'a': 'actions', 's1': 'states_next'}

    def eval_info_gain(self, inputs: torch.Tensor, targets: torch.Tensor):
        self._model.save_old_params()

        for _ in range(self._n_info_gain_update_steps):
            self._model.train_for_info_gain(inputs, targets)

        with torch.no_grad():
            info_gain = self._model.kl_div_new_old().numpy()

        self._model.reset_to_old_params()

        return np.clip(info_gain, 0, self._max_inf_gain)

    def __call__(self,
                 states: np.ndarray,
                 actions: np.ndarray,
                 states_next: np.ndarray):
        if not self._model.training:
            self._model.train()

        if self._target_is_state_diff:
            target = states_next - states
        else:
            target = states_next

        if self._target_transform is not None:
            target = self._target_transform(target)

        states = torch.from_numpy(states)
        actions = torch.from_numpy(actions)
        target = torch.from_numpy(target)

        inp = torch.cat((states, actions), dim=-1)

        scores = np.zeros(len(states))
        for idx in range(0, len(states), self._batch_size):
            info_gain = self.eval_info_gain(inp[idx:idx + self._batch_size],
                                            target[idx:idx + self._batch_size])
            scores[idx:idx + self._batch_size] = info_gain

        if self._normalize_scores:
            self._last_scores.append(scores)
            median = np.median(np.concatenate(self._last_scores))
            self._median_queue.append(median)

            scores = scores / np.mean(np.asarray(self._median_queue))

        return scores


@gin.configurable
class EnsembleDisagreementScorer(cid.memory.score_based.TransitionScorer):
    """Scorer that uses variance in model prediction as the score"""
    def __init__(self, model: torch.nn.Module = None):
        assert isinstance(model, models_.Ensemble)
        self._model = model

    @property
    def memory_to_scorer_keys(self) -> Dict[str, str]:
        return {'s': 'states', 'a': 'actions'}

    def __call__(self,
                 states: np.ndarray,
                 actions: np.ndarray) -> np.ndarray:
        if self._model.training:
            self._model.eval()

        states_and_actions = np.concatenate((states, actions), axis=-1)
        states_and_actions = torch.from_numpy(states_and_actions)
        states = torch.from_numpy(states)

        with torch.no_grad():
            pred = self._model(states_and_actions)
            if isinstance(pred, tuple):
                pred = pred[0]

            # pred has shape E x B x D
            mean_pred = torch.mean(pred, axis=0)

            ens_mse = torch.mean((pred - mean_pred[None])**2, axis=-1)
            mean_mse = torch.mean(ens_mse, axis=0).numpy()

        score = mean_mse

        return score
