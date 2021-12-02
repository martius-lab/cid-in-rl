from collections import OrderedDict
from typing import Callable, Dict, Sequence, Set

import gin
import numpy as np
import torch
import torch.utils.data.dataloader


@gin.configurable(blacklist=['memory'])
class ForwardGoalDataset(torch.utils.data.Dataset):
    """Dataset using achieved goals of transition target as targets"""
    def __init__(self,
                 memory: Sequence[Dict[str, np.ndarray]],
                 include_action=True,
                 subtract_action_noise=False,
                 use_state_noise=False,
                 use_goal_diff_as_target=True,
                 use_full_state_as_target=False,
                 use_only_random_actions=False,
                 use_only_contacts=False,
                 use_only_no_contacts=False,
                 use_only_actions_smaller_than=None,
                 use_only_goal_diffs_larger_than=None,
                 state_transform=None,
                 target_transform=None,
                 target_scale=None,
                 state_noise_fn=None,
                 ag_noise_fn=None,
                 copy_memory=False):
        """
        :param memory: Sequence over elements of this dataset
        :param include_action: If `True`, add action to input
        :param subtract_action_noise: If `True`, subtract action noise from
            action
        :param use_state_noise: If `True`, get state noise from memory
        :param use_goal_diff_as_target: If `True`, return difference
            `achieved_goal_next - achieved_goal` as target
        :param use_full_state_as_target: If `True`, use full next state as
            target instead of goals
        :param use_only_random_actions: If `True`, the used sequence should
            only use random actions. This is enforced by returning a
            corresponding `sequence_filter`
        :param use_only_contacts: If `True`, the used sequence should only
            consist of contact transitions. This is enforced by returning a
            corresponding `sequence_filter`
        :param use_only_no_contacts: If `True`, the used sequence should only
            consist of no contact transitions. This is enforced by returning a
            corresponding `sequence_filter`
        :param use_only_actions_smaller_than: If a number, the used sequence
            should only consist of actions whose L2 norm is smaller equal to
            the number. This is enforced by returning a corresponding
            `sequence_filter`
        :param use_only_goal_diffs_larger_than: If a number, the used sequence
            should only consist of transitions where the L2 norm of the goal
            diff is larger equal to the number. This is enforced by returning a
            corresponding `sequence_filter`
        :param state_transform: Callable that receives the input state
            and returns a transformed version
        :param target_transform: Callable that receives the target and returns
            a transformed version
        :param state_noise_fn: Callable that receives state and noise and
            returns transformed state
        :param ag_noise_fn: Callable that receives goals and noise and
            returns transformed goal
        :param target_scale: Scalar value that target gets multiplied with
        :param copy_memory: Make a copy of memory to get a contiguous array
        """
        super().__init__()
        assert not (use_only_contacts and use_only_no_contacts), \
            ('`use_only_contacts` and `use_only_no_contacts` can not both be '
             'set to `True`')
        self._include_action = include_action
        self._subtract_action_noise = subtract_action_noise
        self._use_state_noise = use_state_noise
        self._use_goal_diff_as_target = use_goal_diff_as_target
        self._use_full_state_as_target = use_full_state_as_target
        self._use_only_random_actions = use_only_random_actions
        self._use_only_contacts = use_only_contacts
        self._use_only_no_contacts = use_only_no_contacts
        self._use_only_actions_smaller_than = use_only_actions_smaller_than
        self._use_only_goal_diffs_larger_than = use_only_goal_diffs_larger_than
        self._state_transform = state_transform
        self._target_transform = target_transform
        self._target_scale = target_scale
        if use_state_noise:
            self._state_noise_fn = state_noise_fn
            self._ag_noise_fn = ag_noise_fn
        else:
            self._state_noise_fn = None
            self._ag_noise_fn = None

        if copy_memory:
            self._memory = {key: np.stack([memory[idx][key]
                                           for idx in range(len(memory))])
                            for key in self.required_keys
                            if len(memory) > 0}
            self._memory_layout_by_index = False
        else:
            self._memory = memory
            self._memory_layout_by_index = True

    @property
    def shapes(self):
        example = self[0]
        return example[0].shape, example[1].shape

    @property
    def required_keys(self) -> Set[str]:
        keys = {'s'}

        if self._use_state_noise:
            keys.add('s_noise0')
            keys.add('s_noise1')

        if self._use_full_state_as_target:
            keys.add('s1')
        else:
            keys.add('ag1')
            if self._use_goal_diff_as_target:
                keys.add('ag0')

        if self._include_action:
            keys.add('a')
        if self._subtract_action_noise:
            keys.add('action_noise')

        return keys

    @property
    def sequence_filter(self):
        if (self._use_only_random_actions
                or self._use_only_contacts
                or self._use_only_no_contacts
                or self._use_only_actions_smaller_than is not None
                or self._use_only_goal_diffs_larger_than is not None):
            def _filter(buffers):
                shape = next(iter(buffers.values())).shape
                selection = np.ones(shape[:2], dtype=np.bool)
                if self._use_only_random_actions:
                    selection &= (buffers['rand_a'].astype(bool)
                                                   .squeeze(axis=-1))
                if self._use_only_contacts:
                    selection &= (buffers['contact'].astype(bool)
                                                    .squeeze(axis=-1))
                if self._use_only_no_contacts:
                    selection &= ~(buffers['contact'].astype(bool)
                                                     .squeeze(axis=-1))
                if self._use_only_actions_smaller_than is not None:
                    action_norm = np.linalg.norm(buffers['a'], ord=2, axis=-1)
                    selection &= (action_norm
                                  <= self._use_only_actions_smaller_than)
                if self._use_only_goal_diffs_larger_than is not None:
                    ag_next = np.roll(buffers['ag'], -1, axis=1)
                    goal_diff = ag_next - buffers['ag']
                    goal_diff_norm = np.linalg.norm(goal_diff, ord=2, axis=-1)
                    selection &= (goal_diff_norm
                                  >= self._use_only_goal_diffs_larger_than)

                return selection

            return _filter
        else:
            return None

    def __len__(self):
        if self._memory_layout_by_index:
            return len(self._memory)
        else:
            return len(self._memory['s'])

    def __getitem__(self, idx: int):
        if self._memory_layout_by_index:
            data = self._memory[idx]
            state = data['s']
            if self._use_state_noise:
                state_noise = data['s_noise0'][idx]
                state_next_noise = data['s_noise1'][idx]
            if self._include_action:
                action = data['a']
                if self._subtract_action_noise:
                    action_noise = data['action_noise']
            if self._use_full_state_as_target:
                state_next = data['s1']
            else:
                ag1 = data['ag1']
                if self._use_goal_diff_as_target:
                    ag0 = data['ag0']
        else:
            state = self._memory['s'][idx]
            if self._use_state_noise:
                state_noise = self._memory['s_noise0'][idx]
                state_next_noise = self._memory['s_noise1'][idx]
            if self._include_action:
                action = self._memory['a'][idx]
                if self._subtract_action_noise:
                    action_noise = self._memory['action_noise'][idx]
            if self._use_full_state_as_target:
                state_next = self._memory['s1'][idx]
            else:
                ag1 = self._memory['ag1'][idx]
                if self._use_goal_diff_as_target:
                    ag0 = self._memory['ag0'][idx]

        if self._state_transform is not None:
            state = self._state_transform(state)

        if self._state_noise_fn is not None:
            state = self._state_noise_fn(state, state_noise)

        if self._include_action:
            if self._subtract_action_noise:
                action = action - action_noise
            inp = np.concatenate((state, action), axis=-1)
        else:
            inp = state

        if self._use_full_state_as_target:
            target1 = state_next
            if self._state_noise_fn is not None:
                target1 = self._state_noise_fn(target1, state_next_noise)
            if self._use_goal_diff_as_target:
                target0 = state
        else:
            target1 = ag1
            if self._ag_noise_fn is not None:
                target1 = self._ag_noise_fn(target1, state_next_noise)
            if self._use_goal_diff_as_target:
                target0 = ag0
                if self._ag_noise_fn is not None:
                    target0 = self._ag_noise_fn(target0, state_noise)

        if self._use_goal_diff_as_target:
            target = target1 - target0
        else:
            target = target1

        if self._target_transform is not None:
            target = self._target_transform(target)

        if self._target_scale is not None:
            target = target * self._target_scale

        return inp.astype(np.float32), target.astype(np.float32)


@gin.configurable(blacklist=['memory'])
class FactorizedForwardDataset(torch.utils.data.Dataset):
    """Dataset with factorized grouping of the state

    Uses the full next state as target.
    """
    def __init__(self,
                 memory: Sequence[Dict[str, np.ndarray]],
                 factorizer: Callable[[np.ndarray], Dict[str, np.ndarray]],
                 target_factorizer: Callable[[np.ndarray],
                                             Dict[str, np.ndarray]] = None,
                 include_action=True,
                 use_state_noise=False,
                 use_state_diff_as_target=True,
                 use_only_random_actions=False,
                 target_keys_postfix='',
                 unwrap_target=False,
                 target_scale=None,
                 state_noise_fn=None,
                 copy_memory=False):
        """
        :param memory: Sequence over elements of this dataset
        :param factorizer: Callable that returns dictionary of named
            state groups
        :param target_factorizer: Callable that returns dictionary of named
            state groups for target variable. If `None`, use `factorizer`.
        :param include_action: If `True`, add action to input
        :param use_state_noise: If `True`, get state noise from memory
        :param use_state_diff_as_target: If `True`, return difference
            `achieved_goal_next - achieved_goal` as target
        :param use_only_random_actions: If `True`, the used sequence should
            only use random actions. This is enforced by returning a
            corresponding `sequence_filter`
        :param target_keys_postfix: String to append to keys of the target
        :param target_scale: Scalar value that target gets multiplied with
        :param unwrap_target: Turn target dictionary into vector
        :param state_noise_fn: Callable that receives state and noise and
            returns transformed state
        :param copy_memory: Make a copy of memory to get a contiguous array
        """
        super().__init__()
        self._memory = memory
        self._factorizer = factorizer
        if target_factorizer is not None:
            self._target_factorizer = target_factorizer
        else:
            self._target_factorizer = factorizer
        self._include_action = include_action
        self._use_state_noise = use_state_noise
        self._use_state_diff_as_target = use_state_diff_as_target
        self._use_only_random_actions = use_only_random_actions
        self._target_keys_postfix = target_keys_postfix
        self._unwrap_target = unwrap_target
        self._target_scale = target_scale if target_scale is not None else 1.0
        if use_state_noise:
            self._state_noise_fn = state_noise_fn
        else:
            self._state_noise_fn = None

        if copy_memory:
            self._memory = {key: np.stack([memory[idx][key]
                                           for idx in range(len(memory))])
                            for key in self.required_keys
                            if len(memory) > 0}
            self._memory_layout_by_index = False
        else:
            self._memory = memory
            self._memory_layout_by_index = True

    @property
    def shapes(self):
        example = self[0]
        inp_shapes = {name: val.shape for name, val in example[0].items()}

        if self._unwrap_target:
            target_shapes = example[1].shape
        else:
            target_shapes = {name: val.shape
                             for name, val in example[1].items()}

        return inp_shapes, target_shapes

    @property
    def required_keys(self) -> Set[str]:
        keys = {'s0', 's1', 'a'}

        if self._use_state_noise:
            keys.add('s_noise0')
            keys.add('s_noise1')

        return keys

    @property
    def sequence_filter(self):
        if self._use_only_random_actions:
            def _filter(buffers):
                shape = next(iter(buffers.values())).shape
                selection = np.ones(shape[:2], dtype=np.bool)
                if self._use_only_random_actions:
                    selection &= (buffers['rand_a'].astype(bool)
                                                   .squeeze(axis=-1))

                return selection

            return _filter
        else:
            return None

    def __len__(self):
        if self._memory_layout_by_index:
            return len(self._memory)
        else:
            return len(self._memory['s0'])

    def __getitem__(self, idx: int):
        if self._memory_layout_by_index:
            data = self._memory[idx]
            state = data['s0']
            state_next = data['s1']
            if self._use_state_noise:
                state_noise = data['s_noise0'][idx]
                state_next_noise = data['s_noise1'][idx]
            if self._include_action:
                action = data['a']
        else:
            state = self._memory['s0'][idx]
            state_next = self._memory['s0'][idx]
            if self._use_state_noise:
                state_noise = self._memory['s_noise0'][idx]
                state_next_noise = self._memory['s_noise1'][idx]
            if self._include_action:
                action = self._memory['a'][idx]

        if self._state_noise_fn is not None:
            state = self._state_noise_fn(state, state_noise)
            state_next = self._state_noise_fn(state_next, state_next_noise)

        state_unfactorized = state
        state = self._factorizer(state)
        state_next = self._target_factorizer(state_next)

        if self._use_state_diff_as_target:
            state_as_target = self._target_factorizer(state_unfactorized)
            target = {name + self._target_keys_postfix:
                      (self._target_scale
                       * (state_next[name] - state_as_target[name]))
                      for name in state_next}
        else:
            target = {name + self._target_keys_postfix:
                      self._target_scale * value
                      for name, value in state_next.items()}

        if self._include_action:
            state['a'] = action

        inp = state

        if self._unwrap_target:
            target = np.concatenate([v for v in target.values()], axis=0)

        return inp, target


@gin.configurable(blacklist=['memory'])
class FactorizedDataset(torch.utils.data.Dataset):
    """Dataset with factorized grouping of the state

    Input and target are specified by keys.
    """
    def __init__(self,
                 memory: Sequence[Dict[str, np.ndarray]],
                 inp_keys: Sequence[str],
                 target_keys: Sequence[str],
                 next_keys_postfix: str,
                 factorizer: Callable[[np.ndarray], Dict[str, np.ndarray]],
                 next_factorizer: Callable[[np.ndarray],
                                           Dict[str, np.ndarray]] = None,
                 use_only_random_actions=False,
                 extra_dataset_keys: Sequence[str]=None,
                 unwrap_target=False):
        super().__init__()
        self._memory = memory
        self._inp_keys = frozenset(inp_keys)
        self._target_keys = frozenset(target_keys)
        self._next_keys_postfix = next_keys_postfix
        self._factorizer = factorizer
        self._next_factorizer = next_factorizer
        self._use_only_random_actions = use_only_random_actions
        self._unwrap_target = unwrap_target

        required_keys = {'s0', 's1', 'a'}
        if extra_dataset_keys is not None:
            required_keys |= set(extra_dataset_keys)
        self._required_keys = frozenset(required_keys)

    @property
    def shapes(self):
        example = self[0]
        assert all(key in example[0] for key in self._inp_keys)
        inp_shapes = {name: val.shape for name, val in example[0].items()}

        if self._unwrap_target:
            target_shapes = example[1].shape
        else:
            assert all(key in example[1] for key in self._target_keys)
            target_shapes = {name: val.shape
                             for name, val in example[1].items()}

        return inp_shapes, target_shapes

    @property
    def required_keys(self) -> Set[str]:
        return self._required_keys

    @property
    def sequence_filter(self):
        if self._use_only_random_actions:
            def _filter(buffers):
                shape = next(iter(buffers.values())).shape
                selection = np.ones(shape[:2], dtype=np.bool)
                if self._use_only_random_actions:
                    selection &= (buffers['rand_a'].astype(bool)
                                                   .squeeze(axis=-1))
                return selection
            return _filter
        else:
            return None

    def __len__(self):
        return len(self._memory)

    def _build_dict(self, keys, state, state_next, extra_data):
        res = OrderedDict()
        for key, val in state.items():
            if key in keys:
                res[key] = val
        if state_next is not None:
            for key, val in state_next.items():
                next_key = key + self._next_keys_postfix
                if next_key in keys:
                    res[next_key] = val
        for key, val in extra_data.items():
            if key in keys:
                res[key] = val

        return res

    def __getitem__(self, idx: int):
        data = self._memory[idx]

        state = self._factorizer(data['s0'])

        if self._next_factorizer is not None:
            state_next = self._next_factorizer(data['s1'])
        else:
            state_next = None

        inp = self._build_dict(self._inp_keys, state, state_next, data)
        target = self._build_dict(self._target_keys, state, state_next, data)

        if self._unwrap_target:
            target = np.concatenate([v for v in target.values()], axis=0)

        return inp, target


class SeededRandomSampler(torch.utils.data.sampler.Sampler):
    """Sampler that follows Pytorch's RandomSampler, but is seeded"""
    def __init__(self, data_source, seed=0):
        self.data_source = data_source
        self._num_samples = None
        self._rng = np.random.RandomState(seed)

    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        return iter(self._rng.permutation(n))

    def __len__(self):
        return self.num_samples


class SeededWeightedRandomSampler(torch.utils.data.sampler.Sampler):
    """Sampler that follows Pytorch's WeightedRandomSampler, but is seeded"""
    def __init__(self, weights, num_samples, replacement=True, seed=0):
        self._weights = np.array(weights, dtype=np.float64)
        self._weights /= np.sum(self._weights)
        self._num_samples = num_samples
        self._replacement = replacement
        self._rng = np.random.RandomState(seed)

    def __iter__(self):
        return iter(self._rng.choice(len(self._weights),
                                     self._num_samples,
                                     replace=self._replacement,
                                     p=self._weights))

    def __len__(self):
        return self._num_samples
