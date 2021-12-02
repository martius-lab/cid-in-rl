import abc
from typing import Any, Dict, TYPE_CHECKING

from cid import utils
from cid.memory import BaseReplayMemory

if TYPE_CHECKING:
    import gym
    import numpy


class BaseAgent(abc.ABC):
    @abc.abstractmethod
    def __init__(self, state_dim: int, action_space: 'gym.Space'):
        self._state_dim = state_dim
        self._action_space = action_space

    @property
    def state_dim(self) -> int:
        return self._state_dim

    @property
    def action_space(self) -> 'gym.Space':
        return self._action_space

    @abc.abstractmethod
    def get_action(self, state, evaluate: bool = False):
        ...

    def update_parameters(self,
                          replay_memory: BaseReplayMemory,
                          batch_size: int,
                          n_updates: int) -> Dict[str, Any]:
        total_stats = {}
        for _ in range(n_updates):
            batch = replay_memory.sample_batch(batch_size)
            stats = self.update_step(batch)
            if stats is not None:
                utils.update_dict_of_lists(total_stats, stats)

        return total_stats

    @abc.abstractmethod
    def update_step(self, batch: 'numpy.ndarray') -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def get_state(self) -> Any:
        ...

    @abc.abstractmethod
    def set_state(self, state: Any):
        ...

    @property
    def unwrapped(self) -> 'BaseAgent':
        return self

    @abc.abstractmethod
    def save(self, path: str) -> str:
        """Save model parameters and return file path

        :param path: Filename to which the data is saved. A `.pth` extension
        will be appended to the file name if it does not already have one
        """
        ...

    @abc.abstractmethod
    def load(self, path: str):
        """Load model parameters"""
        ...
