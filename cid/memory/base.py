import abc
from typing import Any, Callable, Dict, Iterable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

FilterT = Callable[[Dict[str, 'np.ndarray']], 'np.ndarray']


class BaseReplayMemory(abc.ABC):
    @property
    @abc.abstractmethod
    def current_size(self) -> int:
        ...

    @abc.abstractmethod
    def store_episodes(self, episodes):
        ...

    @abc.abstractmethod
    def sample_batch(self, batch_size: int):
        ...

    @abc.abstractmethod
    def report_statistics(self) -> Dict[str, Any]:
        ...

    @abc.abstractmethod
    def to_transition_sequence(self, keys: Iterable[str] = None,
                               filter_fn: FilterT = None) \
            -> Sequence[Dict[str, Any]]:
        """Return sequence over transitions of this memory

        Sequence is guaranteed to stay unchanged as long as the underlying
        replay memory is unchanged.

        :param keys: Keys to include in the sequence. By convention, keys
            ending with `0` refer to the start state of the transition, keys
            ending with `1` refer to the end state of the transition, and keys
            ending without any number refer to the start state of the
            transition.
        :param filter_fn: A callable that receives the buffer dictionary and
            should return a 2D-boolean array which contains `True` for every
            buffer entry to include.
        """
        ...

    @abc.abstractmethod
    def __iter__(self):
        """Iterate over elements of this buffer"""
        ...

    @abc.abstractmethod
    def save(self, path: str):
        """Save buffer content to disk"""
        ...

    @abc.abstractmethod
    def load(self, path: str):
        """Load buffer content from disk"""
        ...
