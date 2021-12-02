import numpy as np


class Normalizer:
    def __init__(self, eps_std=0.01):
        self._eps_var = eps_std**2
        self._mean = 0.0
        self._std = 1.0
        self._num_updates = 0

        self._sum = None
        self._sum_of_squares = None

    @property
    def num_updates(self) -> int:
        return self._num_updates

    @property
    def mean(self) -> np.ndarray:
        return self._mean

    @property
    def std(self) -> np.ndarray:
        return self._std

    def update(self, data: np.ndarray):
        """Update normalizer with a batch of data

        :param data: Batch of data of at least two dimensions, where the first
            dimension is the batch dimension
        """
        if self._sum is None:
            self._sum = np.zeros_like(data[0])
            self._sum_of_squares = np.zeros_like(data[0])

        self._sum += np.sum(data, axis=0)
        self._sum_of_squares += np.sum(np.square(data), axis=0)
        self._num_updates += data.shape[0]

        mean = self._sum / self._num_updates
        var = self._sum_of_squares / self._num_updates - np.square(mean)
        std = np.sqrt(np.maximum(self._eps_var, var))

        self._mean = np.expand_dims(mean, axis=0)
        self._std = np.expand_dims(std, axis=0)

    def __call__(self, inp: np.ndarray):
        return (inp - self._mean) / self._std

    def state_dict(self):
        return {
            'mean': self._mean,
            'std': self._std,
            'sum': self._sum,
            'sum_of_squares': self._sum_of_squares,
            'num_updates': self._num_updates
        }

    def load_state_dict(self, state_dict):
        self._mean = state_dict['mean']
        self._std = state_dict['std']
        self._sum = state_dict['sum']
        self._sum_of_squares = state_dict['sum_of_squares']
        self._num_updates = state_dict['num_updates']
