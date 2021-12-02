from typing import Dict

import numpy as np


def buffer_from_example(example: Dict[str, np.ndarray],
                        leading_dims) -> Dict[str, np.ndarray]:
    buf = {}
    for key, value in example.items():
        buf[key] = np.zeros(leading_dims + value.shape, dtype=value.dtype)

    return buf


def get_leading_dims(dictionary, n_dims=1):
    values = iter(dictionary.values())
    leading_dims = next(values).shape[:n_dims]

    if not all(leading_dims == value.shape[:n_dims] for value in values):
        key, shape = [(key, value.shape[:n_dims])
                      for key, value in dictionary.items()
                      if leading_dims != value.shape[:n_dims]][0]
        raise ValueError((f'Dimensions do not match: {leading_dims} vs. '
                          f'{shape} (for key `{key}`)'))

    return leading_dims
