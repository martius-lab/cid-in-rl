import gin
from torch import nn


def init_module(m, w_init, b_init):
    if hasattr(m, 'initialized'):
        return
    if (hasattr(m, 'weight') and not hasattr(m, 'weight_initialized')
            and m.weight is not None and w_init is not None):
        w_init(m.weight)
    if (hasattr(m, 'bias') and not hasattr(m, 'bias_initialized')
            and m.bias is not None and b_init is not None):
        b_init(m.bias)
