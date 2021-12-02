import collections
import itertools

import gin
import torch
from torch import nn

from cid.models.utils import init_module


@gin.configurable(blacklist=['inp_dim', 'outp_dim'])
class MLP(nn.Module):
    def __init__(self,
                 inp_dim,
                 outp_dim,
                 hidden_dims,
                 hidden_activation=nn.ReLU,
                 outp_layer=nn.Linear,
                 outp_activation=nn.Identity,
                 outp_scaling=1.0,
                 weight_init=None,
                 bias_init=None,
                 weight_init_last=None,
                 bias_init_last=None,
                 bn_first=False,
                 use_bn=False,
                 use_layer_norm=False,
                 use_spectral_norm=False,
                 legacy_bn_first=True,
                 add_bn_input=False):
        super().__init__()
        self.w_init = weight_init
        self.b_init = bias_init
        self.w_init_last = weight_init_last
        self.b_init_last = bias_init_last
        self.outp_scaling = outp_scaling
        self.add_bn_input = add_bn_input
        if add_bn_input:
            assert not legacy_bn_first

        layers = []
        self.input_bn = None
        if bn_first:
            bn = nn.BatchNorm1d(inp_dim, momentum=0.1, affine=False)
            if legacy_bn_first:
                layers.append(bn)
            else:
                self.input_bn = bn

        current_dim = inp_dim + (inp_dim if self.add_bn_input else 0)
        for idx, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(current_dim,
                                    hidden_dim))
            if use_spectral_norm:
                layers[-1] = nn.utils.spectral_norm(layers[-1])
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
            elif use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim,
                                           elementwise_affine=True))
            layers.append(hidden_activation())

            current_dim = hidden_dim

        layers.append(outp_layer(current_dim, outp_dim))
        if outp_activation is not None:
            layers.append(outp_activation())

        self.layers = nn.Sequential(*layers)
        self.init()

    def init(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.initialized = True
            if isinstance(module, nn.LayerNorm):
                module.initialized = True
        self.layers.apply(lambda m: init_module(m, self.w_init, self.b_init))
        self.layers[-2].apply(
            lambda m: init_module(m, self.w_init_last, self.b_init_last))

    def forward(self, inp):
        if self.input_bn is not None:
            normed_inp = self.input_bn(inp)
            if self.add_bn_input:
                inp = torch.cat((inp, normed_inp), dim=-1)
            else:
                inp = normed_inp

        x = inp
        for idx, layer in enumerate(self.layers):
            x = layer(x)

        if self.outp_scaling != 1:
            x = self.outp_scaling * x

        return x


@gin.configurable(blacklist=['inp_dim', 'outp_dim'])
class FactorizedMLP(MLP):
    """MLP taking factorized inputs"""
    def __init__(self, inp_dim, outp_dim, *args, **kwargs):
        self.inp_dims = collections.OrderedDict(inp_dim)
        if isinstance(inp_dim, dict):
            inp_dim = sum(shape[0] for shape in inp_dim.values())
        if isinstance(outp_dim, dict):
            outp_dim = sum(shape[0] for shape in outp_dim.values())

        super().__init__(inp_dim, outp_dim, *args, **kwargs)

    def forward(self, inp):
        if isinstance(inp, dict):
            inp = torch.cat([inp[key] for key in self.inp_dims], axis=-1)

        # Factorizing outputs not supported here yet
        return super().forward(inp)


@gin.configurable
def load_mlp(path, mlp_cls):
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    # TODO: this is kind of an unsafe way to detect the input/output dims
    keys = list(state_dict.keys())
    mlp = mlp_cls(inp_dim=state_dict[keys[0]].shape[1],
                  outp_dim=state_dict[keys[-1]].shape[0])
    mlp.load_state_dict(state_dict)

    return mlp


@gin.configurable(blacklist=['model'])
def get_params_from_mlp(model, param_type):
    if param_type == 'stem':
        return model.layers[:-2].parameters()
    elif param_type == 'mean_head':
        return model.layers[-2].mean.parameters()
    elif param_type == 'var_head':
        return model.layers[-2].var.parameters()
    elif param_type == 'stem+mean_head':
        return itertools.chain(model.layers[:-2].parameters(),
                               model.layers[-2].mean.parameters())
    else:
        raise ValueError('Unknown param_type `{}`'.format(param_type))
