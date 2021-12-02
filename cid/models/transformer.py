import itertools
from collections import OrderedDict
from typing import Any

import gin
import torch
from torch import nn


@gin.configurable(blacklist=['inp_dim', 'outp_dim'])
class Transformer(nn.Module):
    def __init__(self,
                 inp_dim: 'OrderedDict[str, Any]',
                 outp_dim: 'OrderedDict[str, Any]',
                 embedding_dim,
                 n_layers,
                 n_heads=1,
                 fc_dim=64,
                 outp_layer=nn.Linear,
                 dropout_probs=None,
                 bn_first=False,
                 **kwargs):
        super().__init__()

        self.outp_groups = {name for name in inp_dim if name in outp_dim}
        self.only_inp_groups = {name for name in inp_dim
                                if name not in outp_dim}

        self.inp_projs = nn.ModuleDict()
        for name in itertools.chain(self.outp_groups, self.only_inp_groups):
            shape = inp_dim[name]
            bn = None
            if bn_first:
                bn = nn.BatchNorm1d(shape[0], momentum=0.1, affine=False)

            if bn is None:
                inp_proj = nn.Linear(shape[0], embedding_dim)
            else:
                inp_proj = nn.Sequential(bn,
                                         nn.Linear(shape[0], embedding_dim))
            self.inp_projs[name] = inp_proj

        self.outp_projs = nn.ModuleDict()
        for name in self.outp_groups:
            shape = outp_dim[name]
            self.outp_projs[name] = outp_layer(embedding_dim, shape[0])

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = TransformerEncoderLayer(embedding_dim,
                                            nhead=n_heads,
                                            dim_feedforward=fc_dim,
                                            dropout=0.0)
            self.layers.append(layer)

        if dropout_probs is None:
            dropout_probs = {}

        probs = [[dropout_probs.get(name, 0) for name in self.inp_projs]]
        self.register_buffer('dropout_probs',
                             torch.tensor(probs, dtype=torch.float))

    def load_state_dict(self, state_dict):
        if state_dict['dropout_probs'].shape != self.dropout_probs.shape:
            state_dict['dropout_probs'] = self.dropout_probs

        cur_state_dict = self.state_dict()
        missing_inp_keys = {name for name in state_dict
                            if (name.startswith('inp_projs')
                                and name not in cur_state_dict)}
        for key in missing_inp_keys:
            del state_dict[key]

        super().load_state_dict(state_dict)

    def get_input_index(self, key):
        """Index of group named by key in input"""
        return list(self.inp_projs).index(key)

    def get_output_index(self, key):
        """Index of group named by key in output"""
        return list(self.outp_projs).index(key)

    def get_mask(self, x):
        embeddings = [proj(x[name]) for name, proj in self.inp_projs.items()]
        embeddings = torch.stack(embeddings, dim=0)

        joint_mask = None
        for layer in self.layers:
            # Mask is encoded as B x Out x Inp, where Out is over the values,
            # and Inp is over the key-query products.
            embeddings, mask = layer(embeddings,
                                     return_attention_weights=True)
            if joint_mask is None:
                joint_mask = mask.transpose(1, 2)
            else:
                joint_mask = joint_mask.bmm(mask.transpose(1, 2))

        # Returned mask is encoded as B x Inp x Out
        return joint_mask[:, :, :len(self.outp_projs)]

    def forward(self, x):
        embeddings = [proj(x[name]) for name, proj in self.inp_projs.items()]
        embeddings = torch.stack(embeddings, dim=0)

        if self.training and self.dropout_probs is not None:
            bs = embeddings.shape[1]
            probs = self.dropout_probs.expand(bs, -1)
            mask = torch.bernoulli(probs)
        else:
            mask = None

        for layer in self.layers:
            embeddings = layer(embeddings, src_key_padding_mask=mask)

        outp = OrderedDict()
        for idx, (name, proj) in enumerate(self.outp_projs.items()):
            # `embeddings` encoded as Out x B x Dim
            outp[name] = proj(embeddings[idx])

        return outp


class TransformerEncoderLayer(nn.TransformerEncoderLayer):
    """Same as `nn.TransformerEncoderLayer`, but returns attention weights"""
    def __init__(self, *args, **kwargs):
        super(TransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None,
                return_attention_weights=False):
        src2, mask = self.self_attn(src, src, src,
                                    attn_mask=src_mask,
                                    need_weights=return_attention_weights,
                                    key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        if return_attention_weights:
            return src, mask
        else:
            return src


@gin.configurable(blacklist=['inp_dim', 'outp_dim'])
class ModelFactorizingWrapper(nn.Module):
    """Compatibility wrapper for models using factorized inputs/outputs

    Turns vector inputs into factorized inputs and factorized outputs into
    vector outputs.
    """
    def __init__(self, inp_dim, outp_dim, model_cls,
                 factorize_fn=None, join_fn=None, unwrap_dict=False):
        super().__init__()
        self.model = model_cls(inp_dim, outp_dim)
        self.factorize_fn = factorize_fn
        self.join_fn = join_fn
        self.unwrap_dict = unwrap_dict

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    def forward(self, x):
        if self.factorize_fn is not None:
            x = self.factorize_fn(x)

        x = self.model(x)

        if self.join_fn is not None:
            return self.join_fn(x)
        elif self.unwrap_dict:
            elem = next(iter(x.values()))
            if isinstance(elem, (list, tuple)):
                x = [torch.cat([v[idx] for v in x.values()], dim=1)
                     for idx in range(len(elem))]
            else:
                x = torch.cat([v for v in x.values()], dim=1)
            return x
        else:
            return x
