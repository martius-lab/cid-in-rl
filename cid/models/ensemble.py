import gin
import torch
from torch import nn


@gin.configurable(blacklist=['inp_dim', 'outp_dim'])
class Ensemble(nn.Module):
    def __init__(self, inp_dim, outp_dim, model_cls, n_models):
        super().__init__()

        self._models = nn.ModuleList()
        for _ in range(n_models):
            self._models.append(model_cls(inp_dim, outp_dim))

    @property
    def models(self):
        return self._models

    def forward(self, inp):
        pred_per_model = [model(inp) for model in self._models]

        if isinstance(pred_per_model[0], (list, tuple)):
            outp = [torch.stack([pred[idx] for pred in pred_per_model], dim=0)
                    for idx in range(len(pred_per_model[0]))]
        else:
            outp = torch.stack(pred_per_model, dim=0)

        return outp
