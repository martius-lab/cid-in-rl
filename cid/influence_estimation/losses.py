import math

import gin
import torch
from torch import distributions as D
from torch.nn import functional as F

_LOG_2PI = math.log(2 * math.pi)


@gin.configurable
def mse_loss(pred, target):
    return torch.mean((pred - target)**2, axis=-1)


@gin.configurable
def gaussian_log_likelihood_loss(pred, target, with_logvar=False,
                                 fixed_variance=None, detach_mean=False,
                                 detach_var=False):
    mean = pred[0]
    if detach_mean:
        mean = mean.detach()

    if with_logvar:
        logvar = pred[1]
        if detach_var:
            logvar = logvar.detach()

        if fixed_variance is not None:
            logvar = torch.ones_like(mean) * math.log(fixed_variance)
        ll = -0.5 * ((target - mean)**2 * (-logvar).exp() + logvar + _LOG_2PI)
    else:
        var = pred[1]
        if detach_var:
            var = var.detach()

        if fixed_variance is not None:
            var = torch.ones_like(mean) * fixed_variance
        ll = -0.5 * ((target - mean)**2 / var + torch.log(var) + _LOG_2PI)

    return -torch.sum(ll, axis=-1)


@gin.configurable
def make_factorized_loss_fn(loss_fn):
    def wrapper_loss_fn(preds, targets):
        losses = [loss_fn(preds[name], targets[name]) for name in targets]
        return torch.sum(torch.stack(losses, dim=0), dim=0)

    return wrapper_loss_fn


factorized_mse_loss = make_factorized_loss_fn(mse_loss)


def compute_l1_penalty(model):
    loss = 0

    for param in model.parameters():
        if param.dim() >= 2:  # Exclude bias parameters
            loss += F.l1_loss(param,
                              target=torch.zeros_like(param),
                              reduction='mean')

    return loss


def collect_module_losses(model):
    losses = {}

    for module in model.modules():
        if hasattr(module, 'loss'):
            module_losses = module.loss()
            for key, value in module_losses.items():
                if key in losses:
                    losses[key] += value
                else:
                    losses[key] = value

    return losses


class LossScheduler:
    def __init__(self, losses):
        self._losses = losses
        self._n_steps = 0
        self._training = True

    def step(self):
        self._n_steps += 1
        for loss in self._losses:
            if isinstance(loss, LossScheduler):
                loss.step()

    def train(self):
        self._training = True
        for loss in self._losses:
            if isinstance(loss, LossScheduler):
                loss.train()

    def eval(self):
        self._training = False
        for loss in self._losses:
            if isinstance(loss, LossScheduler):
                loss.eval()

    def __call__(self, *args, **kwargs):
        pass


@gin.configurable
class LossWarmupScheduler(LossScheduler):
    def __init__(self, loss_fn, warmup_loss_fn,
                 warmup_steps=0, annealing_steps=0, annealing='linear'):
        super().__init__([loss_fn, warmup_loss_fn])
        self._loss_fn = loss_fn
        self._warmup_loss_fn = warmup_loss_fn
        self._warmup_steps = warmup_steps
        self._annealing_steps = annealing_steps
        self._annealing = annealing

    def __call__(self, *args, **kwargs):
        if not self._training:
            return self._loss_fn(*args, **kwargs)

        if self._n_steps < self._warmup_steps:
            warmup_weight = 1.0
        elif self._n_steps < self._warmup_steps + self._annealing_steps:
            steps = self._n_steps - self._warmup_steps
            if self._annealing == 'linear':
                warmup_weight = 1.0 - steps / self._annealing_steps
            elif self._annealing == 'cosine':
                warmup_weight = 1.0 - 0.5 * (1 + math.cos(math.pi * steps
                                             / self._annealing_steps))
        else:
            warmup_weight = 0.0

        if warmup_weight > 0:
            warmup_loss = self._warmup_loss_fn(*args, **kwargs)
        else:
            warmup_loss = 0.0

        if warmup_weight < 1:
            loss = self._loss_fn(*args, **kwargs)
        else:
            loss = 0.0

        return (1 - warmup_weight) * loss + warmup_weight * warmup_loss


@gin.configurable
class AlternatingLossScheduler(LossScheduler):
    def __init__(self, eval_loss_fn, loss_fns, n_steps_per_loss=1):
        assert isinstance(loss_fns, list)
        super().__init__(loss_fns)
        self._loss_fns = loss_fns
        self._eval_loss_fn = eval_loss_fn
        self._n_steps_per_loss = n_steps_per_loss
        self._cur_loss_idx = 0

    def step(self):
        super().step()
        if (self._n_steps % self._n_steps_per_loss) == 0:
            self._cur_loss_idx = (self._cur_loss_idx + 1) % len(self._loss_fns)

    def __call__(self, *args, **kwargs):
        if not self._training and self._eval_loss_fn is not None:
            return self._eval_loss_fn(*args, **kwargs)

        return self._loss_fns[self._cur_loss_idx](*args, **kwargs)
