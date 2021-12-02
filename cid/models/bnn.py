"""Bayesian NN for VIME

Adapted from https://github.com/alec-tschantz/vime under MIT License
"""
import math

import gin
import numpy as np
import torch
from torch import autograd
from torch import nn

_HALF_LOG_2PI = 0.5 * math.log(2 * math.pi)


@torch.jit.script
def _log_to_std(rho):
    return torch.log(1 + torch.exp(rho))


def _std_to_log(std):
    return np.log(np.exp(std) - 1)


@torch.jit.script
def _kl_div(p_mean, p_std, q_mean, q_std):
    numerator = (p_mean - q_mean)**2 + p_std**2
    denominator = 2 * q_std**2 + 1e-8
    kl = numerator / denominator + torch.log(q_std) - torch.log(p_std) - 0.5

    return kl.sum()


@gin.configurable
def neg_log_prob_normal(pred, target, sigma=1.):
    # pred: batch_size x n_samples x dim
    # target: batch_size x dim
    target = target.unsqueeze(1)
    log_sigma = math.log(sigma)

    log_prob = (-log_sigma - _HALF_LOG_2PI
                - (target - pred)**2 / (2 * (sigma**2)))

    return -torch.sum(log_prob, dim=-1).mean(dim=-1)


@gin.configurable
def mse(pred, target):
    # pred: batch_size x n_samples x dim
    # target: batch_size x dim
    target = target.unsqueeze(1)

    return torch.mean((target - pred)**2, dim=-1).mean(dim=-1)


class BNNLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs, prior_std):
        super(BNNLayer, self).__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.prior_mean = torch.Tensor([0.0])
        self.prior_std = torch.Tensor([prior_std])

        self.W = np.random.normal(0., prior_std,
                                  (self.n_inputs, self.n_outputs))
        self.b = np.zeros((self.n_outputs,), dtype=np.float)

        W_mu = torch.Tensor(self.n_inputs, self.n_outputs)
        W_mu = nn.init.normal_(W_mu, mean=0., std=0.2)
        self.W_mu = nn.Parameter(W_mu)

        W_rho = torch.Tensor(self.n_inputs, self.n_outputs)
        W_rho = nn.init.constant_(W_rho, _std_to_log(prior_std))
        self.W_rho = nn.Parameter(W_rho)

        b_mu = torch.Tensor(self.n_outputs,)
        b_mu = nn.init.zeros_(b_mu)
        #b_mu = nn.init.normal_(b_mu, mean=0., std=0.2)
        self.b_mu = nn.Parameter(b_mu)

        b_rho = torch.Tensor(self.n_outputs,)
        b_rho = nn.init.constant_(b_rho, _std_to_log(prior_std))
        self.b_rho = nn.Parameter(b_rho)

        self.W_mu_old = torch.Tensor(self.n_inputs, self.n_outputs).detach()
        self.W_rho_old = torch.Tensor(self.n_inputs, self.n_outputs).detach()
        self.b_mu_old = torch.Tensor(self.n_outputs,).detach()
        self.b_rho_old = torch.Tensor(self.n_outputs,).detach()

    def forward(self, X, infer=False):
        if infer:
            output = torch.mm(X, self.W_mu) + \
                self.b_mu.expand(X.size()[0], self.n_outputs)
            return output

        W = self.get_W()
        b = self.get_b()
        output = torch.mm(X, W) + b.expand(X.size()[0], self.n_outputs)

        return output

    def get_W(self):
        epsilon = torch.Tensor(self.n_inputs, self.n_outputs)
        epsilon = nn.init.normal_(epsilon, mean=0., std=1.)
        epsilon = autograd.Variable(epsilon)
        self.W = self.W_mu + _log_to_std(self.W_rho) * epsilon

        return self.W

    def get_b(self):
        epsilon = torch.Tensor(self.n_outputs)
        epsilon = nn.init.normal_(epsilon, mean=0., std=1.)
        epsilon = autograd.Variable(epsilon)
        self.b = self.b_mu + _log_to_std(self.b_rho) * epsilon
        return self.b

    def kl_div_new_prior(self):
        kl_div = _kl_div(self.W_mu, _log_to_std(self.W_rho),
                         self.prior_mean, self.prior_std)
        kl_div += _kl_div(self.b_mu, _log_to_std(self.b_rho),
                          self.prior_mean, self.prior_std)
        return kl_div

    def kl_div_new_old(self):
        kl_div = _kl_div(self.W_mu, _log_to_std(self.W_rho),
                         self.W_mu_old, _log_to_std(self.W_rho_old))
        kl_div += _kl_div(self.b_mu, _log_to_std(self.b_rho),
                          self.b_mu_old, _log_to_std(self.b_rho_old))
        return kl_div

    def loss(self):
        """Regularization loss, called by ModelTrainer"""
        return {'kl_div': self.kl_div_new_prior()}

    def save_old_params(self):
        self.W_mu_old = self.W_mu.clone()
        self.W_rho_old = self.W_rho.clone()
        self.b_mu_old = self.b_mu.clone()
        self.b_rho_old = self.b_rho.clone()

    def reset_to_old_params(self):
        self.W_mu.data = self.W_mu_old.data
        self.W_rho.data = self.W_rho_old.data
        self.b_mu.data = self.b_mu_old.data
        self.b_rho.data = self.b_rho_old.data


@gin.configurable(blacklist=['inp_dim', 'outp_dim'])
class BNN(nn.Module):
    def __init__(self, inp_dim, outp_dim, hidden_dims,
                 std_prior=0.5,
                 std_likelihood=5.0,
                 n_samples=10,
                 bn_first=False):
        super(BNN, self).__init__()
        self.std_likelihood = std_likelihood
        self.n_samples = n_samples

        if bn_first:
            self.input_bn = nn.BatchNorm1d(inp_dim, momentum=0.1, affine=False)
        else:
            self.input_bn = None

        cur_dim = inp_dim
        layers = []
        for hidden_dim in hidden_dims:
            layers.append(BNNLayer(cur_dim, hidden_dim, std_prior))
            cur_dim = hidden_dim

        layers.append(BNNLayer(cur_dim, outp_dim, std_prior))

        self.layers = nn.ModuleList(layers)
        self.opt = None

    def set_optimizer(self, optimizer):
        self.opt = optimizer

    def forward(self, inp, infer=False):
        if not self.training:
            infer = True

        if self.input_bn is not None:
            inp = self.input_bn(inp)

        n_samples = 1 if infer else self.n_samples

        outputs = []
        for _ in range(n_samples):
            x = inp
            for layer in self.layers[:-1]:
                x = torch.relu(layer(x, infer))

            x = self.layers[-1](x, infer)
            outputs.append(x)

        if infer:
            return outputs[0]
        else:
            return torch.stack(outputs, dim=1)

    def loss_old(self, inp, target):
        _log_p_D_given_W = []
        for _ in range(self.n_samples):
            prediction = self(inp)
            _log_p_D_given_W.append(neg_log_prob_normal(target, prediction,
                                                        self.std_likelihood))

        kl = self.kl_div_new_prior()
        return kl / self.n_batches + sum(_log_p_D_given_W) / self.n_samples

    def loss_last_sample(self, inp, target):
        prediction = self(inp)

        log_p_D_given_w = neg_log_prob_normal(target, prediction,
                                              self.std_likelihood)

        return self.kl_div_new_old() + log_p_D_given_w.sum()

    def train_fn(self, inputs, targets):
        inputs = torch.from_numpy(inputs).float()
        targets = torch.from_numpy(targets).float()

        self.opt.zero_grad()
        loss = self.loss(inputs, targets)
        loss.backward()
        self.opt.step()

        return loss.item()

    def train_for_info_gain(self, inputs, targets):
        if self.input_bn and self.input_bn.training:
            # Don't want to update BN stats during info gain computation
            self.input_bn.eval()

        self.opt.zero_grad()
        loss = self.loss_last_sample(inputs, targets)
        loss.backward()
        self.opt.step()

        return loss.item()

    def pred_fn(self, inputs):
        with torch.no_grad():
            _out = self(inputs, infer=True)

        return _out

    def kl_div_new_old(self):
        """KL divergence KL[new_params || old_param] aka info gain"""
        kl_divs = [l.kl_div_new_old() for l in self.layers]
        return sum(kl_divs)

    def kl_div_new_prior(self):
        """KL divergence KL[new_params || prior]"""
        kl_divs = [l.kl_div_new_prior() for l in self.layers]
        return sum(kl_divs)

    def save_old_params(self):
        self._optimizer_state = self.opt.state_dict()
        for l in self.layers:
            l.save_old_params()

    def reset_to_old_params(self):
        self.opt.load_state_dict(self._optimizer_state)
        for l in self.layers:
            l.reset_to_old_params()
