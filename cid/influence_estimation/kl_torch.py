import math

import torch


@torch.jit.script
def log_2pi():
    return math.log(2 * math.pi)  # Optimized (hopefully) to a constant by JIT


@torch.jit.script
def inf():
    return float('inf')  # Optimized (hopefully) to a constant by JIT


@torch.jit.script
def gaussian_entropy(m, v):
    """Entropy of Gaussian"""
    d = v.shape[-1]
    return 0.5 * (d * (1 + log_2pi()) + torch.sum(torch.log(v), dim=-1))


@torch.jit.script
def gaussian_prod_logconst(m1, v1, m2, v2):
    """Log normalization constant of product of two Gaussians"""
    d = m1.shape[-1]
    v_sum = v1 + v2
    return (-0.5 * (d * log_2pi()
            + torch.sum(torch.log(v_sum), dim=-1)
            + torch.sum((m1 - m2)**2 / v_sum, dim=-1)))


@torch.jit.script
def kl_div(m1, v1, m2, v2):
    """KL divergence between two Gaussians"""
    d = m1.shape[-1]
    return (0.5 * (-d + ((v1 + (m2 - m1)**2) / v2
            + torch.log(v2) - torch.log(v1)).sum(dim=-1)))


@torch.jit.script
def _kl_div_mixture_app(m1, v1, m2, v2):
    m1_ = m1.unsqueeze(-2)
    v1_ = v1.unsqueeze(-2)

    log_n_mixtures = math.log(m2.shape[-2])

    # Variational approximation
    inner_kls = kl_div(m1_, v1_, m2, v2)
    kls_var = log_n_mixtures - torch.logsumexp(-inner_kls, dim=-1)

    # Product approximation
    log_constants = gaussian_prod_logconst(m1_, v1_, m2, v2)
    kls_prod = (log_n_mixtures - gaussian_entropy(m1, v1)
                - torch.logsumexp(log_constants, dim=-1))
    kls_prod = torch.max(kls_prod, torch.zeros(1))

    kls_app = 0.5 * (kls_var + kls_prod)

    return kls_app, kls_var, kls_prod


@torch.jit.script
def _kl_div_mixture_app_with_upper_bound(m1, v1, m2, v2):
    # For the upper bound, we ignore the Gaussian on the right side that
    # corresponds to the Gaussian on the left side that is evaluated.
    m1_ = m1.unsqueeze(-2)
    v1_ = v1.unsqueeze(-2)

    log_n_mixtures = math.log(m2.shape[-2])

    # Variational approximation
    inner_kls = kl_div(m1_, v1_, m2, v2)
    kls_var = log_n_mixtures - torch.logsumexp(-inner_kls, dim=-1)

    torch.diagonal(inner_kls, dim1=-2, dim2=-1)[:] = inf()
    kls_var_upper = log_n_mixtures - torch.logsumexp(-inner_kls, dim=-1)

    # Product approximation
    n_mixtures_minus_entropy = log_n_mixtures - gaussian_entropy(m1, v1)
    log_constants = gaussian_prod_logconst(m1_, v1_, m2, v2)

    kls_prod = (n_mixtures_minus_entropy
                - torch.logsumexp(log_constants, dim=-1))
    kls_prod = torch.max(kls_prod, torch.zeros(1))

    torch.diagonal(log_constants, dim1=-2, dim2=-1)[:] = -inf()
    kls_prod_upper = (n_mixtures_minus_entropy
                      - torch.logsumexp(log_constants, dim=-1))
    kls_prod_upper = torch.max(kls_prod_upper, torch.zeros(1))

    kls_app = 0.5 * (kls_var + kls_prod)
    kls_app_upper = 0.5 * (kls_var_upper + kls_prod_upper)

    return (kls_app, kls_var, kls_prod,
            kls_app_upper, kls_var_upper, kls_prod_upper)


def kl_div_mixture_app(m1, v1, m2, v2,
                       return_approximations=False,
                       return_upper_bound=False):
    """Approximate KL divergence between Gaussian and mixture of Gaussians

    See Durrieu et al, 2012: "Lower and upper bounds for approximation of the
    Kullback-Leibler divergence between Gaussian Mixture Models"
    https://serval.unil.ch/resource/serval:BIB_513DF4E21898.P001/REF

    Both the variational and the product approximation are simplified here
    compared to the paper, as we assume to have a single Gaussian as the first
    argument.

    m1: ([batch_dims], data_dims)
    v1: ([batch_dims], data_dims)
    m2: ([batch_dims], mixtures, data_dims)
    v2: ([batch_dims], mixtures, data_dims)
    """
    assert m1.ndim + 1 == m2.ndim

    if return_upper_bound:
        res = _kl_div_mixture_app_with_upper_bound(m1, v1, m2, v2)
        if return_approximations:
            return res
        else:
            return res[0], res[3]
    else:
        kls_app, kls_var, kls_prod = _kl_div_mixture_app(m1, v1, m2, v2)
        if return_approximations:
            return kls_app, kls_var, kls_prod
        else:
            return kls_app
