import torch
import math


def log_beta_function(x: torch.Tensor)-> torch.float:
    x_0 = torch.sum(x)
    return torch.sum(torch.lgamma(x)) - torch.lgamma(x_0)


def cayleys_formula(n, log=False):
    out = -1
    if log:
        out = (n - 2) * math.log(n)
    else:
        out = n ** (n - 2)
    return out