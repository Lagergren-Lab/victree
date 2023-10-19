import torch
import math


def nanmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    output = tensor.nan_to_num(min_value).amax(dim=dim, keepdim=keepdim)
    return output


def nanmin(tensor, dim=None, keepdim=False):
    max_value = torch.finfo(tensor.dtype).max
    output = tensor.nan_to_num(max_value).amin(dim=dim, keepdim=keepdim)
    return output


def nanlogsumexp(tensor, dim=None, keepdim=False):
    if dim is None:
        dim = tuple(range(tensor.ndim))
    output = tensor.nan_to_num(-torch.inf).logsumexp(dim=dim, keepdim=keepdim)
    return output


def log_beta_function(x: torch.Tensor) -> torch.float:
    x_0 = torch.sum(x)
    return torch.sum(torch.lgamma(x)) - torch.lgamma(x_0)


def cayleys_formula(n: int, log: bool = False) -> int | float:
    """
Cayley's formula for number of labeled unrooted trees with n vertices
    Parameters
    ----------
    n int, number of vertices
    log bool, computation and output in log-scale

    Returns
    -------
    number of distinct labeled trees
    """
    if log:
        out: float = (n - 2) * math.log(n)
    else:
        out: int = n ** (n - 2)
    return out


def cayleys_formula_rooted(n: int, log: bool = False) -> int:
    """
Rooted trees version of Cayley's formula (see Corollary 2 at
https://math.berkeley.edu/~mhaiman/math172-spring10/matrixtree.pdf )
    Parameters
    ----------
    n int, number of vertices
    log bool, computation and output in log-scale

    Returns
    -------
    number of distinct labeled rooted trees
    """
    if log:
        out: float = (n - 1) * math.log(n)
    else:
        out: int = n ** (n - 1)
    return out


def log_factorial(x: torch.Tensor):
    log_x_factorial = torch.lgamma(x + 1)
    return torch.tensor(log_x_factorial)


def inverse_decay_function(x: torch.Tensor, a, b, c):
    """
    Returns f(x) = a * 1 / (x - b)^c
    If x-b < 1. returns a
    This function is used for tempering schemes and step size schemes.
    """
    z = torch.max(torch.tensor(1.), x - b)
    return a * z ** (-c)
