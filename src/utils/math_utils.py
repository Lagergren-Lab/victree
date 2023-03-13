import torch
import math


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
