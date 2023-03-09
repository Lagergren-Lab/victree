import torch
import math


def log_beta_function(x: torch.Tensor) -> torch.float:
    x_0 = torch.sum(x)
    return torch.sum(torch.lgamma(x)) - torch.lgamma(x_0)


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
    out = -1
    if log:
        out = (n - 1) * math.log(n)
    else:
        out = n ** (n - 1)
    return out