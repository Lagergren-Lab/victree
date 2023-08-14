import torch
import math
import numpy as np
import scipy.special as sp_spec
import scipy.stats as sp_stats


def remap_tensor(x: torch.Tensor, perm) -> torch.float:
    index = torch.LongTensor(perm)
    y = torch.zeros_like(x)
    y[index] = x
    return y
