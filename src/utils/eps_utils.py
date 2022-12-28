"""
Set of functions and class that manage the discrete probability distribution
over copy number configurations according to the rules defined,
differentiating between
- common cases: current copy number evolves from parent with the same shift as the
        previuous site's shift
- rare cases: not common
- impossible cases: current copy number is > 0 while parent's is 0
"""

import torch
import numpy as np
import itertools

from typing import Tuple


def iter_pair_states(n_states):
    # TODO: remove impossible cases from this iterable
    for pair_state in itertools.combinations_with_replacement(range(n_states), 2):
        yield pair_state


def iter_quad_states(n_states):
    for quad_state in itertools.combinations_with_replacement(range(n_states), 2):
        yield quad_state


# TODO: differentiate impossible cases to mask
def get_zipping_mask(n_states) -> torch.Tensor:
    """Build a mask on the i - i' == j - j' condition

    Parameters
    ----------
    n_states :
        Number of total copy number states (cn = 0, 1, ..., n_states - 1)
    Returns
    -------
    Torch boolean tensor of shape (n_states, n_states, n_states, n_states),
    true where the indices satisfy the condition.
    Idx order is `mask[j', j, i', i]`
    """
    ind_arr = np.indices((n_states, n_states))
    # i - j 
    imj = ind_arr[0] - ind_arr[1]
    # i - j == k - l
    mask = imj == imj[:, :, np.newaxis, np.newaxis]
    return torch.tensor(mask)


def get_zipping_mask0(n_states) -> torch.Tensor:
    """Build a mask on the i == j condition

    Parameters
    ----------
    n_states :
        Number of total copy number states (cn = 0, 1, ..., n_states - 1)
    Returns
    -------
    Torch boolean tensor of shape (n_states, n_states),
    true where the indices satisfy the condition
    """
    ind_arr = np.indices((n_states, n_states))
    # i = j (diagonal)
    mask = ind_arr[0] == ind_arr[1]
    return torch.tensor(mask)


def h_eps(n_states: int, eps: float) -> torch.Tensor:
    mask_arr = get_zipping_mask(n_states=n_states)
    # put 1-eps where j'-j = i'-i
    a = mask_arr * (1 - eps)
    # put either 1 or 1-eps in j'-j != i'-i  and divide by the cases
    b = (1 - torch.sum(a, dim=0)) / torch.sum(~mask_arr, dim=0)
    # combine the two arrays
    out_arr = b * (~mask_arr) + a
    return out_arr


def normalized_zipping_constant(n_states: int) -> torch.Tensor:
    # FIXME: output shape has to be properly matched with zipping tensor
    mask = get_zipping_mask(n_states)
    return torch.sum(~mask, dim=0, keepdim=True)


def normalized_zipping_constant0(n_states: int) -> torch.Tensor:
    # FIXME: output shape has to be properly matched with zipping tensor
    mask = get_zipping_mask0(n_states)
    return torch.sum(~mask, dim=0, keepdim=True)


def is_rare_case(jj, j, ii, i):
    return (j != 0 or jj == 0) and (i != 0 or ii == 0) and (jj - j != ii - i)


def is_common_case(jj, j, ii, i):
    return (j != 0 or jj == 0) and (i != 0 or ii == 0) and (jj - j == ii - i)


def compute_n_cases(n_copy_states) -> Tuple[torch.Tensor, torch.Tensor]:
    rare_cnt = torch.zeros((n_copy_states,) * 3)
    common_cnt = torch.zeros((n_copy_states,) * 3)
    # iterates over all configurations
    for cp4 in itertools.product(*(range(n_copy_states),) * 4):
        # for each of the conditioned states combination
        # count the number of rare and common states with that specific config
        rare_cnt[cp4[1], cp4[2], cp4[3]] += is_rare_case(*cp4)
        common_cnt[cp4[1], cp4[2], cp4[3]] += is_common_case(*cp4)

    return rare_cnt, common_cnt


class TreeHMM:

    def __init__(self, n_copy_states, eps=torch.tensor(1e-2), delta=torch.tensor(1e-10)):
        self.eps = eps
        self.delta = delta
        self.n_copy_states = n_copy_states

        self.n_rare_cases, self.n_common_cases = compute_n_cases(n_copy_states)

        # h(jj, j, ii, i) = p(c_u_m | c_u_m-1, c_p_m, c_p_m-1)
        self.cpd_table = {cp4: 0. for cp4 in itertools.product(*(range(self.n_copy_states),) * 4)}
        # h(j, i) = p(c_u_0 | c_p_0)
        self.cpd_pair_table = {cp2: 0. for cp2 in itertools.product(*(range(self.n_copy_states),) * 4)}
        self.compute_cpds()

    def compute_cpds(self):
        # FIXME: 0 absorption is not working properly
        for cp4 in itertools.product(*(range(self.n_copy_states),) * 4):
            self.cpd_table[cp4] = self.h(*cp4)

        for cp2 in itertools.product(*(range(self.n_copy_states),) * 2):
            if cp2[1] == 0 and cp2[0] == 0:
                self.cpd_pair_table[cp2] = 1.
            elif cp2[1] == 0 and cp2[0] != 0:
                self.cpd_pair_table[cp2] = self.delta
            elif cp2[0] == cp2[1]:
                self.cpd_pair_table[cp2] = 1. - self.eps
            else:
                self.cpd_pair_table[cp2] = self.eps / (self.n_copy_states - 1)

    def h(self, jj, j, ii, i):
        if self.n_rare_cases[j, ii, i] == 0:
            # every configuration is impossible regardless the value of jj
            return self.delta

        local_eps = self.eps

        if self.n_common_cases[j, ii, i] == 0:
            # either it's a rare case or an impossible one
            # eps should then be set to 1 and prob is distributed
            # over the rare cases
            local_eps = torch.tensor(1.)

        p = torch.tensor(-1)
        if (j == 0 and jj != 0) or (i == 0 and ii != 0):
            p = self.delta
        elif jj - j == ii - i:
            p = (1. - local_eps) / self.n_common_cases[j, ii, i]
        else:
            p = local_eps / self.n_rare_cases[j, ii, i]

        return p
