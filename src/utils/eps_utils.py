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
    A = n_states
    co_mut_mask = torch.zeros((A, A, A, A), dtype=torch.bool)
    anti_sym_mask = torch.zeros((A, A, A, A), dtype=torch.bool)
    absorbing_state_mask = torch.zeros((A, A, A, A), dtype=torch.bool)
    # TODO: Find effecient way of indexing i-j = k-l
    for jj, j, ii, i in itertools.product(range(A), range(A), range(A), range(A)):
        if (ii == 0 and jj != 0) or (i == 0 and j != 0):
            absorbing_state_mask[jj, j, ii, i] = 1
        elif jj - j == ii - i:
            co_mut_mask[jj, j, ii, i] = 1
        else:
            anti_sym_mask[jj, j, ii, i] = 1
    return co_mut_mask, anti_sym_mask, absorbing_state_mask

def get_zipping_mask_old(n_states) -> torch.Tensor:
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
    """
Zipping function tensor for given epsilon. In arc u->v, for each
combination, P(Cv_m=j'| Cv_{m-1}=j, Cu_m=i', Cu_{m-1}=i) = h(j'|j, i', i).
Indexing order: [j', j, i', i]. Invariant: sum(dim=0) = 1.
    Args:
        n_states: total number of copy number states
        eps: arc distance parameter

    Returns:
        tensor of shape (A x A x A x A) with A = n_states
    """
    stable_zero = 1e-8
    # init with nans so to make sure every case is covered
    out_arr = torch.full((n_states, ) * 4, torch.nan)
    for i, ii, j in itertools.product(range(n_states), repeat=3):
        # all cases given i, ii, j
        # CASE 0: parent cn is 0 at site m-1 and evolves to some cn > 0 (not-feasible)
        if i == 0 and j != 0:
            out_arr[:, j, ii, i] = stable_zero
        # CASE 1: parent cn is 0 at site m, it can only stay 0 at site m (forced transition)
        elif ii == 0:
            out_arr[0, j, ii, i] = 1. - stable_zero
            out_arr[1:, j, ii, i] = stable_zero
        # CASE 2: feasible co-mutation i.e. there exists jj s.t. ii - i == jj - j
        elif 0 <= ii - i + j < n_states:
            out_arr[:, j, ii, i] = eps / (n_states - 1)
            out_arr[ii - i + j, j, ii, i] = 1 - eps
        # CASE 3: unfeasible co-mutation i.e. child cn should evolve to state out of 0, ... ,  n_states-1
        elif ii - i + j >= n_states or ii - i + j < 0:
            out_arr[:, j, ii, i] = 1. / n_states

    return out_arr


def feasible_state_mask(n_states: int):
    # returns the triplets j, ii, i for which a jj exists
    # used in einsum operations together with h_eps
    mask = torch.ones((n_states, ) * 3, dtype=torch.bool)
    mask[1:, :, 0] = False
    return mask


def h_eps0(n_states: int, eps0: float) -> torch.Tensor:
    """
Simple zipping function tensor. P(Cv_1=j| Cu_1=i) = h0(j|i)
    Args:
        n_states: total number of copy number states
        eps: arc distance parameter

    Returns:
        tensor of shape (A x A) with A = n_states, indexing [j, i]
    """
    stable_zero = 1e-8
    heps0_arr = torch.full((n_states, ) * 2, torch.nan)
    diag_mask = get_zipping_mask0(n_states)
    heps0_arr[diag_mask] = 1 - eps0
    heps0_arr[~diag_mask] = eps0 / (n_states - 1)
    # zero absorption
    heps0_arr[0, 0] = 1. - stable_zero
    heps0_arr[1:, 0] = stable_zero
    return heps0_arr


def normalizing_zipping_constant(n_states: int) -> torch.Tensor:
    # out shape (n_states, n_states, n_states, n_states)
    out_tensor = torch.empty((n_states, ) * 4)
    comut_mask, no_comut_mask, abs_mask = get_zipping_mask(n_states)
    out_tensor[...] = torch.sum(no_comut_mask, dim=0, keepdim=True)
    return out_tensor


def normalizing_zipping_constant0(n_states: int) -> torch.Tensor:
    # out shape (n_states, n_states)
    out_tensor = torch.empty(n_states, n_states)
    mask = get_zipping_mask0(n_states)
    out_tensor[...] = torch.sum(~mask, dim=0, keepdim=True)
    return out_tensor


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


# obsolete
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
