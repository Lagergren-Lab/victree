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
import itertools

from typing import Tuple


def is_rare_case(jj, j, ii, i):
    return (j != 0 or jj == 0) and (i != 0 or ii == 0) and (jj - j != ii - i)

def is_common_case(jj, j, ii, i):
    return (j != 0 or jj == 0) and (i != 0 or ii == 0) and (jj - j == ii - i)

def compute_n_cases(n_copy_states) -> Tuple[torch.Tensor, torch.Tensor]:
    rare_cnt = torch.zeros((n_copy_states,) * 3)
    common_cnt = torch.zeros((n_copy_states,) * 3)
    # iterates over all configurations
    for cp4 in itertools.product(*(range(n_copy_states), ) * 4):

        # for each of the conditioned states combination
        # count the number of rare and common states with that specific config
        rare_cnt[cp4[1], cp4[2], cp4[3]] += is_rare_case(*cp4)
        common_cnt[cp4[1], cp4[2], cp4[3]] += is_common_case(*cp4)

    return rare_cnt, common_cnt


class TreeHMM:

    def __init__(self, n_copy_states, eps = torch.tensor(1e-2), delta = torch.tensor(1e-10)):
        self.eps = eps
        self.delta = delta
        self.n_copy_states = n_copy_states

        self.n_rare_cases,  self.n_common_cases = compute_n_cases(n_copy_states)

        # h(jj, j, ii, i) = p(c_u_m | c_u_m-1, c_p_m, c_p_m-1)
        self.cpd_table = { cp4: 0. for cp4 in itertools.product(*(range(self.n_copy_states), ) * 4) }
        # h(j, i) = p(c_u_0 | c_p_0)
        self.cpd_pair_table = { cp2: 0. for cp2 in itertools.product(*(range(self.n_copy_states), ) * 4) }
        self.compute_cpds()


    def compute_cpds(self):
        # FIXME: 0 absorption is not working properly
        for cp4 in itertools.product(*(range(self.n_copy_states), ) * 4):
            self.cpd_table[cp4] = self.h(*cp4)

        for cp2 in itertools.product(*(range(self.n_copy_states), ) * 2):
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
