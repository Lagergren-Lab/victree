import itertools

import torch

from utils.eps_utils import compute_n_cases


class CopyNumberTreeHMM:

    def __init__(self, n_copy_states, eps=torch.tensor(1e-2), delta=torch.tensor(1e-10)):
        self.eps = eps
        self.delta = delta
        self.n_copy_states = n_copy_states

        self.n_rare_cases, self.n_common_cases = compute_n_cases(n_copy_states)

        # h(jj, j, ii, i) = p(c_u_m | c_u_m-1, c_p_m, c_p_m-1)
        self.cpd_table = torch.empty((n_copy_states,) * 4)
        # h(j, i) = p(c_u_0 | c_p_0)
        self.cpd_pair_table = torch.empty((n_copy_states,) * 2)
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
