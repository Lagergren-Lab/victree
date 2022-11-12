import itertools

from utils.config import Config
import torch
from typing import Optional, Union
from utils.eps_utils import get_zipping_mask, get_zipping_mask0

from utils import tree_utils
from variational_distributions.variational_distribution import VariationalDistribution


class qEpsilon(VariationalDistribution):

    def __init__(self, config: Config, alpha_0: float = 1, beta_0: float = 1):
        self.alpha_prior = alpha_0
        self.beta_prior = beta_0
        self.alpha = alpha_0
        self.beta = alpha_0
        super().__init__(config)

    def set_params(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def initialize(self):
        # TODO: implement (over set params)
        return super().initialize()

    def create_masks(self, A):
        co_mut_mask = torch.zeros((A, A, A, A))
        anti_sym_mask = torch.zeros((A, A, A, A))
        # TODO: Find effecient way of indexing i-j = k-l
        for i, j, k, l in itertools.combinations_with_replacement(range(A), 4):
            if i - j == k - l:
                co_mut_mask[i, j, k, l] = 1
            else:
                anti_sym_mask[i, j, k, l] = 1
        return co_mut_mask, anti_sym_mask

    def update(self, T_list, w_T, q_C_pairwise_marginals):
        self.update_CAVI(T_list, w_T, q_C_pairwise_marginals)
        super().update()
    
    def update_CAVI(self, T_list: list, w_T: torch.Tensor, q_C_pairwise_marginals: torch.Tensor):
        N, M, A, A = q_C_pairwise_marginals.size()
        alpha = self.alpha_prior
        beta = self.beta_prior
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(T_list, N_nodes=N)

        E_CuCv_a = torch.zeros((N, N))
        E_CuCv_b = torch.zeros((N, N))
        co_mut_mask, anti_sym_mask = self.create_masks(A)
        for uv in unique_edges:
            u, v = uv
            E_CuCv_a[u, v] = torch.einsum('mij, mkl, ijkl -> ', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v], co_mut_mask)
            E_CuCv_b[u, v] = torch.einsum('mij, mkl, ijkl -> ', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v], anti_sym_mask)

        for (k, T) in enumerate(T_list):
            for uv in T.edges:
                u, v = uv
                alpha += w_T[k] * E_CuCv_a[u, v]
                beta += w_T[k] * E_CuCv_b[u, v]

        self.set_params(alpha, beta)
        return alpha, beta


    def h_eps0(self, i: Optional[int] = None, j: Optional[int] = None) -> Union[float, torch.Tensor]:
        # FIXME: add normalization constant A0
        if i is not None and j is not None:
            return 1. - self.config.eps0 if i == j else self.config.eps0
        else:
            heps0_arr = self.config.eps0 * torch.ones((self.config.n_states, self.config.n_states))
            diag_mask = get_zipping_mask0(self.config.n_states)
            heps0_arr[diag_mask] = 1 - self.config.eps0
            if i is None and j is not None:
                return heps0_arr[:, j]
            elif i is not None and j is None:
                return heps0_arr[i, :]
            else:
                return heps0_arr

    def exp_zipping(self):
        # TODO: implement
        copy_mask = get_zipping_mask(self.config.n_states)

        # FIXME: add normalization (division by A constant)
        norm_const = self.config.n_states

        out_arr = torch.ones(copy_mask.shape) *\
                (torch.digamma(torch.tensor(self.beta)) -\
                torch.digamma(torch.tensor(self.alpha + self.beta)))
        out_arr[~copy_mask] -= norm_const
        return out_arr














