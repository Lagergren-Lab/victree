import torch
import torch.distributions as dist

from utils import tree_utils


class qEpsilon():

    def __init__(self, alpha_0: torch.Tensor, beta_0: torch.Tensor):
        self.alpha_prior = alpha_0
        self.beta_prior = beta_0
        self.alpha = alpha_0
        self.beta = alpha_0

    def set_params(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def create_masks(self, A):
        co_mut_mask = torch.zeros((A, A, A, A))
        anti_sym_mask = torch.zeros((A, A, A, A))
        # TODO: Find effecient way of indexing i-j = k-l
        for i in range(0, A):
            for j in range(0, A):
                for k in range(0, A):
                    for l in range(0, A):
                        if i - j == k - l:
                            co_mut_mask[i, j, k, l] = 1
                        else:
                            anti_sym_mask[i, j, k, l] = 1
        return co_mut_mask, anti_sym_mask

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



