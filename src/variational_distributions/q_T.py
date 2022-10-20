import torch

from src.utils import tree_utils


class q_T():

    def __init__(self, L):
        self.L = L

    def update(self, T_list, q_C_pairwise_marginals: torch.Tensor, q_C, q_epsilon):
        q_T = self.update_CAVI(T_list, q_C_pairwise_marginals, q_C, q_epsilon)
        return q_T

    def update_CAVI(self, T_list: list, q_C_pairwise_marginals: torch.Tensor, q_C, q_epsilon):
        """
        log q(T) =
        (1) E_{C^r}[log p(C^r)] +
        (2) sum_{uv in A(T)} E_{C^u, C^v, epsilon}[log p(C^u | C^v, epsilon)] +
        (3) log p(T)
        :param T_list: list of tree topologies (L x 1)
        :param q_C_pairwise_marginals: (N x M-1 x A x A)
        :param q_C: q(C) variational distribution object
        :param q_epsilon: q(epsilon) variational distribution object
        :return: log_q_T_tensor
        """
        K = len(T_list)
        N, M, A, A = q_C_pairwise_marginals.size()
        log_q_T_tensor = torch.zeros((K,))
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(T_list, N)

        # Term (1) - expectation over root node
        # Constant w.r.t. T, can be omitted
        # Term (3)
        # Constant w.r.t T, can be omitted

        # Term (2)
        # TODO: Move to q_epsilon and make u, v specific
        E_eps_h_comut = torch.digamma(q_epsilon) - torch.digamma(q_C + q_epsilon)
        E_eps_h_diff = torch.digamma(q_epsilon) - torch.digamma(q_C + q_epsilon) - torch.log(torch.tensor(A))
        E_eps_h = torch.ones((A, A, A, A)) * E_eps_h_diff
        # idx = torch.ones((A, A, A, A)) * E_eps_h_diff
        # torch.range(0, A)
        # E_eps_h.fill_diagonal_(E_eps_h_comut)
        # TODO: Find effecient way of indexing i-j = k-l
        for i in range(0, A):
            for j in range(0, A):
                for k in range(0, A):
                    for l in range(0, A):
                        if i - j == k - l:
                            E_eps_h[i, j, k, l] = E_eps_h_comut

        E_CuCveps = torch.zeros((N, N))
        for uv in unique_edges:
            u, v = uv
            E_CuCveps[u, v] = torch.einsum('mij, mkl, ijkl  -> ', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v],
                                           E_eps_h)

        for (k, T) in enumerate(T_list):
            for uv in T.edges:
                u, v = uv
                log_q_T_tensor[k] += E_CuCveps[u, v]

        return log_q_T_tensor
