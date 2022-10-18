import torch

from src.utils import tree_utils
from variational_distributions.variational_distribution import VariationalDistribution


class q_T(VariationalDistribution):

    def __init__(self, L):
        self.L = L

    def update(self, T_list, q_C_pairwise_marginals: torch.Tensor, q_C, q_epsilon):
        q_T = self.update_CAVI(T_list, q_C_pairwise_marginals, q_C, q_epsilon)
        return q_T

    def update_CAVI(self, T_list, q_C_pairwise_marginals: torch.Tensor, q_C, q_epsilon):
        """
        log q(T) =
        (1) E_{C^r}[log p(C^r)] +
        (2) sum_{uv in A(T)} E_{C^u, C^v, epsilon}[log p(C^u | C^v, epsilon)] +
        (3) log p(T)
        :param T_list: list of tree topologies (L x 1)
        :param q_C_pairwise_marginals: (M-1 x A x A)
        :param q_C: q(C) variational distribution object
        :param q_epsilon: q(epsilon) variational distribution object
        :return:
        """
        K = len(T_list)
        N, M, A, = q_C_pairwise_marginals.size()
        q_T_tensor = torch.zeros((K,))
        # pre-steps: get unique edges
        unique_edges = tree_utils.get_unique_edges(T_list)

        # Term (1) - expectation over root node


        # Term (2)
        E_eps_h_comut = torch.digamma(q_epsilon) - torch.digamma(q_C + q_epsilon)
        E_eps_h_diff = torch.digamma(q_epsilon) - torch.digamma(q_C + q_epsilon) - torch.log(A)
        E_eps_h = torch.ones(M, A, A) * E_eps_h_diff
        E_eps_h.fill_diagonal_(E_eps_h_comut)

        for uv in unique_edges:
            u, v = uv
            E_CuCveps = torch.einsum('ij,  -> k', q_C_pairwise_marginals[u], q_C_pairwise_marginals[v], E_eps_h)

        return q_T_tensor