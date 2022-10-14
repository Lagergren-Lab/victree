import torch

from utils import tree_utils
from variational_distributions.variational_distribution import VariationalDistribution


class q_T(VariationalDistribution):

    def __init__(self, L):
        self.L = L

    def update(self, T_list, q_C):
        q_T = self.update_CAVI(T_list, q_C)
        return q_T

    def update_CAVI(self, T_list, q_C: torch.Tensor, a, b):
        """
        log q(T) =
        (1) E_{C^r}[log p(C^r)] +
        (2) sum_{uv in A(T)} E_{C^u, C^v, epsilon}[log p(C^u | C^v, epsilon)] +
        (3) log p(T)
        :param T_list: list of tree topologies (L x 1)
        :param q_C: (M-1 x A x A)
        :param a:
        :param b:
        :return:
        """

        N, M, A, = q_C.size()
        # pre-steps: get unique edges
        unique_edges = tree_utils.get_unique_edges(T_list)

        # Term (1) - expectation over root node


        # Term (2)
        E_eps_h_comut = torch.digamma(b) - torch.digamma(a + b)
        E_eps_h_diff = torch.digamma(b) - torch.digamma(a + b) - torch.log(A)
        E_eps_h = torch.ones(M, A, A) * E_eps_h_diff
        E_eps_h.fill_diagonal_(E_eps_h_comut)

        for uv in unique_edges:
            u, v = uv
            E_CuCveps = torch.einsum('mij -> k', q_C, E_eps_h)

        return q_T