import unittest

import networkx as nx
import torch

from utils import tree_utils
from variational_distributions.q_epsilon import qEpsilon


class qEpsilonTestCase(unittest.TestCase):

    def setUp(self) -> None:
        L = 5
        a = 1
        b = 1
        self.q_epsilon = qEpsilon(a, b)

    def test_simple_T(self):
        # Arange
        T_1 = nx.DiGraph()
        T_2 = nx.DiGraph()
        T_1.add_edge(0, 1)
        T_1.add_edge(0, 2)
        T_2.add_edge(0, 1)
        T_2.add_edge(1, 2)
        T_list = [T_1, T_2]
        w_T = torch.tensor([0.3, 0.7])
        M = 20
        A = 5
        N = 3
        q_C_0_init = torch.zeros(A)
        q_C_0_init[2] = 1.0
        q_C_0_transitions = torch.rand((M, A, A))
        q_C_0_pairwise_marginals = tree_utils.forward_backward_markov_chain(q_C_0_init, q_C_0_transitions, M)
        q_C_1_init = torch.rand(A)
        q_C_1_transitions = torch.rand((M, A, A))
        q_C_1_pairwise_marginals = tree_utils.forward_backward_markov_chain(q_C_1_init, q_C_1_transitions, M)
        q_C_2_init = torch.rand(A)
        q_C_2_transitions = torch.rand((M, A, A))
        q_C_2_pairwise_marginals = tree_utils.forward_backward_markov_chain(q_C_2_init, q_C_2_transitions, M)
        q_C_pairwise_marginals = torch.zeros(N, M-1, A, A)
        q_C_pairwise_marginals[0] = q_C_0_pairwise_marginals
        q_C_pairwise_marginals[1] = q_C_1_pairwise_marginals
        q_C_pairwise_marginals[2] = q_C_2_pairwise_marginals

        q_C = torch.tensor([1.0])
        q_epsilon = torch.tensor([2.0])
        # Act
        log_a, log_b = self.q_epsilon.update_CAVI(T_list, w_T, q_C_pairwise_marginals)

        # Assert
        print(log_q_T)