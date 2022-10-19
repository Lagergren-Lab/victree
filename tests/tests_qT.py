import unittest

import networkx as nx
import torch

from utils import tree_utils
from variational_distributions.q_T import q_T


class qTTestCase(unittest.TestCase):

    def setUp(self) -> None:
        L = 5
        self.q_T = q_T(L)

    def test_simple_T(self):
        T = nx.DiGraph()
        T.add_edge(0, 1)
        T.add_edge(0, 2)
        M = 5
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
        q_epsilon = torch.tensor([1.0])
        self.q_T.update_CAVI([T], q_C_pairwise_marginals, q_C, q_epsilon)