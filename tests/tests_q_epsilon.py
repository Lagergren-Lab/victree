import unittest

import networkx as nx
import torch

from tests import utils_testing
from utils import tree_utils
from variational_distributions.q_epsilon import qEpsilon


class qEpsilonTestCase(unittest.TestCase):

    def setUp(self) -> None:
        L = 5
        a = 1
        b = 1
        self.q_epsilon = qEpsilon(a, b)

    def test_q_epsilon_running_for_two_simple_Ts_random_qC(self):
        # Arange
        M=20
        A=5
        T_list, q_C_pairwise_marginals = utils_testing.get_two_simple_trees_with_random_qCs(M, A)
        w_T = torch.tensor([0.3, 0.7])

        # Act
        a, b = self.q_epsilon.update_CAVI(T_list, w_T, q_C_pairwise_marginals)

        # Assert
        print(f"Beta param a: {a}")
        print(f"Beta param b: {b}")