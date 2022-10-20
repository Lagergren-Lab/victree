import unittest

import networkx as nx
import torch

from tests import utils_testing
from utils import tree_utils
from variational_distributions.q_T import q_T


class qTTestCase(unittest.TestCase):

    def setUp(self) -> None:
        L = 5
        self.q_T = q_T(L)

    def test_simple_T(self):
        M = 20
        A = 5
        T_list, q_C_pairwise_marginals = utils_testing.get_two_simple_trees_with_random_qCs(M, A)
        q_C = torch.tensor([1.0])
        q_epsilon = torch.tensor([2.0])
        # Act
        log_q_T = self.q_T.update_CAVI(T_list, q_C_pairwise_marginals, q_C, q_epsilon)

        # Assert
        print(log_q_T)