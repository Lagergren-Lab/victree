import unittest

import networkx as nx
import torch

from variational_distributions.q_T import q_T


class qTTestCase(unittest.TestCase):

    def setUp(self) -> None:
        L = 5
        self.q_T = q_T(L)

    def tearDown(self) -> None:
        self.q_T.dispose()


    def test_simple_T(self):
        T = nx.DiGraph()
        T.add_edge(0, 1)
        T.add_edge(0, 2)
        M = 5
        A = 5
        N = 3
        q_C_0 = torch.zeros((M, A))
        q_C_0[:, 2] = 1
        q_C_1 = torch.zeros((M, A))
        q_C_1[:, 2] = 1
        q_C_2 = torch.zeros((M, A))
        q_C_2[:, 2] = 1
        q_C = torch.zeros((N, M, A))
        q_C[0] = q_C_0
        q_C[1] = q_C_1
        q_C[2] = q_C_2
        self.q_T.update_CAVI([T], q_C)