import unittest

import networkx as nx
import torch

from tests import utils_testing
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qC, qEpsilon


class qTTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_q_T_running_for_two_simple_Ts_random_qC(self):
        M = 20
        A = 5
        N = 3
        self.config = Config(chain_length=M, n_states=A, n_nodes=N)
        self.q_T = qT(config=self.config)
        T_list, q_C_pairwise_marginals = utils_testing.get_two_simple_trees_with_random_qCs(M, A, N)
        q_C = qC(config=self.config)
        q_C.couple_filtering_probs = q_C_pairwise_marginals
        q_epsilon = qEpsilonMulti(config=self.config)
        # Act
        log_q_T = self.q_T.update_CAVI(T_list, q_C, q_epsilon)

        # Assert
        print(f"log_q_T of T_1 and T_2: {log_q_T}")
