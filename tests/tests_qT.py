import unittest

import networkx as nx
import torch

from variational_distributions.variational_hmm import CopyNumberHmm
from variational_distributions.q_epsilon import qEpsilon
from utils import tree_utils
from tests import utils_testing
from utils.config import Config
from variational_distributions.q_T import q_T


class qTTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_q_T_running_for_two_simple_Ts_random_qC(self):
        M = 20
        A = 5
        self.config = Config(chain_length=M, n_states=A, n_nodes=3)
        self.q_T = q_T(config=self.config)
        T_list, q_C_pairwise_marginals = utils_testing.get_two_simple_trees_with_random_qCs(M, A)
        q_C = CopyNumberHmm(config=self.config)
        q_epsilon = qEpsilon(config=self.config)
        # Act
        log_q_T = self.q_T.update_CAVI(T_list, q_C_pairwise_marginals, q_C, q_epsilon)

        # Assert
        print(f"log_q_T of T_1 and T_2: {log_q_T}")
