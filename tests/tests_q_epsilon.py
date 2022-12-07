import unittest
import torch

from utils.config import Config
from tests import utils_testing
from variational_distributions.var_dists import qEpsilon, qEpsilonMulti


class qEpsilonTestCase(unittest.TestCase):

    def setUp(self) -> None:
        L = 5
        a = 1
        b = 1
        self.config = Config()
        # self.qeps = qEpsilon(self.config, a, b)
        self.qeps = qEpsilonMulti(self.config, a, b)

    def test_q_epsilon_running_for_two_simple_Ts_random_qC(self):
        # Arange
        M=20
        A=5
        T_list, q_C_pairwise_marginals = utils_testing.get_two_simple_trees_with_random_qCs(M, A)
        w_T = torch.tensor([0.3, 0.7])

        # Act
        a, b = self.qeps.update_CAVI(T_list, w_T, q_C_pairwise_marginals)

        # Assert
        print(f"Beta param a: {a}")
        print(f"Beta param b: {b}")

    def test_expectation_size(self):
        for u in range(self.config.n_nodes):
            for v in range(self.config.n_nodes):
                if u != v:
                    exp_zipping = self.qeps.exp_zipping((u,v))
                    self.assertEqual(exp_zipping.shape, (self.config.n_states,) * 4)


