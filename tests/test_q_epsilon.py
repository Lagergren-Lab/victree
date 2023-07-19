import itertools
import unittest

import networkx as nx
import torch

from utils.config import Config
from tests import utils_testing
from variational_distributions.var_dists import qEpsilon, qEpsilonMulti, qC


class qEpsilonTestCase(unittest.TestCase):

    def setUp(self) -> None:
        a = 1.
        b = 1.
        self.config = Config()
        # self.qeps = qEpsilon(self.config, a, b)
        self.qeps = qEpsilonMulti(self.config, a, b).initialize()

    def test_q_epsilon_running_for_two_simple_Ts_random_qC(self):
        # Arange
        M=20
        A=5
        N=3

        config = Config(n_nodes=N, n_states=A, chain_length=M)

        T_list, q_C_pairwise_marginals = utils_testing.get_random_q_C(M, A)
        w_T = torch.tensor([0.3, 0.7])

        qc = qC(config)
        qc.couple_filtering_probs = q_C_pairwise_marginals
        # Act
        a, b = self.qeps.update_CAVI(T_list, w_T, qc)

        # Assert
        print(f"Beta param a: {a}")
        print(f"Beta param b: {b}")

        elbo = self.qeps.compute_elbo([T_list[0]], w_T[0])
        print(f"ELBO: {elbo}")

    def test_expectation_size(self):
        for u in range(self.config.n_nodes):
            for v in range(self.config.n_nodes):
                if u != v and v != 0:
                    exp_zipping = self.qeps.exp_log_zipping((u, v))
                    self.assertEqual(exp_zipping.shape, (self.config.n_states,) * 4)

    def test_h_eps0(self):
        heps_0_marg = torch.sum(self.qeps.h_eps0(), dim=-1)
        self.assertTrue(torch.allclose(heps_0_marg,
                                       torch.ones(self.config.n_states)))

    def test_q_epsilon_ELBO_larger_for_no_mutation(self):
        # Arange
        K = 2
        M = 10
        A = 5

        config = Config(n_nodes=K, n_states=A, chain_length=M)
        q_eps1 = qEpsilonMulti(config, alpha_prior=1., beta_prior=100.)
        q_eps2 = qEpsilonMulti(config, alpha_prior=1., beta_prior=100.)
        q_eps1.initialize("non_mutation")
        q_eps2.initialize("non_mutation")
        T = nx.DiGraph()
        T.add_edge(0, 1)
        w_T = [1.]
        q_C1 = qC(config)
        q_C2 = qC(config)
        q_C1.couple_filtering_probs[0] = torch.diag(torch.ones(A))
        q_C1.couple_filtering_probs[1] = torch.diag(torch.ones(A))
        q_C2.couple_filtering_probs[0] = torch.rand((A, A))
        q_C2.couple_filtering_probs[1] = torch.rand((A, A))

        # Act
        q1_partial_elbo = q_eps1.compute_elbo([T], w_T)
        q2_partial_elbo = q_eps2.compute_elbo([T], w_T)
        self.assertEqual(q1_partial_elbo, q2_partial_elbo)

        q_eps1.update([T], w_T, q_C1)
        q_eps2.update([T], w_T, q_C2)
        elbo_1 = q_eps1.compute_elbo([T], w_T)
        elbo_2 = q_eps2.compute_elbo([T], w_T)

        # Assert
        self.assertTrue(elbo_1 > elbo_2, msg="ELBO for no q(Epsilon) lower for ")