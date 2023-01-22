import unittest
import torch

from utils.config import Config
from tests import utils_testing
from variational_distributions.var_dists import qZ, qPi, qMuTau, qC


class qZTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.alpha_prior = 1
        self.K = 10
        self.N = 20
        self.M = 5
        self.A = 4
        self.config = Config(n_nodes=self.K, chain_length=self.M, n_cells=self.N, n_states=self.A)
        self.q_Z_test = qZ(self.config)
        self.q_Z_test.initialize()
        self.q_pi_test = qPi(self.config)
        self.q_mu_tau_test = qMuTau(self.config)
        self.q_mu_tau_test.initialize()
        self.q_C_test = qC(self.config)
        self.q_C_test.eta1 = torch.zeros_like(self.q_C_test.eta1) - torch.log(torch.tensor(self.A))
        self.q_C_test.eta2 = torch.zeros_like(self.q_C_test.eta2) - torch.log(torch.tensor(self.A))

    def test_q_Z_uniform_prior_and_observations(self):
        observations = torch.ones((self.M, self.N))
        self.q_Z_test.update(self.q_mu_tau_test, self.q_C_test, self.q_pi_test, observations)

        # same cat probs for first cell against all others ?
        # print(self.q_Z_test.pi)
        # FIXME: nans in pi for some cells
        self.assertTrue(torch.allclose(self.q_Z_test.pi, self.q_Z_test.pi[0, :]),
                        msg=f"Categorical probabilites of Z not equal for uniform prior and observations: {self.q_Z_test.pi}")

    def test_ELBO_greater_for_uniform_qZ_than_skewed_qZ_when_pi_uniform(self):
        res_1 = self.q_Z_test.elbo(self.q_pi_test)
        config_2 = Config(n_nodes=self.K, chain_length=self.M, n_cells=self.N, n_states=self.A)
        q_Z_2 = qZ(config_2)
        q_Z_2.initialize()
        q_Z_2.pi = torch.zeros((self.N, self.K))
        q_Z_2.pi[:, 0] = 1
        res_2 = q_Z_2.elbo(self.q_pi_test)
        self.assertGreater(res_1, res_2, f"ELBO for uniform assignment over clusters smaller than all probability to one cluster")

    def test_ELBO_lower_for_uniform_qZ_than_skewed_qZ_when_pi_skewed(self):
        skew_cluster_idx = 2
        pi_1 = qPi(self.config)
        pi_1.concentration_param[skew_cluster_idx] = 100
        pi_2 = qPi(self.config)
        pi_2.concentration_param[skew_cluster_idx] = 100

        q_Z_1 = qZ(self.config)
        q_Z_1.initialize()
        q_Z_2 = qZ(self.config)
        q_Z_2.initialize()

        q_Z_2.pi = torch.zeros((self.N, self.K))
        q_Z_2.pi[:, skew_cluster_idx] = 1

        res_1 = q_Z_1.elbo(pi_1)
        res_2 = q_Z_2.elbo(pi_2)
        print(f"Uniform: {res_1} \n Skewed: {res_2}")
        self.assertLess(res_1, res_2, f"ELBO for uniform assignment over clusters greater than all probability to one cluster")

    def test_ELBO_lower_for_uniform_qZ_than_slightly_skewed_qZ_when_pi_small_skew(self):
        skew_cluster_idx = 2
        pi_1 = qPi(self.config)
        pi_1.concentration_param[skew_cluster_idx] = 10
        pi_2 = qPi(self.config)
        pi_2.concentration_param[skew_cluster_idx] = 10

        q_Z_1 = qZ(self.config)
        q_Z_1.initialize()
        q_Z_2 = qZ(self.config)
        q_Z_2.initialize()

        q_Z_2.pi = torch.ones((self.N, self.K)) / self.K
        q_Z_2.pi[:, skew_cluster_idx] = 0.15

        res_1 = q_Z_1.elbo(pi_1)
        res_2 = q_Z_2.elbo(pi_2)
        print(f"Uniform: {res_1} \n Skewed: {res_2}")
        self.assertLess(res_1, res_2, f"ELBO for uniform assignment over clusters greater than all probability to one cluster")

