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
        config = Config(n_nodes=self.K, chain_length=self.M, n_cells=self.N, n_states=self.A)
        self.q_Z_test = qZ(config)
        self.q_Z_test.initialize()
        self.q_pi_test = qPi(config)
        self.q_mu_tau_test = qMuTau(config)
        self.q_C_test = qC(config)
        self.q_C_test.eta1 = torch.ones((self.K, self.A)) / self.A
        self.q_C_test.eta2 = torch.ones((self.K, self.M, self.A, self.A)) / self.A

    def test_q_Z_uniform_prior_and_observations(self):
        observations = torch.ones((self.M, self.N))
        self.q_Z_test.update(self.q_mu_tau_test, self.q_C_test, self.q_pi_test, observations)

        self.assertTrue(torch.allclose(self.q_Z_test.pi, self.q_Z_test.pi[0]),
                        msg="Categorical probabilites of Z not equal for uniform prior and observations")


    def test_ELBO(self):
        res = self.q_Z_test.elbo(self.q_pi_test)
        print(f"ELBO: {res}")
