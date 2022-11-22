import unittest
import torch

from utils.config import Config
from tests import utils_testing
import variational_distributions.q_pi as q_pi
import variational_distributions.q_Z as q_Z
import variational_distributions.variational_normal as q_mutau
import variational_distributions.variational_hmm as q_C


class qEpsilonTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.alpha_prior = 1
        self.K = 10
        self.N = 20
        self.M = 5
        config = Config(n_nodes=self.K)
        self.q_Z_test = q_Z.qZ(config)
        self.q_pi_test = q_pi.qPi(config, torch.ones(self.K,))
        self.q_mu_tau_test = q_mutau.qMuTau(config)
        self.q_C_test = q_C.CopyNumberHmm(config)

    def test_q_Z(self):
        observations = torch.ones((self.M, self.N))
        q_C_marginals = self.q_C_test.get
        self.q_Z_test.update(self.q_pi_test, self.q_mu_tau_test, q_C_marginals, observations)
