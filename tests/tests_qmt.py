import unittest

import torch

from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qEpsilon, qZ, qMuTau, qC


class qmtTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # build config
        self.K = 3
        self.M = 100
        self.N = 100
        self.A = 7
        self.config = Config(n_nodes=self.K, chain_length=self.M, n_cells=self.N, n_states=self.A)
        self.qc = qC(self.config)
        self.qc.initialize()
        self.qt = qT(self.config)
        self.qeps = qEpsilonMulti(self.config, 2, 5)  # skewed towards 0
        self.qz = qZ(self.config)
        self.qz.initialize()
        self.mu_init = 100
        self.prec_factor_init = .1
        self.alpha_init = 100
        self.beta_init = 100
        self.qmt = qMuTau(self.config, loc=self.mu_init, precision_factor=self.prec_factor_init, shape=self.alpha_init,
                          rate=self.beta_init)

    def test_update_mu_greater_than_init_for_observations_greater_than_init(self):
        obs = torch.randint(low=200, high=250, size=(self.config.chain_length, self.config.n_cells), dtype=torch.float)
        sum_M_y2 = torch.sum(obs ** 2, dim=0)
        self.qc.single_filtering_probs = torch.zeros((self.K, self.M, self.A))
        self.qc.single_filtering_probs[:, :, 1] = 2
        mu, lmbda, alpha, beta = self.qmt.update(qc=self.qc, qz=self.qz, obs=obs, sum_M_y2=sum_M_y2)
        self.assertTrue(torch.greater_equal(torch.mean(mu), self.mu_init), msg=f"mu smaller after update for "
                                                                               f"observations larger than mu init. "
                                                                               f" \n mu updated: {mu} - mu init {self.mu_init}")

    def test_update_beta(self):
        n_iter = 10
        obs = torch.randint(low=200, high=250, size=(self.config.chain_length, self.config.n_cells), dtype=torch.float)
        sum_M_y2 = torch.sum(obs ** 2, dim=0)
        self.qc.single_filtering_probs = torch.zeros((self.K, self.M, self.A))
        self.qc.single_filtering_probs[:, :, 1] = 2
        for i in range(n_iter):
            mu, lmbda, alpha, beta = self.qmt.update(qc=self.qc, qz=self.qz, obs=obs, sum_M_y2=sum_M_y2)
            print(f"mu: {mu[0]} beta: {beta[0]}")
        self.assertTrue(torch.greater_equal(torch.mean(mu), self.mu_init), msg=f"mu smaller after update for "
                                                                               f"observations larger than mu init. "
                                                                               f" \n mu updated: {mu} - mu init {self.mu_init}")

    def test_expectation_size(self):
        obs = torch.randint(low=50, high=150, size=(self.config.chain_length, self.config.n_cells), dtype=torch.float)
        exp_log_emission = self.qmt.exp_log_emission(obs)
        self.assertEqual(exp_log_emission.shape, (self.config.n_cells,
                                                  self.config.chain_length,
                                                  self.config.n_states))
