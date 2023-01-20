import unittest

import torch

from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qEpsilon, qZ, qMuTau, qC


class qmtTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # build config
        self.K = 3
        self.M = 150
        self.N = 100
        self.A = 7
        self.config = Config(n_nodes=self.K, chain_length=self.M, n_cells=self.N, n_states=self.A)
        self.qc = qC(self.config)
        self.qc.initialize()
        self.qt = qT(self.config)
        self.qeps = qEpsilonMulti(self.config, 2, 5)  # skewed towards 0
        self.qz = qZ(self.config)
        self.qz.initialize()
        self.mu_init = 0
        self.prec_factor_init = 1
        self.alpha_init = 1
        self.beta_init = 3
        self.qmt = qMuTau(self.config)
        self.qmt.initialize(loc=self.mu_init, precision_factor=self.prec_factor_init,
                            shape=self.alpha_init, rate=self.beta_init)

    def test_update_mu_greater_than_init_for_observations_greater_than_init(self):
        obs = torch.randint(low=200, high=250, size=(self.config.chain_length, self.config.n_cells), dtype=torch.float)
        self.qc.single_filtering_probs = torch.zeros((self.K, self.M, self.A))
        self.qc.single_filtering_probs[:, :, 1] = 2
        mu, lmbda, alpha, beta = self.qmt.update(qc=self.qc, qz=self.qz, obs=obs)
        self.assertTrue(torch.greater_equal(torch.mean(mu), self.mu_init), msg=f"mu smaller after update for "
                                                                               f"observations larger than mu init. "
                                                                               f" \n mu updated: {mu} - mu init {self.mu_init}")
        L = 2
        tau_dist = torch.distributions.Gamma(alpha, beta)
        tau_sample = tau_dist.sample((L, 1))
        mu_dist = torch.distributions.Normal(mu, 1./(tau_sample*lmbda))
        mu_sample = mu_dist.sample((L, 1))
        print(f"mu: {mu_sample}")

    def test_update_beta(self):
        n_iter = 3
        #obs = torch.randint(low=200, high=250, size=(self.config.chain_length, self.config.n_cells), dtype=torch.float)
        obs_rv = torch.distributions.Normal(loc=0, scale=3)
        obs = obs_rv.sample((self.config.chain_length, self.config.n_cells))
        sum_M_y2 = torch.sum(obs ** 2, dim=0)
        self.qc.single_filtering_probs = torch.zeros((self.K, self.M, self.A))
        self.qc.single_filtering_probs[:, :, 1] = 2
        for i in range(n_iter):
            mu, lmbda, alpha, beta = self.qmt.update(qc=self.qc, qz=self.qz, obs=obs)
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

    def test_log_emissions(self):
        C = torch.ones(self.M) * 2.
        C[0:50] = 6.
        mu = torch.ones(self.N) * 1
        tau = 1
        obs_rv = torch.distributions.Normal(loc=torch.outer(C, mu), scale=1/tau)
        #obs_gc_scaled = torch.empty((self.M, self.N))
        #torch.nn.init.trunc_normal_(obs_gc_scaled, mean=torch.outer(C, mu), std=1/tau)
        obs = obs_rv.sample()
        qmt = qMuTau(config=self.config)
        mu_init = 1
        prec_factor_init = 1
        alpha_init = 10
        beta_init = 10
        qmt.initialize(loc=mu_init, precision_factor=prec_factor_init,
                       shape=alpha_init, rate=beta_init)
        exp_log_emission = qmt.exp_log_emission(obs)
        print(f"E[log p(y|C)] {exp_log_emission[0,0,:]}")
        print(f"E[log p(y|C)] {exp_log_emission[0,51,:]}")
        # Log emission always largest for C=0 - is that correct?

    def test_update(self):
        # design simple test: fix all other variables
        # and update the q_mutau params
        cells_per_clone = 10
        cfg = Config(n_nodes=3, n_states=5, n_cells=3 * cells_per_clone, chain_length=10,
                     wis_sample_size=2, debug=True)
        qmt = qMuTau(cfg)
        # uninformative initialization of mu0, tau0, alpha0, beta0
        qmt.initialize(loc=0, precision_factor=.1, rate=.5, shape=.5)

        # obs with 15 cells, 5 each to different clone
        # in order, clone 0, 1, 2
        cn_profile = torch.tensor(
            [[2] * 10,
             [2] * 4 + [3] * 6,
             [1] * 8 + [2] * 2]
        )
        # cell assignments
        true_z = torch.tensor([0] * cells_per_clone +
                              [1] * cells_per_clone +
                              [2] * cells_per_clone)

        cell_cn_profile = cn_profile[true_z, :].float()
        self.assertEqual(cell_cn_profile.shape, (cfg.n_cells, cfg.chain_length))

        true_mu = torch.distributions.Normal(1, .1).rsample((cfg.n_cells, 1))
        obs = (cell_cn_profile * true_mu).T

        self.assertEqual(obs.shape, (cfg.chain_length, cfg.n_cells))

        # give true values to the other required dists
        # i.e. qc, qz
        fix_qc = qC(cfg, true_params={
            "c": cn_profile
        })
        fix_qz = qZ(cfg, true_params={
            "z": true_z
        })

        for i in range(10):
            qmt.update(fix_qc, fix_qz, obs)

        # this should be equal (or close) to 100.
        print(f'beta: {qmt.beta}')
        print(qmt.nu)
        # this should be very high (there is almost no variance in the emissions)
        print(qmt.exp_tau())
        print(true_mu)


