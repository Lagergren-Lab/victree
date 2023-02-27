import unittest

import torch

from simul import generate_dataset_var_tree
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
        self.mu_init = 100
        self.prec_factor_init = 1
        self.alpha_init = 1
        self.beta_init = 3
        self.qmt = qMuTau(self.config)
        self.qmt.initialize(loc=self.mu_init, precision_factor=self.prec_factor_init,
                            shape=self.alpha_init, rate=self.beta_init)

    def test_update_mu_greater_than_init_for_observations_greater_than_init(self):
        obs = torch.randint(low=200, high=250, size=(self.config.chain_length, self.config.n_cells), dtype=torch.float)
        self.qc.single_filtering_probs = torch.zeros((self.K, self.M, self.A))
        self.qc.single_filtering_probs[:, :, 1] = 1.
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
        obs_rv = torch.distributions.Normal(loc=200, scale=3)
        obs = obs_rv.sample((self.config.chain_length, self.config.n_cells))
        sum_M_y2 = torch.sum(obs ** 2, dim=0)
        self.qc.single_filtering_probs = torch.zeros((self.K, self.M, self.A))
        self.qc.single_filtering_probs[:, :, 1] = 2
        for i in range(n_iter):
            mu, lmbda, alpha, beta = self.qmt.update(qc=self.qc, qz=self.qz, obs=obs)
            print(f"mu: {mu[0]} beta: {beta[0]}")
        # FIXME: mu_init = 100, while observation are Normal(0, 9)
        #   probably not a good idea to keep self.mu_init for all tests
        #   solution: initialize qmt in this function and check that mu approaches observations
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
        K = 3
        M = 10
        N = 5
        A = 7
        config = Config(n_nodes=K, chain_length=M, n_cells=N, n_states=A)
        C = torch.ones(M) * 2.
        C[0:50] = 6.
        mu = torch.ones(N) * 1
        tau = 1
        obs_rv = torch.distributions.Normal(loc=torch.outer(C, mu), scale=1/tau)
        #obs_gc_scaled = torch.empty((self.M, self.N))
        #torch.nn.init.trunc_normal_(obs_gc_scaled, mean=torch.outer(C, mu), std=1/tau)
        obs = torch.ones((M, N))*6. #obs_rv.sample()
        qmt = qMuTau(config=config)
        mu_init = 1
        prec_factor_init = 10
        alpha_init = 2
        beta_init = 1
        qmt.initialize(loc=mu_init, precision_factor=prec_factor_init,
                       shape=alpha_init, rate=beta_init)
        exp_log_emission = qmt.exp_log_emission(obs)

        for n in range(N):
            for m in range(M):
                self.assertTrue(torch.argmax(exp_log_emission) == 6, msg=f"E_mu_tau[log p(y_{m,n}|C)] {exp_log_emission[n,m,:]}")
        # Log emission always largest for C=0 - is that correct?


    def test_entropy(self):
        obs = torch.randint(low=10, high=20, size=(self.config.chain_length, self.config.n_cells), dtype=torch.float)
        eps = 1e-5
        self.qc.single_filtering_probs = torch.zeros((self.K, self.M, self.A)) + eps
        self.qc.single_filtering_probs[:, :, 1] = 1 - eps
        self.qmt.update(qc=self.qc, qz=self.qz, obs=obs)
        elbo_qmt = self.qmt.elbo()
        print(f"ELBO(mu, tau): {elbo_qmt}")

    def test_elbo(self):

        joint_q = generate_dataset_var_tree(Config())
        qmt = qMuTau(joint_q.config).initialize(method='fixed')  # with defaults
        for i in range(10):
            alt_elbo_qmt = self.qmt.elbo_alt()
            elbo_qmt = self.qmt.elbo()
            qmt.update(qc=joint_q.c, qz=joint_q.z, obs=joint_q.obs)
            print(f"ELBO(mu, tau): {elbo_qmt}")
            print(f"new ELBO(mu, tau): {alt_elbo_qmt}")
