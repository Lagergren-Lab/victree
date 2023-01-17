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
        self.q_C_test = qC(self.config)
        self.q_C_test.eta1 = torch.ones((self.K, self.A)) / self.A
        self.q_C_test.eta2 = torch.ones((self.K, self.M, self.A, self.A)) / self.A

    def test_q_Z_uniform_prior_and_observations(self):
        observations = torch.ones((self.M, self.N))
        self.q_Z_test.update(self.q_mu_tau_test, self.q_C_test, self.q_pi_test, observations)

        self.assertTrue(torch.allclose(self.q_Z_test.pi, self.q_Z_test.pi[0]),
                        msg="Categorical probabilites of Z not equal for uniform prior and observations")

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

    def test_update(self):
        # design simple test: fix all other variables
        # and update the q_z params
        cells_per_clone = 10
        cfg = Config(n_nodes=3, n_states=5, n_cells=3 * cells_per_clone, chain_length=10,
                     wis_sample_size=2, debug=True)
        qz = qZ(cfg)
        # uniform initialization of pi
        qz.initialize(method='random')

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
        true_pi = torch.nn.functional.one_hot(true_z, num_classes=cfg.n_nodes).float()

        cell_cn_profile = cn_profile[true_z, :]
        self.assertEqual(cell_cn_profile.shape, (cfg.n_cells, cfg.chain_length))

        obs = (cell_cn_profile * 100).T
        # introduce some randomness
        obs += torch.distributions.normal.Normal(0, 10).sample(obs.shape).int()

        # give true values to the other required dists
        # i.e. qmt, qc, qpi
        fix_qmt = qMuTau(cfg, true_params={
            "mu": 100 * torch.ones(cfg.n_cells),
            "tau": 1 * torch.ones(cfg.n_cells)
        })

        fix_qc = qC(cfg, true_params={
            "c": cn_profile
        })

        fix_qpi = qPi(cfg, true_params={
            "pi": torch.ones(cfg.n_nodes) / 3.
        })

        for i in range(3):
            qz.update(fix_qmt, fix_qc, fix_qpi, obs)

        self.assertTrue(torch.allclose(true_pi, qz.pi))

