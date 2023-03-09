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

    def test_q_Z_uniform_prior_and_observations_update_gives_uniform_q_Z(self):
        self.q_Z_test.initialize('uniform')
        self.q_C_test.compute_filtering_probs()
        observations = torch.ones((self.M, self.N)) * 10.
        self.q_Z_test.update(self.q_mu_tau_test, self.q_C_test, self.q_pi_test, observations)

        self.assertTrue(torch.allclose(self.q_Z_test.pi, self.q_Z_test.pi[:, :]),
                        msg=f"Categorical probabilites of Z not equal for uniform prior and observations: {self.q_Z_test.pi[0,:]}")

    def test_ELBO_greater_for_uniform_qZ_than_confident_qZ_when_pi_skewed_away_from_confident_cluster(self):
        # Arrange
        q_pi = qPi(self.config)
        q_pi.concentration_param = torch.ones(self.K)
        q_pi.concentration_param[1:] = 10
        config_1 = Config(n_nodes=self.K, chain_length=self.M, n_cells=self.N, n_states=self.A)
        q_Z_1 = qZ(config_1)
        q_Z_1.initialize(method='uniform')
        config_2 = Config(n_nodes=self.K, chain_length=self.M, n_cells=self.N, n_states=self.A)
        q_Z_2 = qZ(config_2)
        q_Z_2.initialize()
        q_Z_2.pi = torch.zeros((self.N, self.K))
        q_Z_2.pi[:, 0] = 1

        # Act
        res_1 = q_Z_1.elbo(q_pi)
        res_2 = q_Z_2.elbo(q_pi)

        # Assert
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
        pi_1.concentration_param[skew_cluster_idx] = 1.1
        pi_2 = qPi(self.config)
        pi_2.concentration_param[skew_cluster_idx] = 1.1

        q_Z_1 = qZ(self.config)
        q_Z_2 = qZ(self.config)
        q_Z_1.initialize('uniform')
        q_Z_2.initialize('uniform')

        q_Z_2.pi = torch.ones((self.N, self.K)) / self.K
        q_Z_2.pi[:, skew_cluster_idx] = 0.11
        q_Z_2.pi = q_Z_2.pi / q_Z_2.pi.sum(dim=1, keepdim=True)

        res_1 = q_Z_1.elbo(pi_1)
        res_2 = q_Z_2.elbo(pi_2)
        print(f"Uniform: {res_1} \n Skewed: {res_2}")
        self.assertLess(res_1, res_2, f"ELBO for uniform assignment over clusters greater than all probability to one cluster")

    def test_qZ_kmeans_init(self):
        # Arrange
        sim_joint_q = utils_testing.generate_test_dataset_fixed_tree()
        obs = sim_joint_q.obs

        qz = qZ(sim_joint_q.config).initialize(method='uniform')
        unif_elbo = qz.elbo(sim_joint_q.pi)

        qz.initialize(method='random')
        rand_elbo = qz.elbo(sim_joint_q.pi)

        qz.initialize(method='kmeans', obs=obs)
        kmeans_elbo = qz.elbo(sim_joint_q.pi)

        print(unif_elbo)
        print(rand_elbo)
        print(kmeans_elbo)
        self.assertGreater(kmeans_elbo, unif_elbo, f"ELBO for uniform assignment over clusters is greater than KMeans")
        self.assertGreater(kmeans_elbo, rand_elbo, f"ELBO for random assignment over clusters is greater than KMeans")

    def test_summary(self):
        print(self.q_Z_test)

