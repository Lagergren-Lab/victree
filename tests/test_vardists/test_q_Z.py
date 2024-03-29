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
        self.config = Config(n_nodes=self.K, n_states=self.A, n_cells=self.N, chain_length=self.M)
        self.q_Z_test = qZ(self.config)
        self.q_Z_test.initialize()
        self.q_pi_test = qPi(self.config)
        self.q_pi_test.initialize()
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
        config_1 = Config(n_nodes=self.K, n_states=self.A, n_cells=self.N, chain_length=self.M)
        q_Z_1 = qZ(config_1)
        q_Z_1.initialize(z_init='uniform')
        config_2 = Config(n_nodes=self.K, n_states=self.A, n_cells=self.N, chain_length=self.M)
        q_Z_2 = qZ(config_2)
        q_Z_2.initialize()
        q_Z_2.pi = torch.zeros((self.N, self.K))
        q_Z_2.pi[:, 0] = 1

        # Act
        res_1 = q_Z_1.compute_elbo(q_pi)
        res_2 = q_Z_2.compute_elbo(q_pi)

        # Assert
        self.assertGreater(res_1, res_2, f"ELBO for uniform assignment over clusters smaller than all probability to one cluster")

    def test_ELBO_lower_for_uniform_qZ_than_skewed_qZ_when_pi_skewed(self):
        skew_cluster_idx = 2
        pi_1 = qPi(self.config)
        pi_1.concentration_param = torch.ones(self.config.n_nodes)
        pi_1.concentration_param[skew_cluster_idx] = 100
        pi_2 = qPi(self.config)
        pi_2.concentration_param = torch.ones(self.config.n_nodes)
        pi_2.concentration_param[skew_cluster_idx] = 100

        q_Z_1 = qZ(self.config)
        q_Z_1.initialize()
        q_Z_2 = qZ(self.config)
        q_Z_2.initialize()

        q_Z_2.pi = torch.zeros((self.N, self.K))
        q_Z_2.pi[:, skew_cluster_idx] = 1

        res_1 = q_Z_1.compute_elbo(pi_1)
        res_2 = q_Z_2.compute_elbo(pi_2)
        print(f"Uniform: {res_1} \n Skewed: {res_2}")
        self.assertLess(res_1, res_2, f"ELBO for uniform assignment over clusters greater than all probability to one cluster")

    def test_ELBO_lower_for_uniform_qZ_than_slightly_skewed_qZ_when_pi_small_skew(self):
        skew_cluster_idx = 2
        pi_1 = qPi(self.config)
        pi_1.concentration_param = torch.ones(self.config.n_nodes)
        pi_1.concentration_param[skew_cluster_idx] = 1.1
        pi_2 = qPi(self.config)
        pi_2.concentration_param = torch.ones(self.config.n_nodes)
        pi_2.concentration_param[skew_cluster_idx] = 1.1

        q_Z_1 = qZ(self.config)
        q_Z_2 = qZ(self.config)
        q_Z_1.initialize('uniform')
        q_Z_2.initialize('uniform')

        q_Z_2.pi = torch.ones((self.N, self.K)) / self.K
        q_Z_2.pi[:, skew_cluster_idx] = 0.11
        q_Z_2.pi = q_Z_2.pi / q_Z_2.pi.sum(dim=1, keepdim=True)

        res_1 = q_Z_1.compute_elbo(pi_1)
        res_2 = q_Z_2.compute_elbo(pi_2)
        print(f"Uniform: {res_1} \n Skewed: {res_2}")
        self.assertLess(res_1, res_2, f"ELBO for uniform assignment over clusters greater than all probability to one cluster")

    @unittest.skip('broken-test')
    def test_qZ_kmeans_init(self):
        # this tests asserts that, after one cavi update,
        # the distribution initialized by kmeans improves by a
        # smaller amount than the others, i.e. it's closer
        # to convergence
        sim_joint_q = utils_testing.generate_test_dataset_fixed_tree()
        obs = sim_joint_q.obs

        qz = qZ(sim_joint_q.config).initialize(z_init='uniform')
        elbo0 = qz.compute_elbo(sim_joint_q.pi)
        qz.update(sim_joint_q.mt, sim_joint_q.c, sim_joint_q.pi, sim_joint_q.obs)
        unif_elbo_impr = - (qz.compute_elbo(sim_joint_q.pi) - elbo0) / elbo0

        qz = qZ(sim_joint_q.config).initialize(z_init='random')
        elbo0 = qz.compute_elbo(sim_joint_q.pi)
        qz.update(sim_joint_q.mt, sim_joint_q.c, sim_joint_q.pi, sim_joint_q.obs)
        rand_elbo_impr = - (qz.compute_elbo(sim_joint_q.pi) - elbo0) / elbo0

        qz = qZ(sim_joint_q.config).initialize(z_init='gmm', data=obs)
        elbo0 = qz.compute_elbo(sim_joint_q.pi)
        qz.update(sim_joint_q.mt, sim_joint_q.c, sim_joint_q.pi, sim_joint_q.obs)
        kmeans_elbo_impr = - (qz.compute_elbo(sim_joint_q.pi) - elbo0) / elbo0

        print(unif_elbo_impr)
        print(rand_elbo_impr)
        print(kmeans_elbo_impr)
        self.assertLess(kmeans_elbo_impr, unif_elbo_impr, f"ELBO for uniform assignment over clusters is greater than GMM")
        self.assertLess(kmeans_elbo_impr, rand_elbo_impr, f"ELBO for random assignment over clusters is greater than GMM")

    def test_summary(self):
        print(self.q_Z_test)

