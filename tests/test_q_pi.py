import unittest
import torch

from utils.config import Config
from variational_distributions.var_dists import qZ, qPi

class qPiTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.K = 5
        self.config = Config(n_nodes=self.K)
        self.q_pi = qPi(self.config)

    def test_q_pi_params_increase_with_more_uniform_data(self):
        # Arrange
        N_1 = 10
        N_2 = 100
        config_1 = Config(n_cells=N_1)
        config_2 = Config(n_cells=N_2)
        q_pi_1 = qPi(config_1)
        q_pi_2 = qPi(config_2)

        q_z_1 = qZ(config_1)  # uniform
        q_z_2 = qZ(config_2)  # uniform

        # Act
        q_pi_1.update(q_z_1)
        q_pi_2.update(q_z_2)

        # Assert
        assert [a1 < a2 for (a1, a2) in zip(q_pi_1.concentration_param, 
                                            q_pi_2.concentration_param)]

    def test_q_pi_higher_entropy_for_uniform_data_vs_random(self):
        # Arrange
        N = 200
        K = 10
        config = Config(n_nodes=K, n_cells=N)
        q_pi_1 = qPi(config)
        q_pi_2 = qPi(config)

        q_z_1 = qZ(config)  # uniform
        q_z_1.initialize(z_init='uniform')
        q_z_probs_2 = torch.rand((N, K))
        q_z_probs_2 = q_z_probs_2 / torch.sum(q_z_probs_2, dim=-1).unsqueeze(-1)
        q_z_2 = qZ(config)
        q_z_2.pi = q_z_probs_2

        # Act
        q_pi_1.update(q_z_1)
        q_pi_2.update(q_z_2)

        ent_1 = q_pi_1.entropy()
        ent_2 = q_pi_2.entropy()

        # Assert
        print(f"ent_1 (uniform): {ent_1}")
        print(f"ent_2 (random): {ent_2}")
        assert ent_1 > ent_2

    def test_q_pi_increased_weight_for_observed_k(self):
        # Arrange
        N = 20
        K = 10
        config = Config(n_nodes=K, n_cells=N)
        q_pi = qPi(config)

        q_z = qZ(config)  # uniform
        q_z_probs = torch.zeros(N, K)
        q_z_probs[:, 1] = 1
        q_z.pi = q_z_probs

        # Act
        q_pi.update(q_z)
        delta = q_pi.concentration_param

        # Assert
        assert all(delta[1] >= delta)
        print(q_pi)

    def test_cross_entropy(self):
        self.q_pi.concentration_param = torch.ones(self.K)*2
        res = self.q_pi.neg_cross_entropy()
        print(f"Cross entropy: {res}")

    def test_entropy(self):
        self.q_pi.concentration_param = torch.ones(self.K)*2
        res = self.q_pi.entropy()
        print(f"Entropy: {res}")

    def test_ELBO(self):
        self.q_pi.concentration_param = torch.ones(self.K)*2
        res = self.q_pi.compute_elbo()
        print(f"ELBO: {res}")
