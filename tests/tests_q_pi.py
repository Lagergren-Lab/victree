import unittest
import torch

from utils.config import Config
from tests import utils_testing
import variational_distributions.q_pi as q_pi
import variational_distributions.q_Z as q_Z


class qEpsilonTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.alpha_prior = 1

    def test_q_pi_params_increase_with_more_uniform_data(self):
        # Arrange
        N_1 = 10
        N_2 = 100
        config_1 = Config(n_cells=N_1)
        config_2 = Config(n_cells=N_2)
        q_pi_1 = q_pi.qPi(config_1, self.alpha_prior)
        q_pi_2 = q_pi.qPi(config_2, self.alpha_prior)

        q_z_1 = q_Z.qZ(config_1)  # uniform
        q_z_2 = q_Z.qZ(config_2)  # uniform
        q_z_probs_1 = q_z_1.pi
        q_z_probs_2 = q_z_2.pi

        # Act
        alpha_1 = q_pi_1.update(q_z_probs_1)
        alpha_2 = q_pi_2.update(q_z_probs_2)

        # Assert
        assert [a1 < a2 for (a1, a2) in zip(alpha_1, alpha_2)]

    def test_q_pi_higher_entropy_for_uniform_data_vs_random(self):
        # Arrange
        N = 20
        K = 10
        config = Config(n_cells=N, n_nodes=K)
        q_pi_1 = q_pi.qPi(config, self.alpha_prior)
        q_pi_2 = q_pi.qPi(config, self.alpha_prior)

        q_z_1 = q_Z.qZ(config)  # uniform
        q_z_probs_1 = q_z_1.pi
        q_z_probs_2 = torch.rand((N, K))
        q_z_probs_2 = q_z_probs_2 / torch.sum(q_z_probs_2, dim=-1).unsqueeze(-1)

        # Act
        alpha_1 = q_pi_1.update(q_z_probs_1)
        alpha_2 = q_pi_2.update(q_z_probs_2)

        ent_1 = -torch.sum(q_pi_1.exp_log_pi())
        ent_2 = -torch.sum(q_pi_2.exp_log_pi())

        # Assert
        print(f"ent_1: {ent_1}")
        print(f"ent_2: {ent_2}")
        assert ent_1 < ent_2

    def test_q_pi_increased_weight_for_observed_k(self):
        # Arrange
        N = 20
        K = 10
        config = Config(n_cells=N, n_nodes=K)
        q_pi_1 = q_pi.qPi(config, self.alpha_prior)

        q_z_1 = q_Z.qZ(config)  # uniform
        q_z_probs_1 = torch.zeros(N, K)
        q_z_probs_1[:, 1] = 1

        # Act
        alpha = q_pi_1.update(q_z_probs_1)

        # Assert
        assert all(alpha[1] >= alpha)
