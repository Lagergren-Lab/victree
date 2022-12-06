import unittest
import torch
from model.generative_model import GenerativeModel

from utils.config import Config
from variational_distributions.var_dists import qZ, qPi

class qEpsilonTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.alpha_prior = 1

    def test_q_pi_params_increase_with_more_uniform_data(self):
        # Arrange
        N_1 = 10
        N_2 = 100
        config_1 = Config(n_cells=N_1)
        config_2 = Config(n_cells=N_2)
        p_1 = GenerativeModel(config_1)
        p_2 = GenerativeModel(config_2)
        q_pi_1 = qPi(config_1)
        q_pi_2 = qPi(config_2)

        q_z_1 = qZ(config_1)  # uniform
        q_z_2 = qZ(config_2)  # uniform

        # Act
        q_pi_1.update(q_z_1, p_1.delta_pi)
        q_pi_2.update(q_z_2, p_2.delta_pi)

        # Assert
        assert [a1 < a2 for (a1, a2) in zip(q_pi_1.concentration_param, 
                                            q_pi_2.concentration_param)]

    def test_q_pi_higher_entropy_for_uniform_data_vs_random(self):
        # Arrange
        N = 20
        K = 10
        config = Config(n_cells=N, n_nodes=K)
        p = GenerativeModel(config)
        q_pi_1 = qPi(config)
        q_pi_2 = qPi(config)

        q_z_1 = qZ(config)  # uniform
        q_z_1.initialize()
        q_z_probs_2 = torch.rand((N, K))
        q_z_probs_2 = q_z_probs_2 / torch.sum(q_z_probs_2, dim=-1).unsqueeze(-1)
        q_z_2 = qZ(config)
        q_z_2.pi = q_z_probs_2

        # Act
        q_pi_1.update(q_z_1, p.delta_pi)
        q_pi_2.update(q_z_2, p.delta_pi)

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
        p = GenerativeModel(config)
        q_pi = qPi(config)

        q_z = qZ(config)  # uniform
        q_z_probs = torch.zeros(N, K)
        q_z_probs[:, 1] = 1
        q_z.pi = q_z_probs

        # Act
        q_pi.update(q_z, p.delta_pi)
        alpha = q_pi.concentration_param

        # Assert
        assert all(alpha[1] >= alpha)
