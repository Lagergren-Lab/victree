import torch
import torch.nn.functional as f
import unittest

from simul import simulate_full_dataset
import tests.utils_testing
from utils.config import Config


class simulTestCase(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()


    def test_simulate_full_dataset(self):
        n_cells = 200
        n_sites = 20
        n_copy_states = 5
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        n_nodes = len(tree.nodes)
        mu_0 = 10.0
        lmbda_0 = 2.
        alpha0 = 10.
        beta0 = 40.
        a0 = 10.
        b0 = 40.
        dir_alpha0 = [1., 3., 3.]
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites)
        output_sim = simulate_full_dataset(config, eps_a=a0, eps_b=b0, mu0=mu_0, lambda0=lmbda_0, alpha0=alpha0,
                                           beta0=beta0, dir_delta=dir_alpha0)
        y = output_sim['obs']
        C = output_sim['c']
        z = output_sim['z']
        pi = output_sim['pi']
        mu = output_sim['mu']
        tau = output_sim['tau']
        eps = output_sim['eps']

        # Assert Z
        z_one_hot = f.one_hot(z, num_classes=n_nodes)
        z_avg = torch.sum(z_one_hot, dim=0) / n_cells

        print(f"z_pi: {z_avg} - pi: {pi}")
        if n_cells >= 100:
            assert torch.allclose(z_avg, pi, atol=1e-01)

        # Assert mu and tau
        mu_avg = torch.mean(mu)
        print(f"mu mean: {mu_avg} - expected mu mean: {mu_0}")
        if n_cells >= 100:
            assert torch.isclose(mu_avg, torch.tensor(mu_0), rtol=3 * 1e-01)

        for m in range(n_sites):
            avg_y_m = torch.mean(y[m, :])
            exp_res_m = 0
            for n in range(n_cells):
                exp_res_m += C[z[n], m] * mu[n] * 1 / n_cells

            print(f"avg_y_{m}: {avg_y_m} - expected average: {exp_res_m}")
            if n_cells >= 100:
                self.assertTrue(torch.isclose(avg_y_m, exp_res_m, rtol=3 * 1e-01),
                                msg=f"Diff larger than rel tolerance:- avg_y_{m}: {avg_y_m} - expected average: {exp_res_m}")

    def test_simple_simulation(self):
        config = Config(4, 5, n_cells=20, chain_length=10)
        simul = simulate_full_dataset(config)
        print(simul['obs'])