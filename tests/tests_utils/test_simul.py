import torch
import torch.nn.functional as f
import unittest

from pyro import poutine

import simul
import tests.utils_testing


class simulTestCase(unittest.TestCase):

    def setUp(self) -> None:
        return super().setUp()


    def test_model_tree_markov_full(self):
        n_cells = 1000
        n_sites = 20
        n_copy_states = 5
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        n_nodes = len(tree.nodes)
        mu_0 = 10.0
        nu_0 = 0.1
        alpha0 = 10.
        beta0 = 40.
        a0 = 10.
        b0 = 40.
        dir_alpha0 = 1.
        data = torch.ones((n_sites, n_cells))
        model_tree_markov = simul.model_tree_markov_full
        unconditioned_model = poutine.uncondition(model_tree_markov)
        C, y, z, pi, mu, tau, eps = unconditioned_model(data, n_cells, n_sites, n_copy_states, tree, mu_0, nu_0, alpha0, beta0, a0, b0, dir_alpha0)


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
            assert torch.isclose(mu_avg, torch.tensor(mu_0), rtol=3*1e-01)

        for m in range(n_sites):
            avg_y_m = torch.mean(y[m, :])
            exp_res_m = 0
            for n in range(n_cells):
                exp_res_m += C[z[n], m] * mu[n] * 1 / n_cells

            #print(f"avg_y_{m}: {avg_y_m} - expected average: {exp_res_m}")
            if n_cells >= 100:
                self.assertTrue(torch.isclose(avg_y_m, exp_res_m, rtol=3*1e-01), msg=f"Diff larger than rel tolerance:- avg_y_{m}: {avg_y_m} - expected average: {exp_res_m}")