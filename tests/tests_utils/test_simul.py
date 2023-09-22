import torch
import torch.nn.functional as f
import unittest

import simul
import utils
from simul import simulate_full_dataset
import tests.utils_testing
from utils.config import Config
from variational_distributions.var_dists import qC, qCMultiChrom


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

    @unittest.skip("experiment")
    def test_copy_tree_sim(self):
        torch.manual_seed(0)
        K = 5
        M = 2000
        A = 7
        eps_a = 1.
        eps_b = 10.
        eps_0 = 0.1
        tree = utils.tree_utils.star_tree(K)
        eps, c = simul.simulate_copy_tree_data(K, M, A, tree, eps_a, eps_b, eps_0)
        torch.equal(c[0, :], torch.ones((M,), dtype=torch.int) * 2)

        # For Large M, assert that the clone with most variance is gained from the edge with highest epsilon
        variances = torch.std(c.float(), dim=1)
        max_eps_arc = max(eps, key=eps.get)
        self.assertTrue(torch.argmax(variances) == max_eps_arc[1])

    def test_one_edge_tree(self):
        torch.manual_seed(0)
        tree = tests.utils_testing.get_two_node_tree()
        n_nodes = len(tree.nodes)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        dir_delta = torch.tensor([1., 3.])
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(10.0)
        b0 = torch.tensor(200.0)
        R_0 = 100.

        out = simul.simulate_data_total_GC_urn_model(tree, n_cells, n_sites, n_nodes, n_copy_states, R_0, eps_a=a0,
                                                     eps_b=b0, eps_0=1., alpha0=alpha0, beta0=beta0,
                                                     dir_delta=dir_delta)
        x = out['x']
        R = out['R']
        gc = out['gc']
        phi = out['phi']
        c = out['c']
        z = out['z']
        pi = out['pi']
        eps = out['eps']
        eps_0 = out['eps0']

        torch.allclose(torch.mean(R.float()), torch.tensor(R_0))
        # think of interesting cases here

    def test_real_full_dataset(self):
        full_length = 200

        # default = real 24 chr
        config = Config(chain_length=full_length)
        chr_df = simul.generate_chromosome_binning(config.chain_length)
        dat = simul.simulate_full_dataset(config, chr_df=chr_df)
        self.assertEqual(len(dat['chr_idx']), 23)
        self.assertTrue(all(dat['chr_idx'][i] <= dat['chr_idx'][i+1] for i in range(len(dat['chr_idx']) - 1)))
        self.assertGreaterEqual(dat['obs'].shape[0], full_length)

        # total synthetic
        n_chr = 10

        config = Config(chain_length=full_length)
        chr_df = simul.generate_chromosome_binning(config.chain_length, method='uniform', n_chr=n_chr)
        dat = simul.simulate_full_dataset(config, chr_df=chr_df)
        self.assertEqual(len(dat['chr_idx']), n_chr - 1)
        self.assertTrue(all(dat['chr_idx'][i] <= dat['chr_idx'][i+1] for i in range(len(dat['chr_idx']) - 1)))
        self.assertEqual(dat['obs'].shape[0], full_length)

    def test_generate_var_dataset_multichr(self):
        # test balanced chromosomes
        config = Config()
        joint_q_3chr = simul.generate_dataset_var_tree(config, chrom=3)
        self.assertTrue(isinstance(joint_q_3chr.c, qCMultiChrom))
        self.assertEqual(config.n_chromosomes, 3)
        self.assertTrue(joint_q_3chr.fixed)
        for qc in joint_q_3chr.c.qC_list:
            self.assertTrue(qc.fixed)
        self.assertTrue(joint_q_3chr.c.fixed)
        concat_single_c = torch.concat([qc.true_params['c'] for qc in joint_q_3chr.c.qC_list], dim=1)
        self.assertTrue(torch.all(joint_q_3chr.c.true_params['c'] == concat_single_c))

        # test hg19 real chromosomes
        config = Config()
        joint_q_hg19 = simul.generate_dataset_var_tree(config, chrom='real')
        self.assertTrue(isinstance(joint_q_hg19.c, qCMultiChrom))
        self.assertEqual(config.n_chromosomes, 24)

        # test 1 chr
        config = Config()
        joint_q_1chr = simul.generate_dataset_var_tree(config, chrom=1)
        self.assertTrue(isinstance(joint_q_1chr.c, qC))
        self.assertEqual(config.n_chromosomes, 1)



