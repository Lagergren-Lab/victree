import logging
import random
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as f
from pyro import poutine

import simul
import tests.utils_testing
import utils.config
from inference.copy_tree import VarDistFixedTree, CopyTree, JointVarDist
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils, tree_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class VICtreeTreePostOptimizationTestCase(unittest.TestCase):

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)
        return qc, qt, qeps, qz, qpi, qmt

    def test_one_edge_tree(self):
        torch.manual_seed(0)
        logging.getLogger().setLevel("INFO")
        tree = tests.utils_testing.get_two_node_tree()
        n_nodes = len(tree.nodes)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        dir_alpha = torch.tensor([1., 3.])
        nu_0 = torch.tensor(10.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(5.0)
        b0 = torch.tensor(200.0)

        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        debug=False, diagnostics=True)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = JointVarDist(config, y, qc, qz, qt, qeps, qmt, qpi)
        q.initialize()
        copy_tree = CopyTree(config, q, y)

        # Act
        copy_tree.run(5)

        """
        Assert - in case of root + one edge, all sampled trees should be equal to true tree.
        """
        T_list, w_T_list = qt.get_trees_sample(sample_size=10)
        for T in T_list:
            self.assertEqual(tree.edges(), T.edges())

    def test_three_node_tree(self):
        torch.manual_seed(0)
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        n_nodes = len(tree.nodes)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        dir_alpha = torch.tensor([1., 3., 3.])
        nu_0 = torch.tensor(10.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(10.0)
        b0 = torch.tensor(800.0)
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        print(f"Epsilon: {eps}")
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        debug=False, diagnostics=True)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = JointVarDist(config, y, qc, qz, qt, qeps, qmt, qpi)
        q.initialize()
        copy_tree = CopyTree(config, q, y)

        # Act
        copy_tree.run(50)

        """
        Assert - in case of root + two edges, all sampled trees should be equal to true tree.
        """
        T_list, w_T_list = qt.get_trees_sample(sample_size=10)
        for T in T_list:
            self.assertEqual(tree.edges(), T.edges())
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)

    def test_large_tree(self):
        torch.manual_seed(0)
        K = 7
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 100
        n_sites = 200
        n_copy_states = 7
        dir_alpha0 = torch.ones(K) * 2.
        nu_0 = torch.tensor(10.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(5.0)
        b0 = torch.tensor(200.0)
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha0)
        print(f"Epsilon: {eps}")
        config = Config(step_size=0.3, n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                        diagnostics=False)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self.id().replace(".", "/"))

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = JointVarDist(config, y, qc, qz, qt, qeps, qmt, qpi)
        q.initialize()
        copy_tree = CopyTree(config, q, y)

        # Act
        copy_tree.run(5)

        """
        Assert - in case of root + K edges.
        """
        T_list, w_T_list = qt.get_trees_sample(sample_size=100)
        ari, best_perm, acc = model_variational_comparisons.compare_qZ_and_true_Z(z, copy_tree.q.z)
        T_list_remapped = tree_utils.relabel_trees(T_list, best_perm)
        T_undirected_list = tree_utils.to_undirected(T_list_remapped)
        prufer_list = tree_utils.to_prufer_sequences(T_undirected_list)
        unique_seq, unique_seq_idx = tree_utils.unique_trees(prufer_list)
        print(f"N unique trees: {len(unique_seq_idx)}")
        distances = tree_utils.distances_to_true_tree(tree, [T_list[l] for l in unique_seq_idx], best_perm)
        sorted_dist = np.sort(distances)
        sorted_dist_idx = np.argsort(distances)
        visualization_utils.visualize_T_given_true_tree_and_distances(T_list, w_T_list, distances, )
        print(f"Distances to true tree: {distances}")
        print(f"Weights: {w_T_list}")
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)


    def test_large_tree_init_true_params(self):
        logger = logging.getLogger()
        logger.level = logging.INFO
        K = 10
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        nu_0 = torch.tensor(10.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(10.0)
        b0 = torch.tensor(200.0)
        dir_alpha0 = torch.ones(K) * 2.
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha0)
        config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        diagnostics=True)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()

        copy_tree = CopyTree(config, q, y)
        copy_tree.q.pi.concentration_param = dir_alpha0
        copy_tree.q.z.pi[...] = f.one_hot(z, num_classes=K)
        copy_tree.q.c.single_filtering_probs[...] = f.one_hot(C.long(), num_classes=n_copy_states).float()

        copy_tree.run(50)

        # Assert
        diagnostics_dict = q.diagnostics_dict
        visualization_utils.plot_diagnostics_to_pdf(diagnostics_dict,
                                                    cells_to_vis_idxs=[0, 10, 20, 30, int(n_cells / 2), int(n_cells / 3),
                                                                       n_cells - 1],
                                                    clones_to_vis_idxs=[1, 0, 2, 3, 4],
                                                    edges_to_vis_idxs=tree.edges,
                                                    save_path=test_dir_name + '/diagnostics.pdf')
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)

    def test_large_tree_init_close_to_true_params(self):
        logger = logging.getLogger()
        logger.level = logging.INFO
        utils.config.set_seed(0)
        K = 5
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 1000
        n_sites = 200
        n_copy_states = 7
        nu_0 = torch.tensor(10.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(10.0)
        b0 = torch.tensor(200.0)
        dir_alpha0 = torch.ones(K) * 2.
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha0)
        config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        diagnostics=True)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = VarDistFixedTree(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()

        copy_tree = CopyTree(config, q, y)
        copy_tree.q.pi.concentration_param = dir_alpha0

        # Init Z with off-set from true params
        z_one_hot = f.one_hot(z, num_classes=K)
        off_set_z = 0.1
        z_perturbed = z_one_hot + off_set_z
        copy_tree.q.z.pi[...] = z_perturbed / z_perturbed.sum(1, keepdims=True)
        print(f'q(Z) init offset: {off_set_z * 100}%')

        # Init single filter probs with off-set from true params
        c_one_hot = f.one_hot(C.long(), num_classes=n_copy_states).float()
        off_set_c = 0.1
        c_perturbed = c_one_hot + off_set_c
        copy_tree.q.c.single_filtering_probs[...] = c_perturbed / c_perturbed.sum(dim=-1, keepdims=True)
        print(f'q(C) marginals init offset: {off_set_z * 100}%')

        # Init single filter probs with off-set from true params
        off_set_mutau = 0.1
        mu_perturbed = mu + mu * off_set_mutau
        tau_perturbed = tau + tau * off_set_mutau
        alpha_perturbed = torch.ones(n_cells) * 50.
        beta_perturbed = alpha_perturbed / tau_perturbed
        copy_tree.q.mt.nu[...] = mu_perturbed
        copy_tree.q.mt.alpha[...] = alpha_perturbed
        copy_tree.q.mt.beta[...] = beta_perturbed
        print(f'q(mu, tau) param init offset: {off_set_mutau * 100}%')

        # Act
        copy_tree.run(50)

        # Assert
        diagnostics_dict = q.diagnostics_dict
        visualization_utils.plot_diagnostics_to_pdf(diagnostics_dict,
                                                    cells_to_vis_idxs=[0, 10, 20, 30, int(n_cells / 2), int(n_cells / 3),
                                                                       n_cells - 1],
                                                    clones_to_vis_idxs=[1, 0, 2, 3, 4],
                                                    edges_to_vis_idxs=tree.edges,
                                                    save_path=test_dir_name + '/diagnostics.pdf')
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt,
                                                          q_eps=copy_tree.q.eps)
