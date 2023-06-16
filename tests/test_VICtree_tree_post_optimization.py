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
from inference.copy_tree import CopyTree
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils, tree_utils
from utils.config import Config
from variational_distributions.joint_dists import VarTreeJointDist
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
        dir_alpha = [1., 3.]
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
        q = VarTreeJointDist(config, y, qc, qz, qt, qeps, qmt, qpi)
        q.initialize()
        copy_tree = CopyTree(config, q, y)

        # Act
        copy_tree.run(n_iter=5)

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
        dir_alpha = [1., 3., 3.]
        nu_0 = torch.tensor(10.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(10.0)
        b0 = torch.tensor(200.0)
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        print(f"Epsilon: {eps}")
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        debug=False, diagnostics=False)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = VarTreeJointDist(config, y, qc, qz, qt, qeps, qmt, qpi)
        q.initialize()
        copy_tree = CopyTree(config, q, y)

        # Act
        copy_tree.run(n_iter=100)

        """
        Assert - in case of root + two edges, all sampled trees should be equal to true tree.
        """
        T_list, w_T_list, log_g_t = qt.get_trees_sample(sample_size=20, add_log_g=True)
        T_unique, indexes, multiplicity = tree_utils.get_unique_trees_and_multiplicity(T_list)
        print(f"Unique trees: {[tree_utils.tree_to_newick(T, 0) for T in T_unique]}")
        print(f"Multiplicity: {multiplicity}")
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)

    def test_large_tree(self):
        torch.manual_seed(0)
        L_train = 40
        L_eval = 50
        K = 10
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 200
        n_sites = 200
        n_copy_states = 7
        dir_alpha0 = list(np.ones(K) * 2.)
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
                        diagnostics=False, wis_sample_size=L_train)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self.id().replace(".", "/"))

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = VarTreeJointDist(config, y, qc, qz, qt, qeps, qmt, qpi)
        q.initialize()
        copy_tree = CopyTree(config, q, y)

        # Act
        copy_tree.run(n_iter=100)

        """
        Assert - in case of root + K edges.
        """
        T_list, w_T_list, log_g_list = qt.get_trees_sample(sample_size=L_eval, add_log_g=True)
        ari, best_perm, acc = model_variational_comparisons.compare_qZ_and_true_Z(z, copy_tree.q.z)
        T_list_remapped = tree_utils.relabel_trees(T_list, best_perm)
        T_undirected_list = tree_utils.to_undirected(T_list_remapped)
        prufer_list = tree_utils.to_prufer_sequences(T_undirected_list)
        unique_seq, unique_seq_idx, multiplicity = tree_utils.unique_trees_and_multiplicity(prufer_list)
        print(f"N unique trees: {len(unique_seq_idx)}")
        distances = tree_utils.distances_to_true_tree(tree, [T_list_remapped[l] for l in unique_seq_idx], best_perm)
        sorted_dist = np.sort(distances)
        sorted_dist_idx = np.argsort(distances)
        T_list_unique = [T_list_remapped[i] for i in unique_seq_idx]
        w_T_list_unique = [w_T_list[i] for i in unique_seq_idx]
        w_T_list_unique_with_multiplicity = list(np.multiply(w_T_list_unique, multiplicity))
        visualization_utils.visualize_and_save_T_plots(test_dir_name, tree, T_list_unique, w_T_list_unique_with_multiplicity, distances)
        print(f"Distances to true tree: {distances}")
        print(f"Weights: {w_T_list_unique_with_multiplicity}")
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt,
                                                          q_eps=copy_tree.q.eps)

    def test_large_tree_small_epsilon(self):
        torch.manual_seed(0)
        L_train = 40
        L_eval = 50
        K = 9
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 200
        n_sites = 50
        n_copy_states = 7
        dir_alpha0 = list(np.ones(K) * 2.)
        nu_0 = torch.tensor(10.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(5.0)
        b0 = torch.tensor(200.0)
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0,
                                                                        beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha0)
        print(f"Epsilon: {eps}")
        config = Config(step_size=0.3, n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                        diagnostics=False, wis_sample_size=L_train)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self.id().replace(".", "/"))

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = VarTreeJointDist(config, y, qc, qz, qt, qeps, qmt, qpi)
        q.initialize()
        copy_tree = CopyTree(config, q, y)

        # Act
        copy_tree.run(n_iter=50)

        """
        Assert - in case of root + K edges.
        """
        T_list, w_T_list, log_g_list = qt.get_trees_sample(sample_size=L_eval, add_log_g=True)
        ari, best_perm, acc = model_variational_comparisons.compare_qZ_and_true_Z(z, copy_tree.q.z)
        T_list_remapped = tree_utils.relabel_trees(T_list, best_perm)
        T_undirected_list = tree_utils.to_undirected(T_list_remapped)
        prufer_list = tree_utils.to_prufer_sequences(T_undirected_list)
        unique_seq, unique_seq_idx, multiplicity = tree_utils.unique_trees_and_multiplicity(prufer_list)
        print(f"N unique trees: {len(unique_seq_idx)}")
        distances = tree_utils.distances_to_true_tree(tree, [T_list_remapped[l] for l in unique_seq_idx], best_perm)
        T_list_unique = [T_list_remapped[i] for i in unique_seq_idx]
        w_T_list_unique = [w_T_list[i] for i in unique_seq_idx]
        log_g_list_unique = [log_g_list[i] for i in unique_seq_idx]
        w_T_list_unique_with_multiplicity = list(np.multiply(w_T_list_unique, multiplicity))
        visualization_utils.visualize_and_save_T_plots(test_dir_name, tree, T_list_unique,
                                                       w_T_list_unique_with_multiplicity, distances, log_g_list_unique)
        print(f"Distances to true tree: {distances}")
        print(f"Weights: {w_T_list_unique_with_multiplicity}")
        print(f"Weights: {log_g_list_unique}")
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                          q_mt=copy_tree.q.mt,
                                                          q_eps=copy_tree.q.eps)


    def test_large_tree_init_true_params(self):
        torch.manual_seed(0)
        K = 7
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 500
        n_sites = 500
        n_copy_states = 7
        dir_alpha0 = list(np.ones(K) * 2.)
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
        config = Config(step_size=1., n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                        diagnostics=False)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self.id().replace(".", "/"))

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = VarTreeJointDist(config, y, qc, qz, qt, qeps, qmt, qpi)
        q.initialize()

        copy_tree = CopyTree(config, q, y)
        copy_tree.q.pi.concentration_param = torch.tensor(dir_alpha0)
        copy_tree.q.z.pi[...] = f.one_hot(z, num_classes=K)
        copy_tree.q.c.single_filtering_probs[...] = f.one_hot(C.long(), num_classes=n_copy_states).float()

        copy_tree.run(n_iter=20)

        """
        Assert - in case of root + K edges.
        """
        T_list, w_T_list, log_g_list = qt.get_trees_sample(sample_size=20, add_log_g=True)
        ari, best_perm, acc = model_variational_comparisons.compare_qZ_and_true_Z(z, copy_tree.q.z)
        T_list_remapped = tree_utils.relabel_trees(T_list, best_perm)
        T_undirected_list = tree_utils.to_undirected(T_list_remapped)
        prufer_list = tree_utils.to_prufer_sequences(T_undirected_list)
        unique_seq, unique_seq_idx, multiplicity = tree_utils.unique_trees_and_multiplicity(prufer_list)
        print(f"N unique trees: {len(unique_seq_idx)}")
        distances = tree_utils.distances_to_true_tree(tree, [T_list_remapped[l] for l in unique_seq_idx], best_perm)
        sorted_dist = np.sort(distances)
        sorted_dist_idx = np.argsort(distances)
        T_list_unique = [T_list_remapped[i] for i in unique_seq_idx]
        w_T_list_unique = [w_T_list[i] for i in unique_seq_idx]
        w_T_list_unique_with_multiplicity = list(np.multiply(w_T_list_unique, multiplicity))
        visualization_utils.visualize_and_save_T_plots(test_dir_name, tree, T_list_unique, w_T_list_unique_with_multiplicity, distances)
        print(f"Distances to true tree: {distances}")
        print(f"Weights: {w_T_list_unique_with_multiplicity}")
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt,
                                                          q_eps=copy_tree.q.eps)