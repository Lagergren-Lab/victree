import logging
import random
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as f

import simul
import tests.utils_testing
import utils.config
from inference.victree import VICTree
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class TemperingVICtreeFixedTreeTestCase(unittest.TestCase):

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
        n_cells = 100
        n_sites = 20
        n_copy_states = 7
        dir_alpha = [1., 3.]
        nu_0 = 10.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 200.0
        n_iter = 50
        start_temp = 200
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        debug=False, diagnostics=False)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()
        copy_tree = VICTree(config, q, y)

        config_temp = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                             step_size=0.3, annealing=start_temp, debug=False, diagnostics=False)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_temp)
        q_temp = FixedTreeJointDist(config_temp, qc, qz, qeps, qmt, qpi, tree, y)
        q_temp.initialize()
        copy_tree_temp = VICTree(config_temp, q_temp, y)

        # Act
        copy_tree.run(n_iter)
        copy_tree_temp.run(n_iter)

        # Assert
        torch.set_printoptions(precision=2)
        self.assertGreater(copy_tree_temp.q.z.entropy(), copy_tree.q.z.entropy())
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)

        print(f"\n ---- Annealed Model ------")
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree_temp.q.c,
                                                          q_z=copy_tree_temp.q.z, qpi=copy_tree_temp.q.pi, q_mt=copy_tree_temp.q.mt)

    def test_three_node_tree(self):
        torch.manual_seed(0)
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        n_nodes = len(tree.nodes)
        n_cells = 500
        n_sites = 500
        n_copy_states = 7
        dir_alpha = [1., 3., 3.]
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 800.0
        start_temp = 400
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        print(f"Epsilon: {eps}")

        # setup non-annealed model
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        debug=False, diagnostics=False)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)

        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        # initialize all var dists
        q.initialize()
        copy_tree = VICTree(config, q, y)

        # setup annealed model
        config_temp = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                             step_size=0.3, annealing=start_temp, debug=False, diagnostics=False)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_temp)
        q_temp = FixedTreeJointDist(config_temp, qc, qz, qeps, qmt, qpi, tree, y)
        q_temp.initialize()
        copy_tree_temp = VICTree(config_temp, q_temp, y)

        # Act
        copy_tree.run(n_iter=100)
        copy_tree_temp.run(n_iter=100)

        # Assert
        self.assertGreater(copy_tree_temp.q.z.entropy(), copy_tree.q.z.entropy())

        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)

        print(f"\n ---- Annealed Model ------")
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree_temp.q.c,
                                                          q_z=copy_tree_temp.q.z, qpi=copy_tree_temp.q.pi, q_mt=copy_tree_temp.q.mt)

    def test_large_tree(self):
        torch.manual_seed(0)
        K = 5
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 500
        n_sites = 500
        n_copy_states = 7
        start_temp = 40.
        dir_alpha0 = 2.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 5.0
        b0 = 200.0
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha0)
        config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.1,
                        diagnostics=False)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()
        copy_tree = VICTree(config, q, y)

        # setup annealed model
        config_temp = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                             step_size=0.1, annealing=start_temp, debug=False, diagnostics=False)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_temp)
        q_temp = FixedTreeJointDist(config_temp, qc, qz, qeps, qmt, qpi, tree, y)
        q_temp.initialize()
        copy_tree_temp = VICTree(config_temp, q_temp, y)

        # Act
        copy_tree.run(n_iter=100)
        copy_tree_temp.run(n_iter=100)

        # Assert
        self.assertGreater(copy_tree_temp.q.z.entropy(), copy_tree.q.z.entropy())

        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)

        print(f"\n ---- Annealed Model ------")
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree_temp.q.c,
                                                          q_z=copy_tree_temp.q.z, qpi=copy_tree_temp.q.pi, q_mt=copy_tree_temp.q.mt)


