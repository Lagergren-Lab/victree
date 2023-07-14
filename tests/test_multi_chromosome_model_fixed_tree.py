import logging
import os.path
import random
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as f
from pyro import poutine

import simul
import tests.utils_testing
import utils.config
from inference.victree import VICTree
from model.multi_chromosome_model import MultiChromosomeGenerativeModel
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils, data_handling
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent, \
    qCMultiChrom


class MultiChromosomeTestCase(unittest.TestCase):

    def set_up_q(self, config):
        qc = qCMultiChrom(config)
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
        a0 = torch.tensor(10.0)
        b0 = torch.tensor(200.0)

        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.3,
                        debug=False, diagnostics=True)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()
        copy_tree = VICTree(config, q, y)

        # Act
        copy_tree.run(n_iter=80)

        # Assert
        # FIXME: use q.params_history for each distribution of interest
        diagnostics_dict = q.params_history
        visualization_utils.plot_diagnostics_to_pdf(diagnostics_dict,
                                                    cells_to_vis_idxs=[0, int(n_cells / 2), int(n_cells / 3),
                                                                       n_cells - 1],
                                                    clones_to_vis_idxs=[1, 0],
                                                    edges_to_vis_idxs=[(0, 1)],
                                                    save_path=test_dir_name + '/diagnostics.pdf')
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)

    def test_three_node_tree(self):
        torch.manual_seed(0)
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        K = len(tree.nodes)
        N = 1000
        M = 2000
        chromosome_indexes = [int(M / 10), int(M / 10 * 5), int(M / 10 * 8)]
        n_chromosomes = len(chromosome_indexes) + 1
        A = 7
        dir_alpha = torch.tensor([1., 3., 3.])
        nu_0 = torch.tensor(1.)
        lambda_0 = torch.tensor(5.)
        alpha0 = torch.tensor(50.)
        beta0 = torch.tensor(5.)
        a0 = torch.tensor(10.0)
        b0 = torch.tensor(800.0)
        eps0_a = torch.tensor(10.0)
        eps0_b = torch.tensor(800.0)
        config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M,
                        chromosome_indexes=chromosome_indexes, n_chromosomes=n_chromosomes,
                        step_size=0.3,
                        debug=False, diagnostics=False)
        model = MultiChromosomeGenerativeModel(config)
        out_simul = model.simulate_data(tree, a0=a0, b0=b0, eps0_a=eps0_a, eps0_b=eps0_b, delta=dir_alpha,
                                        nu0=nu_0, lambda0=lambda_0, alpha0=alpha0, beta0=beta0)
        y = out_simul['obs']
        c = out_simul['c']
        z = out_simul['z']
        pi = out_simul['pi']
        mu = out_simul['mu']
        tau = out_simul['tau']
        eps = out_simul['eps']
        eps0 = out_simul['eps0']

        print(f"Epsilon: {eps}")

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)

        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        # initialize all var dists
        q.initialize()
        copy_tree = VICTree(config, q, y)

        copy_tree.run(n_iter=50)

        # Assert
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)

    def test_large_tree(self):
        torch.manual_seed(0)
        K = 9
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_cells = 500
        n_sites = 500
        n_copy_states = 5
        dir_alpha0 = 10.
        nu_0 = torch.tensor(1.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)
        a0 = torch.tensor(10.0)
        b0 = torch.tensor(200.0)
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha0)
        print(f"C: {C}")
        # print(f"z: {z}")
        # print(f"y: {y}")
        # print(f"mu: {mu}")

        out_dir = "./test_output/" + self._testMethodName
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.1,
                        diagnostics=False, out_dir=out_dir, annealing=1.)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()
        copy_tree = VICTree(config, q, y)

        copy_tree.run(n_iter=500)
        copy_tree.step()
        print(q.c)

        # Assert
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
        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()

        copy_tree = VICTree(config, q, y)
        copy_tree.q.pi.concentration_param = dir_alpha0
        copy_tree.q.z.pi[...] = f.one_hot(z, num_classes=K)
        copy_tree.q.c.single_filtering_probs[...] = f.one_hot(C.long(), num_classes=n_copy_states).float()

        copy_tree.run(n_iter=50)

        # Assert
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)