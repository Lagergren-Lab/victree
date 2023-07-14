import logging
import os.path
import random
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as f

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


    def test_three_node_tree(self):
        torch.manual_seed(0)
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        K = len(tree.nodes)
        N = 100
        M = 200
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
        print(f"Mu in range: [{mu.min()}, {mu.max()}] ")

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


    def test_qC_vs_qCMutliChrom_random_node_tree(self):
        torch.manual_seed(0)
        K = 5
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
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
        print(f"Mu in range: [{mu.min()}, {mu.max()}]")
        print(f"tau in range: [{tau.min()}, {tau.max()}]")

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)

        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        # initialize all var dists
        q.initialize()
        copy_tree = VICTree(config, q, y)

        copy_tree.run(n_iter=100)

        qc2, qt2, qeps2, qz2, qpi2, qmt2 = self.set_up_q(config)
        qc2 = qC(config)
        q2 = FixedTreeJointDist(config, qc2, qz2, qeps2, qmt2, qpi2, tree, y)
        q2.initialize()
        copy_tree2 = VICTree(config, q2, y)

        copy_tree2.run(n_iter=100)

        # Assert
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)

        # Assert
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree2.q.c,
                                                          q_z=copy_tree2.q.z, qpi=copy_tree2.q.pi, q_mt=copy_tree2.q.mt)

    @unittest.skip("long exec time")
    def test_large_tree(self):
        torch.manual_seed(0)
        K = 9
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
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
        print(f"Mu in range: [{mu.min()}, {mu.max()}] ")

        out_dir = "./test_output/" + self._testMethodName
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()
        copy_tree = VICTree(config, q, y)

        copy_tree.run(n_iter=500)
        copy_tree.step()
        print(q.c)

        # Assert
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)

