import logging
import sys
import unittest

import torch

import tests.utils_testing
from inference.victree import VICTree
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simulate_full_dataset_no_pyro
from utils.config import Config, set_seed
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC


class VICtreeQcSmoothingFixedTreeTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(101)
        # setup logging debug in tests
        stream_handler = logging.StreamHandler(sys.stdout)
        logger = logging.getLogger()
        logger.level = logging.INFO
        logger.addHandler(stream_handler)
        self.logger = logger
        self.stream_handler = stream_handler

    def tearDown(self) -> None:
        self.logger.removeHandler(self.stream_handler)

    def set_up_q(self, config):
        logging.getLogger().debug("setting up joint q for tests")
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)  # ), alpha_prior=1., beta_prior=10.)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)  # , nu_prior=1.0, lambda_prior=10., alpha_prior=50., beta_prior=5.)
        return qc, qt, qeps, qz, qpi, qmt

    def test_three_node_tree(self):
        set_seed(0)
        tree = tests.utils_testing.get_tree_three_nodes_balanced()
        n_nodes = len(tree.nodes)
        n_cells = 500
        n_sites = 2000
        n_copy_states = 7
        dir_alpha = [5., 10., 10.]
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 5.0
        b0 = 250.0
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        print(f"Epsilon: {eps}")
        print(f"pi: {pi}")
        print(f"mu: [{mu.min()}, {mu.max()}]")
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                        step_size=0.3, n_run_iter=40,
                        qc_smoothing=True, debug=True)
        config2 = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                         step_size=0.3, n_run_iter=40,
                         qc_smoothing=False, debug=True)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        qc2, qt2, qeps2, qz2, qpi2, qmt2 = self.set_up_q(config2)

        q = FixedTreeJointDist(y, config, qc, qz, qeps, qmt, qpi, tree)
        q2 = FixedTreeJointDist(y, config2, qc2, qz2, qeps2, qmt2, qpi2, tree)
        # initialize all var dists
        q.initialize()
        q2.initialize()

        copy_tree = VICTree(config, q, y, draft=True)
        copy_tree2 = VICTree(config2, q2, y, draft=True)

        copy_tree.run()
        copy_tree2.run()

        # Assert
        torch.set_printoptions(precision=2)
        print("Results with qc smoothing")
        out1 = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                                 true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                                 q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                                 q_mt=copy_tree.q.mt,
                                                                 q_eps=copy_tree.q.eps)

        print("Results without qc smoothing")
        out2 = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                                 true_tau=tau, true_epsilon=eps, q_c=copy_tree2.q.c,
                                                                 q_z=copy_tree2.q.z, qpi=copy_tree2.q.pi,
                                                                 q_mt=copy_tree2.q.mt,
                                                                 q_eps=copy_tree2.q.eps)

    @unittest.skip("long exec time")
    def test_large_tree(self):
        set_seed(2)
        K = 7
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        n_nodes = len(tree.nodes)
        n_cells = 500
        n_sites = 1000
        n_copy_states = 7
        dir_alpha = [10.] * K
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 250.0
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_alpha)
        set_seed(0)
        print(f"Epsilon: {eps}")
        print(f"pi: {pi}")
        print(f"mu: [{mu.min()}, {mu.max()}]")
        config = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                        step_size=0.1,
                        qc_smoothing=True)
        config2 = Config(n_nodes=n_nodes, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites,
                         step_size=0.1,
                         qc_smoothing=False)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
        qc2, qt2, qeps2, qz2, qpi2, qmt2 = self.set_up_q(config2)

        q = FixedTreeJointDist(y, config, qc, qz, qeps, qmt, qpi, tree)
        q2 = FixedTreeJointDist(y, config2, qc2, qz2, qeps2, qmt2, qpi2, tree)
        # initialize all var dists
        q.initialize()
        q2.initialize()

        copy_tree = VICTree(config, q, y)
        copy_tree2 = VICTree(config2, q2, y)

        copy_tree.run(n_iter=200)
        copy_tree2.run(n_iter=200)

        # Assert
        torch.set_printoptions(precision=2)
        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                          q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt,
                                                          q_eps=copy_tree.q.eps)

        model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                          true_tau=tau, true_epsilon=eps, q_c=copy_tree2.q.c,
                                                          q_z=copy_tree2.q.z, qpi=copy_tree2.q.pi, q_mt=copy_tree2.q.mt,
                                                          q_eps=copy_tree2.q.eps)
