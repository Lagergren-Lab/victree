import logging
import os.path
import random
import unittest

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as f
import numpy as np

import simul
import tests.utils_testing
import utils.config
from inference.victree import VICTree
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons, utils_testing
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils, data_handling
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class VICTreeFixedTreeTrueInitializationsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(0)

        if not hasattr(self, 'K'):
            self.K = 10
            self.tree = tests.utils_testing.get_tree_K_nodes_random(self.K)
            self.N = 500
            self.M = 500
            self.A = 7
            self.dir_alpha0 = 10.
            self.nu_0 = 1.
            self.lambda_0 = 10.
            self.alpha0 = 500.
            self.beta0 = 50.
            self.a0 = 5.0
            self.b0 = 500.0
            y, c, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(self.N, self.M, self.A, self.tree,
                                                                            nu_0=self.nu_0,
                                                                            lambda_0=self.lambda_0, alpha0=self.alpha0,
                                                                            beta0=self.beta0,
                                                                            a0=self.a0, b0=self.b0,
                                                                            dir_alpha0=self.dir_alpha0)
            self.y = y
            self.c = c
            self.z = z
            self.pi = pi
            self.mu = mu
            self.tau = tau
            self.eps = eps
            self.eps0 = eps0

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)
        return qc, qt, qeps, qz, qpi, qmt

    def test_init_true_Z(self):
        y, c, z, pi, mu, tau, eps, eps0 = (self.y, self.c, self.z, self.pi, self.mu, self.tau, self.eps, self.eps0)

        out_dir = "./test_output/" + self._testMethodName
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        config = Config(n_nodes=self.K, n_states=self.A, n_cells=self.N, chain_length=self.M, step_size=0.1,
                        diagnostics=False, out_dir=out_dir, annealing=1.)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self.id().replace(".", "/"))

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)

        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, self.tree, y)
        q.initialize()

        # Initialize Z to true values.
        qz.initialize('fixed', pi_init=torch.nn.functional.one_hot(z, num_classes=self.K))

        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=qc,
                                                                q_z=qz, qpi=qpi,
                                                                q_mt=qmt, q_eps=qeps, perm=list(range(0, self.K)))
        copy_tree = VICTree(config, q, y, draft=True)
        copy_tree.run(n_iter=100)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=copy_tree.q.c,
                                                                q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                                q_mt=copy_tree.q.mt, q_eps=copy_tree.q.eps,
                                                                perm=list(range(0, self.K)))
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])

        self.assertGreater(ari, 0.95, msg='ari less than 0.95.')

    def test_init_true_mt_and_C(self):
        y, c, z, pi, mu, tau, eps, eps0 = (self.y, self.c, self.z, self.pi, self.mu, self.tau, self.eps, self.eps0)

        out_dir = "./test_output/" + self._testMethodName
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        config = Config(n_nodes=self.K, n_states=self.A, n_cells=self.N, chain_length=self.M, step_size=0.1,
                        diagnostics=False, out_dir=out_dir, annealing=1.)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self.id().replace(".", "/"))

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)

        q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, self.tree, y)
        # initialize all var dists
        q.initialize()

        # Initialize qMuTau to true values.
        qmt.initialize('fixed', loc=mu, precision_factor=100., shape=self.alpha0, rate=self.alpha0 / tau)
        utils_testing.initialize_qc_to_true_values(c, self.A, qc)
        utils_testing.initialize_qepsilon_to_true_values(eps, self.a0, self.b0, qeps)

        # Make sure qZ is updated first based on good values
        qz.update(qmt, qc, qpi, y)

        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=qc,
                                                                q_z=qz, qpi=qpi,
                                                                q_mt=qmt, q_eps=qeps, perm=list(range(0, self.K)))

        copy_tree = VICTree(config, q, y, draft=True)
        copy_tree.run(n_iter=100)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=copy_tree.q.c,
                                                                q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
                                                                q_mt=copy_tree.q.mt, q_eps=copy_tree.q.eps,
                                                                perm=list(range(0, self.K)))
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])

        self.assertGreater(ari, 0.95, msg='ari less than 0.95.')
