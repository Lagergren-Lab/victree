import logging
import os.path
import unittest

import numpy as np
import torch

import simul
import tests.utils_testing
import utils
from inference.copy_tree import CopyTree
from variational_distributions.joint_dists import QuadrupletJointDist
from tests import model_variational_comparisons, utils_testing
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class VICtreeQuadrupletTestCase(unittest.TestCase):

    def set_up_q(self, config):
        qc = qC(config)
        qeps = qEpsilonMulti(config)
        z_assgn = torch.tensor([2, 3])
        qz = qZ(config, true_params={'z': z_assgn})
        qmt = qMuTau(config, nu_prior=1., lambda_prior=1., alpha_prior=100., beta_prior=5.)
        return qc, qeps, qz, qmt

    def test_quadruplet(self):
        utils.config.set_seed(0)
        tree = tests.utils_testing.get_quadtruplet_tree()
        n_cells = 2
        n_sites = 2000
        n_copy_states = 7
        mu_v = 1.2
        mu_w = 0.9
        tau_v = 10.
        tau_w = 5.
        a0 = torch.tensor(5.0)
        b0 = torch.tensor(200.0)
        y, c, mu, tau, eps, eps0 = utils_testing.simulate_quadruplet_data(n_sites, n_copy_states, tree,
                                                                          a0, b0, eps_0=0.1,
                                                                          mu_v=mu_v, mu_w=mu_w,
                                                                          tau_v=tau_v, tau_w=tau_w)

        print(f"C: {c}")

        out_dir = "./test_output/" + self._testMethodName
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        config = Config(n_nodes=4, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.05,
                        diagnostics=False, out_dir=out_dir, annealing=1.)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)
        qc, qeps, qz, qmt = self.set_up_q(config)
        q = QuadrupletJointDist(config, qc, qz, qeps, qmt, tree, y)
        qc.initialize()
        qeps.initialize()
        qmt.initialize(loc=1., precision_factor=1., shape=100., rate=10.)
        copy_tree = CopyTree(config, q, y)

        copy_tree.run(n_iter=230)
        copy_tree.step()
        # print(q.c)

        # Assert
        print(f"Argmax q(C): {q.c.single_filtering_probs.argmax(dim=-1)}")
        print(f"Ground truth c: {c}")
        model_variational_comparisons.compare_qC_and_true_C(true_C=c, q_c=copy_tree.q.c)
        model_variational_comparisons.compare_quadruplet_mu_and_tau(true_mu_v=mu[0], true_mu_w=mu[1],
                                                                    true_tau_v=tau[0], true_tau_w=tau[1],
                                                                    q_mt=copy_tree.q.mt)
        model_variational_comparisons.compare_qEpsilon_and_true_epsilon(eps, copy_tree.q.eps)

    def test_quadruplet_initilization(self):
        utils.config.set_seed(0)
        n_init = 20
        tree = tests.utils_testing.get_quadtruplet_tree()
        n_cells = 2
        n_sites = 2000
        n_copy_states = 7
        mu_v = 1.2
        mu_w = 0.9
        tau_v = 10.
        tau_w = 5.
        a0 = torch.tensor(5.0)
        b0 = torch.tensor(200.0)
        y, c, mu, tau, eps, eps0 = utils_testing.simulate_quadruplet_data(n_sites, n_copy_states, tree,
                                                                          a0, b0, eps_0=0.1,
                                                                          mu_v=mu_v, mu_w=mu_w,
                                                                          tau_v=tau_v, tau_w=tau_w)

        print(f"C: {c}")

        out_dir = "./test_output/" + self._testMethodName
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        config = Config(n_nodes=4, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.05,
                        diagnostics=False, out_dir=out_dir, annealing=1.)
        test_dir_name = tests.utils_testing.create_test_output_catalog(config, self._testMethodName)
        d_mu_v_list = []
        d_mu_w_list = []
        d_tau_v_list = []
        d_tau_w_list = []
        for i in range(n_init):
            utils.config.set_seed(i)
            qc, qeps, qz, qmt = self.set_up_q(config)
            q = QuadrupletJointDist(config, qc, qz, qeps, qmt, tree, y)
            qc.initialize()
            qeps.initialize()
            qmt.initialize(loc=1., precision_factor=1., shape=100., rate=10.)
            copy_tree = CopyTree(config, q, y)

            copy_tree.run(n_iter=230)
            copy_tree.step()
            # print(q.c)

            # Assert
            print(f"Argmax q(C): {q.c.single_filtering_probs.argmax(dim=-1)}")
            print(f"Ground truth c: {c}")
            model_variational_comparisons.compare_qC_and_true_C(true_C=c, q_c=copy_tree.q.c)
            d_mu_v, d_mu_w, d_tau_v, d_tau_w = \
                model_variational_comparisons.compare_quadruplet_mu_and_tau(true_mu_v=mu[0], true_mu_w=mu[1],
                                                                            true_tau_v=tau[0], true_tau_w=tau[1],
                                                                            q_mt=copy_tree.q.mt)
            d_mu_v_list.append(d_mu_v)
            d_mu_w_list.append(d_mu_w)
            d_tau_v_list.append(d_tau_v)
            d_tau_w_list.append(d_tau_w)

        print(f"Min d_mu_v: {np.min(d_mu_v_list)}")
        print(f"Min d_mu_w: {np.min(d_mu_w_list)}")
        print(f"Min d_tau_v: {np.min(d_tau_v_list)}")
        print(f"Min d_tau_w: {np.min(d_tau_w_list)}")