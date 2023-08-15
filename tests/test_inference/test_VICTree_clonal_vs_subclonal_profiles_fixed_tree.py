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


def generate_global_profile_data(A, C, K, M_general, M_local, N, eps, mu, tau, y, z):
    M_tot = M_general + M_local
    tree_temp = nx.DiGraph()
    tree_temp.add_edge(0, 1)
    eps2, c2 = simul.simulate_copy_tree_data(2, M_general, A, tree_temp, eps_a=5., eps_b=M_tot, eps_0=0.01)
    c_general = c2[1].repeat(K, 1)  # Add same global structure to all clones
    c_general[0, :] = 2.  # reset root to 2
    y2 = simul.simulate_observations_Normal(N, M_general, c_general, z, mu, tau)
    eps_tot = {key: eps[key] * (M_local / M_tot) + eps2[(0, 1)] * (M_general / M_tot) for key in eps.keys()}
    c_tot = torch.zeros(K, M_tot)
    c_tot[:, 0:M_general] = c_general
    c_tot[:, M_general:M_tot] = C
    y_tot = torch.zeros((M_tot, N))
    y_tot[0:M_general, :] = y2
    y_tot[M_general:M_tot, :] = y
    return M_tot, c_tot, eps_tot, y_tot


class VICtreeClonalVsSubclonalProfilesFixedTreeTestCase(unittest.TestCase):

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)
        return qc, qt, qeps, qz, qpi, qmt

    def test_seven_node_tree_subclonal_profile_ratio_small(self):
        torch.manual_seed(0)
        K = 7
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        N = 500
        M_local = 300
        M_general = 1700
        A = 7
        dir_delta0 = 10.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 100.0
        y, c, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M_local, A, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_delta0)

        config_local = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_local, step_size=0.3,
                              diagnostics=False, annealing=1.)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_local)
        q = FixedTreeJointDist(config_local, qc, qz, qeps, qmt, qpi, tree, y)
        q.initialize()
        victree = VICTree(config_local, q, y, draft=True)

        victree.run(n_iter=100)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=victree.q.c,
                                                                q_z=victree.q.z, qpi=victree.q.pi,
                                                                q_mt=victree.q.mt, q_eps=qeps)

        # Extend clones with general structure
        M_tot, c_tot, eps_tot, y_tot = generate_global_profile_data(A, c, K, M_general, M_local, N, eps, mu, tau,
                                                                    y, z)

        config_general = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_tot, step_size=0.3,
                                diagnostics=False, annealing=1.)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config_general, self.id().replace(".", "/"),
                                                                       base_dir='./../test_output')

        qc2, qt2, qeps2, qz2, qpi2, qmt2 = self.set_up_q(config_general)
        q2 = FixedTreeJointDist(config_general, qc2, qz2, qeps2, qmt2, qpi2, tree, y_tot)
        q2.initialize()
        victree2 = VICTree(config_general, q2, y_tot, draft=True)

        victree2.run(n_iter=100)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y_tot, true_C=c_tot, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps_tot,
                                                                q_c=victree2.q.c,
                                                                q_z=victree2.q.z, qpi=victree2.q.pi,
                                                                q_mt=victree2.q.mt, q_eps=victree2.q.eps)
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])

        utils_testing.write_inference_test_output(victree, y, c, z, tree, mu, tau, eps, eps0, pi, test_dir_path=test_dir_name)

        c_true_and_qc_viterbi = np.zeros((2, K, M_tot))
        c_true_and_qc_viterbi[0] = np.array(c_tot)
        c_true_and_qc_viterbi[1] = np.array(victree2.q.c.get_viterbi())
        visualization_utils.visualize_copy_number_profiles_of_multiple_sources(c_true_and_qc_viterbi,
                                                                               save_path=test_dir_name +
                                                                                         '/c_plot.png')
        visualization_utils.visualize_observations_copy_number_profiles_of_multiple_sources(c_true_and_qc_viterbi,
                                                                                            y_tot, z,
                                                                                            save_path=test_dir_name +
                                                                                                      '/c_obs_plot.png')

        # self.assertGreater(ari, 0.7, msg='ari less than 0.7.')

    def test_init_clones_to_general_profile(self):
        torch.manual_seed(0)
        K = 6
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        N = 500
        M_local = 100
        M_general = 2000
        A = 7
        dir_delta0 = 10.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 100.0
        y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M_local, A, tree,
                                                                        nu_0=nu_0,
                                                                        lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                        a0=a0, b0=b0, dir_alpha0=dir_delta0)

        # Extend clones with general structure
        M_tot, c_tot, eps_tot, y_tot = generate_global_profile_data(A, C, K, M_general, M_local, N, eps, mu, tau,
                                                                         y, z)

        config_general = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_tot, step_size=0.3,
                                diagnostics=False, annealing=1.)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config_general, self.id().replace(".", "/"),
                                                                       base_dir='../../test_output')

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_general)
        q = FixedTreeJointDist(config_general, qc, qz, qeps, qmt, qpi, tree, y_tot)
        q.initialize()
        utils_testing.initialize_qc_to_true_values(c_tot, A, qc)#), indexes=list(range(0:M_general)))
        q.z.update(qmt, qc, qpi, y_tot)
        q.mt.update(qc, qz, y_tot)

        victree = VICTree(config_general, q, y_tot, draft=True)

        victree.run(n_iter=100)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y_tot, true_C=c_tot, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps_tot,
                                                                q_c=victree.q.c,
                                                                q_z=victree.q.z, qpi=victree.q.pi,
                                                                q_mt=victree.q.mt, q_eps=victree.q.eps)
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])

        c_true_and_qc_viterbi = np.zeros((2, K, M_tot))
        c_true_and_qc_viterbi[0] = np.array(c_tot)
        c_true_and_qc_viterbi[1] = np.array(victree.q.c.get_viterbi())
        visualization_utils.visualize_copy_number_profiles_of_multiple_sources(c_true_and_qc_viterbi,
                                                                               save_path=test_dir_name +
                                                                                         '/c_plot.png')
        visualization_utils.visualize_observations_copy_number_profiles_of_multiple_sources(c_true_and_qc_viterbi,
                                                                                            y_tot, z,
                                                                                            save_path=test_dir_name +
                                                                                                      '/c_obs_plot.png')

        # self.assertGreater(ari, 0.7, msg='ari less than 0.7.')

