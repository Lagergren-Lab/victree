import logging
import random

import matplotlib.pyplot as plt
import networkx as nx
import torch
import torch.nn.functional as f
from pyro import poutine

import simul
import tests.utils_testing
import utils.config
from inference.copy_tree import CopyTree
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class VICtreeFixedTreeExperiment():
    """
    Test class for running small scale, i.e. runnable on local machine, experiments for fixed trees.
    Not using unittest test framework as it is incompatible with matplotlib GUI-backend.
    """

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config)
        return qc, qt, qeps, qz, qpi, qmt

    def ari_as_function_of_K_experiment(self):
        torch.manual_seed(0)
        K_list = [3, 4, 5, 6, 7, 8, 9, 10]
        for K in K_list:
            tree = tests.utils_testing.get_tree_K_nodes_random(K)
            n_cells = 100
            n_sites = 100
            n_copy_states = 7
            dir_alpha0 = torch.ones(K) * 10.
            nu_0 = torch.tensor(1.)
            lambda_0 = torch.tensor(10.)
            alpha0 = torch.tensor(500.)
            beta0 = torch.tensor(50.)
            a0 = torch.tensor(5.0)
            b0 = torch.tensor(200.0)
            y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(n_cells, n_sites, n_copy_states, tree,
                                                                            nu_0=nu_0,
                                                                            lambda_0=lambda_0, alpha0=alpha0, beta0=beta0,
                                                                            a0=a0, b0=b0, dir_alpha0=dir_alpha0)
            config = Config(n_nodes=K, n_states=n_copy_states, n_cells=n_cells, chain_length=n_sites, step_size=0.1,
                            diagnostics=True)
            test_dir_name = tests.utils_testing.create_test_output_catalog(config, self.__class__.__name__)
            qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
            q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
            q.initialize()
            copy_tree = CopyTree(config, q, y)

            copy_tree.run(100)
            print(q.c)

            # Assert
            diagnostics_dict = q.diagnostics_dict
            visualization_utils.plot_diagnostics_to_pdf(diagnostics_dict,
                                                        cells_to_vis_idxs=[0, int(n_cells / 2), int(n_cells / 3),
                                                                           n_cells - 1],
                                                        clones_to_vis_idxs=[1, 0],
                                                        edges_to_vis_idxs=[(0, 1)],
                                                        save_path=test_dir_name + '/diagnostics.pdf')
            model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
                                                              true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
                                                              q_z=copy_tree.q.z, qpi=copy_tree.q.pi, q_mt=copy_tree.q.mt)

