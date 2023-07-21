import logging
import os
import random
import sys

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as f
from sklearn.metrics import adjusted_rand_score

import simul
import tests.utils_testing
import utils.config
from inference.victree import VICTree
from variational_distributions.joint_dists import VarTreeJointDist
from tests import model_variational_comparisons
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils, tree_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class TreeTopologyPostInferenceExperiment():
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

    def edge_probability_experiment(self, save_plot=False):
        utils.config.set_seed(0)

        K_list = list(range(8, 10))
        ari_list = []
        seeds = list(range(0, 2))

        N = 500
        M = 500
        A = 7
        dir_alpha0 = 10.
        nu_0 = torch.tensor(1.)
        lambda_0 = torch.tensor(10.)
        alpha0 = torch.tensor(500.)
        beta0 = torch.tensor(50.)

        for K in K_list:
            tree = tests.utils_testing.get_tree_K_nodes_random(K)

            a0 = torch.tensor(5.0)
            b0 = torch.tensor(200.0)
            y, C, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M, A, tree,
                                                                            nu_0=nu_0,
                                                                            lambda_0=lambda_0, alpha0=alpha0,
                                                                            beta0=beta0,
                                                                            a0=a0, b0=b0, dir_alpha0=dir_alpha0)
            print(f"------------ Data set sanity check ------------")
            print(f"C: {C}")
            print(f"pi: {pi}")
            print(f"eps: {eps}")
            print(f"tree edges: {tree.edges}")
            ari = []
            for seed in seeds:
                utils.config.set_seed(seed)
                config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M, step_size=0.3)
                qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config)
                q = VarTreeJointDist(config, obs=y, qc=qc, qz=qz, qt=qt, qeps=qeps, qmt=qmt, qpi=qpi)
                q.initialize()
                victree = VICTree(config, q, y)

                victree.run(n_iter=100)

                ari_seed, best_perm, accuracy_best = model_variational_comparisons.compare_qZ_and_true_Z(true_Z=z, q_z=victree.q.z)
                ari.append(ari_seed)
                print(f"ARI for K {K} and seed {seed}: {ari_seed}")
                print(f"Best accuracy for K {K} and seed {seed}: {accuracy_best}")
                T_list_seed, w_T_list_seed = victree.q.t.get_trees_sample(sample_size=100)
                unique_seq, unique_seq_idx, multiplicity = tree_utils.unique_trees_and_multiplicity(T_list_seed)
                print(f"N uniques trees sampled: {len(unique_seq)}")
                T_list_seed_remapped = tree_utils.remap_edge_labels(T_list_seed, best_perm)
                unique_edges_list, unique_edges_count = tree_utils.get_unique_edges(T_list_seed_remapped)
                x_axis = list(range(0, len(unique_edges_list)))
                y_axis = [unique_edges_count[e].item() for e in unique_edges_list]
                true_tree_edges = [unique_edges_count[e].item() for e in tree.edges]
                labels = [str(e) if e not in tree.edges else f'[{e[0]}, {e[1]}]' for e in unique_edges_list]
                colors = ['blue' if e not in tree.edges else 'orange' for e in unique_edges_list]
                for i in x_axis:
                    plt.scatter(x_axis[i], y_axis[i], s=30, c=colors[i])
                #plt.plot(x_axis, true_tree_edges, 'x')
                plt.xticks(ticks=x_axis, labels=labels, rotation=60)
                plt.ylabel('Edges count')
                plt.xlabel('Unique edges in sampled trees')
                plt.title(f'Sampled edges experiment seed {seed} - L: {100} K:{K} N: {N} - M: {M} - A: {A}')
                if save_plot:
                    dirs = os.getcwd().split('/')
                    dir_top_idx = dirs.index('experiments')
                    dir_path = dirs[dir_top_idx:]
                    path = os.path.join(*dir_path, self.__class__.__name__, sys._getframe().f_code.co_name)
                    base_dir = '../../test_output'
                    test_dir_name = tests.utils_testing.create_experiment_output_catalog(path, base_dir)
                    plt.savefig(test_dir_name + f"/T_edge_plot_seed{seed}_K{K}_N{N}_M{M}_A{A}.png")
                    plt.close()

            ari_list.append(ari)
            print(f"mean ARI for K {K}: {np.array(ari).mean()} ({np.array(ari).std()})")
            # Assert


        np_ari = np.array(ari_list)
        ari_means = np_ari.mean(axis=-1)
        ari_stds = np_ari.std(axis=-1)
        plt.errorbar(K_list, ari_means, ari_stds, linestyle='None', marker='^')
        plt.xlabel('K - number of clones')
        plt.ylabel('ARI score')
        plt.title(f'Fixed tree clustering performance - N: {N} - M: {M} - A: {A}')
        if save_plot:
            dirs = os.getcwd().split('/')
            dir_top_idx = dirs.index('experiments')
            dir_path = dirs[dir_top_idx:]
            path = os.path.join(*dir_path, self.__class__.__name__, sys._getframe().f_code.co_name)
            base_dir = '../../test_output'
            test_dir_name = tests.utils_testing.create_experiment_output_catalog(path, base_dir)
            plt.savefig(test_dir_name + f"/ari_plot_N{N}_M{M}_A{A}.png")
            plt.close()


if __name__ == '__main__':
    experiment_class = TreeTopologyPostInferenceExperiment()
    experiment_class.edge_probability_experiment(save_plot=True)