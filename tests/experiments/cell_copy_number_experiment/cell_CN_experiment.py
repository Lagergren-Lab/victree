import logging
import os
import random
import sys

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as f
from sklearn.metrics import adjusted_rand_score

import simul
import tests.utils_testing
import utils.config
from inference.victree import VICTree
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons, utils_testing
from tests.utils_testing import simul_data_pyro_full_model, simulate_full_dataset_no_pyro
from utils import visualization_utils
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent


class CellCNExperiment():
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
        qmt = qMuTau(config, nu_prior=1., lambda_prior=10., alpha_prior=50., beta_prior=5.)
        return qc, qt, qeps, qz, qpi, qmt

    def ari_as_function_of_K_experiment(self, save_plot=False):
        utils.config.set_seed(0)

        ari_list = []
        seeds = list(range(0, 1))

        N_list = [10]
        M = 300
        A = 7
        n_iter = 500
        dir_delta0 = 1.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.

        for N in N_list:
            K = N + 1
            hard_assignments = {'z': torch.eye(N, dtype=torch.long)}
            fixed_z = torch.argmax(hard_assignments['z'], dim=-1) + 1
            tree = tests.utils_testing.get_tree_K_nodes_one_level(K)
            config = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M, step_size=1.0)

            a0 = 5.0
            b0 = 300.0
            out = simul.simulate_full_dataset(config, fixed_z=fixed_z,
                                              tree=tree,
                                              mu0=nu_0,
                                              lambda0=lambda_0, alpha0=alpha0,
                                              beta0=beta0,
                                              eps_a=a0, eps_b=b0, dir_delta=dir_delta0)
            y = out['obs']
            c = out['c']
            z = out['z']
            pi = out['pi']
            mu = out['mu']
            tau = out['tau']
            eps = out['eps']

            print(f"------------ Data set sanity check ------------")
            print(f"C: {c}")
            print(f"pi: {pi}")
            print(f"eps: {eps}")
            print(f"Mu in range: [{mu.min()}, {mu.max()}] ")
            print(f"Tau in range: [{tau.min()}, {tau.max()}] ")
            ari = []
            for seed in seeds:
                utils.config.set_seed(seed)
                qc, qt, qeps, qz_fixed, qpi, qmt = self.set_up_q(config)
                qz_fixed = qZ(config, true_params={'z': fixed_z})
                qmt_fixed = qMuTau(config, true_params={'mu': mu, 'tau': tau})
                # q = FixedTreeJointDist(config, qc, qz, qeps, qmt, qpi, tree, y)
                # q.initialize()
                qeps = qEpsilonMulti(config, true_params={'eps': eps})
                qmt.initialize(loc=mu, precision_factor=10.,
                                shape=5., rate=5./tau)
                qc.initialize()
                # Hard assign q(Z) to each clone
                for i in range(n_iter):
                    qc.update(y, qeps, qz_fixed, qmt, [tree], [1.])
                    qmt.update(qc, qz_fixed, y)
                    if i % 100 == 0:
                        print(qc.compute_elbo([tree], [1.], qeps))

                c_score_seed = model_variational_comparisons.compare_qC_and_true_C(c, qc, qz_perm=list(range(0, N+1)))
                mutau_score_seed = model_variational_comparisons.compare_qMuTau_with_true_mu_and_tau(mu, tau, qmt)
                c_all = np.zeros((2, K, M))
                c_all[0] = np.array(c)
                c_all[1] = np.array(qc.single_filtering_probs.argmax(dim=-1))
                visualization_utils.visualize_observations_copy_number_profiles_of_multiple_sources(c_all, y, z,
                                                                                                    title_suff='1')
                ari.append(c_score_seed)
                print(f"ARI for N {N} and seed {seed}: {c_score_seed}")
                print(f"E[q(mu)]: {[(i, qmt.nu[i].item(), mu[i].item()) for i in range(len(mu))]}")

            ari_list.append(ari)
            print(f"mean ARI for K {N}: {np.array(ari).mean()} ({np.array(ari).std()})")
            # Assert
            # model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=C, true_Z=z, true_pi=pi, true_mu=mu,
            #                                                  true_tau=tau, true_epsilon=eps, q_c=copy_tree.q.c,
            #                                                  q_z=copy_tree.q.z, qpi=copy_tree.q.pi,
            #                                                  q_mt=copy_tree.q.mt)

        return
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
            df = pd.DataFrame({
                'm': K_list,
                'ari': ari_means,
                'sd': ari_stds
            })
            df.to_csv(os.path.join(test_dir_name, f"k_ari_N{N}_M{M}_A{A}.csv"))


if __name__ == '__main__':
    experiment_class = CellCNExperiment()
    experiment_class.ari_as_function_of_K_experiment(save_plot=False)
