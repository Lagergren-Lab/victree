import logging
import os.path
import random
import sys
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
from utils.data_handling import DataHandler
from variational_distributions.joint_dists import FixedTreeJointDist
from tests import model_variational_comparisons, utils_testing
from tests.utils_testing import simulate_full_dataset_no_pyro
from utils import visualization_utils, data_handling
from utils.config import Config
from variational_distributions.var_dists import qEpsilonMulti, qT, qZ, qPi, qMuTau, qC, qMuAndTauCellIndependent, \
    qCMultiChrom


def generate_clonal_profile_data(A, C, K, M_clonal, M_subclonal, N, eps, mu, tau, y, z):
    M_tot = M_clonal + M_subclonal
    tree_temp = nx.DiGraph()
    tree_temp.add_edge(0, 1)
    eps2, c2 = simul.simulate_copy_tree_data(2, M_clonal, A, tree_temp, eps_a=5., eps_b=M_tot, eps_0=0.01)
    c_clonal = c2[1].repeat(K, 1)  # Add same global structure to all clones
    c_clonal[0, :] = 2.  # reset root to 2
    y2 = simul.simulate_observations_Normal(N, M_clonal, c_clonal, z, mu, tau)
    eps_tot = {key: eps[key] * (M_subclonal / M_tot) + eps2[(0, 1)] * (M_clonal / M_tot) for key in eps.keys()}
    c_tot = torch.zeros(K, M_tot)
    c_tot[:, 0:M_clonal] = c_clonal
    c_tot[:, M_clonal:M_tot] = C
    y_tot = torch.zeros((M_tot, N))
    y_tot[0:M_clonal, :] = y2
    y_tot[M_clonal:M_tot, :] = y
    return M_tot, c_tot, eps_tot, y_tot, c_clonal


#@unittest.skip('long exec test')
class VICtreeClonalVsSubclonalProfilesFixedTreeTestCase(unittest.TestCase):

    def set_up_q(self, config):
        qc = qC(config)
        qt = qT(config)
        qeps = qEpsilonMulti(config)
        qz = qZ(config)
        qpi = qPi(config)
        qmt = qMuTau(config, nu_prior=1.0, lambda_prior=100., alpha_prior=500., beta_prior=50.)
        return qc, qt, qeps, qz, qpi, qmt

    def test_seven_node_tree_subclonal_profile_ratio_small(self):
        """
        Demonstrates the poor performance of VICTree when running on clonal + subclonal data vs only running
        on subclonal data.
        """
        torch.manual_seed(0)
        n_iter = 100
        K = 7
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        N = 500
        M_subclonal = 300
        M_clonal = 1700
        A = 7
        dir_delta0 = 10.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 300.0
        y, c_local, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M_subclonal, A, tree,
                                                                              nu_0=nu_0,
                                                                              lambda_0=lambda_0, alpha0=alpha0,
                                                                              beta0=beta0,
                                                                              a0=a0, b0=b0, dir_alpha0=dir_delta0)

        config_local = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_subclonal, step_size=0.3,
                              diagnostics=False, annealing=1.)
        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_local)
        q = FixedTreeJointDist(y, config_local, qc, qz, qeps, qmt, qpi, tree)
        q.initialize()
        victree = VICTree(config_local, q, y, draft=True)

        victree.run(n_iter=n_iter)

        # Assert
        torch.set_printoptions(precision=2)
        out = model_variational_comparisons.fixed_T_comparisons(obs=y, true_C=c_local, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=victree.q.c,
                                                                q_z=victree.q.z, qpi=victree.q.pi,
                                                                q_mt=victree.q.mt, q_eps=qeps)

        # Extend clones with general structure
        M_tot, c_tot, eps_tot, y_tot, c_clonal = generate_clonal_profile_data(A, c_local, K, M_clonal, M_subclonal, N,
                                                                              eps, mu,
                                                                              tau, y, z)

        config_general = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_tot, step_size=0.3,
                                diagnostics=False, annealing=1.)

        test_dir_name = tests.utils_testing.create_test_output_catalog(config_general, self.id().replace(".", "/"),
                                                                       base_dir='./../test_output')

        qc2, qt2, qeps2, qz2, qpi2, qmt2 = self.set_up_q(config_general)
        q2 = FixedTreeJointDist(y_tot, config_general, qc2, qz2, qeps2, qmt2, qpi2, tree)
        q2.initialize()
        victree2 = VICTree(config_general, q2, y_tot, draft=True)

        victree2.run(n_iter=n_iter)

        # Assert
        torch.set_printoptions(precision=2)
        out2 = model_variational_comparisons.fixed_T_comparisons(obs=y_tot, true_C=c_tot, true_Z=z, true_pi=pi,
                                                                 true_mu=mu,
                                                                 true_tau=tau, true_epsilon=eps_tot,
                                                                 q_c=victree2.q.c,
                                                                 q_z=victree2.q.z, qpi=victree2.q.pi,
                                                                 q_mt=victree2.q.mt, q_eps=victree2.q.eps)
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])
        ari2, perm2, acc2 = (out2['ari'], out2['perm'], out2['acc'])
        c_local_remapped = utils_testing.remap_tensor(c_local, perm)
        c_tot_remapped = utils_testing.remap_tensor(c_tot, perm2)
        utils_testing.write_inference_test_output(victree, y, c_local_remapped, z, tree, mu, tau, eps, eps0, pi,
                                                  test_dir_path=test_dir_name, file_name_prefix='subclonal_')
        utils_testing.write_inference_test_output(victree2, y_tot, c_tot_remapped, z, tree, mu, tau, eps, eps0, pi,
                                                  test_dir_path=test_dir_name, file_name_prefix='clonal_and_subclonal_')

    def test_init_clones_to_true_clonal_profile(self):
        torch.manual_seed(0)
        n_iter = 100
        K = 7
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        N = 500
        M_subclonal = 300
        M_clonal = 1700
        A = 7
        dir_delta0 = 10.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 300.0
        y, c_local, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M_subclonal, A, tree,
                                                                              nu_0=nu_0,
                                                                              lambda_0=lambda_0, alpha0=alpha0,
                                                                              beta0=beta0,
                                                                              a0=a0, b0=b0, dir_alpha0=dir_delta0)

        # Extend clones with general structure
        M_tot, c_tot, eps_tot, y_tot, c_clonal = generate_clonal_profile_data(A, c_local, K, M_clonal, M_subclonal, N,
                                                                              eps, mu,
                                                                              tau, y, z)

        # Run VICTree using normal initialization
        config_init1 = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_tot, step_size=0.3,
                              diagnostics=False, annealing=1.)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_init1)
        q = FixedTreeJointDist(y_tot, config_init1, qc, qz, qeps, qmt, qpi, tree)
        q.initialize()
        victree = VICTree(config_init1, q, y_tot, draft=True)
        victree.run(n_iter=n_iter)

        out = model_variational_comparisons.fixed_T_comparisons(obs=y_tot, true_C=c_tot, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=victree.q.c,
                                                                q_z=victree.q.z, qpi=victree.q.pi,
                                                                q_mt=victree.q.mt, q_eps=qeps)

        # Run VICTree using init to clonal structure
        config_init2 = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_tot, step_size=0.3,
                              diagnostics=False, annealing=1.)

        qc2, qt2, qeps2, qz2, qpi2, qmt2 = self.set_up_q(config_init2)
        q2 = FixedTreeJointDist(y_tot, config_init2, qc2, qz2, qeps2, qmt2, qpi2, tree)
        q2.initialize()
        c_clonal_extended = torch.ones_like(c_tot) * 2.
        c_clonal_extended[:, 0:M_clonal] = c_clonal
        utils_testing.initialize_qc_to_true_values(c_clonal_extended, A, qc2)
        victree2 = VICTree(config_init2, q2, y_tot, draft=True)

        victree2.run(n_iter=n_iter)

        # Assert
        test_dir_name = tests.utils_testing.create_test_output_catalog(config_init2, self.id().replace(".", "/"),
                                                                       base_dir='./../test_output')
        torch.set_printoptions(precision=2)
        out2 = model_variational_comparisons.fixed_T_comparisons(obs=y_tot, true_C=c_tot, true_Z=z, true_pi=pi,
                                                                 true_mu=mu,
                                                                 true_tau=tau, true_epsilon=eps_tot,
                                                                 q_c=victree2.q.c,
                                                                 q_z=victree2.q.z, qpi=victree2.q.pi,
                                                                 q_mt=victree2.q.mt, q_eps=victree2.q.eps)
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])
        ari2, perm2, acc2 = (out2['ari'], out2['perm'], out2['acc'])
        c_tot_remapped = c_tot[perm]
        c_tot_remapped2 = c_tot[perm2]
        z_remapped = torch.tensor([perm[i] for i in z])
        z2_remapped = torch.tensor([perm2[i] for i in z])
        utils_testing.write_inference_test_output(victree, y_tot, c_tot_remapped, z_remapped, tree, mu, tau, eps, eps0, pi,
                                                  test_dir_path=test_dir_name, file_name_prefix='default_init_')
        utils_testing.write_inference_test_output(victree2, y_tot, c_tot_remapped2, z2_remapped, tree, mu, tau, eps, eps0, pi,
                                                  test_dir_path=test_dir_name, file_name_prefix='clonal_init_')

    def test_clonal_profile_init(self):
        utils.config.set_seed(0)
        n_iter = 200
        K = 7
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        N = 500
        M_subclonal = 300
        M_clonal = 1700
        A = 7
        dir_delta0 = 10.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 300.0
        y, c_local, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M_subclonal, A, tree,
                                                                              nu_0=nu_0,
                                                                              lambda_0=lambda_0, alpha0=alpha0,
                                                                              beta0=beta0,
                                                                              a0=a0, b0=b0, dir_alpha0=dir_delta0)

        # Extend clones with general structure
        M_tot, c_tot, eps_tot, y_tot, c_clonal = generate_clonal_profile_data(A, c_local, K, M_clonal, M_subclonal, N,
                                                                              eps, mu,
                                                                              tau, y, z)

        # Run VICTree using normal initialization
        config_init1 = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_tot, step_size=0.3,
                              diagnostics=False, annealing=1.)

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_init1)
        q = FixedTreeJointDist(y_tot, config_init1, qc, qz, qeps, qmt, qpi, tree)
        q.initialize()
        victree = VICTree(config_init1, q, y_tot, draft=True)
        victree.run(n_iter=n_iter)

        out = model_variational_comparisons.fixed_T_comparisons(obs=y_tot, true_C=c_tot, true_Z=z, true_pi=pi,
                                                                true_mu=mu,
                                                                true_tau=tau, true_epsilon=eps,
                                                                q_c=victree.q.c,
                                                                q_z=victree.q.z, qpi=victree.q.pi,
                                                                q_mt=victree.q.mt, q_eps=qeps)
        # Run VICTree using init to clonal structure
        config_init2 = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_tot, step_size=0.3,
                              diagnostics=False, annealing=1.)

        qc2, qt2, qeps2, qz2, qpi2, qmt2 = self.set_up_q(config_init2)
        q2 = FixedTreeJointDist(y_tot, config_init2, qc2, qz2, qeps2, qmt2, qpi2, tree)
        q2.initialize()
        qc2.initialize(method='clonal', obs=y_tot)
        victree2 = VICTree(config_init2, q2, y_tot, draft=True)

        victree2.run(n_iter=n_iter)

        # Assert
        test_dir_name = tests.utils_testing.create_test_output_catalog(config_init2, self.id().replace(".", "/"),
                                                                       base_dir='./../test_output')
        torch.set_printoptions(precision=2)
        out2 = model_variational_comparisons.fixed_T_comparisons(obs=y_tot, true_C=c_tot, true_Z=z, true_pi=pi,
                                                                 true_mu=mu,
                                                                 true_tau=tau, true_epsilon=eps_tot,
                                                                 q_c=victree2.q.c,
                                                                 q_z=victree2.q.z, qpi=victree2.q.pi,
                                                                 q_mt=victree2.q.mt, q_eps=victree2.q.eps)
        ari, perm, acc = (out['ari'], out['perm'], out['acc'])
        ari2, perm2, acc2 = (out2['ari'], out2['perm'], out2['acc'])
        c_tot_remapped = c_tot[perm]
        c_tot_remapped2 = c_tot[perm2]
        z_remapped = torch.tensor([perm[i] for i in z])
        z2_remapped = torch.tensor([perm2[i] for i in z])
        utils_testing.write_inference_test_output(victree, y_tot, c_tot_remapped, z_remapped, tree, mu, tau, eps, eps0, pi,
                                                  test_dir_path=test_dir_name, file_name_prefix='default_init_')
        utils_testing.write_inference_test_output(victree2, y_tot, c_tot_remapped2, z2_remapped, tree, mu, tau, eps, eps0, pi,
                                                  test_dir_path=test_dir_name, file_name_prefix='clonal_init_')

    @unittest.skip('long exec test')
    def test_clonal_profile_init_real_data(self):
        torch.manual_seed(0)

        n_iter = 200
        K = 7
        tree = nx.DiGraph()
        tree.add_edge(0, 1)
        tree.add_edge(0, 2)
        tree.add_edge(1, 3)
        tree.add_edge(1, 4)
        tree.add_edge(3, 5)
        tree.add_edge(3, 6)
        A = 7
        dir_delta0 = 10.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 300.0

        file_path ='../../data/x_data/P01-066_cn_data.h5ad'
        data_handler = DataHandler(file_path)
        y = data_handler.norm_reads
        M, N = y.shape


        # Run VICTree using normal initialization
        config_init1 = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M, step_size=0.3,
                              diagnostics=False, annealing=1., split=True, chromosome_indexes=data_handler.get_chr_idx())

        test_dir_name = tests.utils_testing.create_test_output_catalog(config_init1, self.id().replace(".", "/"),
                                                                       base_dir='./../test_output')
        orig_stdout = sys.stdout
        f = open(test_dir_name + '/out.txt', 'w')
        sys.stdout = f

        qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_init1)
        qc = qCMultiChrom(config_init1)
        q = FixedTreeJointDist(y, config_init1, qc, qz, qeps, qmt, qpi, tree)
        q.initialize()
        qmt.initialize(method='fixed', loc=1.0, precision_factor=100., shape=5000., rate=500.)
        victree = VICTree(config_init1, q, y, draft=True)
        victree.run(n_iter=n_iter)


        # Run VICTree using init to clonal structure
        config_init2 = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M, step_size=0.3,
                              diagnostics=False, annealing=1., split=True, chromosome_indexes=data_handler.get_chr_idx())

        qc2, qt2, qeps2, qz2, qpi2, qmt2 = self.set_up_q(config_init2)
        qc2 = qCMultiChrom(config_init2)
        q2 = FixedTreeJointDist(y, config_init2, qc2, qz2, qeps2, qmt2, qpi2, tree)
        q2.initialize()
        qc2.initialize(method='clonal', obs=y)
        qmt2.initialize(method='fixed', loc=1.0, precision_factor=100., shape=5000., rate=500.)
        qz2.update(qmt2, qc2, qpi2, y)
        victree2 = VICTree(config_init2, q2, y, draft=True)

        victree2.run(n_iter=n_iter)

        # Assert

        torch.set_printoptions(precision=2)

        print(f"Dir param 1: {victree.q.pi}")
        print(f"Dir param 2: {victree2.q.pi}")
        z1 = victree.q.z.pi
        z2 = victree2.q.z.pi
        print(f"Avg assignment 1: {z1.mean(dim=0)}")
        print(f"Avg assignment 2: {z2.mean(dim=0)}")

        visualization_utils.visualize_qC_qZ_and_obs(victree.q.c, victree.q.z, y,
                                                    save_path=test_dir_name + '/default_init_qc_qz_obs_plot')
        visualization_utils.visualize_qC_qZ_and_obs(victree2.q.c, victree2.q.z, y,
                                                    save_path=test_dir_name + '/clonal_init_qc_qz_obs_plot')

        visualization_utils.visualize_qMuTau(victree.q.mt, save_path=test_dir_name + '/defualt_init_qmt_plot')
        visualization_utils.visualize_qMuTau(victree2.q.mt, save_path=test_dir_name + '/clonal_init_qmt_plot')

        print(f"ELBO 1: {victree.elbo}")
        print(f"ELBO 2: {victree2.elbo}")
        sys.stdout = orig_stdout
        f.close()

    @unittest.skip('long exec test')
    def test_clonal_profile_init_and_split_vs_no_split(self):
        utils.config.set_seed(0)
        seeds = list(range(3, 4))
        data_seeds = list(range(0, 2))
        n_iter = 50
        K = 6
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        N = 500
        M_subclonal = 300
        M_clonal = 1700
        A = 7
        dir_delta0 = 10.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 300.0

        ari_list_list = []
        ari2_list_list = []
        for data_seed in data_seeds:
            utils.config.set_seed(data_seed)
            y, c_local, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M_subclonal, A, tree,
                                                                                  nu_0=nu_0,
                                                                                  lambda_0=lambda_0, alpha0=alpha0,
                                                                                  beta0=beta0,
                                                                                  a0=a0, b0=b0, dir_alpha0=dir_delta0)

            # Extend clones with general structure
            M_tot, c_tot, eps_tot, y_tot, c_clonal = generate_clonal_profile_data(A, c_local, K, M_clonal, M_subclonal, N,
                                                                                  eps, mu,
                                                                                  tau, y, z)

            ari_list = []
            ari_list2 = []
            for i in range(len(seeds)):
                # Run VICTree using normal initialization
                utils.config.set_seed(seeds[i])
                config_init1 = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_tot, step_size=0.3,
                                      diagnostics=False, annealing=1., split=True)

                qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_init1)
                q = FixedTreeJointDist(y_tot, config_init1, qc, qz, qeps, qmt, qpi, tree)
                q.initialize()
                victree = VICTree(config_init1, q, y_tot, draft=True)
                victree.run(n_iter=n_iter)

                out = model_variational_comparisons.fixed_T_comparisons(obs=y_tot, true_C=c_tot, true_Z=z, true_pi=pi,
                                                                        true_mu=mu,
                                                                        true_tau=tau, true_epsilon=eps,
                                                                        q_c=victree.q.c,
                                                                        q_z=victree.q.z, qpi=victree.q.pi,
                                                                        q_mt=victree.q.mt, q_eps=qeps)
                # Run VICTree using init to clonal structure
                config_init2 = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_tot, step_size=0.3,
                                      diagnostics=False, annealing=1., split=False)

                qc2, qt2, qeps2, qz2, qpi2, qmt2 = self.set_up_q(config_init2)
                q2 = FixedTreeJointDist(y_tot, config_init2, qc2, qz2, qeps2, qmt2, qpi2, tree)
                q2.initialize()
                #qc2.initialize(method='clonal', obs=y_tot)
                victree2 = VICTree(config_init2, q2, y_tot, draft=True)
                victree2.run(n_iter=n_iter)

                # Assert
                test_dir_name = tests.utils_testing.create_test_output_catalog(config_init2, self.id().replace(".", "/")
                                                                               + f'/SVI/data_seed_{data_seed}',
                                                                               base_dir='./../test_output')
                torch.set_printoptions(precision=2)
                out2 = model_variational_comparisons.fixed_T_comparisons(obs=y_tot, true_C=c_tot, true_Z=z, true_pi=pi,
                                                                         true_mu=mu,
                                                                         true_tau=tau, true_epsilon=eps_tot,
                                                                         q_c=victree2.q.c,
                                                                         q_z=victree2.q.z, qpi=victree2.q.pi,
                                                                         q_mt=victree2.q.mt, q_eps=victree2.q.eps)
                ari, perm, acc = (out['ari'], out['perm'], out['acc'])
                ari2, perm2, acc2 = (out2['ari'], out2['perm'], out2['acc'])
                c_tot_remapped = c_tot[perm]
                c_tot_remapped2 = c_tot[perm2]
                z_remapped = torch.tensor([perm[i] for i in z])
                z2_remapped = torch.tensor([perm2[i] for i in z])
                utils_testing.write_inference_test_output(victree, y_tot, c_tot_remapped, z_remapped, tree, mu, tau, eps, eps0, pi,
                                                          test_dir_path=test_dir_name, file_name_prefix=f'seed{seeds[i]}_split_')
                utils_testing.write_inference_test_output(victree2, y_tot, c_tot_remapped2, z2_remapped, tree, mu, tau, eps, eps0, pi,
                                                          test_dir_path=test_dir_name, file_name_prefix=f'seed{seeds[i]}_no_split_')

                ari_list.append(ari)
                ari_list2.append(ari2)

            print(f'ARI scores split: {ari_list}')
            print(f'ARI scores no split: {ari_list2}')
            ari_list_list.append(ari_list)
            ari2_list_list.append(ari_list2)

        print(f'SPLIT ARIs: {ari_list_list}')
        print(f'NO SPLIT ARIs: {ari2_list_list}')
        print(f'SPLIT - ARI mean over inference seed: {np.mean(ari_list_list, axis=-1)} ({np.std(ari_list_list, axis=-1)})')
        print(f'NO SPLIT - ARI mean over inference seed: {np.mean(ari2_list_list, axis=-1)} ({np.std(ari2_list_list, axis=-1)})')

    #@unittest.skip('long exec test')
    def test_SVI_split_and_clonal_init(self):
        utils.config.set_seed(0)
        seeds = list(range(10, 11))
        data_seeds = list(range(0, 2))
        n_iter = 30
        K = 6
        tree = tests.utils_testing.get_tree_K_nodes_random(K)
        N = 500
        M_subclonal = 300
        M_clonal = 1700
        A = 7
        dir_delta0 = 3.
        nu_0 = 1.
        lambda_0 = 10.
        alpha0 = 500.
        beta0 = 50.
        a0 = 10.0
        b0 = 300.0

        ari_list_list = []
        for data_seed in data_seeds:
            utils.config.set_seed(data_seed)
            y, c_local, z, pi, mu, tau, eps, eps0 = simulate_full_dataset_no_pyro(N, M_subclonal, A, tree,
                                                                                  nu_0=nu_0,
                                                                                  lambda_0=lambda_0, alpha0=alpha0,
                                                                                  beta0=beta0,
                                                                                  a0=a0, b0=b0, dir_alpha0=dir_delta0)

            # Extend clones with general structure
            M_tot, c_tot, eps_tot, y_tot, c_clonal = generate_clonal_profile_data(A, c_local, K, M_clonal, M_subclonal, N,
                                                                                  eps, mu,
                                                                                  tau, y, z)

            ari_list = []
            for i in range(len(seeds)):
                # Run VICTree using SVI, split and init to clonal structure
                config_init = Config(n_nodes=K, n_states=A, n_cells=N, chain_length=M_tot, step_size=0.05,
                                     step_size_scheme='inverse', batch_size=100,
                                     diagnostics=True, annealing=1., split=True, SVI=True)
                test_dir_name = tests.utils_testing.create_test_output_catalog(config_init, self.id().replace(".", "/")
                                                                               + f'/data_seed_{data_seed}',
                                                                               base_dir='./../test_output')
                config_init.out_dir = test_dir_name

                qc, qt, qeps, qz, qpi, qmt = self.set_up_q(config_init)
                q = FixedTreeJointDist(y_tot, config_init, qc, qz, qeps, qmt, qpi, tree)
                q.initialize()
                qc.initialize(method='clonal', obs=y_tot)
                victree = VICTree(config_init, q, y_tot, draft=True)
                victree.run(n_iter=n_iter)

                # Assert
                torch.set_printoptions(precision=2)
                out = model_variational_comparisons.fixed_T_comparisons(obs=y_tot, true_C=c_tot, true_Z=z, true_pi=pi,
                                                                        true_mu=mu,
                                                                        true_tau=tau, true_epsilon=eps_tot,
                                                                        q_c=victree.q.c,
                                                                        q_z=victree.q.z, qpi=victree.q.pi,
                                                                        q_mt=victree.q.mt, q_eps=victree.q.eps)
                ari, perm, acc = (out['ari'], out['perm'], out['acc'])
                c_tot_remapped = c_tot[perm]
                z2_remapped = torch.tensor([perm[i] for i in z])
                utils_testing.write_inference_test_output(victree, y_tot, c_tot_remapped, z2_remapped, tree, mu, tau, eps, eps0, pi,
                                                          test_dir_path=test_dir_name, file_name_prefix=f'seed{seeds[i]}')

                ari_list.append(ari)

            print(f'ARI scores SVI: {ari_list}')
            ari_list_list.append(ari_list)

        print(f'SPLIT ARIs: {ari_list_list}')
        print(f'ARI mean over inference seed: {np.mean(ari_list_list, axis=-1)} ({np.std(ari_list_list, axis=-1)})')

