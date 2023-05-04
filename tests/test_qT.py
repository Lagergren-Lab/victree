import itertools
import unittest

import numpy as np
import torch

import simul
import utils.config
from tests import utils_testing
from utils import tree_utils, visualization_utils
from utils.config import Config
from utils.tree_utils import tree_to_newick
from variational_distributions.var_dists import qEpsilonMulti, qT, qC, qEpsilon


class qTTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_q_T_running_for_two_simple_Ts_random_qC(self):
        M = 20
        A = 5
        N = 5
        config = Config(n_nodes=N, n_states=A, chain_length=M)
        q_T = qT(config=config)
        T_list, q_C_pairwise_marginals = utils_testing.get_two_simple_trees_with_random_qCs(M, A, N)
        q_C = qC(config=config)
        q_C.couple_filtering_probs = q_C_pairwise_marginals
        q_epsilon = qEpsilonMulti(config=config)
        q_epsilon.initialize()
        # Act
        log_q_T = q_T.update_CAVI(T_list, q_C, q_epsilon)

        # Assert
        print(f"log_q_T of T_1 and T_2: {log_q_T}")

    def test_tree_enumeration(self):
        qt5: qT = qT(config=Config(n_nodes=5))
        qt5.initialize()
        trees, trees_log_prob = qt5.enumerate_trees()
        print(len(trees))
        self.assertAlmostEqual(trees_log_prob.exp().sum().item(), 1., places=5,
                               msg='q(T) is not normalized')

    def test_tree_sampling_accuracy(self):
        qt5: qT = qT(config=Config(n_nodes=5))
        qt5.initialize()
        trees, trees_log_prob = qt5.enumerate_trees()
        dsl_trees, dsl_trees_log_weights = qt5.get_trees_sample(sample_size=1000,
                                                                     torch_tensor=True, log_scale=True)
        self.assertAlmostEqual(dsl_trees_log_weights.exp().sum().item(), 1., places=5)
        unique_dsl_trees = {}
        for t, lw in zip(dsl_trees, dsl_trees_log_weights):
            t_str = tree_to_newick(t)
            if t_str not in unique_dsl_trees:
                unique_dsl_trees[t_str] = lw
            else:
                unique_dsl_trees[t_str] = np.logaddexp(unique_dsl_trees[t_str], lw)

        k = 10
        # sort enumerated trees
        _, topk_idx = torch.topk(trees_log_prob, k=k)
        print(f"TOP {k} enumerated trees")
        for i in topk_idx:
            print(f"{tree_to_newick(trees[i])}: {trees_log_prob[i].exp()}")

        # sort unique sampled trees
        topk_sampled_str = [None] * k
        sampled_lw = torch.zeros(k) - torch.inf
        for i in range(k):
            for t_str, lw in unique_dsl_trees.items():
                if lw > sampled_lw[i]:
                    sampled_lw[i] = lw
                    topk_sampled_str[i] = t_str

            unique_dsl_trees.pop(topk_sampled_str[i])

        print(f"TOP {k} sampled trees")
        for t_str, lw in zip(topk_sampled_str, sampled_lw):
            print(f"{t_str}: {lw.exp()}")


    def test_qT_update_low_weights_for_improbable_epsilon(self):
        utils.config.set_seed(0)
        N = 100
        M = 100
        K = 10
        A = 7
        eps_a = 5.
        eps_b = 200.
        true_tree = utils_testing.get_tree_K_nodes_random(K)
        config = Config(n_cells=N, chain_length=M, n_nodes=K, n_states=A)
        out_simul = simul.simulate_full_dataset(config, tree=true_tree, eps_a=eps_a, eps_b=eps_b, mu0=1., lambda0=10.,
                          alpha0=500., beta0=50., dir_alpha=10.)
        c = out_simul["c"]
        eps = out_simul["eps"]
        q_T = qT(config=config)
        q_C = qC(config=config)
        q_C_pairwise_marginals = utils_testing.get_two_sliced_marginals_from_one_slice_marginals(c, A)
        q_C.couple_filtering_probs = q_C_pairwise_marginals
        q_epsilon = qEpsilonMulti(config=config)
        gedges = [(u, v) for u, v in itertools.product(range(config.n_nodes),
                                                       range(config.n_nodes)) if v != 0 and u != v]

        # Set epsilon parameters close to true parameters for edges in true tree and far away for edges not in tree
        eps_alpha_dict = {e: torch.tensor(1.) for e in gedges}
        eps_beta_dict = {e: 1./eps[e] if e in eps.keys() else torch.tensor(eps_b / 100.) for e in gedges}
        q_epsilon.initialize(method="fixed", eps_alpha_dict=eps_alpha_dict, eps_beta_dict=eps_beta_dict)
        q_T.initialize()
        q_T.update(q_C, q_epsilon)
        print(q_T)

        # Expect weights of edges in true tree to be larger than edges not in true tree
        min_weight = np.min([q_T.weight_matrix[u, v] for (u, v) in true_tree.edges()])

        for u in range(K):
            for v in range(K):
                if (u, v) not in true_tree.edges() and v != 0 and u != v:
                    self.assertGreater(min_weight, q_T.weight_matrix[u, v])

    def test_qT_given_true_parameters(self):
        utils.config.set_seed(0)
        N = 100
        M = 100
        K = 5
        A = 7
        L = 100
        eps_a = 5.
        eps_b = 200.
        off_set_factor = 1 / 100.
        true_tree = utils_testing.get_tree_K_nodes_random(K)
        config = Config(n_cells=N, chain_length=M, n_nodes=K, n_states=A, wis_sample_size=L)
        out_simul = simul.simulate_full_dataset(config, tree=true_tree, eps_a=eps_a, eps_b=eps_b, mu0=1., lambda0=10.,
                          alpha0=500., beta0=50., dir_alpha=10.)
        c = out_simul["c"]
        eps = out_simul["eps"]
        q_T = qT(config=config)
        q_C = qC(config=config)
        q_C_pairwise_marginals = utils_testing.get_two_sliced_marginals_from_one_slice_marginals(c, A)
        q_C.couple_filtering_probs = q_C_pairwise_marginals
        q_epsilon = qEpsilonMulti(config=config)
        gedges = [(u, v) for u, v in itertools.product(range(config.n_nodes),
                                                       range(config.n_nodes)) if v != 0 and u != v]
        eps_alpha_dict = {e: torch.tensor(eps_a) for e in gedges}
        eps_beta_dict = {e: eps_a/eps[e] if e in eps.keys() else torch.tensor(eps_b * off_set_factor) for e in gedges}
        q_epsilon.initialize(method="fixed", eps_alpha_dict=eps_alpha_dict, eps_beta_dict=eps_beta_dict)
        q_T.initialize()
        q_T.update(q_C, q_epsilon)

        test_dir_name = utils_testing.create_test_output_catalog(config, self.id().replace(".", "/"))

        print(q_T)
        print(f"True tree: {tree_utils.tree_to_newick(true_tree, 0)}")
        T_list, w_T_list, log_g_T_list = q_T.get_trees_sample(sample_size=L)
        g_T_list = np.exp(log_g_T_list)
        print(f"g(T): {g_T_list}")


        T_undirected_list = tree_utils.to_undirected(T_list)
        prufer_list = tree_utils.to_prufer_sequences(T_undirected_list)
        unique_seq, unique_seq_idx = tree_utils.unique_trees(prufer_list)
        print(f"N unique trees: {len(unique_seq_idx)}")
        T_list_unique = [T_list[i] for i in unique_seq_idx]
        w_T_list_unique = [w_T_list[i] for i in unique_seq_idx]
        g_T_list_unique = [g_T_list[i] for i in unique_seq_idx]
        distances = tree_utils.distances_to_true_tree(true_tree, T_list_unique)
        visualization_utils.visualize_and_save_T_plots(test_dir_name, true_tree, T_list_unique, w_T_list_unique, distances, g_T_list_unique)
        print(f"Distances to true tree: {distances}")
        print(f"Weights: {w_T_list_unique}")
        print(f"g(T): {g_T_list_unique}")
        tree_of_distance_0 = T_list_unique[np.where(distances == 0.)[0][0]]
        tree_of_distance_2 = T_list_unique[np.where(distances == 2.)[0][0]]
        tree_of_distance_4 = T_list_unique[np.where(distances == 4.)[0][0]]
        if len(tree_of_distance_0) != 0:
            print(f"Tree distance 0: {tree_utils.tree_to_newick(tree_of_distance_0, 0)}")
        print(f"Tree distance 2: {tree_utils.tree_to_newick(tree_of_distance_2, 0)}")
        print(f"Tree distance 4: {tree_utils.tree_to_newick(tree_of_distance_4, 0)}")

