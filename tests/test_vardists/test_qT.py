import itertools
import unittest

import numpy as np
import torch

import simul
from tests import utils_testing
from utils import tree_utils, visualization_utils
from utils.config import Config, set_seed
from utils.tree_utils import tree_to_newick
from variational_distributions.var_dists import qEpsilonMulti, qT, qC


class qTTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(1010)

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

    def test_inference_fixed_ceps(self):
        # build ad-hoc c
        c = torch.tensor([
            [2] * 20,
            [3] * 10 + [2] * 10,
            [2] * 5 + [3] * 5 + [2] * 10,
            [2] * 3 + [3] * 3 + [1] * 3 + [2] * 3 + [3] * 3 + [2] * 5
        ])
        config = Config(4, n_states=7, eps0=1e-2, chain_length=c.shape[1],
                        wis_sample_size=100, step_size=.2, debug=True)

        qt = qT(config).initialize()
        qeps = qEpsilonMulti(config).initialize()
        qc = qC(config, true_params={'c': c})

        for i in range(10):
            qt.update(qc, qeps)
            t, w = qt.get_trees_sample()
            qeps.update(t, w, qc)

        # print(qc)
        # print(qt)
        # print(qeps)
        gt_tree_newick = "((2)1,3)0"
        tol = 3
        qt_pmf = qt.get_pmf_estimate()
        sorted_newick = sorted(qt_pmf, key=qt_pmf.get, reverse=True)
        self.assertTrue(gt_tree_newick in sorted_newick[:tol],
                        msg=f"true " + gt_tree_newick + f" not in the first {tol} trees. those are:"
                                                        f"{sorted_newick[0]} | {sorted_newick[1]} | {sorted_newick[2]}")

    def test_qT_one_step_connections_more_probable_than_two_step_connections_on_simulated_data(self):
        N = 10
        M = 200
        K = 10
        A = 7
        T = utils_testing.get_tree_K_nodes_random(K)

        a0 = 5.
        b0 = 100.
        alpha0 = 100.
        beta0 = 100.
        y, c, z, pi, mu, tau, eps, eps0 = utils_testing.simulate_full_dataset_no_pyro(N, M, A, T, a0=a0, b0=b0, alpha0=alpha0, beta0=beta0)

        config = Config(n_cells=N, chain_length=M, n_nodes=K, n_states=A)

        qc = qC(config=config)
        qc.initialize()
        two_slice_marginals = utils_testing.get_two_sliced_marginals_from_one_slice_marginals(c, A, offset=0.1)
        qc.couple_filtering_probs = two_slice_marginals
        qeps = qEpsilonMulti(config=config)
        qeps.initialize('non-mutation')  # eps_alpha_dict=eps*10., eps_beta_dict=eps/10.)

        qt = qT(config=config)
        qt.initialize()
        qt.update(qc, qeps)
        W = qt.weight_matrix
        two_step_connections = tree_utils.get_all_two_step_connections(T)
        for (u, v) in T.edges:
            for w in two_step_connections[u]:
                self.assertGreater(W[u, v], W[u, w], msg='One step connection weaker than two step connection'
                                                         ' (based on true tree)')

    def test_qT_small_epsilon_simulated_data(self):
        N = 10
        M = 200
        K = 10
        A = 7
        T = utils_testing.get_tree_K_nodes_random(K)

        a0 = 5.
        b0 = 100.
        alpha0 = 100.
        beta0 = 100.
        y, c, z, pi, mu, tau, eps, eps0 = utils_testing.simulate_full_dataset_no_pyro(N, M, A, T, a0=a0, b0=b0, alpha0=alpha0, beta0=beta0)

        config = Config(n_cells=N, chain_length=M, n_nodes=K, n_states=A)

        qc = qC(config=config)
        qc.initialize()
        two_slice_marginals = utils_testing.get_two_sliced_marginals_from_one_slice_marginals(c, A, offset=10.0)
        qc.couple_filtering_probs = two_slice_marginals

        small_eps_a = 1.0
        small_eps_b = 10.
        qeps = qEpsilonMulti(config=config)
        gedges = [(u, v) for u, v in itertools.product(range(config.n_nodes),
                                                       range(config.n_nodes)) if v != 0 and u != v]
        eps_alpha_dict = {e: torch.tensor(small_eps_a) for e in gedges}
        eps_beta_dict = {e: small_eps_a / eps[e] if e in eps.keys() else torch.tensor(small_eps_b) for e in gedges}
        eps_beta_dict = {e: small_eps_a / torch.tensor(small_eps_b) for e in gedges}
        qeps.initialize('fixed', eps_alpha_dict=eps_alpha_dict, eps_beta_dict=eps_beta_dict)

        qt = qT(config=config)
        qt.initialize()
        qt.update(qc, qeps)
        W = qt.weight_matrix
        print(W)

        T_list, w_T_list = qt.get_trees_sample(sample_size=10)
        unique_edges, unique_edges_count = tree_utils.get_unique_edges(T_list)

        for (u, v) in gedges:
            if (u, v) in T.edges:
                print(f"True edge: {unique_edges_count[u, v]}")
            else:
                print(f"Non edge: {unique_edges_count[u, v]}")

    def test_qT_given_true_parameters(self):
        set_seed(0)
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
        out_simul = simul.simulate_full_dataset(config, eps_a=eps_a, eps_b=eps_b, mu0=1., lambda0=10., alpha0=500.,
                                                beta0=50., dir_delta=10., tree=true_tree)
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
        nx_trees_sample, log_w_t, log_g_t = q_T.get_trees_sample(sample_size=L, add_log_g=True)
        print(f"g(T): {log_g_t}")

        T_undirected_list = tree_utils.to_undirected(nx_trees_sample)
        prufer_list = tree_utils.to_prufer_sequences(T_undirected_list)
        unique_seq, unique_seq_idx = tree_utils.unique_trees(prufer_list)
        print(f"N unique trees: {len(unique_seq_idx)}")
        t_list_unique = [nx_trees_sample[i] for i in unique_seq_idx]
        log_w_t_unique = [log_w_t[i] for i in unique_seq_idx]
        log_g_t_unique = [log_g_t[i] for i in unique_seq_idx]
        distances = tree_utils.distances_to_true_tree(true_tree, t_list_unique)
        visualization_utils.visualize_and_save_T_plots(test_dir_name, true_tree, t_list_unique, log_w_t_unique, distances)
        print(f"Distances to true tree: {distances}")
        print(f"Weights: {log_w_t_unique}")
        print(f"g(T): {log_g_t_unique}")
        tree_of_distance_0 = t_list_unique[np.where(distances == 0.)[0][0]]
        tree_of_distance_2 = t_list_unique[np.where(distances == 2.)[0][0]]
        tree_of_distance_4 = t_list_unique[np.where(distances == 4.)[0][0]]
        print(f"True tree: {tree_utils.tree_to_newick(true_tree, 0)}")
        print(f"Tree distance 0: {tree_utils.tree_to_newick(tree_of_distance_0, 0)}")
        print(f"Tree distance 2: {tree_utils.tree_to_newick(tree_of_distance_2, 0)}")
        print(f"Tree distance 4: {tree_utils.tree_to_newick(tree_of_distance_4, 0)}")

    def test_tree_confidence_factor(self):

        config = Config(n_nodes=6, n_states=7, n_cells=500, chain_length=1000,
                        wis_sample_size=50, step_size=1., debug=True)
        joint_q = simul.generate_dataset_var_tree(config)
        true_tree_newick = tree_to_newick(joint_q.t.true_params['tree'])
        print(f"true tree: {true_tree_newick}")
        for tcf in [1., 10., 50., 100., 500., 1000.]:
            config.tree_confidence_factor = tcf

            qt = qT(config)
            qt.initialize()

            for i in range(10):
                qt.update(joint_q.c, joint_q.eps)

            print(f"factor: {tcf}")
            print(qt.get_pmf_estimate(normalized=True, desc_sorted=True))


