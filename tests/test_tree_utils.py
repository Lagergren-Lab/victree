import unittest

import networkx as nx

import simul
from utils import tree_utils


class treeUtilTestCase(unittest.TestCase):

    def setUp(self) -> None:
        simul.set_seed(101)

    def test_get_unique_edges_correct_edges_and_count_returned(self):
        T_1 = nx.DiGraph()
        T_2 = nx.DiGraph()
        T_1.add_edge(0, 1)
        T_1.add_edge(0, 2)
        T_2.add_edge(0, 1)
        T_2.add_edge(1, 2)
        T_list = [T_1, T_2]
        edges_list, edges_count = tree_utils.get_unique_edges(T_list, N_nodes=3)
        expected_edges = [(0, 1), (0, 2), (1, 2)]
        for e in expected_edges:
            self.assertTrue(e in edges_list)
        self.assertEqual(edges_count[0, 1], 2)
        self.assertEqual(edges_count[0, 2], 1)
        self.assertEqual(edges_count[1, 2], 1)


    @unittest.skip("wrong code")
    def test_tree_metrics(self):
        K = 5
        T_1 = tree_utils.generate_fixed_tree(K, seed=0)
        T_2 = tree_utils.generate_fixed_tree(K, seed=1)
        T_3 = tree_utils.generate_fixed_tree(K, seed=2)
        print(f" Tree 1: {tree_utils.networkx_tree_to_dendropy(T_1, 0)}")
        print(f" Tree 2: {tree_utils.networkx_tree_to_dendropy(T_2, 0)}")
        print(f" Tree 3: {tree_utils.networkx_tree_to_dendropy(T_3, 0)}")
        graph_dist12 = tree_utils.calculate_graph_distance(T_1, T_2)
        graph_dist13 = tree_utils.calculate_graph_distance(T_1, T_3)
        print(f"Graph dist T2 to T1: {graph_dist12}")
        print(f"Graph dist T3 to T1: {graph_dist13}")
        rf_dist = tree_utils.calculate_Labeled_Robinson_Foulds_distance(T_1, T_2)
        print(f'RF dist: {rf_dist}')

        spr_dist = tree_utils.calculate_SPR_distance(T_1, T_2)
        print(f'SPR dist: {spr_dist}')

    def test_get_unique_trees(self):
        K = 5
        T_1 = tree_utils.generate_fixed_tree(K, seed=0)
        T_2 = tree_utils.generate_fixed_tree(K, seed=1)
        T_3 = tree_utils.generate_fixed_tree(K, seed=2)
        T_4 = tree_utils.generate_fixed_tree(K, seed=2)
        T_list = [T_1, T_2, T_3, T_4]
        T_undirected_list = tree_utils.to_undirected(T_list)
        prufer_list = tree_utils.to_prufer_sequences(T_undirected_list)
        unique_seq, unique_seq_idx = tree_utils.unique_trees(prufer_list)
        n_unique = len(unique_seq)
        self.assertEqual(n_unique, 3)
        self.assertEqual(unique_seq_idx, [0, 1, 2])

    def test_get_distances(self):
        K = 10
        T = tree_utils.generate_fixed_tree(K, seed=0)
        T_1 = tree_utils.generate_fixed_tree(K, seed=0)
        T_2 = tree_utils.generate_fixed_tree(K, seed=1)
        T_3 = tree_utils.generate_fixed_tree(K, seed=2)
        T_4 = tree_utils.generate_fixed_tree(K, seed=2)
        T_list = [T_1, T_2, T_3, T_4]
        distances = tree_utils.distances_to_true_tree(T, T_list)
        self.assertEqual(distances[0], 0.)
        self.assertGreaterEqual(distances[1], distances[0])
        self.assertGreaterEqual(distances[2], distances[0])
        self.assertEqual(distances[2], distances[3])

    def test_unique_trees_and_multiplicity(self):
        T_0 = nx.DiGraph()
        T_1 = nx.DiGraph()
        T_2 = nx.DiGraph()
        T_3 = nx.DiGraph()
        # T_0 and T_1: Two identical trees
        T_0.add_edge(0, 1)
        T_0.add_edge(0, 2)
        T_1.add_edge(0, 1)
        T_1.add_edge(0, 2)
        # T_2: different topology should give unique tree
        T_2.add_edge(0, 1)
        T_2.add_edge(1, 2)
        # T_3: Order in which edges are added shouldn't matter
        T_3.add_edge(0, 2)
        T_3.add_edge(0, 1)

        T_list = [T_0, T_1, T_2, T_3]
        T_list_undirected = tree_utils.to_undirected(T_list)
        prufer_seqs = tree_utils.to_prufer_sequences(T_list_undirected)
        unique_Ts, unique_Ts_idx, multiplicity = tree_utils.unique_trees_and_multiplicity(prufer_seqs)
        self.assertEqual(len(unique_Ts), 2)
        self.assertEqual(unique_Ts_idx, [0, 2])
        self.assertEqual(multiplicity, [3, 1])

    def test_tree_remap(self):
        T = nx.DiGraph()
        T.add_edge(0, 1)
        T.add_edge(0, 2)
        T.add_edge(2, 3)
        T.add_edge(3, 4)

        perm = [0, 1, 2, 3]

        # Test identity mapping
        T_map = tree_utils.remap_edge_labels([T], perm)[0]
        self.assertEqual(T.edges, T_map.edges, msg='Identity mapping failed.')

        perm1 = [0, 2, 3, 1]
        T_map = tree_utils.remap_edge_labels([T], perm1)[0]
        self.assertNotEqual(T.edges, T_map.edges, msg='Mapped tree still equal to original after mapping.')

        perm1_inv = [0, 3, 1, 2]
        T_map_inv = tree_utils.remap_edge_labels([T_map], perm1_inv)[0]
        self.assertEqual(T.edges, T_map_inv.edges)

    def test_generate_all_directed_unlabeled_tree_topologies(self):
        # Example usage:
        K = 3
        all_directed_trees = tree_utils.get_all_tree_topologies(K)
        nw_trees = [tree_utils.tree_to_newick(T, 0) for T in all_directed_trees]
        self.assertTrue('(1,2)0' in nw_trees)
        self.assertTrue('((2)1)0' in nw_trees)
        self.assertTrue('((1)2)0' in nw_trees)
