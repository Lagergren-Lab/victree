import unittest

import networkx as nx

from utils import tree_utils


class treeUtilTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

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
            assert e in edges_list
        assert edges_count[0, 1] == 2
        assert edges_count[0, 2] == 1
        assert edges_count[1, 2] == 1
