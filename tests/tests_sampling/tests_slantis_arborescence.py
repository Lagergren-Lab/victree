import random
import sys
import os
from os import path

import networkx as nx
import numpy as np
from matplotlib.pyplot import logging

import torch
import unittest

from networkx import maximum_spanning_arborescence
from networkx.algorithms.tree import Edmonds

from sampling import slantis_arborescence
from sampling.slantis_arborescence import create_fully_connected_graph, new_graph_with_arc, \
    sample_arborescence_from_weighted_graph
from utils.config import set_seed
from utils.tree_utils import tree_to_newick


class slantisArborescenceTestCase(unittest.TestCase):

    def setUp(self) -> None:
        # create output dir (for graph and logfile)
        self.output_dir = "./test_out"
        if not path.exists(self.output_dir):
            os.mkdir(self.output_dir)

        # setup logger
        self.logger = logging.getLogger("slantis_test_log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger.level = logging.DEBUG
        self.fh = logging.FileHandler(path.join(self.output_dir, "slantis_test.log"))
        self.fh.setFormatter(formatter)
        self.logger.addHandler(self.fh)
        set_seed(42)

        return super().setUp()

    def test_slantis_random_weight_matrix(self):
        n_nodes = 10
        W = torch.rand((n_nodes, n_nodes))
        log_W = torch.log(W)
        log_W_root = torch.rand((n_nodes,))
        T, log_T = slantis_arborescence.sample_arborescence(log_W=log_W, root=0, debug=True)

        # save sampled tree on img
        slantis_arborescence.draw_graph(T, to_file=path.join(self.output_dir, 
                                                             "slantis_random_sample.png"))
        self.logger.debug(f"log_T: {torch.exp(log_T)}")

    def test_edmonds(self):
        graph = nx.DiGraph(directed=True)
        weighted_edges = [
            (0, 1, .7), (0, 2, .3), (0, 3, .5), (0, 4, -np.inf),
            (1, 2, .2), (1, 3, .1), (1, 4, .1),
            (2, 1, .5), (2, 3, 1.),
            (4, 1, .1)  # this edge cannot be reached because of (0, 4, -inf)
        ]
        graph.add_weighted_edges_from(weighted_edges)
        edmonds_arb: nx.DiGraph = maximum_spanning_arborescence(graph, preserve_attrs=True)

        self.assertAlmostEqual(edmonds_arb.size(weight='weight'), 2., delta=.1)
        self.assertTrue(nx.is_arborescence(edmonds_arb))
        self.assertTrue((0, 1) in edmonds_arb.edges)
        self.assertTrue((0, 2) in edmonds_arb.edges)
        self.assertTrue((2, 3) in edmonds_arb.edges)
        self.assertFalse((0, 4) in edmonds_arb.edges)
        self.assertFalse((4, 1) in edmonds_arb.edges)
        self.assertEqual([n for n, d in edmonds_arb.in_degree() if d == 0], [0])
        self.assertEqual(tree_to_newick(edmonds_arb), '((4)1,(3)2)0')

    def test_include_arc_edmonds(self):
        graph = nx.DiGraph(directed=True)
        weighted_edges = [
            (0, 1, .1), (0, 3, .1),
            (1, 2, .1),
            (2, 1, 1.), (2, 3, .1),
            (3, 2, 1.)
        ]
        graph.add_weighted_edges_from(weighted_edges)
        edmonds_arb: nx.DiGraph = maximum_spanning_arborescence(graph, preserve_attrs=True)
        # print(tree_to_newick(edmonds_arb))
        self.assertFalse((0, 1) in edmonds_arb.edges)

        new_graph_with_arc(0, 1, graph)

        edmonds_arb_with_included = maximum_spanning_arborescence(graph, preserve_attrs=True)
        # print(tree_to_newick(edmonds_arb_with_included))
        self.assertTrue((0, 1) in edmonds_arb_with_included.edges)


    def test_tree_sampling(self):
        graph = nx.DiGraph(directed=True)
        weighted_edges = [
            (0, 1, .1), (0, 2, .1), (0, 3, .3),
            (1, 2, .1), (1, 3, .1),
            (2, 1, .1), (2, 3, .1),
            (3, 2, .7), (3, 1, .8)
        ]
        graph.add_weighted_edges_from(weighted_edges)

        l = 1000
        print('samples')
        e03_cnt = 0
        e31_cnt = 0
        e32_cnt = 0
        e13_cnt = 0
        for i in range(l):
            s, log_iws = sample_arborescence_from_weighted_graph(graph)
            if (0, 3) in s.edges:
                e03_cnt += 1
            if (3, 1) in s.edges:
                e31_cnt += 1
            if (1, 3) in s.edges:
                e13_cnt += 1
            if (3, 2) in s.edges:
                e32_cnt += 1
            # print(f'[{i}] {tree_to_newick(s)} -> {log_iws}')

        print(f'(0,3): {e03_cnt / l}')
        print(f'(3,1): {e31_cnt / l}')
        print(f'(1,3): {e13_cnt / l}')
        print(f'(3,2): {e32_cnt / l}')


    def test_tree_to_newick(self):
        graph = nx.DiGraph(directed=True)
        weighted_edges = [
            (0, 1, .7), (0, 2, .3),
            (1, 3, .1), (1, 4, .9)
        ]
        graph.add_weighted_edges_from(weighted_edges)
        self.assertEqual(tree_to_newick(graph, weight='weight'), '((3:0.1,4:0.9)1:0.7,2:0.3)0')
        self.assertEqual(tree_to_newick(graph), '((3,4)1,2)0')
        self.assertEqual(tree_to_newick(graph, root=0), '((3,4)1,2)0')

    def tearDown(self) -> None:
        self.logger.removeHandler(self.fh)
        return super().tearDown()
