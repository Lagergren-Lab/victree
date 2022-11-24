import random
import sys
import os
from os import path
from matplotlib.pyplot import logging

import torch
import unittest
from sampling import slantis_arborescence


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

        return super().setUp()

    def test_slantis_random_weight_matrix(self):
        n_nodes = 10
        torch.manual_seed(0)
        random.seed(0)
        W = torch.rand((n_nodes, n_nodes))
        log_W = torch.log(W)
        log_W_root = torch.rand((n_nodes,))
        T, log_T = slantis_arborescence.sample_arborescence(log_W=log_W, root=0, debug=True)

        # save sampled tree on img
        slantis_arborescence.draw_graph(T, to_file=path.join(self.output_dir, 
                                                             "slantis_random_sample.png"))
        self.logger.debug(f"log_T: {torch.exp(log_T)}")

    def tearDown(self) -> None:
        self.logger.removeHandler(self.fh)
        return super().tearDown()
