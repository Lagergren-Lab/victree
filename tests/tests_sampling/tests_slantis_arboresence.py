import torch
import unittest
from sampling import slantis_arborescence


class slantisArborescenceTestCase(unittest.TestCase):

    def test_slantis_random_weight_matrix(self):
        n_nodes = 10
        W = torch.rand((n_nodes, n_nodes))
        log_W = torch.log(W)
        log_W_root = torch.rand((n_nodes,))
        T, log_T = slantis_arborescence.sample_arborescence(log_W=log_W, root=0)
