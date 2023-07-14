import unittest

import numpy as np

from model.multi_chromosome_model import MultiChromosomeGenerativeModel
from tests import utils_testing
from utils.config import set_seed, Config


class MultiChromosomeModelTestCase(unittest.TestCase):

    def setUp(self) -> None:
        set_seed(0)

    def test_generate_data(self):
        N, M, K, A = (10, 20, 2, 5)
        chromosome_indexes = [int(M/10), int(M/10*5), int(M/10*8)]
        n_chromosomes = len(chromosome_indexes) + 1
        config = Config(n_cells=N, chain_length=M, n_nodes=K, n_states=A,
                        chromosome_indexes=chromosome_indexes, n_chromosomes=n_chromosomes)
        model = MultiChromosomeGenerativeModel(config)
        T = utils_testing.get_two_node_tree()
        a0 = 10.
        b0 = 100.
        eps0_a = 5.
        eps0_b = 20.
        delta = 1.
        nu0 = 1.
        lambda0 = 10.
        alpha0 = 100.
        beta0 = 5.
        out_simul = model.simulate_data(T, a0, b0, eps0_a, eps0_b, delta, nu0, lambda0, alpha0, beta0)
        y = out_simul['obs']
        c = out_simul['c']
        z = out_simul['z']
        pi = out_simul['pi']
        mu = out_simul['mu']
        tau = out_simul['tau']
        eps = out_simul['eps']
        eps0 = out_simul['eps0']
