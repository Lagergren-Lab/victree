import unittest
import time
import itertools
import numpy as np
import torch

from utils import eps_utils
from model.tree_hmm import CopyNumberTreeHMM
from utils.config import Config
from utils.eps_utils import h_eps, h_eps0
from variational_distributions.var_dists import qEpsilonMulti


class zippingTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.A = 7

    def test_numpy_mask_indexing(self):
        arr = np.arange(self.A ** 4).reshape([self.A] * 4)


        # compute it manually
        start = time.time_ns()
        ll = self.manual_mask(arr)
        t = time.time_ns() - start
        print(f"manual time {t}")

        start = time.time_ns()
        idx_ll = self.indexing_mask(arr)
        t = time.time_ns() - start
        print(f"indexing-method time {t}")

        self.assertEqual(len(ll), len(idx_ll))
        self.assertTrue(np.alltrue(idx_ll == ll))

    def manual_mask(self, arr):
        ll = []
        for i, j, k, l in itertools.product(*(range(self.A),) * 4):
            if i - j == k - l:
                ll.append(arr[i, j, k, l])
        return ll

    def indexing_mask(self, arr):
        mask = eps_utils.get_zipping_mask(self.A)
        return arr[mask]

    def test_normalization(self):
        h_eps = eps_utils.h_eps(self.A, eps=.1)
        self.assertTrue(torch.allclose(h_eps.sum(dim=0), torch.ones((self.A, ) * 3)))

    def test_mask_absorbing_state(self):
        # indexes: jj, j, ii, i (jj child position m+1, ii parent position m+1, etc)
        A = 4
        config = Config()
        qeps = qEpsilonMulti(config)
        comut, no_comut, abs_state = qeps.create_masks(A)
        self.assertTrue(comut[2, 1, 3, 2] == 1)
        self.assertTrue(comut[1, 2, 2, 3] == 1)
        self.assertFalse(comut[1, 1, 1, 2] == 1)
        self.assertTrue(no_comut[1, 1, 1, 2] == 1)
        self.assertTrue(abs_state[1, 1, 0, 2] == 1, msg="Transitioning from absorbing state should be True in absorbing state mask.")
        self.assertEqual(abs_state[0, 2, 1, 1], 0, msg="Transitioning to absorbing state should be False in absorbing state mask.")
        self.assertEqual(abs_state[0, 1, 0, 1], 0)

    def test_masks_mutually_exclusive(self):
        A = 4
        config = Config()
        qeps = qEpsilonMulti(config)
        comut, no_comut, abs_state = qeps.create_masks(A)
        self.assertTrue(((comut == no_comut) == (abs_state == 1)).all())
        self.assertTrue(((comut == abs_state) == (no_comut == 1)).all())
        self.assertTrue(((no_comut == abs_state) == (comut == 1)).all())

    def test_zipping(self):
        A = 4
        K = 2
        config = Config(n_nodes=K, n_states=A)
        qeps = qEpsilonMulti(config)
        qeps.initialize("non_mutation")
        comut, no_comut, abs_state = qeps.create_masks(A)
        exp_log_zip = qeps.exp_log_zipping((0, 1))
        print(exp_log_zip)

    def test_h_eps_absorption(self):
        n_states = 6
        h_eps_test = h_eps(n_states, .01)
        self.assertFalse(torch.isnan(h_eps_test).any())
        # exclude non-feasible states, all rest should sum up to 1.
        unfeas_configs = torch.zeros((n_states, ) * 3, dtype=torch.bool)
        unfeas_configs[1:, :, 0] = True
        sum_h_eps = h_eps_test.sum(dim=0)[~unfeas_configs]
        self.assertTrue(torch.allclose(sum_h_eps, torch.ones_like(sum_h_eps)))

    def test_h_eps0_absorption(self):
        n_states = 6
        h_eps0_test = h_eps0(n_states, .01)
        self.assertFalse(torch.isnan(h_eps0_test).any())
        sum_h_eps0 = h_eps0_test.sum(dim=0)
        self.assertTrue(torch.allclose(sum_h_eps0, torch.ones_like(sum_h_eps0)))


