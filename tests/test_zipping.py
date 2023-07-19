import unittest
import time
import itertools
import numpy as np
import torch

from utils import eps_utils
from model.tree_hmm import CopyNumberTreeHMM
from utils.config import Config
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
        self.assertTrue(abs_state[1, 1, 0, 2] == 1, msg="Transitioning from absorbing state should be True in absorbing state mask.")
        self.assertFalse(abs_state[0, 2, 1, 1] == 1, msg="Transitioning to absorbing state should be False in absorbing state mask.")
        self.assertFalse(abs_state[0, 1, 0, 1] == 1)

    def test_masks_different(self):
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
