import unittest
import time
import itertools
import numpy as np
import torch

from utils import eps_utils

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

