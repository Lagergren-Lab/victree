import unittest

import networkx as nx
import torch

from utils import math_utils


class MathUtilsTestCase(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_inverse_decay_function_1_over_x(self):
        a = torch.tensor(1.)
        b = torch.tensor(0.)
        c = torch.tensor(1.)
        x = torch.tensor(8.23)
        y = math_utils.inverse_decay_function(x, a, b, c)
        self.assertEqual(1./x, y)

    def test_inverse_decay_function_a_over_x(self):
        a = torch.tensor(230.)
        b = torch.tensor(0.)
        c = torch.tensor(1.)
        x = torch.tensor(8.23)
        y = math_utils.inverse_decay_function(x, a, b, c)
        self.assertEqual(a / x, y)

    def test_inverse_decay_function_almost_zero_for_large_x_and_c_equal_to_3(self):
        """
        Tests that the decay function quickly goes to zero when c is "large"
        """
        a = torch.tensor(5000.)
        b = torch.tensor(0.)
        c = torch.tensor(3.)
        x = torch.tensor(100.)
        y = math_utils.inverse_decay_function(x, a, b, c)
        self.assertLess(y, 0.1)

    @unittest.skip("Plotting test")
    def test_inverse_decay_function_for_vectors(self):
        """
        Manual check of the decay function for different values. Matplotlib backend set to run in Pycharm.
        """
        a = torch.tensor(50.)
        b = torch.tensor(20.)
        c = torch.tensor(1.)
        x = torch.arange(0, 100)
        y = math_utils.inverse_decay_function(x, a, b, c)

        import matplotlib
        matplotlib.use('module://backend_interagg')
        import matplotlib.pyplot as plt
        plt.plot(y)
        plt.plot(torch.ones_like(y))
        plt.show()