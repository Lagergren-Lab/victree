import math
import unittest
from random import random

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

    def test_inverse_decay_function_equal_to_val_at_max_iter(self):
        """
        Manual check of the decay function for different values. Matplotlib backend set to run in Pycharm.
        """
        max_iter = 100
        a = torch.tensor(50.)
        b = torch.tensor(max_iter * 0.2)
        d = torch.tensor(1.)  # desired value
        c = math_utils.inverse_decay_function_calculate_c(a, b, d, max_iter)
        x = torch.arange(0, max_iter+1)
        y = math_utils.inverse_decay_function(x, a, b, c)

        self.assertAlmostEqual(d, y[-1], delta=0.01)

        @unittest.skip("Plotting test")
        def test_inverse_decay_function_as_tempering_scheme(self):
            """
            Manual check of the decay function for different values. Matplotlib backend set to run in Pycharm.
            """
            max_iter = 50
            a = torch.tensor(50.)
            b = torch.tensor(max_iter * 0.1)
            d = torch.tensor(1.)
            c = math_utils.inverse_decay_function_calculate_c(a, b, d, max_iter)
            x = torch.arange(0, max_iter)
            y = math_utils.inverse_decay_function(x, a, b, c)

            import matplotlib
            matplotlib.use('module://backend_interagg')
            import matplotlib.pyplot as plt
            plt.plot(y)
            plt.plot(torch.ones_like(y))
            plt.show()