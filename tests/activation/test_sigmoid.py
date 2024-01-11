import unittest

import numpy as np

from PyNeuralNet.activation import Sigmoid


class TestSigmoid(unittest.TestCase):
    """
    Tests the class Sigmoid.
    """

    def setUp(self):
        self.X = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ])
        self.dL_dZ = np.array([
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ])
        self.act_fn = Sigmoid()

    def test_call(self):
        """
        Tests the magic method __call__()
        """
        Z = self.act_fn(self.X)
        np.testing.assert_almost_equal(np.array([
            [0.5, 0.73105858, 0.88079708],
            [0.95257413, 0.98201379, 0.99330715],
            [0.99752738, 0.99908895, 0.99966465],
        ]), Z)

    def test_forward(self):
        """
        Tests the method forward().
        """
        Z = self.act_fn.forward(self.X)
        np.testing.assert_almost_equal(np.array([
            [0.5, 0.73105858, 0.88079708],
            [0.95257413, 0.98201379, 0.99330715],
            [0.99752738, 0.99908895, 0.99966465],
        ]), Z)

    def test_backward(self):
        """
        Tests the method backward().
        """
        self.act_fn.forward(self.X)
        dL_dX = self.act_fn.backward(self.dL_dZ)
        np.testing.assert_almost_equal(np.array([
            [0.25, 0.19661193, 0.10499359],
            [0.04517666, 0.01766271, 0.00664806],
            [0.00246651, 0.00091022, 0.00033524],
        ]), dL_dX)

    def test_parameters(self):
        """
        Tests the method parameters().
        """
        self.assertEqual({"module_0": {}}, self.act_fn.parameters())

    def test_gradients(self):
        """
        Tests the method gradients().
        """
        self.assertEqual({"module_0": {}}, self.act_fn.gradients())


if __name__ == "__main__":
    unittest.main()
