import unittest

import numpy as np

from PyNeuralNet.activation import ReLU, LeakyReLU


class TestReLU(unittest.TestCase):
    """
    Tests the class ReLU.
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
        self.act_fn = ReLU()

    def test_call(self):
        """
        Tests the magic method __call__()
        """
        Z = self.act_fn(self.X)
        np.testing.assert_almost_equal(np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]), Z)

    def test_forward(self):
        """
        Tests the method forward().
        """
        Z = self.act_fn.forward(self.X)
        np.testing.assert_almost_equal(np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]), Z)

    def test_backward(self):
        """
        Tests the function backward().
        """
        self.act_fn.forward(self.X)
        dL_dX = self.act_fn.backward(self.dL_dZ)
        np.testing.assert_almost_equal(np.array([
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
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


class TestLeakyReLU(unittest.TestCase):
    """
    Tests the class LeakyReLU.
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
        self.act_fn = LeakyReLU(slope=0.1)

    def test_call(self):
        """
        Tests the magic method __call__()
        """
        Z = self.act_fn(self.X)
        np.testing.assert_almost_equal(np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]), Z)

    def test_forward(self):
        """
        Tests the method forward().
        """
        Z = self.act_fn.forward(self.X)
        np.testing.assert_almost_equal(np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
        ]), Z)

    def test_backward(self):
        """
        Tests the method backward().
        """
        self.act_fn.forward(self.X)
        dL_dX = self.act_fn.backward(self.dL_dZ)
        np.testing.assert_almost_equal(np.array([
            [0.1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
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
