import unittest

import numpy as np

from PyNeuralNet.activation import Tanh


class TestTanh(unittest.TestCase):
    """
    Tests the class Tanh.
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
        self.act_fn = Tanh()

    def test_call(self):
        """
        Tests the magic method __call__()
        """
        Z = self.act_fn(self.X)
        np.testing.assert_almost_equal(np.array([
            [0., 0.76159416, 0.96402758],
            [0.99505475, 0.9993293, 0.9999092],
            [0.99998771, 0.99999834, 0.99999977],
        ]), Z)

    def test_forward(self):
        """
        Tests the method forward().
        """
        Z = self.act_fn(self.X)
        np.testing.assert_almost_equal(np.array([
            [0., 0.76159416, 0.96402758],
            [0.99505475, 0.9993293, 0.9999092],
            [0.99998771, 0.99999834, 0.99999977],
        ]), Z)

    def test_backward(self):
        """
        Tests the method backward().
        """
        self.act_fn.forward(self.X)
        dL_dX = self.act_fn.backward(self.dL_dZ)
        np.testing.assert_almost_equal(np.array([
            [1.00000000e+00, 4.19974342e-01, 7.06508249e-02],
            [9.86603717e-03, 1.34095068e-03, 1.81583231e-04],
            [2.45765474e-05, 3.32610934e-06, 4.50140597e-07],
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
