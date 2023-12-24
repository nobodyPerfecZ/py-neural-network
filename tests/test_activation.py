import unittest
import numpy as np

from PyNeuralNet.activation import ReLU, LeakyReLU, Sigmoid, Tanh


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

    def test_gradient(self):
        """
        Tests the method gradient().
        """
        dZ_dX = self.act_fn.gradient(self.X)
        np.testing.assert_almost_equal(np.array([
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]), dZ_dX)


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

    def test_gradient(self):
        """
        Tests the method gradient().
        """
        dZ_dX = self.act_fn.gradient(self.X)
        np.testing.assert_almost_equal(np.array([
            [0.1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]), dZ_dX)


class TestSigmoid(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
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

    def test_gradient(self):
        """
        Tests the method gradient().
        """
        dZ_dX = self.act_fn.gradient(self.X)
        np.testing.assert_almost_equal(np.array([
            [0.25, 0.19661193, 0.10499359],
            [0.04517666, 0.01766271, 0.00664806],
            [0.00246651, 0.00091022, 0.00033524],
        ]), dZ_dX)


class TestTanh(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
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

    def test_gradient(self):
        """
        Tests the method gradient().
        """
        dZ_dX = self.act_fn.gradient(self.X)
        np.testing.assert_almost_equal(np.array([
            [1.00000000e+00, 4.19974342e-01, 7.06508249e-02],
            [9.86603717e-03, 1.34095068e-03, 1.81583231e-04],
            [2.45765474e-05, 3.32610934e-06, 4.50140597e-07],
        ]), dZ_dX)


if __name__ == '__main__':
    unittest.main()
