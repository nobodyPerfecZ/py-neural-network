import unittest

import numpy as np

from PyNeuralNet.layer import Linear


class TestLinear(unittest.TestCase):
    """
    Tests the class Linear.
    """

    def setUp(self):
        self.model = Linear(
            in_features=3,
            out_features=2,
            bias=True,
            weight_initializer="ones",
            bias_initializer="zeros",
        )
        self.X = np.array([
            [1, 2, 3],
            [3, 4, 5],
            [6, 7, 8],
        ])
        self.dL_dZ = np.array([
            [1, 1],
            [1, 1],
            [1, 1],
        ])

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        Z = self.model(self.X)
        np.testing.assert_almost_equal(np.array([
            [1, 2],
            [3, 4],
            [6, 7],
        ]), Z)

    def test_forward(self):
        """
        Tests the method forward().
        """
        Z = self.model.forward(self.X)
        np.testing.assert_almost_equal(np.array([
            [1, 2],
            [3, 4],
            [6, 7],
        ]), Z)

    def test_backward(self):
        """
        Tests the method backward().
        """
        self.model.forward(self.X)
        dL_dX = self.model.backward(self.dL_dZ)
        np.testing.assert_almost_equal(np.array([
            [1, 1, 0],
            [1, 1, 0],
            [1, 1, 0],
        ]), dL_dX)

    def test_parameters(self):
        """
        Tests the method parameters().
        """
        parameters = self.model.parameters()
        np.testing.assert_almost_equal(np.array(
            [0, 0]
        ), parameters["module_0"]["bias"])
        np.testing.assert_almost_equal(np.eye(3, 2), parameters["module_0"]["weights"])

    def test_gradients(self):
        """
        Tests the method gradients().
        """
        self.model.forward(self.X)
        self.model.backward(self.dL_dZ)
        gradients = self.model.gradients()
        np.testing.assert_almost_equal(np.array([1, 1]), gradients["module_0"]["bias"])
        np.testing.assert_almost_equal(np.array([
            [3.33333333, 3.33333333],
            [4.33333333, 4.33333333],
            [5.33333333, 5.33333333],
        ]), gradients["module_0"]["weights"])


if __name__ == "__main__":
    unittest.main()
