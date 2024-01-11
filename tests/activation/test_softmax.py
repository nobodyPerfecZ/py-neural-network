import unittest

import numpy as np

from PyNeuralNet.activation import Softmax, LogSoftmax


class TestSoftmax(unittest.TestCase):
    """
    Tests the class Softmax.
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
        self.act_fn = Softmax()

    def test_call(self):
        """
        Tests the magic method __call__()
        """
        Z = self.act_fn(self.X)
        np.testing.assert_almost_equal(np.array([
            [2.12078996e-04, 5.76490482e-04, 1.56706360e-03],
            [4.25972051e-03, 1.15791209e-02, 3.14753138e-02],
            [8.55587737e-02, 2.32572860e-01, 6.32198578e-01],
        ]), Z)
        np.testing.assert_almost_equal(np.array([1.0]), np.sum(Z))

        Z = self.act_fn(self.X, axis=0)
        np.testing.assert_almost_equal(np.array([
            [0.00235563, 0.00235563, 0.00235563],
            [0.04731416, 0.04731416, 0.04731416],
            [0.95033021, 0.95033021, 0.95033021],
        ]), Z)
        np.testing.assert_almost_equal(np.array([1.0, 1.0, 1.0]), np.sum(Z, axis=0))

        Z = self.act_fn(self.X, axis=1)
        np.testing.assert_almost_equal(np.array([
            [0.09003057, 0.24472847, 0.66524096],
            [0.09003057, 0.24472847, 0.66524096],
            [0.09003057, 0.24472847, 0.66524096],
        ]), Z)
        np.testing.assert_almost_equal(np.array([1.0, 1.0, 1.0]), np.sum(Z, axis=1))

    def test_forward(self):
        """
        Tests the method forward().
        """
        Z = self.act_fn(self.X)
        np.testing.assert_almost_equal(np.array([
            [2.12078996e-04, 5.76490482e-04, 1.56706360e-03],
            [4.25972051e-03, 1.15791209e-02, 3.14753138e-02],
            [8.55587737e-02, 2.32572860e-01, 6.32198578e-01],
        ]), Z)
        np.testing.assert_almost_equal(np.array([1.0]), np.sum(Z))

        Z = self.act_fn(self.X, axis=0)
        np.testing.assert_almost_equal(np.array([
            [0.00235563, 0.00235563, 0.00235563],
            [0.04731416, 0.04731416, 0.04731416],
            [0.95033021, 0.95033021, 0.95033021],
        ]), Z)
        np.testing.assert_almost_equal(np.array([1.0, 1.0, 1.0]), np.sum(Z, axis=0))

        Z = self.act_fn(self.X, axis=1)
        np.testing.assert_almost_equal(np.array([
            [0.09003057, 0.24472847, 0.66524096],
            [0.09003057, 0.24472847, 0.66524096],
            [0.09003057, 0.24472847, 0.66524096],
        ]), Z)
        np.testing.assert_almost_equal(np.array([1.0, 1.0, 1.0]), np.sum(Z, axis=1))

    def test_backward(self):
        """
        Tests the method backward().
        """
        self.act_fn.forward(self.X, axis=1)
        dZ_dX = self.act_fn.backward(self.dL_dZ, axis=1)
        np.testing.assert_almost_equal(np.array([
            [0.08192507, 0.18483645, 0.22269543],
            [0.08192507, 0.18483645, 0.22269543],
            [0.08192507, 0.18483645, 0.22269543],
        ]), dZ_dX)

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


class TestLogSoftmax(unittest.TestCase):
    """
    Tests the class LogSoftmax.
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
        self.act_fn = LogSoftmax()

    def test_call(self):
        """
        Tests the magic method __call__()
        """
        Z = self.act_fn(self.X)
        np.testing.assert_almost_equal(np.array([
            [-8.45855173, -7.45855173, -6.45855173],
            [-5.45855173, -4.45855173, -3.45855173],
            [-2.45855173, -1.45855173, -0.45855173],
        ]), Z)

        Z = self.act_fn(self.X, axis=0)
        np.testing.assert_almost_equal(np.array([
            [-6.05094576, -6.05094576, -6.05094576],
            [-3.05094576, -3.05094576, -3.05094576],
            [-0.05094576, -0.05094576, -0.05094576],
        ]), Z)

        Z = self.act_fn(self.X, axis=1)
        np.testing.assert_almost_equal(np.array([
            [-2.40760596, -1.40760596, -0.40760596],
            [-2.40760596, -1.40760596, -0.40760596],
            [-2.40760596, -1.40760596, -0.40760596],
        ]), Z)

    def test_forward(self):
        """
        Tests the method forward().
        """
        Z = self.act_fn(self.X)
        np.testing.assert_almost_equal(np.array([
            [-8.45855173, -7.45855173, -6.45855173],
            [-5.45855173, -4.45855173, -3.45855173],
            [-2.45855173, -1.45855173, -0.45855173],
        ]), Z)

        Z = self.act_fn(self.X, axis=0)
        np.testing.assert_almost_equal(np.array([
            [-6.05094576, -6.05094576, -6.05094576],
            [-3.05094576, -3.05094576, -3.05094576],
            [-0.05094576, -0.05094576, -0.05094576],
        ]), Z)

        Z = self.act_fn(self.X, axis=1)
        np.testing.assert_almost_equal(np.array([
            [-2.40760596, -1.40760596, -0.40760596],
            [-2.40760596, -1.40760596, -0.40760596],
            [-2.40760596, -1.40760596, -0.40760596],
        ]), Z)

    def test_backward(self):
        """
        Tests the method backward().
        """
        self.act_fn.forward(self.X, axis=1)
        dL_dX = self.act_fn.backward(self.dL_dZ)
        np.testing.assert_almost_equal(np.array([
            [-0.90996943, -0.75527153, -0.33475904],
            [-0.90996943, -0.75527153, -0.33475904],
            [-0.90996943, -0.75527153, -0.33475904],
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


if __name__ == '__main__':
    unittest.main()
