import unittest

import numpy as np

from PyNeuralNet.loss import MSE, MAE, BCEWithLogits, CEWithLogits


class TestMSE(unittest.TestCase):
    """
    Tests the class MSE.
    """
    def setUp(self):
        self.y_pred = np.array([
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
        ])
        self.y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.loss_fn = MSE()

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        loss = self.loss_fn(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [1],
            [0],
            [1],
            [4],
            [9],
            [16],
            [25],
            [36],
            [49],
        ]), loss)

    def test_forward(self):
        """
        Tests the method forward().
        """
        loss = self.loss_fn.forward(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [1],
            [0],
            [1],
            [4],
            [9],
            [16],
            [25],
            [36],
            [49],
        ]), loss)

    def test_gradient(self):
        """
        Tests the method gradient().
        """
        dL_dy_pred = self.loss_fn.gradient(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [-2],
            [0],
            [2],
            [4],
            [6],
            [8],
            [10],
            [12],
            [14],
        ]), dL_dy_pred)


class TestMAE(unittest.TestCase):
    """
    Tests the class MAE
    """

    def setUp(self):
        self.y_pred = np.array([
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
        ])
        self.y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.loss_fn = MAE()

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        loss = self.loss_fn(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [1],
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
        ]), loss)

    def test_forward(self):
        """
        Tests the method forward().
        """
        loss = self.loss_fn.forward(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [1],
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
        ]), loss)

    def test_gradient(self):
        """
        Tests the method gradient().
        """
        dL_dy_pred = self.loss_fn.gradient(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [-1],
            [0],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
            [1],
        ]), dL_dy_pred)


class TestBCEWithLogits(unittest.TestCase):
    """
    Tests the class BCEWithLogits.
    """

    def setUp(self):
        self.y_pred = np.array([
            [0],
            [1],
            [2],
            [3],
            [4],
            [5],
            [6],
            [7],
            [8],
        ])
        self.y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.loss_fn = BCEWithLogits()

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        loss = self.loss_fn(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [6.93147181e-01],
            [3.13261688e-01],
            [1.26928011e-01],
            [4.85873516e-02],
            [1.81499279e-02],
            [6.71534849e-03],
            [2.47568514e-03],
            [9.11466454e-04],
            [3.35406373e-04],
        ]), loss)

    def test_forward(self):
        """
        Tests the method forward().
        """
        loss = self.loss_fn.forward(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [6.93147181e-01],
            [3.13261688e-01],
            [1.26928011e-01],
            [4.85873516e-02],
            [1.81499279e-02],
            [6.71534849e-03],
            [2.47568514e-03],
            [9.11466454e-04],
            [3.35406373e-04],
        ]), loss)

    def test_gradient(self):
        """
        Tests the method gradient().
        """
        dL_dz = self.loss_fn.gradient(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [-5.00000000e-01],
            [-2.68941421e-01],
            [-1.19202922e-01],
            [-4.74258732e-02],
            [-1.79862100e-02],
            [-6.69285092e-03],
            [-2.47262316e-03],
            [-9.11051194e-04],
            [-3.35350130e-04],
        ]), dL_dz)


class TestCEWithLogits(unittest.TestCase):

    def setUp(self):
        self.y_pred = np.array([
            [0, 1, 2],
            [3, 4, 5],
            [6, 7, 8],
            [9, 10, 11],
            [12, 13, 14],
            [15, 16, 17],
            [18, 19, 20],
            [21, 22, 23],
            [24, 25, 26],
        ])
        self.y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.loss_fn = CEWithLogits()

    def test_call(self):
        """
        Tests the magic method __call__().
        """
        loss = self.loss_fn(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
        ]), loss)

    def test_forward(self):
        """
        Tests the method forward().
        """
        loss = self.loss_fn.forward(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
            [1.40760596],
        ]), loss)

    def test_gradient(self):
        """
        Tests the method gradient().
        """
        dL_dz = self.loss_fn.gradient(self.y_pred, self.y_true)
        np.testing.assert_almost_equal(np.array([
            [-2.40760596, -2.40760596, -0.40760596],
            [-2.40760596, -2.40760596, -0.40760596],
            [-2.40760596, -2.40760596, -0.40760596],
            [-2.40760596, -2.40760596, -0.40760596],
            [-2.40760596, -2.40760596, -0.40760596],
            [-2.40760596, -2.40760596, -0.40760596],
            [-2.40760596, -2.40760596, -0.40760596],
            [-2.40760596, -2.40760596, -0.40760596],
            [-2.40760596, -2.40760596, -0.40760596],
        ]), dL_dz)


if __name__ == '__main__':
    unittest.main()
