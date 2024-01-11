import unittest

import numpy as np

from PyNeuralNet.loss import MSE


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
        loss1 = self.loss_fn(self.y_pred, self.y_true, reduction="none")
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
        ]), loss1)

        loss2 = self.loss_fn(self.y_pred, self.y_true, reduction="mean")
        np.testing.assert_almost_equal(np.array([15.66666667]), loss2)

        loss3 = self.loss_fn(self.y_pred, self.y_true, reduction="sum")
        np.testing.assert_almost_equal(np.array([141.0]), loss3)

    def test_forward(self):
        """
        Tests the method forward().
        """
        loss1 = self.loss_fn.forward(self.y_pred, self.y_true, reduction="none")
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
        ]), loss1)

        loss2 = self.loss_fn.forward(self.y_pred, self.y_true, reduction="mean")
        np.testing.assert_almost_equal(np.array([15.66666667]), loss2)

        loss3 = self.loss_fn.forward(self.y_pred, self.y_true, reduction="sum")
        np.testing.assert_almost_equal(np.array([141.0]), loss3)

    def test_backward(self):
        """
        Tests the method backward().
        """
        self.loss_fn.forward(self.y_pred, self.y_true)
        dL_dy_pred = self.loss_fn.backward()
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

    def test_parameters(self):
        """
        Tests the method parameters().
        """
        self.assertEqual({"module_0": {}}, self.loss_fn.parameters())

    def test_gradients(self):
        """
        Tests the method gradients().
        """
        self.assertEqual({"module_0": {}}, self.loss_fn.gradients())


if __name__ == "__main__":
    unittest.main()
