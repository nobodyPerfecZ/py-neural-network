import unittest

import numpy as np

from PyNeuralNet.loss import BCEWithLogits


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
        loss1 = self.loss_fn(self.y_pred, self.y_true, reduction="none")
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
        ]), loss1)

        loss2 = self.loss_fn(self.y_pred, self.y_true, reduction="mean")
        np.testing.assert_almost_equal(np.array([0.13450134]), loss2)

        loss3 = self.loss_fn(self.y_pred, self.y_true, reduction="sum")
        np.testing.assert_almost_equal(np.array([1.21051207]), loss3)

    def test_forward(self):
        """
        Tests the method forward().
        """
        loss1 = (self.loss_fn.forward(self.y_pred, self.y_true, reduction="none"))
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
        ]), loss1)

        loss2 = self.loss_fn.forward(self.y_pred, self.y_true, reduction="mean")
        np.testing.assert_almost_equal(np.array([0.13450134]), loss2)

        loss3 = self.loss_fn.forward(self.y_pred, self.y_true, reduction="sum")
        np.testing.assert_almost_equal(np.array([1.21051207]), loss3)

    def test_backward(self):
        """
        Tests the method backward().
        """
        self.loss_fn.forward(self.y_pred, self.y_true)
        dL_dz = self.loss_fn.backward()
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
