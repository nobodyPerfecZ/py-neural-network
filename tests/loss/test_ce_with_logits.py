import unittest

import numpy as np

from PyNeuralNet.loss import CEWithLogits


class TestCEWithLogits(unittest.TestCase):
    """
    Tests the class CEWithLogits.
    """

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
        loss1 = self.loss_fn(self.y_pred, self.y_true, reduction="none")
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
        ]), loss1)

        loss2 = self.loss_fn(self.y_pred, self.y_true, reduction="mean")
        np.testing.assert_almost_equal(np.array([1.40760596]), loss2)

        loss3 = self.loss_fn(self.y_pred, self.y_true, reduction="sum")
        np.testing.assert_almost_equal(np.array([12.66845368]), loss3)

    def test_forward(self):
        """
        Tests the method forward().
        """
        loss1 = self.loss_fn.forward(self.y_pred, self.y_true, reduction="none")
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
        ]), loss1)

        loss2 = self.loss_fn.forward(self.y_pred, self.y_true, reduction="mean")
        np.testing.assert_almost_equal(np.array([1.40760596]), loss2)

        loss3 = self.loss_fn.forward(self.y_pred, self.y_true, reduction="sum")
        np.testing.assert_almost_equal(np.array([12.66845368]), loss3)

    def test_backward(self):
        """
        Tests the method backward().
        """
        self.loss_fn.forward(self.y_pred, self.y_true)
        dL_dZ = self.loss_fn.backward()
        np.testing.assert_almost_equal(np.array([
            [0.09003057, -0.75527153, 0.66524096],
            [0.09003057, -0.75527153, 0.66524096],
            [0.09003057, -0.75527153, 0.66524096],
            [0.09003057, -0.75527153, 0.66524096],
            [0.09003057, -0.75527153, 0.66524096],
            [0.09003057, -0.75527153, 0.66524096],
            [0.09003057, -0.75527153, 0.66524096],
            [0.09003057, -0.75527153, 0.66524096],
            [0.09003057, -0.75527153, 0.66524096],
        ]), dL_dZ)

    def test_parameters(self):
        """
        Tests the method parameters().
        """
        self.assertEqual({"module_0": {}}, self.loss_fn.parameters())

    def test_gradients(self):
        """
        Tests the method gradients().
        """
        self.assertEqual({"module_0": {}}, self.loss_fn.parameters())


if __name__ == "__main__":
    unittest.main()
