import copy
import unittest

import numpy as np

from PyNeuralNet.activation import ReLU
from PyNeuralNet.layer import Sequential, Linear
from PyNeuralNet.loss import MSE
from PyNeuralNet.optimizer import SGD


class TestSGD(unittest.TestCase):
    """
    Tests the class SGD.
    """

    def setUp(self):
        self.model = Sequential(*[
            Linear(in_features=3, out_features=2, bias=True, weight_initializer="ones", bias_initializer="zeros"),
            ReLU(),
            Linear(in_features=2, out_features=2, bias=True, weight_initializer="ones", bias_initializer="zeros"),
        ])
        self.optimizer = SGD(self.model, lr=1e-3, random_state=42)
        self.X = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ])
        self.y = np.array([
            [1, 2],
            [3, 4],
            [5, 6],
        ])
        self.loss_fn = MSE()

    def test_step(self):
        """
        Tests the method step().
        """
        # Calculate the predicted values
        y_pred = self.model(self.X)

        # Calculate the loss
        loss = self.loss_fn(y_pred, self.y)

        # Do the backward propagation (Calculate the gradients)
        dL_dy_pred = self.loss_fn.backward()
        _ = self.model.backward(dL_dy_pred)

        old_parameters = copy.deepcopy(self.model.parameters())

        # Update the parameters of the model
        self.optimizer.step()

        # Check if the parameters got updated
        new_parameters = self.model.parameters()

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_almost_equal(old_parameters["module_0"]["bias"], new_parameters["module_0"]["bias"])

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_almost_equal(old_parameters["module_0"]["weights"], new_parameters["module_0"]["weights"])

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_almost_equal(old_parameters["module_2"]["bias"], new_parameters["module_2"]["bias"])

        with np.testing.assert_raises(AssertionError):
            np.testing.assert_almost_equal(old_parameters["module_2"]["weights"], new_parameters["module_2"]["weights"])


if __name__ == '__main__':
    unittest.main()
