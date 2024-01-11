import numpy as np

from PyNeuralNet.activation.abstract_activation import Activation


class Sigmoid(Activation):
    """
    Sigmoid Activation Module.
    It should be used for introducing non-linearity in neural networks.

    The formula for the Sigmoid function is:
        - Sigmoid(X) := Z = 1 / 1 + e^(⁻X)

    The gradients dZ/dX for the Sigmoid function is:
        - dZ/dX = Sigmoid(X) * (1 - Sigmoid(X))
    """

    def __init__(self):
        self.X = None

    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self.X = X
        return 1 / (1 + np.exp(-X))

    def backward(self, dL_dZ: np.ndarray, **kwargs) -> np.ndarray:
        if self.X is None:
            raise ValueError("You need to call forward() first, before to compute the gradients!")
        # dZ_dX = self.forward(self.X) * (1 - self.forward(self.X))
        return dL_dZ * (self.forward(self.X) * (1 - self.forward(self.X)))
