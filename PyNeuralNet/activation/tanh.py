import numpy as np

from PyNeuralNet.activation.abstract_activation import Activation


class Tanh(Activation):
    """
    Tangens Hyperbolic (Tanh) Module.
    It should be used for introducing non-linearity in neural networks.

    The formula for the Tanh function is:
        - Tanh(X) := Z = (e^X - e^-X) / (e^X + e^-X)

    The gradients dZ/dX for the Tanh function is:
        - dZ/dX = 1 - Tanh(X)^2
    """

    def __init__(self):
        self.X = None

    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self.X = X
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

    def backward(self, dL_dZ: np.ndarray, **kwargs) -> np.ndarray:
        if self.X is None:
            raise ValueError("You need to call forward() first, before to compute the gradients!")
        # dZ_dX = 1 - self.forward(self.X) ** 2
        return dL_dZ * (1 - self.forward(self.X) ** 2)
