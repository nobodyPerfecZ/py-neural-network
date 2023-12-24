import numpy as np

from PyNeuralNet.module import Module


class ReLU(Module):
    """
    Rectified Linear Unit (ReLU) Activation Module.
    It should be used for introducing non-linearity in neural networks.

    The formula for the ReLU function is:
        - ReLU(X) := Z = max(0, X)

    The gradients dZ/dX for the ReLU function is:
        - dZ/dX = 1 if X > 0 else 0
    """
    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.where(X <= 0, 0, X)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        return np.where(X <= 0, 0, 1)


class LeakyReLU(Module):
    """
    Leaky Rectified Linear Unit (LeakyReLU) Activation Module.
    It should be used for introducing non-linearity in neural networks.

    The formula for the LeakyReLU function is:
        - LeakyReLU(X) := Z = max(0,X) + slope * min(0, X)

    The gradients dZ/dX for the LeakyReLU function is:
        - dZ/dX = 1 if X > 0 else slope

    Args:
        slope (float):
            Defines the slope for the case if X <= 0
    """

    def __init__(self, slope: float = 0.1):
        assert 0.0 <= slope <= 1.0, f"Illegal slope {slope}. The argument should be in between [0.0, 1.0]!"
        self.slope = slope

    def forward(self, X: np.ndarray) -> np.ndarray:
        return np.where(X <= 0, self.slope * X, X)

    def gradient(self, X: np.ndarray) -> np.ndarray:
        return np.where(X <= 0, self.slope, 1)


class Sigmoid(Module):
    """
    Sigmoid Activation Module.
    It should be used for introducing non-linearity in neural networks.

    The formula for the Sigmoid function is:
        - Sigmoid(X) := Z = 1 / 1 + e^(⁻X)

    The gradients dZ/dX for the LeakyReLU function is:
        - dZ/dX = Sigmoid(X) * (1 - Sigmoid(X))
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-X))

    def gradient(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X) * (1 - self.forward(X))


class Tanh(Module):
    """
    Tangens Hyperbolic Module.
    It should be used for introducing non-linearity in neural networks.

    The formula for the Tanh function is:
        - Tanh(X) := Z = (e^X - e^-X) / (e^X + e^-X)

    The gradients dZ/dX for the LeakyReLU function is:
        - dZ/dX = 1 - Tanh(X)^2
    """

    def forward(self, X: np.ndarray) -> np.ndarray:
        return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

    def gradient(self, X: np.ndarray) -> np.ndarray:
        return 1 - self.forward(X) ** 2
