from abc import abstractmethod
from typing import Optional

import numpy as np

from PyNeuralNet.module import Module


class Activation(Module):
    """
    A module representing an activation function.
    """

    @abstractmethod
    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        The forward pass of the activation function.

        Args:
            X (np.ndarray):
                The input of the activation function

        Returns:
            np.ndarray:
                The output of the activation function (Z)
        """
        pass

    @abstractmethod
    def backward(self, dL_dZ: np.ndarray, **kwargs) -> np.ndarray:
        """
        Backward propagation of the activation function.

        It returns the gradients dL/dX := dL/dZ * dZ/dX of the current activation function.

        Args:
            dL_dZ (np.ndarray):
                The gradient dL/dZ of the previous module.

        Returns:
            np.ndarray:
                The gradient dL/dX of the activation function
        """
        pass

    def update(self, params: dict[str, np.ndarray]):
        pass


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU) Activation Module.
    It should be used for introducing non-linearity in neural networks.

    The formula for the ReLU function is:
        - ReLU(X) := Z = max(0, X)

    The gradients dZ/dX for the ReLU function is:
        - dZ/dX = 1 if X > 0 else 0
    """

    def __init__(self):
        self.X = None

    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self.X = X
        return np.where(self.X <= 0, 0, self.X)

    def backward(self, dL_dZ: np.ndarray, **kwargs) -> np.ndarray:
        if self.X is None:
            raise ValueError("You need to call forward() first, before to compute the gradients!")
        # dZ_dX = np.where(self.X <= 0, 0, 1)
        return dL_dZ * np.where(self.X <= 0, 0, 1)


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit (LeakyReLU) Activation Module.
    It should be used for introducing non-linearity in neural networks.

    The formula for the LeakyReLU function is:
        - LeakyReLU(X) := Z = max(0,X) + slope * min(0, X)

    The gradients dZ/dX for the LeakyReLU function is:
        - dZ/dX = 1 if X > 0 else slope

    Args:
        slope (float):
            The slope for the case if X <= 0
    """

    def __init__(self, slope: float = 0.1):
        if slope < 0.0 or slope > 1.0:
            raise ValueError(f"Illegal slope {slope}. The argument should be in between [0.0, 1.0]!")
        self.slope = slope
        self.X = None

    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self.X = X
        return np.where(X <= 0, self.slope * X, X)

    def backward(self, dL_dZ: np.ndarray, **kwargs) -> np.ndarray:
        if self.X is None:
            raise ValueError("You need to call forward() first, before to compute the gradients!")
        # dZ_dX = np.where(self.X <= 0, self.slope, 1)
        return dL_dZ * np.where(self.X <= 0, self.slope, 1)


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


class Softmax(Activation):
    """
    Softmax Module.
    It should be used for converting predicted logits into a probability distribution.
    It is recommended to not use the backward() directly, because of numerical instability.
    Instead, use the loss function that directly handles logits inside their loss (e.g. BCEWithLogits, CEWithLogits).

    The formula for the Softmax function is:
        - Softmax(X) := Z = e^X_i / sum_j^N e^X_j

    The gradients dZ/dX for the Softmax function is:
        - dZ/dX = Softmax(X) * (1 - Softmax(X))
    """
    def __init__(self):
        self.X = None
        self.axis = None

    def forward(self, X: np.ndarray, axis: Optional[int] = None, **kwargs) -> np.ndarray:
        self.X = X
        self.axis = axis
        X_max = np.max(X, axis=axis, keepdims=True)
        e_X = np.exp(X - X_max)
        return e_X / np.sum(e_X, axis=axis, keepdims=True)

    def backward(self, dL_dZ: np.ndarray, **kwargs) -> np.ndarray:
        if self.X is None:
            raise ValueError("You need to call forward() first to compute the gradients!")
        S = self.forward(self.X, axis=self.axis)
        return dL_dZ * (S * (1 - S))


class LogSoftmax(Softmax):
    """
    Log-Softmax Module.
    It should be used for converting predicted logits into a probability distribution.
    It is recommended to not use the backward() directly, because of numerical instability.
    Instead, use the loss function that directly handles logits inside their loss (e.g. BCEWithLogits, CEWithLogits).

    The formula for the Log-Softmax function is:
        - Log-Softmax(X) := Z = Log(e^X_i / sum_j^N e^X_j)

    The gradients dZ/dX for the Log-Softmax function is:
        - dZ/dX = Softmax(X) - 1
    """

    def forward(self, X: np.ndarray, axis: Optional[int] = None, **kwargs) -> np.ndarray:
        return np.log(super().forward(X, axis=axis, **kwargs))

    def backward(self, dL_dZ: np.ndarray, **kwargs) -> np.ndarray:
        if self.X is None:
            raise ValueError("You need to call forward() first to compute the gradients!")
        S = super().forward(self.X, axis=self.axis)
        return dL_dZ * (S - 1)
