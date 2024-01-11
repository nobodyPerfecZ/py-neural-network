from typing import Optional

import numpy as np

from PyNeuralNet.activation.abstract_activation import Activation


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
