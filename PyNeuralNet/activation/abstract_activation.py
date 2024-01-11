from abc import abstractmethod

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
