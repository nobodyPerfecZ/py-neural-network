import numpy as np

from PyNeuralNet.activation.abstract_activation import Activation


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
        self.slope = 0.0
        self.X = None

    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        self.X = X
        return np.where(X <= 0, self.slope * X, X)

    def backward(self, dL_dZ: np.ndarray, **kwargs) -> np.ndarray:
        if self.X is None:
            raise ValueError("You need to call forward() first, before to compute the gradients!")
        # dZ_dX = np.where(self.X <= 0, self.slope, 1)
        return dL_dZ * np.where(self.X <= 0, self.slope, 1)


class LeakyReLU(ReLU):
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
        super().__init__()
        self.slope = slope
