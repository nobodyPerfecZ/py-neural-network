from abc import ABC, abstractmethod

import numpy as np


class Module(ABC):

    def __call__(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return self.forward(X, **kwargs)

    @abstractmethod
    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        # The forward pass of the model
        pass

    @abstractmethod
    def gradient(self, X: np.ndarray, **kwargs) -> np.ndarray:
        # Computes the gradient dZ/dX ...
        pass

