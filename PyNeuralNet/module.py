from abc import ABC, abstractmethod

import numpy as np


class Module(ABC):

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    @abstractmethod
    def forward(self, X: np.ndarray) -> np.ndarray:
        # The forward pass of the model
        pass

    @abstractmethod
    def gradient(self, X: np.ndarray) -> np.ndarray:
        # Computes the gradient dZ/dX ...
        pass

