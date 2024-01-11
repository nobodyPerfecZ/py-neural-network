from abc import abstractmethod
from typing import Optional, Union

import numpy as np

from PyNeuralNet.module import Module


class Loss(Module):
    """
    A module representing a loss function.
    """

    def __call__(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            reduction: Optional[str] = "mean",
            **kwargs
    ) -> np.ndarray:
        """
        The computation of the loss between the predicted values and the targets.

        Args:
            y_pred (np.ndarray):
                The predicted output of the model

            y_true (np.ndarray):
                The true outputs (targets)

            reduction (str):
                Controls how the loss should be aggregated across the batch.
                If reduction := "mean": It uses np.mean(axis=0) for aggregation
                If reduction := "sum": It uses np.sum(axis=0) for aggregation

        Returns:
            np.ndarray:
                losses (np.ndarray):
                    The (non-)aggregated losses between the predicted values and the targets
        """
        return self.forward(y_pred, y_true, reduction, **kwargs)

    @abstractmethod
    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            reduction: Optional[str] = "mean",
            **kwargs
    ) -> Union[np.ndarray, float]:
        """
        The computation of the loss between the predicted values and the targets.

        Args:
            y_pred (np.ndarray):
                The predicted output of the model

            y_true (np.ndarray):
                The true outputs (targets)

            reduction (str):
                Controls how the loss should be aggregated across the batch.
                If reduction := "mean": It uses np.mean(axis=0) for aggregation
                If reduction := "sum": It uses np.sum(axis=0) for aggregation

        Returns:
            np.ndarray:
                losses (np.ndarray):
                    The (non-)aggregated losses between the predicted values and the targets
        """
        pass

    @abstractmethod
    def backward(self, **kwargs) -> np.ndarray:
        """
        The Backward propagation of the loss function.

        It returns the gradients dL/dy_pred of the previous passed (y_pred, y_true) from forward().

        Returns:
            np.ndarray:
                gradients (np.ndarray):
                    The gradients dL/dy_pred
        """
        pass

    def update(self, params: dict[str, np.ndarray]):
        pass
