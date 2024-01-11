from typing import Optional, Union

import numpy as np

from PyNeuralNet.loss.abstract_loss import Loss


class MSE(Loss):
    """
    Mean Squared Error (MSE) Loss Function.

    The Loss function is defined as:
        - L(y_hat, y) = 1/N sum_i=1^N (y_hat_i - y_i)²

    The gradients of the loss function is defined as:
        - dL/dy_hat_i = 2/N * (y_hat_i - y_i)
    """

    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(
            self,
            y_pred: np.ndarray,
            y_true: np.ndarray,
            reduction: Optional[str] = "mean",
            **kwargs
    ) -> Union[np.ndarray, float]:
        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, axis=-1)

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Illegal y_pred {y_pred.shape} and y_true {y_true.shape}. The shapes should be equal!"
            )

        reduction_wrapper = {
            "mean": np.mean,
            "sum": np.sum,
            "none": lambda x: x,
        }
        if reduction not in reduction_wrapper.keys():
            raise ValueError(f"Illegal reduction {reduction}! The valid options are {reduction_wrapper.keys()}!")

        # Safe the inputs for the computation of gradients
        self.y_pred = y_pred
        self.y_true = y_true

        # Compute the losses for every sample
        losses = np.mean((y_pred - y_true) ** 2, axis=-1, keepdims=True)

        return reduction_wrapper[reduction](losses)

    def backward(self, **kwargs) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise ValueError("You need to call forward() first before to compute the gradients!")
        return (2 / self.y_pred.shape[-1]) * (self.y_pred - self.y_true)
