from typing import Optional, Union

import numpy as np

from PyNeuralNet.activation import Sigmoid
from PyNeuralNet.loss.abstract_loss import Loss


class BCEWithLogits(Loss):
    """
    Binary Cross Entropy (with Logits) (BCE) Loss Function.

    The Loss function is defined as:
        - L(y_logits, y) = - 1/N sum_i=1^N y_i * log(sigmoid(y_logits_i)) + (1 - y_i) * log(sigmoid(1 - y_logits_i))

    The gradients of the loss function is defined as:
        - dL/dy_logits = 1/N * sigmoid(y_logits) - y
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
                f"Shape mismatch. The shape of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) should be equal!"
            )

        reduction_wrapper = {
            "mean": np.mean,
            "sum": np.sum,
            "none": lambda x: x,
        }
        if reduction not in reduction_wrapper.keys():
            raise ValueError(f"Illegal reduction {reduction}! The valid options are {reduction_wrapper.keys()}!")

        # Apply a Sigmoid on y_pred (from logits to probabilities)
        y_pred = Sigmoid()(y_pred)

        self.y_pred = y_pred
        self.y_true = y_true

        zero_term = (1 - y_true) * np.log(1 - y_pred)
        one_term = y_true * np.log(y_pred)

        # Compute the losses for every sample
        losses = -np.mean(zero_term + one_term, axis=-1, keepdims=True)

        return reduction_wrapper[reduction](losses)

    def backward(self, **kwargs) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise ValueError("You need to call forward() first before to compute the gradients!")
        return self.y_pred - self.y_true
