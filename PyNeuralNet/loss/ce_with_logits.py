from typing import Optional, Union

import numpy as np

from PyNeuralNet.activation import Softmax, LogSoftmax
from PyNeuralNet.loss.abstract_loss import Loss


class CEWithLogits(Loss):
    """
    Cross Entropy (with Logits) (CE) Loss Function.

    The Loss function is defined as:
        - L(y_logits, y) = - 1/N sum_i=1^N y_i_c * log(softmax(y_logits_i))

    The gradients of the loss function is defined as:
        - dL/dy_logits = softmax(y_logits) - y_i_c
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
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Shape mismatch. First axis of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) should be equal!"
            )

        reduction_wrapper = {
            "mean": np.mean,
            "sum": np.sum,
            "none": lambda x: x,
        }
        if reduction not in reduction_wrapper.keys():
            raise ValueError(f"Illegal reduction {reduction}! The valid options are {reduction_wrapper.keys()}!")

        self.y_pred = y_pred
        self.y_true = y_true

        # Apply a log-softmax on y_pred (from logits to probabilities)
        y_pred = LogSoftmax()(y_pred, axis=-1)

        # Create a one-hot encoded matrix for y_true
        num_classes = self.y_pred.shape[-1]
        y_true = np.eye(num_classes)[self.y_true]

        # Compute the losses for every sample
        losses = -np.sum(y_pred * y_true, axis=-1, keepdims=True)

        return reduction_wrapper[reduction](losses)

    def backward(self, **kwargs) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise ValueError("You need to call forward() first before to compute the gradients!")

        # Apply a softmax on y_pred (from logits to probabilities)
        y_pred = Softmax()(self.y_pred, axis=-1)

        # Create a one-hot encoded matrix for y_true
        num_classes = self.y_pred.shape[-1]
        y_true = np.eye(num_classes)[self.y_true]

        return y_pred - y_true
