from abc import abstractmethod
from typing import Optional, Union

import numpy as np

from PyNeuralNet.activation import Sigmoid, LogSoftmax
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


class MAE(Loss):
    """
    Mean Absolute Error (MAE) Loss Function.

    The Loss function is defined as:
        - L(y_hat, y) = 1/N sum_i=1^N |y_hat_i - y|

    The gradients of the loss function is defined as:
        - dL/dy_hat = 1/N * sign(y_hat - y)
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

        # Safe the inputs for the computation of gradients
        self.y_pred = y_pred
        self.y_true = y_true

        # Compute the losses for every sample
        losses = np.mean(np.abs(y_pred - y_true), axis=-1, keepdims=True)

        return reduction_wrapper[reduction](losses)

    def backward(self, **kwargs) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise ValueError("You need to call forward() first before to compute the gradients!")
        return (1 / self.y_pred.shape[-1]) * np.sign(self.y_pred - self.y_true)


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
        # Change the shape (N,) to (N, 1)
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


class CEWithLogits(Loss):
    # TODO: Fix Errors
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

        # Apply a log-softmax on y_pred (from logits to probabilities)
        y_pred = LogSoftmax()(y_pred, axis=-1)

        self.y_pred = y_pred
        self.y_true = y_true

        # Get the corresponding probabilities of the true classes (columns)
        rows = np.arange(len(y_pred))
        y_pred = np.expand_dims(y_pred[rows, y_true], axis=-1)

        # Compute the losses for every sample
        losses = np.mean(-y_pred, axis=-1, keepdims=True)

        return reduction_wrapper[reduction](losses)

    def backward(self, **kwargs) -> np.ndarray:
        if self.y_pred is None or self.y_true is None:
            raise ValueError("You need to call forward() first before to compute the gradients!")

        # Apply a log-softmax on y_pred (from logits to probabilities)
        gradients = LogSoftmax()(self.y_pred, axis=-1)

        # Get the corresponding probabilities for each true class
        rows = np.arange(len(self.y_pred))
        gradients[rows, self.y_true] -= 1
        return gradients
