from abc import ABC, abstractmethod

import numpy as np

from PyNeuralNet.activation import Sigmoid, LogSoftmax


class Loss(ABC):

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray):
        return self.forward(y_pred, y_true)

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        pass

    @abstractmethod
    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray):
        pass


class MSE(Loss):

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        if y_true.ndim == 1:
            # Case: y_true has shape (N,)
            y_true = np.expand_dims(y_true, axis=-1)

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch. The shape of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) should be equal!"
            )
        return np.mean((y_pred - y_true) ** 2, axis=-1, keepdims=True)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray):
        if y_true.ndim == 1:
            y_true = np.expand_dims(y_true, axis=-1)

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch. The shape of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) should be equal!"
            )
        return (2 / y_pred.shape[-1]) * (y_pred - y_true)


class MAE(Loss):

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        if y_true.ndim == 1:
            # Case: y_true has shape (N,)
            y_true = np.expand_dims(y_true, axis=-1)

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch. The shape of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) should be equal!"
            )

        return np.mean(np.abs(y_pred - y_true), axis=-1, keepdims=True)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray):
        if y_true.ndim == 1:
            # Case: y_true has shape (N,)
            y_true = np.expand_dims(y_true, axis=-1)

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch. The shape of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) should be equal!"
            )

        return (1 / y_pred.shape[-1]) * np.sign(y_pred - y_true)


class BCEWithLogits(Loss):

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        # Change the shape (N,) to (N, 1)
        y_true = np.expand_dims(y_true, axis=-1)

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch. The shape of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) should be equal!"
            )

        # Apply a Sigmoid on y_pred (from logits to probabilities)
        y_pred = Sigmoid()(y_pred)

        zero_term = (1 - y_true) * np.log(1 - y_pred)
        one_term = y_true * np.log(y_pred)
        return -np.mean(zero_term + one_term, axis=-1, keepdims=True)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray):
        if y_true.ndim == 1:
            # Case: y_true has shape (N,)
            y_true = np.expand_dims(y_true, axis=-1)

        if y_pred.shape != y_true.shape:
            raise ValueError(
                f"Shape mismatch. The shape of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) should be equal!"
            )

        # Apply a Sigmoid on y_pred (from logits to probabilities)
        y_pred = Sigmoid()(y_pred)

        return y_pred - y_true


class CEWithLogits(Loss):

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray):
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Shape mismatch. First axis of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) should be equal!"
            )

        # Apply a log-softmax on y_pred (from logits to probabilities)
        y_pred = LogSoftmax()(y_pred, axis=-1)

        # Get the corresponding probabilities of the true classes (columns)
        rows = np.arange(len(y_pred))
        y_pred = np.expand_dims(y_pred[rows, y_true], axis=-1)

        return np.mean(-y_pred, axis=-1, keepdims=True)

    def gradient(self, y_pred: np.ndarray, y_true: np.ndarray):
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"Shape mismatch. First axis of y_pred ({y_pred.shape}) and y_true ({y_true.shape}) should be equal!"
            )

        # Apply a log-softmax on y_pred (from logits to probabilities)
        gradients = LogSoftmax()(y_pred, axis=-1)

        # Get the corresponding probabilities for each true class
        rows = np.arange(len(y_pred))
        gradients[rows, y_true] -= 1
        return gradients
