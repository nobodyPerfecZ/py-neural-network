import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from PyNeuralNet.activation import ReLU
from PyNeuralNet.layer import Sequential, Linear
from PyNeuralNet.loss import MSE, BCEWithLogits
from PyNeuralNet.optimizer import Adam, SGD


def train(
        dataset: str = "diabetes",
        optimizer_type: str = "sgd",
        normalize_X: bool = True,
):
    # Constant values
    random_state = 42
    batch_size = 64
    num_epochs = 2000

    # Load dataset
    dataset_wrapper = {
        "diabetes": load_diabetes,
        "breast_cancer": load_breast_cancer,
    }
    if dataset not in dataset_wrapper:
        raise ValueError(f"Illegal dataset {dataset}. The valid options are {dataset_wrapper.keys()}!")
    X, y = dataset_wrapper[dataset](return_X_y=True)

    # Normalize the input to 0 mean, 1 variance
    if normalize_X:
        X = StandardScaler().fit_transform(X)

    # Reshape from (N,) to (N, 1)
    y = np.expand_dims(y, axis=-1)

    # Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Get the input and output shape of the dataset
    input_output_wrapper = {
        "diabetes": (10, 1),
        "breast_cancer": (30, 1),
    }
    if dataset not in input_output_wrapper:
        raise ValueError(f"Illegal dataset {dataset}. The valid options are {input_output_wrapper.keys()}!")
    input_dim, output_dim = input_output_wrapper[dataset]

    model = Sequential(
        Linear(in_features=input_dim, out_features=128, weight_initializer="pytorch_uniform", bias=True),
        ReLU(),
        Linear(in_features=128, out_features=128, weight_initializer="pytorch_uniform", bias=True),
        ReLU(),
        Linear(in_features=128, out_features=output_dim, weight_initializer="pytorch_uniform", bias=True),
    )

    # Get the optimizer
    optimizer_wrapper = {
        "sgd": SGD(model, lr=0.001, weight_decay=0.1, momentum=0.1),
        "adam": Adam(model, lr=0.01, weight_decay=0.1)
    }
    if optimizer_type not in optimizer_wrapper:
        raise ValueError(f"Illegal optimizer_type {optimizer_type}. The valid options are {optimizer_wrapper.keys()}!")
    optimizer = optimizer_wrapper[optimizer_type]

    # Get the loss function
    loss_wrapper = {
        "diabetes": MSE(),
        "breast_cancer": BCEWithLogits(),
    }
    if dataset not in loss_wrapper:
        raise ValueError(f"Illegal dataset {dataset}. The valid options are {loss_wrapper.keys()}!")
    loss_fn = loss_wrapper[dataset]

    start_time = time.time()
    loss_per_epoch = []
    for i in range(num_epochs):
        # Randomly shuffle your dataset
        indices = np.random.permutation(len(X_train))
        X_train = X_train[indices]
        y_train = y_train[indices]

        # Split training dataset into batches
        X_train_batch = np.array_split(X_train, len(X_train) // batch_size)
        y_train_batch = np.array_split(y_train, len(y_train) // batch_size)

        loss_per_batch = []
        for input, targets in zip(X_train_batch, y_train_batch):
            y_pred = model(input)

            # Calculate the loss
            loss = loss_fn(y_pred, targets)
            loss_per_batch += [loss.item()]

            # Do the backward propagation (Calculate the gradients)
            dL_dy_pred = loss_fn.backward()
            model.backward(dL_dy_pred)

            # Update the parameters of the model
            optimizer.step()
        loss_per_epoch += [np.mean(loss_per_batch)]
        print(f'Epoch [{i + 1}/{num_epochs}], Loss: {np.mean(loss_per_batch):.4f}')

    end_time = time.time()
    print(f"Total Time: {end_time - start_time}")
    return loss_per_epoch


if __name__ == "__main__":
    loss_per_epoch = train(
        dataset="breast_cancer",
        optimizer_type="adam",
        normalize_X=True,
    )

    plt.plot(loss_per_epoch)
    plt.show()
