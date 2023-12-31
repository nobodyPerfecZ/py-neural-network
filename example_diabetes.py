import time

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from PyNeuralNet.activation import ReLU
from PyNeuralNet.layer import Sequential, Linear
from PyNeuralNet.loss import MSE
from PyNeuralNet.optimizer import Adam, SGD


def train():
    # Constant values
    random_state = 42
    batch_size = 64
    num_epochs = 2000

    # Load iris dataset
    X, y = load_diabetes(return_X_y=True)

    # Normalize the input to 0 mean, 1 variance
    X = StandardScaler().fit_transform(X)

    # Reshape from (N,) to (N, 1)
    y = np.expand_dims(y, axis=-1)

    # Split data into train and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    model = Sequential(
        Linear(in_features=X.shape[1], out_features=128, weight_initializer="pytorch_uniform", bias=True),
        ReLU(),
        Linear(in_features=128, out_features=128, weight_initializer="pytorch_uniform", bias=True),
        ReLU(),
        Linear(in_features=128, out_features=1, weight_initializer="pytorch_uniform", bias=True),
    )

    # optimizer = SGD(model, lr=0.001, weight_decay=0.1, momentum=0.1)
    optimizer = Adam(model, lr=0.01, weight_decay=1e-1)
    loss_fn = MSE()

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
    loss_per_epoch = train()

    plt.plot(loss_per_epoch)
    plt.show()

