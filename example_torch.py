import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class SimpleModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size, bias=True)
        )

    def forward(self, x):
        return self.model.forward(x)


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

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create DataLoader for training set
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Get the input and output shape of the dataset
    input_output_wrapper = {
        "diabetes": (10, 1),
        "breast_cancer": (30, 1),
    }
    if dataset not in input_output_wrapper:
        raise ValueError(f"Illegal dataset {dataset}. The valid options are {input_output_wrapper.keys()}!")
    input_dim, output_dim = input_output_wrapper[dataset]

    # Instantiate the model
    model = SimpleModel(input_dim, 128, output_dim)

    # Get the optimizer
    optimizer_wrapper = {
        "sgd": optim.SGD(model.parameters(), lr=0.001, weight_decay=0.1, momentum=0.1),
        "adam": optim.Adam(model.parameters(), lr=0.01, weight_decay=0.1)
    }
    if optimizer_type not in optimizer_wrapper:
        raise ValueError(f"Illegal optimizer_type {optimizer_type}. The valid options are {optimizer_wrapper.keys()}!")
    optimizer = optimizer_wrapper[optimizer_type]

    # Define loss function and optimizer
    # Get the loss function
    loss_wrapper = {
        "diabetes": nn.MSELoss(),
        "breast_cancer": nn.BCEWithLogitsLoss(),
    }
    if dataset not in loss_wrapper:
        raise ValueError(f"Illegal dataset {dataset}. The valid options are {loss_wrapper.keys()}!")
    loss_fn = loss_wrapper[dataset]

    # Training loop
    start_time = time.time()
    loss_per_epoch = []
    for epoch in range(num_epochs):
        loss_per_batch = []
        for inputs, targets in train_dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss_per_batch += [loss.item()]

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_per_epoch += [np.mean(loss_per_batch)]
        # Print the loss after each epoch
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {np.mean(loss_per_batch):.4f}')
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
