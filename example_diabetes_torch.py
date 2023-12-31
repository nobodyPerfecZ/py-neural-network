import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_diabetes
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

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

    # Create DataLoader for training set
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate the model
    model = SimpleModel(X_train.shape[1], 128, 1)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    # optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=0.1, momentum=0.1)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1)

    # Training loop
    start_time = time.time()
    loss_per_epoch = []
    for epoch in range(num_epochs):
        loss_per_batch = []
        for inputs, targets in train_dataloader:
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
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
    loss_per_epoch = train()
    plt.plot(loss_per_epoch)
    plt.show()
