# py-neural-network
A simple implementation of neural network in Numpy.

# PyNeuralNet
PyNeuralNet is a simple Python Framework for using Multiple Linear Perceptrons (MLPs) in Numpy.
You can find more information about MLPs [here](https://en.wikipedia.org/wiki/Feedforward_neural_network).

### Using MLPs with PyNeuralNet
For the following we want to train a MLP to approximate a regression curve for the [diabetes dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html#sklearn.datasets.load_diabetes)
```python
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np

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
```

The `__init__()` of an MLP model is mostly the same as for Pytorch Sequential model.
Additionally we added two parameters to change the initialization of the weight matrix W and the bias B:
```python
from PyNeuralNet.layer import Linear, Sequential
from PyNeuralNet.activation import ReLU

model = Sequential(
    Linear(in_features=X.shape[1], out_features=128, weight_initializer="pytorch_uniform", bias=True),
    ReLU(),
    Linear(in_features=128, out_features=128, weight_initializer="pytorch_uniform", bias=True),
    ReLU(),
    Linear(in_features=128, out_features=1, weight_initializer="pytorch_uniform", bias=True),
```

Now we have to define our loss function, which we want to minimize.
Our choice is the `MSE`:
```python
from PyNeuralNet.loss import MSE

loss_fn = MSE()
```

Now we have to define which optimizer we use to update the parameters of our MLP model.
Our choice is the `SGD`:
```python
from PyNeuralNet.optimizer import SGD

optimizer = SGD(model, lr=0.001, weight_decay=0.1, momentum=0.1)
```

Now we have to use a training loop to learn the parameters of our MLP model.
```python
import time 

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
```

### Future Features
The following list defines features, that are currently on work:

* [X] Implement Adam optimizer
* [X] Make backward propagation faster
* [X] Fix Errors in BCE/CE Loss 
* [X] Fix Errors in Softmax/LogSoftmax
* [ ] Add more model types to PyNeuralNet