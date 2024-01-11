from typing import Optional

import numpy as np

from PyNeuralNet.module import Module


class Linear(Module):
    """
    Linear layer module for neural networks.

    Args:
        in_features (int):
            The number of input features

        out_features (int):
            THe number of output features

        bias (bool, optional):
            Controls if a bias parameter should be included

        weight_initializer (str, optional):
            The initialization method of the weight matrix.

                - If weight_initializer := "random_uniform":
                    Each weight wij is sampled from a uniform distribution U(-1,1)

                - If weight_initializer := "random_normal":
                    Each weight wij is sampled from a normal distribution N(0,1)

                - If weight_initializer := "pytorch_uniform":
                    Each weight wij is sampled from a uniform distribution (-k,k) with k = 1 / in_features

                - If weight_initializer := "ones":
                    Each weight wij := 0 if i != j else 1

                - If weight_initializer := "log2":
                    Each weight wij := 0 if i != j else log(2)

                - If weight_initializer := "sqrt2":
                    Each weight wij := 0 if i != j else sqrt(2)

        bias_initializer (str, optional):
            The initialization method of the bias matrix.

                - If bias_initializer := "random_uniform":
                    Each bias bi is sampled from a uniform distribution U(-1, 1)

                - If bias_initializer := "random_normal":
                    Each bias bi is sampled from a normal distribution N(0,1)

                - If bias_initializer := "pytorch_uniform":
                    Each weight wij is sampled from a uniform distribution (-k,k) with k = 1 / in_features

                - If bias_initializer := "ones":
                    Each bias bi := 1

                - If bias_initializer := "zeros":
                    Each bias bi := 0

                - If bias_initializer := "log2":
                    Each bias bi := log(2)

                - If bias_initializer := "sqrt2":
                    Each bias bi := sqrt(2)

        random_state (int, optional):
            The seed of the random number generator
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True,
            weight_initializer: str = "random_uniform",
            bias_initializer: str = "random_uniform",
            random_state: Optional[int] = None,
    ):
        if in_features < 1:
            raise ValueError(f"Illegal in_features {in_features}. The in_features should be in range [1,+inf)!")
        if out_features < 1:
            raise ValueError(f"Illegal out_features {out_features}. The out_features should be in range [1, +inf)!")

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.rng = np.random.RandomState(random_state)

        self.B = self.initialize_bias()  # bias of the layer (out_features,)
        self.W = self.initialize_weights()  # weights of the layer (in_features, out_features)

        self.dL_dB = None
        self.dL_dW = None

        self.X = None  # previous input of the model

    def initialize_weights(self) -> np.ndarray:
        """
        Initialize the weights after the given weight_initializer.

        - If bias_initializer := "random_uniform":
            Each bias bi is sampled from a uniform distribution U(-1, 1)

        - If bias_initializer := "random_normal":
            Each bias bi is sampled from a normal distribution N(0,1)

        - If bias_initializer := "pytorch_uniform":
            Each weight wij is sampled from a uniform distribution (-k,k) with k = 1 / in_features

        - If bias_initializer := "ones":
            Each bias bi := 1

        - If bias_initializer := "zeros":
            Each bias bi := 0

        - If bias_initializer := "log2":
            Each bias bi := log(2)

        - If bias_initializer := "sqrt2":
            Each bias bi := sqrt(2)

        Returns:
            np.ndarray:
                The initialized weight matrix W
        """
        weight_initializer_wrapper = {
            "random_uniform": self.rng.uniform(low=-1.0, high=1.0, size=(self.in_features, self.out_features)),
            "random_normal": self.rng.normal(loc=0.0, scale=1.0, size=(self.in_features, self.out_features)),
            "pytorch_uniform": self.rng.uniform(low=-1.0/self.in_features, high=1.0/self.in_features,
                                                size=(self.in_features, self.out_features)),
            "ones": np.eye(self.in_features, self.out_features),
            "log2": np.eye(self.in_features, self.out_features) * np.log(2),
            "sqrt2": np.eye(self.in_features, self.out_features) * np.sqrt(2),
        }

        if self.weight_initializer not in weight_initializer_wrapper:
            raise ValueError(f"Illegal weight_initializer {weight_initializer_wrapper}. "
                             f" The valid options are {weight_initializer_wrapper.keys()}!")
        return weight_initializer_wrapper[self.weight_initializer]

    def initialize_bias(self) -> Optional[np.ndarray]:
        """
        Initialize the bias after the given bias_initializer and if the bias should be used.

        - If weight_initializer := "random_uniform":
            Each weight wij is sampled from a uniform distribution U(-1,1)

        - If weight_initializer := "random_normal":
            Each weight wij is sampled from a normal distribution N(0,1)

        - If weight_initializer := "pytorch_uniform":
            Each weight wij is sampled from a uniform distribution (-k,k) with k = 1 / in_features

        - If weight_initializer := "ones":
            Each weight wij := 0 if i != j else 1

        - If weight_initializer := "log2":
            Each weight wij := 0 if i != j else log(2)

        - If weight_initializer := "sqrt2":
            Each weight wij := 0 if i != j else sqrt(2)

        Returns:
            np.ndarray:
                The initialized weight matrix W
        """
        if not self.bias:
            return None

        bias_initializer_wrapper = {
            "random_uniform": self.rng.uniform(low=-1.0, high=1.0, size=self.out_features),
            "random_normal": self.rng.normal(loc=0.0, scale=1.0, size=self.out_features),
            "pytorch_uniform": self.rng.uniform(low=-1.0 / self.in_features, high=1.0 / self.in_features,
                                                size=self.out_features),
            "ones": np.ones(shape=self.out_features),
            "zeros": np.zeros(shape=self.out_features),
            "log2": np.ones(shape=self.out_features) * np.log(2),
            "sqrt2": np.ones(shape=self.out_features) * np.sqrt(2),
        }

        if self.bias_initializer not in bias_initializer_wrapper:
            raise ValueError(f"Illegal bias_initializer {bias_initializer_wrapper}. "
                             f"The valid options are {bias_initializer_wrapper.keys()}!")
        return bias_initializer_wrapper[self.bias_initializer]

    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if X.ndim == 1:
            X = np.expand_dims(X, axis=-1)

        # Safe the input for the backward propagation
        self.X = X

        # Compute the output Z := X * Weights (+ Bias)
        if self.bias:
            return self.X @ self.W + self.B
        else:
            return self.X @ self.W

    def backward(self, dL_dZ: np.ndarray, **kwargs) -> np.ndarray:
        if self.X is None:
            raise ValueError("You need to call forward() before to compute the gradients!")

        # Compute the partial gradient dZ/dTheta, dZ/dX
        # dZ_dB = I (Identity matrix)
        # dZ_dW = self.X
        # dZ_dX = self.W.T

        # Compute the overall gradient dL/dTheta := dL/dZ * dZ/dTheta, dL/dX = dL/dZ * dZ/dX
        if self.bias:
            self.dL_dB = np.mean(dL_dZ, axis=0)
        self.dL_dW = np.dot(self.X.T, dL_dZ) / dL_dZ.shape[0]
        dL_dX = np.dot(dL_dZ, self.W.T)
        return dL_dX

    def update(self, params: dict[str, dict[str, np.ndarray]]):
        # Get the parameters of the module
        params = list(params.values())[0]

        param_keys = ["bias", "weights"]
        if param_keys[-1] not in params:
            raise ValueError(f"Illegal argument params. The valid keys are {param_keys}!")

        if self.bias:
            if param_keys[0] not in params:
                raise ValueError(f"Illegal argument params. The valid keys are {param_keys}!")

            if params["bias"].shape != self.B.shape:
                raise ValueError(f"Illegal bias {params['bias'].shape}. It should have the shape {self.B.shape}!")

        if params["weights"].shape != self.W.shape:
            raise ValueError(f"Illegal weights {params['weights'].shape}. It should have the shape {self.W.shape}!")

        # Update the parameters of the model
        if self.bias:
            self.B = params["bias"]
        self.W = params["weights"]

    def parameters(self, **kwargs) -> dict[str, dict[str, np.ndarray]]:
        module = {
            "module_0": {
                "weights": self.W,
            }
        }
        if self.bias:
            module["module_0"]["bias"] = self.B

        return module

    def gradients(self, **kwargs) -> dict[str, dict[str, np.ndarray]]:
        if self.dL_dW is None:
            raise ValueError("You need to call backward() before to return the gradients!")
        if self.bias:
            if self.dL_dB is None:
                raise ValueError("You need to call backward() before to return the gradients!")
        module = {
            "module_0": {
                "weights": self.dL_dW,
            }
        }
        if self.bias:
            module["module_0"]["bias"] = self.dL_dB
        return module
