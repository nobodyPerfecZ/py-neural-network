from abc import ABC, abstractmethod

import numpy as np


class Module(ABC):

    def __call__(self, *args, **kwargs) -> np.ndarray:
        """
        The forward pass of the module.

        Args:
            *args:
                The input of the module

        Returns:
            np.ndarray:
                The output of the module (Z)
        """
        return self.forward(*args, **kwargs)

    @abstractmethod
    def forward(self, *args, **kwargs) -> np.ndarray:
        """
        The forward pass of the module.

        Args:
            *args:
                The input of the module

        Returns:
            np.ndarray:
                The output of the module (Z)
        """
        # The forward pass of the model
        pass

    @abstractmethod
    def backward(self, *args, **kwargs) -> np.ndarray:
        """
        The Backward propagation of a module.

        A module is defined as a function f: X -> Z, which
        transforms an input X into the output Z.

        The Backward propagation of a module is defined by
        computing the gradients dL/dX with the chain rule:

        - dL/dX = dL/dZ * dZ/dX

        Args:
            *args:
                The gradients dL/dZ of the output Z

        Returns:
            np.ndarray:
                The gradients dL/dX of the input X
        """
        pass

    @abstractmethod
    def update(self, params: dict[str, dict[str, np.ndarray]]):
        """
        Updates the parameters of a module by the given values.

        Args:
            params (dict[str, dict[str, np.ndarray]]):
                module_name (str):
                    The name of the module

                module (dict[str, np.ndarray]):

                    key (str):
                        The name of the tunable parameter

                    value (np.ndarray):
                        The value of the tunable parameter
        """
        pass

    def parameters(self, **kwargs) -> dict[str, dict[str, np.ndarray]]:
        """
        Returns the parameters of the module, which
        should be optimized with gradient descent.

        Returns:
            dict[str, dict[str, np.ndarray]]:

                module_name (str):
                    The name of the module

                module (dict[str, np.ndarray]):

                    key (str):
                        The name of the tunable parameter

                    value (np.ndarray):
                        The value of the tunable parameter
        """
        return {"module_0": {}}

    def gradients(self, **kwargs) -> dict[str, dict[str, np.ndarray]]:
        """
        Returns the gradients of each parameter of the
        module, which should be optimized with gradient
        descent.

        Returns:
            dict[str, dict[str, np.ndarray]]:

                module_name (str):
                    The name of the module

                module (dict[str, np.ndarray]):

                    key (str):
                        The name of the tunable parameter

                    value (np.ndarray):
                        The gradients dL/dtheta of the tunable parameter
        """
        return {"module_0": {}}
