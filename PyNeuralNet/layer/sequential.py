import functools

import numpy as np

from PyNeuralNet.module import Module


class Sequential(Module):
    """
    A sequential module that applies a sequence of submodules in a linear order.

    This class represents the formal definition of a multi-layer neural network.

    Args:
        *modules (Sequence[Module]):
            Variable number of submodules to be added to the sequential container.
    """

    def __init__(self, *modules):
        self.modules = list(modules)

    def forward(self, X: np.ndarray, **kwargs) -> np.ndarray:
        return functools.reduce(lambda x, module: module.forward(x), self.modules, X)

    def backward(self, dL_dZ: np.ndarray, **kwargs) -> np.ndarray:
        return functools.reduce(lambda dl_dz, module: module.backward(dl_dz), reversed(self.modules), dL_dZ)

    def update(self, params: dict[str, dict[str, np.ndarray]]):
        map(lambda module, module_item: module.update({module_item[0]: module_item[1]}), self.modules, params.items())

    def parameters(self, **kwargs) -> dict[str, dict[str, np.ndarray]]:
        return {f"module_{i}": module.parameters()["module_0"] for i, module in enumerate(self.modules)}

    def gradients(self, **kwargs) -> dict[str, dict[str, np.ndarray]]:
        return {f"module_{i}": module.gradients()["module_0"] for i, module in enumerate(self.modules)}
