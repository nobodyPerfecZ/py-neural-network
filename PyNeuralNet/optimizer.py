from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from PyNeuralNet.module import Module


# TODO: Implement more optimizers (adam, ...)

class Optimizer(ABC):
    """
    A base class for optimization algorithms used for training neural networks.
    """

    @abstractmethod
    def step(self):
        """
        Perform a single optimization step.

        This method updates the parameters based on the computed gradients.
        """
        pass


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    The SGD optimizer optimizes the parameters of each module by a single randomly taken sample.

    The implementation of the optimizer is based on the implementation of PyTorch:
    https://pytorch.org/docs/stable/generated/torch.optim.SGD.html

    Args:
        model (Module):
            The module where we want to optimize the parameters.

        lr (float):
            The learning rate of the update rule

        momentum (float, optional):
            The factor of the momentum

        dampening (float, optional):
            The factor of dampening the current gradient

        weight_decay (float, optional):
            The weight of the L2 penality

        random_state (int, optional):
            The seed of the random number generator
    """

    def __init__(
            self,
            model: Module,
            lr: float,
            momentum: float = 0.0,
            dampening: float = 0.0,
            weight_decay: float = 0.0,
            random_state: Optional[int] = None,
    ):
        if lr < 0.0 or lr > 1.0:
            raise ValueError(f"Illegal lr {lr}. The learning rate should be in range [0.0, 1.0]!")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Illegal momentum {momentum}. The momentum should be in range [0.0, 1.0]!")
        if dampening < 0.0 or dampening > 1.0:
            raise ValueError(f"Illegal dampening {dampening}. The dampening should be in range [0.0, 1.0]!")
        if weight_decay < 0.0 or weight_decay > 1.0:
            raise ValueError(f"Illegal weight_decay {weight_decay}. The weight decay should be in range [0.0, 1.0]!")
        if random_state is not None and random_state < 0:
            raise ValueError(f"Illegal random_state {random_state}. The random_state should be in range [0, +inf]!")

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.dampening = dampening
        self.rng = np.random.RandomState(random_state)

        self.b_t = {}  # Momentum of the previous iteration

    def step(self):
        params = self.model.parameters()
        grads = self.model.gradients()
        index = None

        # Iterate over all modules
        for i, (param_module_key, grad_module_key) in enumerate(zip(params, grads)):
            # Iterate over all trainable parameters in each module
            param_module, grad_module = params[param_module_key], grads[grad_module_key]
            for param_key, grad_key in zip(param_module, grad_module):
                grad = grad_module[grad_key]  # gradients of size (NUM_BATCHES, *THETA_SHAPE)
                theta_t_old = param_module[param_key]  # the parameter itself of shape (*THETA_SHAPE)

                # Get the gradient
                if index is None:
                    # Case: Randomly sample which gradient should be chosen
                    batch_size = len(grad)
                    index = self.rng.randint(low=0, high=batch_size)
                g_t = grad[index]

                if self.weight_decay:
                    # Case: Append L2 Penalty to gt (+ weight_decay * theta)
                    g_t += self.weight_decay * theta_t_old

                if self.momentum:
                    # Case: Use momentum for gt
                    if (param_module_key, param_key) in self.b_t:
                        # Case: At least one optimization iteration is gone
                        self.b_t[(param_module_key, param_key)] = (
                                self.momentum * self.b_t[(param_module_key, param_key)] + (1 - self.dampening) * g_t
                        )
                    else:
                        # Case: Assign the first bt with the gradient gt
                        self.b_t[(param_module_key, param_key)] = g_t
                    g_t = self.b_t[(param_module_key, param_key)]

                # Perform the update rule theta_new := theta_old - lr * g_t
                param_module[param_key] -= self.lr * g_t

        # Update parameters of each module
        self.model.update(params)


