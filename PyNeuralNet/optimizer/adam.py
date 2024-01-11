import numpy as np

from PyNeuralNet.module import Module
from PyNeuralNet.optimizer.abstract_optimizer import Optimizer


class Adam(Optimizer):
    """
    Adam: A Method for Stochastic Optimization optimizer.

    The Adam optimizer optimizes the parameters of each module by adapting the learning rate of the update rule with
    the average and square gradients (first and second moment).

    The implementation of the optimizer is based on the implementation of PyTorch:
    https://pytorch.org/docs/stable/generated/torch.optim.Adam.html

    Args:
        model (Module):
            The module where we want to optimize the parameters.

        lr (float):
            The learning rate of the update rule

        beta_1 (float, optional):
            The coefficient of the first moment

        beta_2 (float, optional):
            The coefficient of the second moment

        eps (float, optional):
            The added noise in the update rule

        weight_decay (float, optional):
            The weight of the L2 penalty
    """
    def __init__(
            self,
            model: Module,
            lr: float,
            beta_1: float = 0.9,
            beta_2: float = 0.999,
            eps: float = 1e-08,
            weight_decay: float = 0.0,
    ):
        if lr < 0.0 or lr > 1.0:
            raise ValueError(f"Illegal lr {lr}. The learning rate should be in range [0.0, 1.0]!")
        if beta_1 < 0.0 or beta_1 > 1.0:
            raise ValueError(f"Illegal beta_1 {beta_1}. The beta_1 should be in range [0.0, 1.0]!")
        if beta_2 < 0.0 or beta_2 > 1.0:
            raise ValueError(f"Illegal beta_2 {beta_2}. The beta_2 should be in range [0.0, 1.0]!")
        if weight_decay < 0.0 or weight_decay > 1.0:
            raise ValueError(f"Illegal weight_decay {weight_decay}. The weight decay should be in range [0.0, 1.0]!")

        self.model = model
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eps = eps
        self.weight_decay = weight_decay

        self.m_t = {}  # first moment
        self.v_t = {}  # second moment
        self.timestep = 1

    def step(self):
        params = self.model.parameters()
        grads = self.model.gradients()

        # Iterate over all modules
        for i, (param_module_key, grad_module_key) in enumerate(zip(params, grads)):
            # Iterate over all trainable parameters in each module
            param_module, grad_module = params[param_module_key], grads[grad_module_key]
            for param_key, grad_key in zip(param_module, grad_module):
                g_t = grad_module[grad_key]  # gradients of shape (*THETA_SHAPE)
                theta_t_old = param_module[param_key]  # the parameter itself of shape (*THETA_SHAPE)
                key = (param_module_key, param_key)

                if self.weight_decay:
                    # Case: Append L2 Penalty to gt (+ weight_decay * theta)
                    g_t += self.weight_decay * theta_t_old

                # Calculate the first moment
                if key in self.m_t:
                    # Case: Calculate the next first moment
                    self.m_t[key] = (
                            self.beta_1 * self.m_t[key] + (1 - self.beta_1) * g_t
                    )
                else:
                    # Case: First time to calculate the first moment
                    self.m_t[key] = (
                            (1 - self.beta_1) * g_t
                    )

                # Calculate the second moment
                if key in self.v_t:
                    # Case: Calculate the next second moment
                    self.v_t[key] = (
                            self.beta_2 * self.v_t[key] + (1 - self.beta_2) * g_t ** 2
                    )
                else:
                    # Case: First time to calculate the second moment
                    self.v_t[key] = (
                            (1 - self.beta_2) * g_t ** 2
                    )

                # Calculate the bias shifted first and second moment
                m_t_hat = self.m_t[key] / (1 - (self.beta_1 ** self.timestep))
                v_t_hat = self.v_t[key] / (1 - (self.beta_2 ** self.timestep))

                # Perform the update rule theta_new := theta_old - lr * m_t_hat / (sqrt(v_t_hat) + eps)
                param_module[param_key] -= self.lr * (m_t_hat / np.sqrt(v_t_hat) + self.eps)

        # Update the timestep
        self.timestep += 1

        # Update parameters of each module
        self.model.update(params)
