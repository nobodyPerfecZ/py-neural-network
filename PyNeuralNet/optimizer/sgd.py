from PyNeuralNet.module import Module
from PyNeuralNet.optimizer.abstract_optimizer import Optimizer


class SGD(Optimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer.

    The SGD optimizer optimizes the parameters of each module by taking a random batch of samples, computes their
    gradients and apply the update rule to each learnable parameter.

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
            The weight of the L2 penalty

        nesterov (bool, optional):
            Controls if Nesterov momentum is enabled
    """

    def __init__(
            self,
            model: Module,
            lr: float,
            momentum: float = 0.0,
            dampening: float = 0.0,
            weight_decay: float = 0.0,
            nesterov: bool = False,
    ):
        if lr < 0.0 or lr > 1.0:
            raise ValueError(f"Illegal lr {lr}. The learning rate should be in range [0.0, 1.0]!")
        if momentum < 0.0 or momentum > 1.0:
            raise ValueError(f"Illegal momentum {momentum}. The momentum should be in range [0.0, 1.0]!")
        if dampening < 0.0 or dampening > 1.0:
            raise ValueError(f"Illegal dampening {dampening}. The dampening should be in range [0.0, 1.0]!")
        if weight_decay < 0.0 or weight_decay > 1.0:
            raise ValueError(f"Illegal weight_decay {weight_decay}. The weight decay should be in range [0.0, 1.0]!")

        self.model = model
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay
        self.nesterov = nesterov

        self.b_t = {}  # Momentum of the previous iteration

    def step(self):
        params = self.model.parameters()
        grads = self.model.gradients()

        # Iterate over all modules
        for i, (param_module_key, grad_module_key) in enumerate(zip(params, grads)):
            # Iterate over all trainable parameters in each module
            param_module, grad_module = params[param_module_key], grads[grad_module_key]
            for param_key, grad_key in zip(param_module, grad_module):
                g_t = grad_module[grad_key]  # (aggregated) gradients of shape (*THETA_SHAPE)
                theta_t_old = param_module[param_key]  # the parameter itself of shape (*THETA_SHAPE)
                key = (param_module_key, param_key)

                if self.weight_decay:
                    # Case: Append L2 Penalty to gt (+ weight_decay * theta)
                    g_t += self.weight_decay * theta_t_old

                if self.momentum:
                    # Case: Use momentum for gt
                    if key in self.b_t:
                        # Case: At least one optimization iteration is gone
                        self.b_t[key] = (
                                self.momentum * self.b_t[key] + (1 - self.dampening) * g_t
                        )
                    else:
                        # Case: Assign the first bt with the gradient gt
                        self.b_t[key] = g_t
                    if self.nesterov:
                        g_t += self.momentum * self.b_t[key]
                    else:
                        g_t = self.b_t[key]

                # Perform the update rule theta_new := theta_old - lr * g_t
                param_module[param_key] -= self.lr * g_t

        # Update parameters of each module
        self.model.update(params)
