from abc import ABC, abstractmethod


class Optimizer(ABC):
    """
    A base class for optimization algorithms used for training parameters of a module.
    """

    @abstractmethod
    def step(self):
        """
        Perform a single optimization step.

        This method updates the parameters based on the computed gradients.
        """
        pass
