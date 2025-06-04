"""
Implementation of the Projected Gradient Descent (PGD) attack.
Reference: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2018)
"""

import torch
import torch.nn as nn
from .base_attack import BaseAttack

class PGD(BaseAttack):
    """
    Projected Gradient Descent (PGD) attack.
    An iterative gradient-based attack that projects adversarial examples
    back into the epsilon-ball after each perturbation step.
    """

    def __init__(self,
                 model: nn.Module,
                 epsilon: float = 0.3,
                 alpha: float = 0.01,
                 num_steps: int = 40,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize PGD attack.

        Args:
            model (nn.Module): Model to attack
            epsilon (float): Maximum perturbation
            alpha (float): Step size per iteration
            num_steps (int): Number of PGD steps
            device (str): Device to use for computations
        """
        super().__init__(model, epsilon, device)
        self.alpha = alpha
        self.num_steps = num_steps

    def generate(self,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 targeted: bool = False) -> torch.Tensor:
        """
        Generate adversarial examples using PGD.

        Args:
            x (Tensor): Input images
            y (Tensor): Target labels
            targeted (bool): Whether to perform a targeted attack

        Returns:
            Tensor: Adversarial examples
        """
        x = x.to(self.device)
        y = y.to(self.device)
        x_adv = x.clone().detach().requires_grad_(True)

        for _ in range(self.num_steps):
            outputs = self.model(x_adv)
            loss = self._compute_loss(outputs, y, targeted)

            self.model.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.zero_()
            loss.backward()

            with torch.no_grad():
                grad_sign = x_adv.grad.sign()
                if targeted:
                    grad_sign = -grad_sign
                x_adv = x_adv + self.alpha * grad_sign
                x_adv = self._clip_perturbation(x, x_adv)
                x_adv = torch.clamp(x_adv, 0, 1)
                x_adv.requires_grad_()

        return x_adv
