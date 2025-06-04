"""
Implementation of the Carlini & Wagner (CW) L2 attack.
Reference: Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (2017)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .base_attack import BaseAttack

class CW(BaseAttack):
    """
    Carlini & Wagner (CW) L2 Attack.
    Optimization-based attack to find minimal perturbations that cause misclassification.
    """

    def __init__(self,
                 model: nn.Module,
                 c: float = 1.0,
                 kappa: float = 0.0,
                 max_iter: int = 1000,
                 lr: float = 0.01,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize CW attack.

        Args:
            model (nn.Module): Model to attack
            c (float): Balancing constant between distance and loss
            kappa (float): Confidence of misclassification
            max_iter (int): Maximum optimization steps
            lr (float): Learning rate for optimizer
            device (str): Computation device
        """
        super().__init__(model, epsilon=None, device=device)
        self.c = c
        self.kappa = kappa
        self.max_iter = max_iter
        self.lr = lr

    def _loss_function(self, logits, target, one_hot_target, targeted):
        """
        Compute the CW loss function.
        """
        real = torch.sum(one_hot_target * logits, dim=1)
        other = torch.max((1 - one_hot_target) * logits - 1e4 * one_hot_target, dim=1)[0]
        if targeted:
            loss = torch.clamp(other - real + self.kappa, min=0)
        else:
            loss = torch.clamp(real - other + self.kappa, min=0)
        return loss

    def generate(self,
                 x: torch.Tensor,
                 y: torch.Tensor,
                 targeted: bool = False) -> torch.Tensor:
        """
        Generate adversarial examples using CW L2 attack.

        Args:
            x (Tensor): Original input
            y (Tensor): Target labels
            targeted (bool): Whether this is a targeted attack

        Returns:
            Tensor: Adversarial examples
        """
        x = x.to(self.device)
        y = y.to(self.device)

        # Define variables
        x_adv = x.clone().detach()
        w = torch.atanh((x_adv * 1.999999 - 1))  # inverse tanh
        w = w.clone().detach().requires_grad_(True)

        optimizer = optim.Adam([w], lr=self.lr)
        one_hot_target = torch.nn.functional.one_hot(y, num_classes=self.model(x).shape[1]).float()

        for _ in range(self.max_iter):
            x_adv = 0.5 * (torch.tanh(w) + 1)
            outputs = self.model(x_adv)

            l2_dist = torch.sum((x_adv - x) ** 2, dim=[1, 2, 3])
            f_loss = self._loss_function(outputs, y, one_hot_target, targeted)
            total_loss = l2_dist + self.c * f_loss

            optimizer.zero_grad()
            total_loss.sum().backward()
            optimizer.step()

        x_adv = 0.5 * (torch.tanh(w) + 1)
        return torch.clamp(x_adv, 0, 1)
