"""
Implementation of the Fast Gradient Sign Method (FGSM) attack.
Reference: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2015)
"""

import torch
import torch.nn as nn
from .base_attack import BaseAttack

class FGSM(BaseAttack):
    """
    Fast Gradient Sign Method (FGSM) attack.
    A simple one-step gradient-based attack that perturbs the input
    in the direction of the gradient of the loss with respect to the input.
    """
    
    def __init__(self,
                 model: nn.Module,
                 epsilon: float = 0.3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize FGSM attack.
        
        Args:
            model (nn.Module): Model to attack
            epsilon (float): Maximum perturbation
            device (str): Device to use for computations
        """
        super().__init__(model, epsilon, device)
    
    def generate(self,
                x: torch.Tensor,
                y: torch.Tensor,
                targeted: bool = False) -> torch.Tensor:
        """
        Generate adversarial examples using FGSM.
        
        Args:
            x (Tensor): Input images
            y (Tensor): Target labels
            targeted (bool): Whether to perform a targeted attack
            
        Returns:
            Tensor: Adversarial examples
        """
        # Move inputs to device
        x = x.to(self.device)
        y = y.to(self.device)
        
        # Get gradient sign
        grad_sign = self._get_grad_sign(x, y, targeted)
        
        # Generate adversarial examples
        x_adv = x + self.epsilon * grad_sign
        
        # Clip to valid image range
        x_adv = torch.clamp(x_adv, 0, 1)
        
        return x_adv 