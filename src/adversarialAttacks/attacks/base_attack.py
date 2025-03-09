"""
Base class for adversarial attacks.
Implements common functionality for all attack methods.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union
import numpy as np

class BaseAttack:
    """Base class for adversarial attacks."""
    
    def __init__(self,
                 model: nn.Module,
                 epsilon: float = 0.3,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the attack.
        
        Args:
            model (nn.Module): Model to attack
            epsilon (float): Maximum perturbation
            device (str): Device to use for computations
        """
        self.model = model
        self.epsilon = epsilon
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def generate(self,
                x: torch.Tensor,
                y: torch.Tensor,
                targeted: bool = False) -> torch.Tensor:
        """
        Generate adversarial examples.
        
        Args:
            x (Tensor): Input images
            y (Tensor): Target labels
            targeted (bool): Whether to perform a targeted attack
            
        Returns:
            Tensor: Adversarial examples
        """
        raise NotImplementedError("Subclasses must implement generate()")
    
    def _clip_perturbation(self,
                          x: torch.Tensor,
                          x_adv: torch.Tensor) -> torch.Tensor:
        """
        Clip perturbation to epsilon constraint.
        
        Args:
            x (Tensor): Original images
            x_adv (Tensor): Adversarial examples
            
        Returns:
            Tensor: Clipped adversarial examples
        """
        perturbation = x_adv - x
        perturbation = torch.clamp(perturbation, -self.epsilon, self.epsilon)
        return torch.clamp(x + perturbation, 0, 1)
    
    def _compute_loss(self,
                     outputs: torch.Tensor,
                     targets: torch.Tensor,
                     targeted: bool = False) -> torch.Tensor:
        """
        Compute loss for gradient computation.
        
        Args:
            outputs (Tensor): Model outputs
            targets (Tensor): Target labels
            targeted (bool): Whether this is a targeted attack
            
        Returns:
            Tensor: Loss value
        """
        if targeted:
            return -nn.functional.cross_entropy(outputs, targets)
        return nn.functional.cross_entropy(outputs, targets)
    
    def _get_grad_sign(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      targeted: bool = False) -> torch.Tensor:
        """
        Get gradient sign for FGSM-like attacks.
        
        Args:
            x (Tensor): Input images
            y (Tensor): Target labels
            targeted (bool): Whether this is a targeted attack
            
        Returns:
            Tensor: Gradient sign
        """
        x.requires_grad = True
        outputs = self.model(x)
        loss = self._compute_loss(outputs, y, targeted)
        loss.backward()
        return x.grad.sign()
    
    def evaluate_attack(self,
                       dataloader: torch.utils.data.DataLoader,
                       num_examples: Optional[int] = None) -> Tuple[float, float, float]:
        """
        Evaluate attack success rate and model accuracy.
        
        Args:
            dataloader (DataLoader): Test data loader
            num_examples (int, optional): Number of examples to evaluate on
            
        Returns:
            tuple: (clean_accuracy, adversarial_accuracy, attack_success_rate)
        """
        self.model.eval()
        total = 0
        clean_correct = 0
        adv_correct = 0
        successful_attacks = 0
        
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                if num_examples and total >= num_examples:
                    break
                    
                batch_size = x.size(0)
                total += batch_size
                
                x, y = x.to(self.device), y.to(self.device)
                
                # Clean accuracy
                clean_outputs = self.model(x)
                clean_pred = clean_outputs.argmax(dim=1)
                clean_correct += (clean_pred == y).sum().item()
                
                # Generate adversarial examples
                x_adv = self.generate(x, y)
                
                # Adversarial accuracy
                adv_outputs = self.model(x_adv)
                adv_pred = adv_outputs.argmax(dim=1)
                adv_correct += (adv_pred == y).sum().item()
                
                # Attack success rate (misclassification rate)
                successful_attacks += (adv_pred != y).sum().item()
        
        clean_accuracy = clean_correct / total
        adv_accuracy = adv_correct / total
        attack_success_rate = successful_attacks / total
        
        return clean_accuracy, adv_accuracy, attack_success_rate
    
    def compute_perturbation_stats(self,
                                 x: torch.Tensor,
                                 x_adv: torch.Tensor) -> dict:
        """
        Compute statistics about the perturbation.
        
        Args:
            x (Tensor): Original images
            x_adv (Tensor): Adversarial examples
            
        Returns:
            dict: Dictionary containing perturbation statistics
        """
        perturbation = (x_adv - x).abs()
        
        return {
            'mean': perturbation.mean().item(),
            'std': perturbation.std().item(),
            'min': perturbation.min().item(),
            'max': perturbation.max().item(),
            'l2_norm': torch.norm(perturbation.view(perturbation.shape[0], -1), p=2, dim=1).mean().item(),
            'linf_norm': perturbation.view(perturbation.shape[0], -1).max(dim=1)[0].mean().item()
        } 