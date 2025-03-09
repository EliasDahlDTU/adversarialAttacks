"""
Base model architecture for adversarial attack experiments.
Implements common functionality for all models.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Dict, Any, Optional

class BaseModel(nn.Module):
    """Base model class with common functionality."""
    
    def __init__(self, 
                 model_name: str = 'resnet50',
                 num_classes: int = 100,
                 pretrained: bool = True):
        """
        Initialize the model.
        
        Args:
            model_name (str): Name of the model architecture
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
        """
        super().__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load pretrained model
        self.model = self._get_model(model_name, pretrained)
        
        # Modify final layer for our number of classes
        self._modify_final_layer()
        
    def _get_model(self, model_name: str, pretrained: bool) -> nn.Module:
        """Get the specified model architecture."""
        if model_name == 'resnet50':
            return models.resnet50(pretrained=pretrained)
        elif model_name == 'vgg16':
            return models.vgg16(pretrained=pretrained)
        elif model_name == 'densenet121':
            return models.densenet121(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _modify_final_layer(self):
        """Modify the final layer to match our number of classes."""
        if self.model_name.startswith('resnet'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, self.num_classes)
        elif self.model_name.startswith('vgg'):
            in_features = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Linear(in_features, self.num_classes)
        elif self.model_name.startswith('densenet'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, self.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.model(x)
    
    def save_checkpoint(self, 
                       path: str,
                       epoch: int,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       metrics: Optional[Dict[str, Any]] = None):
        """
        Save model checkpoint.
        
        Args:
            path (str): Path to save checkpoint
            epoch (int): Current epoch
            optimizer (Optimizer): Optimizer state
            scheduler (LRScheduler, optional): Learning rate scheduler state
            metrics (dict, optional): Training metrics to save
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_name': self.model_name,
            'num_classes': self.num_classes
        }
        
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
            
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, 
                       path: str,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            path (str): Path to checkpoint
            optimizer (Optimizer, optional): Optimizer to load state into
            scheduler (LRScheduler, optional): Learning rate scheduler to load state into
            
        Returns:
            dict: Checkpoint contents
        """
        checkpoint = torch.load(path)
        
        # Verify model architecture matches
        if checkpoint['model_name'] != self.model_name:
            raise ValueError(f"Checkpoint model ({checkpoint['model_name']}) "
                           f"does not match current model ({self.model_name})")
        
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        return checkpoint 