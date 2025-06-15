"""
Training utilities for models.
"""

from .train import train_model
from .fast_train import fast_train_model, get_fast_dataloaders

__all__ = [
    'train_model',
    'fast_train_model',
    'get_fast_dataloaders'
] 