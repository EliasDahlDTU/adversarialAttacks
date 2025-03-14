"""
Dataset handling for ImageNet-100.
"""

import os
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class ImageNet100Dataset(Dataset):
    """ImageNet-100 dataset."""
    
    def __init__(self, 
                 root_dir: str, 
                 split: str = 'train',
                 transform: Optional[transforms.Compose] = None):
        """
        Initialize the ImageNet-100 dataset.
        
        Args:
            root_dir (str): Root directory of the processed dataset
            split (str): 'train' or 'val'
            transform (transforms.Compose, optional): Transformations to apply
        """
        self.root_dir = Path(root_dir) / split
        self.split = split
        self.transform = transform
        
        # Get class folders
        self.class_folders = sorted([d for d in self.root_dir.glob("*") if d.is_dir()])
        
        # Create class to index mapping
        self.class_to_idx = {folder.name: i for i, folder in enumerate(self.class_folders)}
        
        # Get all image paths and their labels
        self.samples = []
        for class_folder in self.class_folders:
            class_idx = self.class_to_idx[class_folder.name]
            for img_path in class_folder.glob("*.JPEG"):
                self.samples.append((str(img_path), class_idx))
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (image, label)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            img = self.transform(img)
        
        return img, label
    
    def get_class_names(self) -> List[str]:
        """Get the list of class names."""
        return [folder.name for folder in self.class_folders]


def get_data_loaders(
    data_dir: str = 'data/processed',
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create data loaders for training and validation.
    
    Args:
        data_dir (str): Directory containing the processed dataset
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        
    Returns:
        tuple: (train_loader, val_loader)
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = ImageNet100Dataset(
        root_dir=data_dir,
        split='train',
        transform=train_transform
    )
    
    val_dataset = ImageNet100Dataset(
        root_dir=data_dir,
        split='val',
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader 