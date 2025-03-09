"""
Script to prepare ImageNet dataset for adversarial attack experiments.
Downloads and processes a subset of ImageNet for our experiments.
"""

import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import random
import numpy as np
from tqdm import tqdm

def set_random_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def get_imagenet_transforms():
    """Get standard ImageNet transforms."""
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
    }

def prepare_imagenet(root='./data', num_classes=100, samples_per_class=50):
    """
    Prepare ImageNet dataset for experiments.
    
    Args:
        root (str): Root directory for data
        num_classes (int): Number of classes to use
        samples_per_class (int): Number of samples per class
    """
    set_random_seed(42)
    
    # Create directories
    raw_dir = os.path.join(root, 'raw')
    processed_dir = os.path.join(root, 'processed')
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # Get transforms
    transforms_dict = get_imagenet_transforms()
    
    # Download ImageNet (this requires manual download due to registration)
    print("Please ensure you have downloaded ImageNet to the raw directory")
    print("You can register and download from: https://image-net.org/download.php")
    
    # Load dataset
    try:
        trainset = torchvision.datasets.ImageNet(
            root=raw_dir,
            split='train',
            transform=transforms_dict['train']
        )
        valset = torchvision.datasets.ImageNet(
            root=raw_dir,
            split='val',
            transform=transforms_dict['val']
        )
    except Exception as e:
        print(f"Error loading ImageNet: {e}")
        print("Please ensure you have downloaded ImageNet to the raw directory")
        return
    
    # Select subset of classes
    all_classes = list(range(len(trainset.classes)))
    selected_classes = random.sample(all_classes, num_classes)
    
    # Create class mapping
    class_to_idx = {cls: idx for idx, cls in enumerate(selected_classes)}
    
    # Select samples for each class
    train_indices = []
    val_indices = []
    
    print("Selecting samples for each class...")
    for cls in tqdm(selected_classes):
        # Get indices for this class
        train_cls_indices = [i for i, (_, label) in enumerate(trainset) if label == cls]
        val_cls_indices = [i for i, (_, label) in enumerate(valset) if label == cls]
        
        # Sample from each
        train_cls_samples = random.sample(train_cls_indices, samples_per_class)
        val_cls_samples = random.sample(val_cls_indices, samples_per_class // 5)  # 20% for validation
        
        train_indices.extend(train_cls_samples)
        val_indices.extend(val_cls_samples)
    
    # Create subsets
    train_subset = Subset(trainset, train_indices)
    val_subset = Subset(valset, val_indices)
    
    # Save subset indices
    torch.save({
        'train_indices': train_indices,
        'val_indices': val_indices,
        'class_mapping': class_to_idx,
        'selected_classes': selected_classes
    }, os.path.join(processed_dir, 'dataset_info.pt'))
    
    print(f"Dataset prepared with {num_classes} classes and {samples_per_class} samples per class")
    print(f"Training samples: {len(train_indices)}")
    print(f"Validation samples: {len(val_indices)}")

if __name__ == '__main__':
    prepare_imagenet() 