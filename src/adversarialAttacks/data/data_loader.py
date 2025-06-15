import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from typing import Tuple, Dict, List, Optional

class ImageDataset(Dataset):
    """Custom Dataset for loading preprocessed images."""
    def __init__(self, data_dir: str, transform: Optional[callable] = None, split: str = 'train'):
        """
        Args:
            data_dir (str): Path to data directory
            transform (callable, optional): Optional transform to be applied on an image
            split (str): 'train', 'val', or 'test' split
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        
        # Get class names from directory structure
        try:
            self.classes = sorted([d for d in os.listdir(self.data_dir) 
                                 if os.path.isdir(os.path.join(self.data_dir, d))])
        except FileNotFoundError:
            raise FileNotFoundError(f"Directory not found: {self.data_dir}")
        
        if not self.classes:
            raise ValueError(f"No class directories found in {self.data_dir}")
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Initialize lists to store paths and labels
        self.images: List[str] = []
        self.labels: List[int] = []
        
        # Load all image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])
        
        # Convert labels to numpy array with explicit dtype
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Args:
            idx (int): Index of the item to get
            
        Returns:
            tuple: (image, label) where image is a tensor and label is an integer
        """
        # Handle negative indexing
        if idx < 0:
            idx = len(self) + idx
            
        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} is out of bounds")
            
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Failed to load image {img_path}: {str(e)}")
            
        label = int(self.labels[idx])  # Explicit conversion to int
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    """Get minimal transforms for already preprocessed images."""
    # Just convert to tensor, since images are already preprocessed
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform, transform

def get_dataloaders(
    data_dir: str, 
    batch_size: int = 32, 
    num_workers: int = 4
) -> Tuple[Dict[str, DataLoader], Dict[str, int], List[str]]:
    """
    Create train, validation and test dataloaders.
    
    Args:
        data_dir (str): Path to data directory containing 'train', 'val', and 'test' subdirectories
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        Tuple containing:
            - Dict with 'train', 'val', and 'test' dataloaders
            - Dict with dataset sizes
            - List of class names
    """
    transform, _ = get_data_transforms()  # Same transform for all splits
    
    # Create datasets
    try:
        train_dataset = ImageDataset(data_dir, transform=transform, split='train')
        val_dataset = ImageDataset(data_dir, transform=transform, split='val')
        test_dataset = ImageDataset(data_dir, transform=transform, split='test')
    except Exception as e:
        raise RuntimeError(f"Failed to create datasets: {str(e)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }
    
    return dataloaders, dataset_sizes, train_dataset.classes

if __name__ == "__main__":
    # Test the dataloaders
    data_dir = "data/processed"  # Adjust this path as needed
    
    try:
        dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir)
        
        print(f"Number of classes: {len(class_names)}")
        print(f"Class names: {class_names}")
        print(f"Training set size: {dataset_sizes['train']}")
        print(f"Validation set size: {dataset_sizes['val']}")
        print(f"Test set size: {dataset_sizes['test']}")
        
        # Test a batch
        for images, labels in dataloaders['train']:
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Labels dtype: {labels.dtype}")  # Check label data type
            break
            
    except Exception as e:
        print(f"Error during testing: {str(e)}") 