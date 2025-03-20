import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image
import numpy as np

class ImageDataset(Dataset):
    """Custom Dataset for loading preprocessed images."""
    def __init__(self, data_dir, transform=None, split='train'):
        """
        Args:
            data_dir (str): Path to data directory
            transform (callable, optional): Optional transform to be applied on an image
            split (str): 'train' or 'val' split
        """
        self.data_dir = os.path.join(data_dir, split)
        self.transform = transform
        self.classes = sorted(os.listdir(self.data_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        # Load all image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
                
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_data_transforms():
    """Get minimal transforms for already preprocessed images."""
    # Just convert to tensor, since images are already preprocessed
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    return transform, transform

def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir (str): Path to data directory containing 'train' and 'val' subdirectories
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        
    Returns:
        dict: Dictionary containing 'train' and 'val' dataloaders and dataset sizes
    """
    transform, _ = get_data_transforms()  # Same transform for both train and val
    
    # Create datasets
    train_dataset = ImageDataset(data_dir, transform=transform, split='train')
    val_dataset = ImageDataset(data_dir, transform=transform, split='val')
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Still shuffle to randomize training order
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataloaders = {
        'train': train_loader,
        'val': val_loader
    }
    
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset)
    }
    
    class_names = train_dataset.classes
    
    return dataloaders, dataset_sizes, class_names

if __name__ == "__main__":
    # Test the dataloaders
    data_dir = "data/processed"  # Adjust this path as needed
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir)
    
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    print(f"Training set size: {dataset_sizes['train']}")
    print(f"Validation set size: {dataset_sizes['val']}")
    
    # Test a batch
    for images, labels in dataloaders['train']:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break 