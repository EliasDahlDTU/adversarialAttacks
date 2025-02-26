"""
Data loading utilities for benchmark datasets.
"""
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from typing import Tuple, Dict

def get_mnist_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_dir: str = "data/raw"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get MNIST training and test data loaders.
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of workers for data loading
        data_dir: Directory to store the data
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST mean and std
    ])
    
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def get_cifar10_loaders(
    batch_size: int = 128,
    num_workers: int = 2,
    data_dir: str = "data/raw"
) -> Tuple[DataLoader, DataLoader]:
    """
    Get CIFAR-10 training and test data loaders.
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of workers for data loading
        data_dir: Directory to store the data
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=True,
        transform=transform_train,
        download=True
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_dir,
        train=False,
        transform=transform_test,
        download=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    return train_loader, test_loader

def get_dataset_info() -> Dict:
    """
    Get information about the datasets.
    
    Returns:
        dict: Dictionary containing dataset information
    """
    return {
        "mnist": {
            "input_channels": 1,
            "input_size": 28,
            "num_classes": 10,
            "classes": list(range(10))
        },
        "cifar10": {
            "input_channels": 3,
            "input_size": 32,
            "num_classes": 10,
            "classes": [
                'airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck'
            ]
        }
    } 