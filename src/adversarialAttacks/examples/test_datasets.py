"""
Example script to test dataset loading.
"""
import torch
from adversarialAttacks.datasets.data_loader import (
    get_mnist_loaders,
    get_cifar10_loaders,
    get_dataset_info
)
import matplotlib.pyplot as plt
import numpy as np

def show_batch(images, labels, dataset_name):
    """Display a batch of images with their labels."""
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.ravel()
    
    # Get dataset info for proper denormalization
    info = get_dataset_info()[dataset_name]
    
    for idx, (img, label) in enumerate(zip(images[:10], labels[:10])):
        if dataset_name == "mnist":
            # Denormalize MNIST
            img = img * 0.3081 + 0.1307
        else:
            # Denormalize CIFAR-10
            img = img * torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1) + \
                  torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        
        img = img.numpy()
        if info["input_channels"] == 1:
            img = img.squeeze()
            axes[idx].imshow(img, cmap='gray')
        else:
            img = np.transpose(img, (1, 2, 0))
            axes[idx].imshow(img)
        
        if isinstance(info["classes"][0], str):
            label_text = info["classes"][label]
        else:
            label_text = str(label.item())
            
        axes[idx].set_title(f'Label: {label_text}')
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Test MNIST loading
    mnist_train, mnist_test = get_mnist_loaders(batch_size=32)
    print(f"MNIST Dataset:")
    print(f"Training samples: {len(mnist_train.dataset)}")
    print(f"Test samples: {len(mnist_test.dataset)}")
    
    # Get a batch of MNIST images
    images, labels = next(iter(mnist_train))
    print("\nMNIST batch shape:", images.shape)
    show_batch(images, labels, "mnist")
    
    # Test CIFAR-10 loading
    cifar_train, cifar_test = get_cifar10_loaders(batch_size=32)
    print(f"\nCIFAR-10 Dataset:")
    print(f"Training samples: {len(cifar_train.dataset)}")
    print(f"Test samples: {len(cifar_test.dataset)}")
    
    # Get a batch of CIFAR-10 images
    images, labels = next(iter(cifar_train))
    print("\nCIFAR-10 batch shape:", images.shape)
    show_batch(images, labels, "cifar10")

if __name__ == "__main__":
    main() 