"""
Fast training script for proof-of-concept.
Optimized for speed with GPU, Adam optimizer, and progress updates.
NOW WITH REDUCED DATASET: 5k train + 1k val samples!
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import random
import numpy as np

from models import get_model
from data_loader import get_dataloaders, ImageDataset

def get_fast_dataloaders(data_dir, batch_size=32, train_samples=5000, val_samples=1000, num_workers=0):
    """
    Create FAST dataloaders with limited samples for proof of concept.
    """
    from data_loader import get_data_transforms
    
    # Get transforms
    transform, _ = get_data_transforms()
    
    # Create full datasets
    train_dataset = ImageDataset(data_dir, transform=transform, split='train')
    val_dataset = ImageDataset(data_dir, transform=transform, split='val')
    
    # Get class names
    class_names = train_dataset.classes
    
    # Create random subsets
    train_indices = random.sample(range(len(train_dataset)), min(train_samples, len(train_dataset)))
    val_indices = random.sample(range(len(val_dataset)), min(val_samples, len(val_dataset)))
    
    # Create subset datasets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    dataloaders = {'train': train_loader, 'val': val_loader}
    dataset_sizes = {'train': len(train_subset), 'val': len(val_subset)}
    
    return dataloaders, dataset_sizes, class_names

def fast_train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=5, device='cuda'):
    """
    Fast training with progress bars and frequent updates.
    """
    since = time.time()
    best_acc = 0.0

    # Create directory for model checkpoints
    os.makedirs('data/best_models', exist_ok=True)
    best_model_path = os.path.join('data/best_models', f'fast_{type(model).__name__}.pth')

    print(f"🚀 Starting FAST training for {num_epochs} epochs...")
    print(f"📊 Train samples: {dataset_sizes['train']}, Val samples: {dataset_sizes['val']}")
    
    for epoch in range(num_epochs):
        print(f'\n🔄 Epoch {epoch+1}/{num_epochs}')
        epoch_start = time.time()
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                desc = f"🏋️  Training"
            else:
                model.eval()
                desc = f"🧪 Validation"

            running_loss = 0.0
            running_corrects = 0
            
            # Progress bar for this phase
            pbar = tqdm(dataloaders[phase], desc=desc, leave=False)
            
            for batch_idx, (inputs, labels) in enumerate(pbar):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar every few batches
                if batch_idx % 5 == 0:  # More frequent updates for smaller dataset
                    current_acc = running_corrects.double() / ((batch_idx + 1) * inputs.size(0))
                    pbar.set_postfix({
                        'loss': f'{loss.item():.3f}',
                        'acc': f'{current_acc:.3f}'
                    })

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"  {phase:5s}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.4f}")

            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, best_model_path)
                print(f"  💾 New best model saved! Acc: {best_acc:.4f}")
        
        epoch_time = time.time() - epoch_start
        print(f"  ⏱️  Epoch completed in {epoch_time:.1f}s")

    time_elapsed = time.time() - since
    print(f'\n✅ Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'🏆 Best val Acc: {best_acc:.4f}')

    # Load best model weights
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📁 Loaded best model from epoch {checkpoint['epoch']}")
    
    return model, best_acc

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(42)
    random.seed(42)
    
    # Set device - prioritize GPU!
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    if device.type == 'cpu':
        print("⚠️  WARNING: Training on CPU will be SLOW! Consider using GPU.")
    else:
        print(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")

    # Get FAST data with limited samples
    data_dir = "data/processed"
    batch_size = 64 if device.type == 'cuda' else 32
    
    print(f"🎯 PROOF OF CONCEPT MODE: Using limited dataset!")
    
    dataloaders, dataset_sizes, class_names = get_fast_dataloaders(
        data_dir, 
        batch_size=batch_size,
        train_samples=5000,  # 🔥 Only 5k training samples
        val_samples=1000,    # 🔥 Only 1k validation samples
        num_workers=0
    )
    
    print(f"📚 Number of classes: {len(class_names)}")
    print(f"📦 Batch size: {batch_size}")
    print(f"⚡ FAST MODE: {dataset_sizes['train']} train + {dataset_sizes['val']} val samples")
    
    # Create model
    model_name = 'vgg16'  # Start with VGG16
    model = get_model(model_name, num_classes=len(class_names)).to(device)
    print(f"🧠 Model: {model_name}")
    
    # Setup training with OPTIMIZATIONS
    criterion = nn.CrossEntropyLoss()
    
    # 🔥 ADAM optimizer instead of SGD (usually faster for fine-tuning)
    if model_name == 'vgg16':
        params = model.model.classifier[6].parameters()
    else:  # resnet50
        params = model.model.fc.parameters()
    
    optimizer = optim.Adam(params, lr=0.01)  # 🚀 Higher LR: 0.01 vs 0.001
    
    print(f"⚙️  Optimizer: Adam with lr=0.01 (10x higher than before)")
    
    # 🎯 Fast training - only 3 epochs for super fast proof of concept
    num_epochs = 3
    print(f"⏰ Training for only {num_epochs} epochs (super fast proof of concept)")
    
    # Train model
    model, best_acc = fast_train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer,
        num_epochs=num_epochs,
        device=device
    )
    
    print(f"\n🎉 PROOF OF CONCEPT COMPLETE!")
    print(f"📈 Best accuracy achieved: {best_acc:.1%}")
    
    if best_acc > 0.1:  # 10% accuracy (random would be 1%)
        print(f"✅ SUCCESS: Model is learning! (Random = 1%, Got = {best_acc:.1%})")
    else:
        print(f"⚠️  Model might need more training or tuning")
    
    # Quick test to verify model works
    print(f"\n🧪 Testing model inference...")
    model.eval()
    with torch.no_grad():
        for images, labels in dataloaders['val']:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            accuracy = (preds == labels).float().mean()
            print(f"✅ Model working! Sample batch accuracy: {accuracy:.1%}")
            break
    
    return model

if __name__ == "__main__":
    main() 