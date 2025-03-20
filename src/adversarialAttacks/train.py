import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import matplotlib.pyplot as plt

from models import get_model
from data_loader import get_dataloaders

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """
    Train a model with learning rate scheduling and model checkpointing.
    
    Args:
        model: PyTorch model
        dataloaders: Dict with 'train' and 'val' dataloaders
        dataset_sizes: Dict with size of 'train' and 'val' datasets
        criterion: Loss function
        optimizer: PyTorch optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
    """
    since = time.time()
    best_acc = 0.0

    # Create directory for model checkpoints if it doesn't exist
    os.makedirs('data/best_models', exist_ok=True)
    best_model_path = os.path.join('data/best_models', f'best_{type(model).__name__}.pth')

    # Training history for plotting
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
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

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Record history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                }, best_model_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Plot training history
    plot_training_history(history)

    # Load best model weights
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, history

def plot_training_history(history):
    """Plot training and validation metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data/training_history.png')
    plt.close()

def visualize_predictions(model, dataloader, class_names, num_images=6, device='cuda'):
    """
    Visualize model predictions on validation data.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for validation data
        class_names: List of class names
        num_images: Number of images to visualize
        device: Device to run inference on
    """
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure(figsize=(12, 8))

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[preds[j]]}\ntrue: {class_names[labels[j]]}')
                
                # Convert tensor to image (no denormalization needed)
                img = inputs.cpu().data[j].permute(1, 2, 0)
                plt.imshow(img)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.tight_layout()
                    plt.savefig('data/predictions.png')
                    plt.close()
                    return
    
    model.train(mode=was_training)
    plt.tight_layout()
    plt.savefig('data/predictions.png')
    plt.close()

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data
    data_dir = "data/processed"  # Adjust this path as needed
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir)
    print(f"Number of classes: {len(class_names)}")
    
    # Create model
    model_name = 'vgg16'  # or 'resnet50'
    model = get_model(model_name, num_classes=len(class_names)).to(device)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    
    # Note: we only optimize the classifier parameters
    if model_name == 'vgg16':
        optimizer = optim.SGD(model.model.classifier[6].parameters(), lr=0.001, momentum=0.9)
    else:  # resnet50
        optimizer = optim.SGD(model.model.fc.parameters(), lr=0.001, momentum=0.9)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    # Train model
    model, history = train_model(
        model,
        dataloaders,
        dataset_sizes,
        criterion,
        optimizer,
        scheduler,
        num_epochs=25,
        device=device
    )
    
    # Visualize some predictions
    visualize_predictions(model, dataloaders['val'], class_names)

if __name__ == "__main__":
    main() 