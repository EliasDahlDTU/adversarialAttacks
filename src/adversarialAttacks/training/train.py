import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import json # New import
from datetime import datetime # New import
import copy
from pathlib import Path

from models import get_model
from data_loader import get_dataloaders

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler, num_epochs=25, device='cuda'):
    """
    Train a model with learning rate scheduling and model checkpointing.
    Uses validation set for model selection.
    
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
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            pbar = tqdm(dataloaders[phase], desc=f'{phase} phase')
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{torch.sum(preds == labels.data).item() / inputs.size(0):.4f}'
                })

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Save history
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # Deep copy the model if it's the best validation accuracy
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_acc': best_acc,
                }, f'data/best_models/best_{type(model.model).__name__}.pth')

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path("data/results")
    results_dir.mkdir(exist_ok=True)
    history_path = results_dir / f"{type(model.model).__name__}_training_history_{timestamp}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=4)
    print(f"Training history saved to {history_path}")

    return model, history

def plot_training_history(history, save_path): # Added save_path argument
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
    plt.savefig(save_path) # Use the save_path argument
    plt.close()
    print(f"Training plot saved to {save_path}") # Added print statement

def visualize_predictions(model, dataloader, class_names, human_readable_label_map, plot_save_path, num_images=6, device='cuda'): # Added human_readable_label_map and plot_save_path
    """
    Visualize model predictions on validation data.
    
    Args:
        model: PyTorch model
        dataloader: DataLoader for validation data
        class_names: List of class directory names (e.g., 'n01234567')
        human_readable_label_map (dict): Mapping from class directory names to human-readable names
        plot_save_path (str): Path to save the predictions plot
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
                
                predicted_folder_name = class_names[preds[j]]
                true_folder_name = class_names[labels[j]]
                
                predicted_display_name = human_readable_label_map.get(predicted_folder_name, predicted_folder_name)
                true_display_name = human_readable_label_map.get(true_folder_name, true_folder_name)
                
                ax.set_title(f'predicted: {predicted_display_name}\\ntrue: {true_display_name}')
                
                # Convert tensor to image (no denormalization needed)
                img = inputs.cpu().data[j].permute(1, 2, 0)
                plt.imshow(img)

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    plt.tight_layout()
                    plt.savefig(plot_save_path) # Use the new save path
                    plt.close()
                    print(f"Predictions plot saved to {plot_save_path}")
                    return
    
    model.train(mode=was_training)
    plt.tight_layout()
    plt.savefig(plot_save_path) # Use the new save path
    plt.close()
    print(f"Predictions plot saved to {plot_save_path}")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Get data
    data_dir = "data/processed"  # Adjust this path as needed
    dataloaders, dataset_sizes, class_names = get_dataloaders(data_dir)
    print(f"Number of classes: {len(class_names)}")

    # Load human-readable labels
    labels_json_path = os.path.join("data", "raw", "Labels.json")
    human_readable_label_map = {}
    if os.path.exists(labels_json_path):
        with open(labels_json_path, 'r') as f:
            human_readable_label_map = json.load(f)
        print(f"Loaded human-readable labels from {labels_json_path}")
    else:
        print(f"Warning: Labels.json not found at {labels_json_path}. Using folder names for predictions plot.")
    
    # Create model
    model_name = 'resnet50' #'vgg16'  # or 'resnet50'
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
        num_epochs=25, # Make sure this is passed to train_model if it's not using the one from main's scope
        device=device
    )
    
    # Visualize some predictions
    model_type_name = type(model.model).__name__ # Get underlying model name for consistency
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join('data', 'results')
    predictions_plot_save_path = os.path.join(results_dir, f'{model_type_name}_predictions_{timestamp}.png')
    visualize_predictions(model, dataloaders['val'], class_names, human_readable_label_map, predictions_plot_save_path, device=device)

if __name__ == "__main__":
    main()