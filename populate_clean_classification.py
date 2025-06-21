import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

from adversarialAttacks.models import get_model

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_clean_classifications(model_name, data_dir, device='cuda', seed=42):
    """
    Get clean classification results for all test images in the same order as CSV files.
    
    Args:
        model_name (str): Either 'vgg16' or 'resnet50'
        data_dir (str): Path to test images directory
        device (str): Device to run evaluation on
        seed (int): Random seed for reproducibility
    
    Returns:
        list: List of boolean values indicating correct classification for each image
    """
    set_seed(seed)
    
    # Load model
    model = get_model(model_name, num_classes=100, pretrained=False).to(device)
    weight_file = 'best_VGG.pth' if model_name == 'vgg16' else 'best_ResNet.pth'
    checkpoint = torch.load(f'data/best_models/{weight_file}', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load data in the same order as CSV files (alphabetical, no shuffle)
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    clean_classifications = []
    
    print(f"Evaluating {model_name} on clean images...")
    for x, y in tqdm(dataloader, desc="Processing clean images"):
        x, y = x.to(device), y.to(device)
        
        # Get clean predictions
        with torch.no_grad():
            logits_clean = model(x)
            probs_clean = F.softmax(logits_clean, dim=1)
            pred_clean = probs_clean.argmax(dim=1)
        
        # Store results for each image in batch
        for i in range(x.size(0)):
            clean_classifications.append((pred_clean[i] == y[i]).item())
    
    return clean_classifications

def populate_perturbation_csv_files():
    """Populate clean_correct_classification column in perturbation analysis CSV files."""
    perturbation_dir = Path('data/perturbation_analysis')
    csv_files = list(perturbation_dir.glob('*.csv'))
    
    # Group files by model
    vgg16_files = [f for f in csv_files if f.name.startswith('vgg16')]
    resnet50_files = [f for f in csv_files if f.name.startswith('resnet50')]
    
    print(f"Found {len(vgg16_files)} VGG16 files and {len(resnet50_files)} ResNet50 files")
    
    # Get clean classifications for each model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nGetting clean classifications for VGG16...")
    vgg16_clean_results = get_clean_classifications('vgg16', 'data/processed/test', device)
    
    print("\nGetting clean classifications for ResNet50...")
    resnet50_clean_results = get_clean_classifications('resnet50', 'data/processed/test', device)
    
    # Update VGG16 files
    print(f"\nUpdating {len(vgg16_files)} VGG16 CSV files...")
    for csv_file in tqdm(vgg16_files, desc="VGG16 files"):
        try:
            df = pd.read_csv(csv_file)
            df['clean_correct_classification'] = vgg16_clean_results
            df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
    
    # Update ResNet50 files
    print(f"\nUpdating {len(resnet50_files)} ResNet50 CSV files...")
    for csv_file in tqdm(resnet50_files, desc="ResNet50 files"):
        try:
            df = pd.read_csv(csv_file)
            df['clean_correct_classification'] = resnet50_clean_results
            df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

def populate_transferability_csv_files():
    """Populate clean_correct_classification column in transferability CSV files."""
    transferability_dir = Path('data/transferability')
    csv_files = list(transferability_dir.glob('*.csv'))
    
    # Group files by target model (the model being tested on adversarial examples)
    vgg16_target_files = [f for f in csv_files if 'trans_to_Vgg16' in f.name]
    resnet50_target_files = [f for f in csv_files if 'trans_to_Resnet50' in f.name]
    
    print(f"Found {len(vgg16_target_files)} files with VGG16 as target")
    print(f"Found {len(resnet50_target_files)} files with ResNet50 as target")
    
    # Get clean classifications for each model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nGetting clean classifications for VGG16...")
    vgg16_clean_results = get_clean_classifications('vgg16', 'data/processed/test', device)
    
    print("\nGetting clean classifications for ResNet50...")
    resnet50_clean_results = get_clean_classifications('resnet50', 'data/processed/test', device)
    
    # Update files with VGG16 as target
    print(f"\nUpdating {len(vgg16_target_files)} files with VGG16 as target...")
    for csv_file in tqdm(vgg16_target_files, desc="VGG16 target files"):
        try:
            df = pd.read_csv(csv_file)
            df['clean_correct_classification'] = vgg16_clean_results
            df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")
    
    # Update files with ResNet50 as target
    print(f"\nUpdating {len(resnet50_target_files)} files with ResNet50 as target...")
    for csv_file in tqdm(resnet50_target_files, desc="ResNet50 target files"):
        try:
            df = pd.read_csv(csv_file)
            df['clean_correct_classification'] = resnet50_clean_results
            df.to_csv(csv_file, index=False)
        except Exception as e:
            print(f"Error processing {csv_file.name}: {e}")

def main():
    print("=== Populate Clean Classification Script ===")
    print("This script will populate the clean_correct_classification column")
    print("by testing models on clean images in the same order as CSV files.")
    
    # Ask for confirmation
    response = input("Do you want to proceed? (yes/no): ")
    if response.lower() != 'yes':
        print("Operation cancelled.")
        return
    
    print("\nPopulating perturbation analysis CSV files...")
    populate_perturbation_csv_files()
    
    print("\nPopulating transferability CSV files...")
    populate_transferability_csv_files()
    
    print("\n=== SUMMARY ===")
    print("Clean classification column population completed!")
    print("\nThe clean_correct_classification column now contains:")
    print("- True if the model correctly classified the clean image")
    print("- False if the model incorrectly classified the clean image")
    print("\nThis allows you to calculate RA as:")
    print("RA = (number of correctly classified adversarial examples) / (number of correctly classified clean examples)")

if __name__ == "__main__":
    main() 