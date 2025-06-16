import os
import sys
# Add src to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

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
from adversarialAttacks.attacks.fgsm import FGSM
from adversarialAttacks.attacks.pgd import PGD
from adversarialAttacks.attacks.cw import CW

def calculate_l2_norm(clean_img, adv_img):
    """Calculate L2 norm of the perturbation."""
    return torch.norm(adv_img - clean_img, p=2).item()

def evaluate_perturbations(model_name, attack_type, attack_param, data_dir, device='cuda'):
    """
    Evaluate model robustness by tracking perturbation sizes and confidence changes.
    
    Args:
        model_name (str): Either 'vgg16' or 'resnet50'
        attack_type (str): Either 'fgsm', 'pgd', or 'cw'
        attack_param (float): epsilon for FGSM/PGD, c for CW
        data_dir (str): Path to test images directory
        device (str): Device to run evaluation on
    """
    # Load model
    model = get_model(model_name, num_classes=100, pretrained=False).to(device)
    checkpoint = torch.load(f'data/best_models/best_{model_name}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup attack
    if attack_type.lower() in ['fgsm', 'pgd']:
        attack_cls = FGSM if attack_type.lower() == 'fgsm' else PGD
        attack = attack_cls(model, device=device, epsilon=attack_param)
    else:  # CW
        attack = CW(model, device=device, c=attack_param)
    
    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize results list
    results = []
    
    # Process each batch
    for x, y in tqdm(dataloader, desc="Generating adversarial examples"):
        x, y = x.to(device), y.to(device)
        
        # Get clean predictions
        with torch.no_grad():
            logits_clean = model(x)
            probs_clean = F.softmax(logits_clean, dim=1)
            pred_clean = probs_clean.argmax(dim=1)
            true_prob_clean = probs_clean.gather(1, y.unsqueeze(1)).squeeze(1)
            max_prob_clean = probs_clean.max(dim=1)[0]
        
        # Generate adversarial examples
        x_adv = attack.generate(x, y)
        
        # Get adversarial predictions
        with torch.no_grad():
            logits_adv = model(x_adv)
            probs_adv = F.softmax(logits_adv, dim=1)
            pred_adv = probs_adv.argmax(dim=1)
            true_prob_adv = probs_adv.gather(1, y.unsqueeze(1)).squeeze(1)
            max_prob_adv = probs_adv.max(dim=1)[0]
        
        # Calculate L2 norms
        l2_norms = [calculate_l2_norm(x[i], x_adv[i]) for i in range(x.size(0))]
        
        # Store results for each image
        for i in range(x.size(0)):
            results.append({
                'correct_classification': (pred_adv[i] == y[i]).item(),
                'true_prob_before': true_prob_clean[i].item(),
                'true_prob_after': true_prob_adv[i].item(),
                'max_prob_before': max_prob_clean[i].item(),
                'max_prob_after': max_prob_adv[i].item(),
                'model': model_name,
                'attack': attack_type,
                'attack_param': attack_param,
                'l2_norm': l2_norms[i]
            })
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    output_dir = Path('data/perturbation_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Create filename with model, attack, and parameter info
    filename = f"{model_name}_{attack_type}_{attack_param:.3f}.csv"
    df.to_csv(output_dir / filename, index=False)
    print(f"Results saved to {output_dir / filename}")
    
    return df

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Example 1: Run all combinations (as before)
    # models = ['vgg16', 'resnet50']
    # attacks = [
    #     ('fgsm', 0.03),
    #     ('pgd', 0.03),
    #     ('cw', 1.0)
    # ]
    
    # Example 2: Run only CW attack on VGG16 with multiple c values
    models = ['vgg16']  # Only VGG16
    attacks = [
        ('cw', 0.1),   # Try different c values
        ('cw', 1.0),
        ('cw', 10.0)
    ]
    
    # Example 3: Run specific combinations
    # models = ['vgg16']
    # attacks = [
    #     ('fgsm', 0.03),
    #     ('pgd', 0.03)
    # ]
    
    for model_name in models:
        for attack_type, param in attacks:
            print(f"\nEvaluating {model_name} with {attack_type} (param={param})")
            df = evaluate_perturbations(
                model_name=model_name,
                attack_type=attack_type,
                attack_param=param,
                data_dir='data/processed/test',
                device=device
            ) 