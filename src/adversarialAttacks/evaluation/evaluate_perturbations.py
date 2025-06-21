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
from pytorch_msssim import ssim  # We'll need to install this

from adversarialAttacks.models import get_model
from adversarialAttacks.attacks.fgsm import FGSM
from adversarialAttacks.attacks.pgd import PGD
from adversarialAttacks.attacks.cw import CW

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_l2_norm(clean_img, adv_img):
    """Calculate L2 norm of the perturbation."""
    return torch.norm(adv_img - clean_img, p=2).item()

def calculate_ssim(clean_img, adv_img):
    """Calculate SSIM between clean and adversarial images."""
    # SSIM expects input in range [0, 1] and shape [B, C, H, W]
    # Add batch dimension if needed
    if clean_img.dim() == 3:
        clean_img = clean_img.unsqueeze(0)
        adv_img = adv_img.unsqueeze(0)
    return ssim(clean_img, adv_img, data_range=1.0).item()

def evaluate_perturbations(model_name, attack_type, attack_param, data_dir, device='cuda', seed=42, alpha=None, steps=None):
    """
    Evaluate model robustness by tracking perturbation sizes and confidence changes.
    
    Args:
        model_name (str): Either 'vgg16' or 'resnet50'
        attack_type (str): Either 'fgsm', 'pgd', or 'cw'
        attack_param (float): epsilon for FGSM/PGD, c for CW
        data_dir (str): Path to test images directory
        device (str): Device to run evaluation on
        seed (int): Random seed for reproducibility
        alpha (float): Step size for PGD
        steps (int): Number of steps for PGD
    """
    # Set random seed
    set_seed(seed)
    
    # Print CUDA status
    print("\n=== CUDA Status ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Using device: {device}")
    print("==================\n")
    
    # Load model
    model = get_model(model_name, num_classes=100, pretrained=False).to(device)
    # Map model names to correct weight filenames
    weight_file = 'best_VGG.pth' if model_name == 'vgg16' else 'best_ResNet.pth'
    checkpoint = torch.load(f'data/best_models/{weight_file}', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Setup attack
    if attack_type.lower() == 'fgsm':
        attack = FGSM(model, device=device, epsilon=attack_param)
    elif attack_type.lower() == 'pgd':
        attack = PGD(model, device=device, epsilon=attack_param, alpha=alpha, num_steps=steps)
    else:  # CW
        print("Initializing CW attack...")
        attack = CW(model, device=device, c=attack_param, max_iter=100)
        print("CW attack initialized")
    
    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # Initialize results list and output directory
    results = []
    output_dir = Path('data/perturbation_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Create filename with model, attack, and parameter info
    filename = f"{model_name}_{attack_type}_{attack_param:.3f}.csv"
    save_interval = 10  # Save every 10 batches
    
    # Process each batch
    for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Generating adversarial examples")):
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
        
        # Calculate L2 norms and SSIM
        l2_norms = [calculate_l2_norm(x[i], x_adv[i]) for i in range(x.size(0))]
        ssim_values = [calculate_ssim(x[i], x_adv[i]) for i in range(x.size(0))]
        
        # Store results for each image
        for i in range(x.size(0)):
            results.append({
                'adv_correct_classification': (pred_adv[i] == y[i]).item(),
                'clean_true_prob': true_prob_clean[i].item(),
                'adv_true_prob': true_prob_adv[i].item(),
                'clean_max_prob': max_prob_clean[i].item(),
                'adv_max_prob': max_prob_adv[i].item(),
                'model': model_name,
                'attack': attack_type,
                'attack_param': attack_param,
                'l2_norm': l2_norms[i],
                'ssim': ssim_values[i]
            })
        
        # Save progress periodically
        if (batch_idx + 1) % save_interval == 0:
            df = pd.DataFrame(results)
            df.to_csv(output_dir / filename, index=False)
            print(f"\nProgress saved: {len(results)} images processed")
            
            # Print perturbation statistics
            #print("\nPerturbation Statistics:")
            #print(f"Average L2 norm: {df['l2_norm'].mean():.4f}")
            #print(f"Average SSIM: {df['ssim'].mean():.4f}")
            #print(f"Attack success rate: {(~df['adv_correct_classification']).mean():.2%}")
            #print(f"Average confidence drop: {(df['clean_true_prob'] - df['adv_true_prob']).mean():.4f}")
    
    # Final save
    df = pd.DataFrame(results)
    df.to_csv(output_dir / filename, index=False)
    print(f"\nFinal results saved to {output_dir / filename}")
    
    return df

if __name__ == "__main__":
    # Set random seed for reproducibility
    set_seed(42)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== GPU Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("=====================\n")

    # PGD: Run for all epsilon values from 0.01 to 0.25 with 0.02 increments, 10 steps, alpha = epsilon/10
    models = ['vgg16', 'resnet50']
    attack_type = 'pgd'
    steps = 10
    epsilons = np.arange(0.01, 0.25+0.001, 0.02)
    for model_name in models:
        for epsilon in epsilons:
            alpha = epsilon / steps
            print(f"\nEvaluating {model_name} with PGD (epsilon={epsilon:.3f}, steps={steps}, alpha={alpha:.4f})")
            df = evaluate_perturbations(
                model_name=model_name,
                attack_type=attack_type,
                attack_param=epsilon,
                data_dir='data/processed/test',
                device=device,
                seed=42,
                alpha=alpha,
                steps=steps
            )
