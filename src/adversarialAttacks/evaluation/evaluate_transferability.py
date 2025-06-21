import os
import sys
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
from pytorch_msssim import ssim

from adversarialAttacks.models import get_model
from adversarialAttacks.attacks.fgsm import FGSM
from adversarialAttacks.attacks.pgd import PGD
from adversarialAttacks.attacks.cw import CW

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_l2_norm(clean_img, adv_img):
    return torch.norm(adv_img - clean_img, p=2).item()

def calculate_ssim(clean_img, adv_img):
    if clean_img.dim() == 3:
        clean_img = clean_img.unsqueeze(0)
        adv_img = adv_img.unsqueeze(0)
    return ssim(clean_img, adv_img, data_range=1.0).item()

def evaluate_transferability(src_model_name, tgt_model_name, attack_type, attack_param, data_dir, device='cuda', seed=42):
    set_seed(seed)
    print(f"\n=== CUDA Status ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA version: {torch.version.cuda}")
    print(f"Using device: {device}")
    print(f"==================\n")

    # Load source model
    src_model = get_model(src_model_name, num_classes=100, pretrained=False).to(device)
    src_weight_file = 'best_VGG.pth' if src_model_name == 'vgg16' else 'best_ResNet.pth'
    src_checkpoint = torch.load(f'data/best_models/{src_weight_file}', map_location=device)
    src_model.load_state_dict(src_checkpoint['model_state_dict'])
    src_model.eval()

    # Load target model
    tgt_model = get_model(tgt_model_name, num_classes=100, pretrained=False).to(device)
    tgt_weight_file = 'best_VGG.pth' if tgt_model_name == 'vgg16' else 'best_ResNet.pth'
    tgt_checkpoint = torch.load(f'data/best_models/{tgt_weight_file}', map_location=device)
    tgt_model.load_state_dict(tgt_checkpoint['model_state_dict'])
    tgt_model.eval()

    # Setup attack (on source model)
    if attack_type.lower() == 'fgsm':
        attack = FGSM(src_model, device=device, epsilon=attack_param)
    elif attack_type.lower() == 'pgd':
        steps = 10
        alpha = attack_param / steps
        attack = PGD(src_model, device=device, epsilon=attack_param, alpha=alpha, num_steps=steps)
    else:
        print("Initializing CW attack...")
        attack = CW(src_model, device=device, c=attack_param, max_iter=100)
        print("CW attack initialized")

    # Load data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    results = []
    output_dir = Path('data/transferability')
    output_dir.mkdir(exist_ok=True)
    filename = f"{src_model_name.capitalize()}_trans_to_{tgt_model_name.capitalize()}_{attack_type}_{attack_param:.3f}.csv"
    save_interval = 10

    for batch_idx, (x, y) in enumerate(tqdm(dataloader, desc="Generating adversarial examples (transferability)")):
        x, y = x.to(device), y.to(device)

        # Get clean predictions (source model)
        with torch.no_grad():
            logits_clean_src = src_model(x)
            probs_clean_src = F.softmax(logits_clean_src, dim=1)
            pred_clean_src = probs_clean_src.argmax(dim=1)
            true_prob_clean_src = probs_clean_src.gather(1, y.unsqueeze(1)).squeeze(1)
            max_prob_clean_src = probs_clean_src.max(dim=1)[0]

        # Generate adversarial examples (on source model)
        x_adv = attack.generate(x, y)

        # Get adversarial predictions (target model)
        with torch.no_grad():
            logits_adv_tgt = tgt_model(x_adv)
            probs_adv_tgt = F.softmax(logits_adv_tgt, dim=1)
            pred_adv_tgt = probs_adv_tgt.argmax(dim=1)
            true_prob_adv_tgt = probs_adv_tgt.gather(1, y.unsqueeze(1)).squeeze(1)
            max_prob_adv_tgt = probs_adv_tgt.max(dim=1)[0]

        # Calculate L2 norms and SSIM (between clean and adv)
        l2_norms = [calculate_l2_norm(x[i], x_adv[i]) for i in range(x.size(0))]
        ssim_values = [calculate_ssim(x[i], x_adv[i]) for i in range(x.size(0))]

        for i in range(x.size(0)):
            results.append({
                'adv_correct_classification_tgt': (pred_adv_tgt[i] == y[i]).item(),
                'clean_true_prob_src': true_prob_clean_src[i].item(),
                'clean_max_prob_src': max_prob_clean_src[i].item(),
                'adv_true_prob_tgt': true_prob_adv_tgt[i].item(),
                'adv_max_prob_tgt': max_prob_adv_tgt[i].item(),
                'model_src': src_model_name,
                'model_tgt': tgt_model_name,
                'attack': attack_type,
                'attack_param': attack_param,
                'l2_norm': l2_norms[i],
                'ssim': ssim_values[i]
            })

        if (batch_idx + 1) % save_interval == 0:
            df = pd.DataFrame(results)
            df.to_csv(output_dir / filename, index=False)
            print(f"\nProgress saved: {len(results)} images processed")

    df = pd.DataFrame(results)
    df.to_csv(output_dir / filename, index=False)
    print(f"\nFinal results saved to {output_dir / filename}")
    return df

if __name__ == "__main__":
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== GPU Information ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("=====================\n")

    # Use only PGD parameter values for transfer attacks
    pgd_epsilons = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23, 0.25]

    src_models = ['resnet50', 'vgg16']
    tgt_models = ['vgg16', 'resnet50']
    attacks = [('pgd', eps) for eps in pgd_epsilons]

    for src_model in src_models:
        for tgt_model in tgt_models:
            if src_model == tgt_model:
                continue  # skip self-transfer
            for attack_type, param in attacks:
                print(f"\nEvaluating transferability: {src_model} -> {tgt_model} with {attack_type} (param={param})")
                df = evaluate_transferability(
                    src_model_name=src_model,
                    tgt_model_name=tgt_model,
                    attack_type=attack_type,
                    attack_param=param,
                    data_dir='data/processed/test',
                    device=device
                ) 