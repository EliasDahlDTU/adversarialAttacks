import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
import math
import pandas as pd

# Import models, metrics, and attack classes
from models import get_model
from save_image import save_extreme_examples
from robustness_vs_norm import plot_robustness_vs_norm
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.cw import CW

def normal_ci(p_hat, N, z=1.96):
    if N == 0:
        return 0.0, 0.0
    se = math.sqrt(p_hat * (1 - p_hat) / N)
    lower = max(0.0, p_hat - z * se)
    upper = min(1.0, p_hat + z * se)
    return lower, upper

def evaluate_metrics(model, attack, dataloader, bound=0.05, num_samples=None):
    model.eval()
    correct_adv = total = rr_count = 0
    for x, y in dataloader:
        if num_samples and total >= num_samples:
            break
        x, y = x.to(attack.device), y.to(attack.device)
        with torch.no_grad():
            logits_clean = model(x)
            probs_clean = F.softmax(logits_clean, dim=1)
            x_adv = attack.generate(x, y)
            logits_adv = model(x_adv)
            probs_adv = F.softmax(logits_adv, dim=1)
            pred_adv = probs_adv.argmax(dim=1)
        correct_adv += (pred_adv == y).sum().item()
        true_clean = probs_clean.gather(1, y.unsqueeze(1)).squeeze(1)
        true_adv   = probs_adv.gather(1, y.unsqueeze(1)).squeeze(1)
        rr_count   += (torch.abs(true_adv - true_clean) <= bound).sum().item()
        total      += x.size(0)
    return correct_adv/total, rr_count/total, total

def main():
    DATA_DIR = Path("data/processed/val")
    
    # Choose device (Windows)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Choose device (MacOS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()
    ])
    test_dataset = ImageFolder(DATA_DIR, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    N = len(test_dataset)

    model_names = ["vgg16", "resnet50"]
    bounds = [i/100 for i in range(1,21)]
    attack_constructors = [
        ("FGSM", FGSM, {"epsilon":0.03}),
        ("PGD", PGD, {"epsilon":0.03,"alpha":0.01,"num_steps":10}),
        ("CW",  CW,  {"c":1.0,"kappa":0.0,"max_iter":50,"lr":0.01})
    ]

    results = []
    for model_name in model_names:
        print(f"\n==== Evaluating Model: {model_name.upper()} ===")
        model = get_model(model_name, num_classes=100, pretrained=False).to(device)
        # load state dict...
        for atk_name, atk_cls, atk_kwargs in attack_constructors:
            print(f"\n -- Attack: {atk_name} --")
            attack = atk_cls(model, device=device, **atk_kwargs)
            
            # Save examples & plots
            print("  [1/4] Saving adversarial examples...", end=" ", flush=True)
            save_extreme_examples(model, attack, test_loader, device,
                                  out_dir="data/adversarial_examples")
            print("done.")
            
            print("  [2/4] Plotting RA/RR vs. norm…", end="", flush=True)
            plot_robustness_vs_norm(model, attack, test_loader, device)
            print(" done.")
            
            # Compute RA
            print("  [3/4] Computing RA", end="", flush=True)
            ra, _, _ = evaluate_metrics(model, attack, test_loader, bound=1.0)
            ra_low, ra_high = normal_ci(ra, N)
            
            results.append({
                "model": model_name,
                "attack": atk_name,
                "bound": 1.00,
                "metric": "RA",
                "value": ra,
                "ci_lower": ra_low,
                "ci_upper": ra_high
            })
            print(f" RA={ra:.4f} [95% CI {ra_low:.4f}–{ra_high:.4f}]")

            # Compute RR at each bound
            print("  [4/4] Computing RR over bounds…")
            for b in bounds:
                _, rr, _ = evaluate_metrics(model, attack, test_loader, bound=b)
                rr_low, rr_high = normal_ci(rr, N)
                results.append({
                    "model": model_name,
                    "attack": atk_name,
                    "bound": b,
                    "metric": "RR",
                    "value": rr,
                    "ci_lower": rr_low,
                    "ci_upper": rr_high
                })
                print(f"    Bound {b:0.2f} → RR={rr:.4f} [CI {rr_low:.4f}–{rr_high:.4f}]")
                
    # Ensure results directory exists
    results_dir = Path(__file__).parent / "results"
    results_dir = results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save to CSV inside results/
    out_path = results_dir / "robustness_results.csv"
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(out_path, index=False)
    print(f"Saved all metrics to {out_path}")

if __name__=="__main__":
    main()
