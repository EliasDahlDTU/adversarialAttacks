# Import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path
import math
import pandas as pd

# Import models
from models import get_model

# Import evaluation functions
from save_image import save_extreme_examples
from robustness_vs_norm import plot_robustness_vs_norm
from RR_RA_vs_tol import plot_robustness_vs_tolerance
from transferability import evaluate_transferability

# Import attacks
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
    correct_adv = 0
    total = 0
    rr_true_count = 0
    rr_pred_count = 0

    for x, y in dataloader:
        if num_samples and total >= num_samples:
            break
        x, y = x.to(attack.device), y.to(attack.device)

        # clean and adv logits
        with torch.no_grad():
            logits_clean = model(x)
            probs_clean = F.softmax(logits_clean, dim=1)
        x_adv = attack.generate(x, y)
        with torch.no_grad():
            logits_adv = model(x_adv)
            probs_adv = F.softmax(logits_adv, dim=1)
        
        # predictions on adv
        pred_adv = probs_adv.argmax(dim=1)
        correct_adv += (pred_adv == y).sum().item()

        # 1) true-class confidences
        p_true_clean = probs_clean.gather(1, y.unsqueeze(1)).squeeze(1)
        p_true_adv   = probs_adv.gather(1, y.unsqueeze(1)).squeeze(1)
        rr_true_count += (torch.abs(p_true_adv - p_true_clean) <= bound).sum().item()

        # 2) predicted-class confidences (using clean pred)
        pred_clean = probs_clean.argmax(dim=1)
        p_pred_clean = probs_clean.gather(1, pred_clean.unsqueeze(1)).squeeze(1)
        p_pred_adv   = probs_adv.gather(1, pred_clean.unsqueeze(1)).squeeze(1)
        rr_pred_count += (torch.abs(p_pred_adv - p_pred_clean) <= bound).sum().item()

        total += x.size(0)

    ra = correct_adv / total
    rr_true = rr_true_count / total
    rr_pred = rr_pred_count / total
    return ra, rr_true, rr_pred


def load_checkpoint(model, model_name):
    """
    Load fine-tuned weights into model based on its name.
    """
    # Map model names to checkpoint paths
    ckpt_map = {
        "vgg16": "data/best_models/fast_ModifiedVGG16.pth",
        "resnet50": "data/best_models/ResNet50.pth"
    }
    key = model_name.lower()
    if key not in ckpt_map:
        raise ValueError(f"No checkpoint configured for model '{model_name}'")
    ckpt_path = ckpt_map[key]
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model

def main():
    DATA_DIR = Path("data/processed/test")
    
    # select device (Windows)
    # device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # select device (MacOS)
    device   = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform    = transforms.Compose([transforms.ToTensor()])
    test_dataset = ImageFolder(DATA_DIR, transform=transform)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    N = len(test_dataset)

    model_names = ["vgg16", "resnet50"]
    bounds      = [i/100 for i in range(1,21)]
    epsilons    = [i/100 for i in range(0,21)]
    attack_constructors = [
        ("FGSM", FGSM, {"epsilon":0.03}),
        ("PGD",  PGD,  {"epsilon":0.03, "alpha":0.01, "num_steps":10}),
        ("CW",   CW,   {"c":1.0, "kappa":0.0, "max_iter":50, "lr":0.01})
    ]

    results = []
    for model_name in model_names:
        print(f"\n==== Evaluating Model: {model_name.upper()} ===")
        # Load source model and its checkpoint
        model_src = get_model(model_name, num_classes=100, pretrained=False).to(device)
        model_src = load_checkpoint(model_src, model_name).to(device)
        model_src.eval()

        for atk_name, atk_cls, atk_kwargs in attack_constructors:
            print(f"\n -- Attack: {atk_name} --")
            # 0/4) Sweep ε tolerance
            print(" [0/4] sweeping ε tolerance:", end="", flush=True)
            for eps in epsilons:
                params = dict(atk_kwargs)
                if atk_name in ('FGSM','PGD'):
                    params['epsilon'] = eps
                attack_eps = atk_cls(model_src, device=device, **params)

                ra_eps, rr_true_eps, rr_pred_eps = evaluate_metrics(model_src, attack_eps, test_loader, bound=0.05)
                results.append({
                    "model": model_name,
                    "attack": atk_name,
                    "tolerance": eps,
                    "metric": "RA_tol",
                    "value": ra_eps,
                    "ci_lower": normal_ci(ra_eps, N)[0],
                    "ci_upper": normal_ci(ra_eps, N)[1]
                })
                # RR true class
                results.append({
                    "model": model_name,
                    "attack": atk_name,
                    "tolerance": eps,
                    "metric": "RR_true_tol",
                    "value": rr_true_eps,
                    "ci_lower": normal_ci(rr_true_eps, N)[0],
                    "ci_upper": normal_ci(rr_true_eps, N)[1]
                })
                ## RR predidcted class
                results.append({
                    "model": model_name,
                    "attack": atk_name,
                    "tolerance": eps,
                    "metric": "RR_pred_tol",
                    "value": rr_pred_eps,
                    "ci_lower": normal_ci(rr_pred_eps, N)[0],
                    "ci_upper": normal_ci(rr_pred_eps, N)[1]
                })
                print(f" {eps:.2f}", end="", flush=True)
            print(" (done)")

            # 1/4) Save examples & norm plots at default ε
            default_eps = epsilons[3]
            params = dict(atk_kwargs)
            if atk_name in ('FGSM','PGD'):
                params['epsilon'] = default_eps
            attack = atk_cls(model_src, device=device, **params)
            print("  [1/4] Saving adversarial examples...", end=" ", flush=True)
            save_extreme_examples(model_src, attack, test_loader, device,
                                  out_dir="data/adversarial_examples")
            print("done.")
            print("  [2/4] Plotting RA/RR vs. norm...", end=" ", flush=True)
            plot_robustness_vs_norm(model_src, attack, test_loader, device)
            print("done.")

            # 3/4) Compute in-model RA & CI
            print("  [3/4] Computing RA/RR...", end=" ", flush=True)
            ra, rr_true, rr_pred = evaluate_metrics(model_src, attack, test_loader, bound=1.0)
            ra_low, ra_high = normal_ci(ra, N)
            results.append({
                "model": model_name,
                "attack": atk_name,
                "bound": 1.0,
                "metric": "RA",
                "value": ra,
                "ci_lower": ra_low,
                "ci_upper": ra_high
            })
            print(f"RA={ra:.4f} [{ra_low:.4f}–{ra_high:.4f}]")
            # Compute true RR over bounds
            for b in bounds:
                _, rt, rp = evaluate_metrics(model_src, attack, test_loader, bound=b)
                results.append({
                    "model": model_name,
                    "attack": atk_name,
                    "bound": b,
                    "metric": "RR_true",
                    "value": rt,
                    "ci_lower": normal_ci(rt, N)[0],
                    "ci_upper": normal_ci(rt, N)[1]
                })
                results.append({
                    "model": model_name,
                    "attack": atk_name,
                    "bound": b,
                    "metric": "RR_pred",
                    "value": rp,
                    "ci_lower": normal_ci(rp, N)[0],
                    "ci_upper": normal_ci(rp, N)[1]
                })
                print(f"RR_true({b})={rt:.4f} [{normal_ci(rt, N)[0]:.4f}–{normal_ci(rt, N)[1]:.4f}]  "
                      f"RR_pred({b})={rp:.4f} [{normal_ci(rp, N)[0]:.4f}–{normal_ci(rp, N)[1]:.4f}]")
            print("done.")

            # 4/4) Evaluate transferability
            print("  [4/4] Evaluating transferability...", end=" ", flush=True)
            for other_name in model_names:
                if other_name == model_name:
                    continue
                model_tgt = get_model(other_name, num_classes=100, pretrained=False).to(device)
                model_tgt = load_checkpoint(model_tgt, other_name).to(device)
                model_tgt.eval()
                transfer_acc = evaluate_transferability(
                    model_src, attack, model_tgt, test_loader
                )
                results.append({
                    "model_src": model_name,
                    "model_tgt": other_name,
                    "attack": atk_name,
                    "metric": "transfer_acc",
                    "value": transfer_acc
                })
                print(f"{atk_name}→{other_name}={transfer_acc:.4f}", end="  ", flush=True)
            print()

    # Convert to DataFrame and display transfer matrix
    df = pd.DataFrame(results)
    df_transfer = df[df['metric']=='transfer_acc']
    pivot = df_transfer.pivot_table(
        index='model_src', columns=['attack','model_tgt'], values='value'
    )
    print("\nTransferability matrix (accuracy):")
    print(pivot)

    # Save full results CSV
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "robustness_results.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved all metrics to {out_path}")

    # Generate tolerance plots
    plot_robustness_vs_tolerance(out_path, out_dir=results_dir / "plots")
    print("Tolerance plots saved in results/plots")

if __name__ == "__main__":
    main()
