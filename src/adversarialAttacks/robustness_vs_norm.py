import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def plot_robustness_vs_norm(model, attack, dataloader, device, num_samples=None):
    """
    Generate and plot Robust Accuracy (RA) as a function of the L2 norm of perturbations.
    
    Args:
        model (nn.Module): the model
        attack (BaseAttack): instantiated attack (FGSM, PGD, or CW)
        dataloader (DataLoader): test loader
        device (torch.device): computation device
        num_samples (int, optional): number of samples to evaluate (None = all)
    """
    model.eval()
    norms = []
    success = []  # 1.0 if classification remains correct after attack

    total = 0
    for x, y in dataloader:
        if num_samples and total >= num_samples:
            break
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            # Generate adversarial example
            x_adv = attack.generate(x, y)
            # Compute logits and predictions on adversarial
            logits_adv = model(x_adv)
            preds_adv = logits_adv.argmax(dim=1)

        # Compute L2 norm of perturbation per sample
        delta = (x_adv - x).view(x.size(0), -1)
        batch_norms = torch.norm(delta, p=2, dim=1).cpu().numpy()
        # Determine success (model still correct?)
        batch_success = (preds_adv == y).cpu().numpy().astype(float)

        norms.extend(batch_norms.tolist())
        success.extend(batch_success.tolist())
        total += x.size(0)

    # Convert to arrays and sort by norm
    norms = np.array(norms)
    success = np.array(success)
    idx = np.argsort(norms)
    norms_sorted = norms[idx]
    success_sorted = success[idx]

    # Compute cumulative RA(δ): fraction correct among those with perturbation ≤ δ
    cumulative = np.cumsum(success_sorted)
    ra_curve = cumulative / np.arange(1, len(success_sorted) + 1)

    # Plot
    plt.figure(figsize=(6, 4))
    plt.plot(norms_sorted, ra_curve, label=f"{attack.__class__.__name__}")
    plt.xlabel("ℓ₂ perturbation norm δ")
    plt.ylabel("Robust Accuracy RA(δ)")
    plt.title(f"RA vs. ℓ₂ norm: {attack.__class__.__name__}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
