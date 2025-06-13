import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def plot_robustness_vs_norm(
    model,
    attack,
    dataloader,
    device,
    confidence_bound=0.05,
    num_samples=None
):
    """
    Plot Robust Accuracy (RA) and Robust Ratio (RR) as functions of L2 perturbation norm.
    
    Args:
        model (nn.Module): the model
        attack (BaseAttack): instantiated attack (FGSM, PGD, or CW)
        dataloader (DataLoader): test loader
        device (torch.device): computation device
        confidence_bound (float): tolerance for RR (|p_adv - p_clean| <= bound)
        num_samples (int, optional): number of samples to evaluate (None = all)
    """
    model.eval()
    norms, ra_flags, rr_flags = [], [], []

    total = 0
    for x, y in dataloader:
        if num_samples and total >= num_samples:
            break
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            x_adv      = attack.generate(x, y)
            logits_cl  = model(x)
            logits_adv = model(x_adv)

            probs_cl = F.softmax(logits_cl,  dim=1)
            probs_ad = F.softmax(logits_adv, dim=1)

            preds_adv = probs_ad.argmax(dim=1)
            p_clean   = probs_cl.gather(1, y.unsqueeze(1)).squeeze(1)
            p_adv     = probs_ad.gather(1, y.unsqueeze(1)).squeeze(1)

        # L2 norm of perturbation
        delta       = (x_adv - x).view(x.size(0), -1)
        batch_norms = torch.norm(delta, p=2, dim=1).cpu().numpy()
        # RA flag: 1.0 if still correct
        batch_ra    = (preds_adv == y).cpu().numpy().astype(float)
        # RR flag: 1.0 if confidence‐drop ≤ bound
        batch_rr    = (torch.abs(p_adv - p_clean) <= confidence_bound) \
                          .cpu().numpy().astype(float)

        norms.extend(batch_norms.tolist())
        ra_flags.extend(batch_ra.tolist())
        rr_flags.extend(batch_rr.tolist())
        total += x.size(0)

    # sort by perturbation magnitude
    norms      = np.array(norms)
    ra_flags   = np.array(ra_flags)
    rr_flags   = np.array(rr_flags)
    order      = np.argsort(norms)
    norms_s    = norms[order]
    ra_sorted  = ra_flags[order]
    rr_sorted  = rr_flags[order]

    # cumulative curves: RA(δ) and RR(δ)
    cum_ra = np.cumsum(ra_sorted) / np.arange(1, len(ra_sorted)+1)
    cum_rr = np.cumsum(rr_sorted) / np.arange(1, len(rr_sorted)+1)

    # plot
    plt.figure(figsize=(6,4))
    plt.plot(norms_s, cum_ra, label="RA(δ)")
    plt.plot(norms_s, cum_rr, label=f"RR(δ) @ Δp≤{confidence_bound}")
    plt.xlabel("ℓ₂ perturbation norm δ")
    plt.ylabel("Metric value")
    plt.title(f"{attack.__class__.__name__}: RA & RR vs δ")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
