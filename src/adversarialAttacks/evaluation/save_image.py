from torchvision.utils import save_image
import torch
from pathlib import Path

def save_extreme_examples(model, attack, dataloader, device,
                           out_dir="adv_examples", num_samples=None):
    """
    For a given model and attack, find and save the least, most, and average-perturbation
    adversarial examples (and their clean inputs) under out_dir/AttackName/.
    """
    model.eval()
    records = []  # will hold tuples (norm, x_clean, x_adv)

    total = 0
    for x, y in dataloader:
        if num_samples and total >= num_samples:
            break
        x, y = x.to(device), y.to(device)

        # generate and record
        with torch.no_grad():
            x_adv = attack.generate(x, y)
        # compute L2 norm per sample
        delta = (x_adv - x).view(x.size(0), -1)
        norms = torch.norm(delta, p=2, dim=1)

        # accumulate
        for i in range(x.size(0)):
            records.append((norms[i].item(),
                            x[i].cpu(), x_adv[i].cpu()))
        total += x.size(0)

    # sort by perturbation size
    records.sort(key=lambda r: r[0])
    norms = [r[0] for r in records]
    mean_norm = sum(norms) / len(norms)

    # pick indices
    idx_least = 0
    idx_most  = len(records) - 1
    idx_avg   = min(range(len(records)),
                    key=lambda i: abs(norms[i] - mean_norm))

    choice = {
        "least": idx_least,
        "avg":   idx_avg,
        "most":  idx_most
    }

    base = Path(out_dir) / attack.__class__.__name__
    base.mkdir(parents=True, exist_ok=True)

    for label, idx in choice.items():
        norm, x_clean, x_adv = records[idx]
        save_image(x_clean, base / f"{label}_clean.png")
        save_image(x_adv,   base / f"{label}_adv.png")
        # optional: the perturbation itself
        save_image((x_adv - x_clean).abs(),
                   base / f"{label}_perturb.png")

    print(f"[{attack.__class__.__name__}] "
          f"saved least/avg/most adv examples in {base}  "
          f"(norms: {norms[0]:.4f}, {mean_norm:.4f}, {norms[-1]:.4f})")