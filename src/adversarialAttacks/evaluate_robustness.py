import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from pathlib import Path

# Import your models and attack classes
from models import get_model
from save_image import save_extreme_examples
from attacks.fgsm import FGSM
from attacks.pgd import PGD
from attacks.cw import CW


def evaluate_metrics(model, attack, dataloader, bound=0.05, num_samples=None):
    """
    Evaluate Robust Accuracy (RA) and Robust Ratio (RR) for a given attack and model.

    Args:
        model (nn.Module): The neural network model (already on correct device).
        attack (BaseAttack): An instantiated attack (FGSM, PGD or CW).
        dataloader (DataLoader): DataLoader for the test dataset.
        bound (float): Tolerance bound 'b' for RR calculation.
        num_samples (int, optional): Max number of samples to evaluate. If None, use entire dataset.

    Returns:
        (ra, rr):
          ra (float): Robust Accuracy = (# of adv‐examples classified correctly) / (total).
          rr (float): Robust Ratio = proportion of samples whose confidence‐drop ≤ bound.
    """
    model.eval()
    correct_adv = 0
    total = 0
    rr_count = 0

    for x, y in dataloader:
        if num_samples and total >= num_samples:
            break
        x, y = x.to(attack.device), y.to(attack.device)

        # 1) Compute clean probabilities
        with torch.no_grad():
            logits_clean = model(x)
            probs_clean = F.softmax(logits_clean, dim=1)

        # 2) Generate adversarial examples
        x_adv = attack.generate(x, y)

        # 3) Compute adversarial probabilities / predictions
        with torch.no_grad():
            logits_adv = model(x_adv)
            probs_adv = F.softmax(logits_adv, dim=1)
            pred_adv = probs_adv.argmax(dim=1)

        # 4) Update Robust Accuracy (RA) counts
        correct_adv += (pred_adv == y).sum().item()

        # 5) For Robust Ratio (RR), measure confidence change on the true class
        true_probs_clean = probs_clean.gather(1, y.unsqueeze(1)).squeeze(1)
        true_probs_adv = probs_adv.gather(1, y.unsqueeze(1)).squeeze(1)
        change = torch.abs(true_probs_adv - true_probs_clean)
        rr_count += (change <= bound).sum().item()

        total += x.size(0)

    ra = correct_adv / total
    rr = rr_count / total
    return ra, rr


def main():
    # --------- Configuration ---------
    # 1) Path to ImageNet-100 test set directory (modify this to wherever your data lives).
    #    It should contain subfolders per class, e.g.:
    #      /path/to/imagenet100/test/class_0/…
    #      /path/to/imagenet100/test/class_1/…
    #    etc.
    DATA_DIR = Path("data/processed/val")

    # 2) Choose device (Windows)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 2) Choose device (MacOS)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3) Dataloader for ImageNet-100 test set
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    test_dataset = ImageFolder(DATA_DIR, transform=transform)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=4
    )

    # 4) Which models to evaluate
    model_names = ["vgg16", "resnet50"]

    # 5) Attack parameters (you can adjust these if desired)
    epsilon = 0.03           # FGSM/PGD max perturbation
    alpha = 0.01             # PGD step size
    pgd_steps = 10           # Number of PGD iterations
    cw_c = 1.0               # CW constant c
    cw_kappa = 0.0           # CW confidence margin κ
    cw_max_iter = 50       # CW maximum optimizer steps
    cw_lr = 0.01             # CW learning rate

    # 6) Which attack types to run
    attack_names = ["fgsm", "pgd", "cw"]

    # 7) Bounds at which to compute Robust Ratio (δ ∈ {0.01, 0.02, …, 0.20})
    bounds = [i / 100 for i in range(1, 21)]

    # ------------------------------------------

    for model_name in model_names:
        print(f"\n==== Evaluating Model: {model_name.upper()} ====")
        # Load the model (all layers frozen except final classifier)
        model = get_model(model_name, num_classes=100, pretrained=False).to(device)
        
        # Load finetuned weights 
        if model_name.lower() == "vgg16":
            checkpoint_path = "data/best_models/fast_ModifiedVGG16.pth" # change to model name
        elif model_name.lower() == "resnet50":
            checkpoint_path = "data/best_models/fast_ModifiedResNet50.pth"  # change to model name
        else:
            raise ValueError(f"No checkpoint configured for {model_name}")

        # Load the .pth (ensure it was saved via torch.save(model.state_dict(), …))
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            # In case someone saved just model.state_dict() without wrapping it:
            model.load_state_dict(checkpoint)
        
        # Move model to device
        model = model.to(device)
        model.eval()

        # Instantiate FGSM, PGD, and CW attacks on that model
        fgsm_attack = FGSM(model, epsilon=epsilon, device=device)
        save_extreme_examples(
            model, fgsm_attack, test_loader, device,
            out_dir="data/adversarial_examples",
        )
        
        pgd_attack = PGD(
            model,
            epsilon=epsilon,
            alpha=alpha,
            num_steps=pgd_steps,
            device=device,
        )
        save_extreme_examples(
            model, pgd_attack, test_loader, device,
            out_dir="data/adversarial_examples",
        )
        
        cw_attack = CW(
            model,
            c=cw_c,
            kappa=cw_kappa,
            max_iter=cw_max_iter,
            lr=cw_lr,
            device=device,
        )
        save_extreme_examples(
            model, cw_attack, test_loader, device,
            out_dir="data/adversarial_examples",
        )    
    

        for atk_name, attack in zip(attack_names, [fgsm_attack, pgd_attack, cw_attack]):
            print(f"\n-- Attack: {atk_name.upper()} --")

            # 1) Compute Robust Accuracy (RA)
            ra, _ = evaluate_metrics(model, attack, test_loader, bound=1.0) # set satisfying bound
            print(f"Robust Accuracy (RA): {ra:.4f}")

            # 2) Compute Robust Ratio (RR) at a range of bounds
            for b in bounds:
                _, rr = evaluate_metrics(model, attack, test_loader, bound=b) 
                print(f"  Bound = {b:0.2f} → Robust Ratio (RR): {rr:.4f}")




if __name__ == "__main__":
    main()
