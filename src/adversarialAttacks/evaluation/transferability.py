import torch

def evaluate_transferability(
    model_src,
    attack,
    model_tgt,
    dataloader,
    num_samples=None
):
    """
    Generate adversarial examples on model_src via `attack`, then
    measure clean accuracy of model_tgt on those adversarial examples.

    Args:
        model_src (nn.Module): model used to craft adversarial examples
        attack (BaseAttack): attack instantiated on model_src
        model_tgt (nn.Module): model that will be evaluated on x_adv
        dataloader (DataLoader): loader for (x, y) pairs
        num_samples (int, optional): cap on # of samples

    Returns:
        float: transfer accuracy = (# correctly classified by model_tgt on x_adv) / total
    """
    model_src.eval()
    model_tgt.eval()
    correct = 0
    total   = 0

    for x, y in dataloader:
        if num_samples and total >= num_samples:
            break
        x, y = x.to(attack.device), y.to(attack.device)

        # 1) craft adversarial on source
        with torch.no_grad():
            x_adv = attack.generate(x, y)

        # 2) eval target on those
        with torch.no_grad():
            preds = model_tgt(x_adv).argmax(dim=1)
        correct += (preds == y).sum().item()
        total   += x.size(0)

    return correct / total
