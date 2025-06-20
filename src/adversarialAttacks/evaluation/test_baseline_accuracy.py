import os
import sys
# Add src to Python path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from tqdm import tqdm

from adversarialAttacks.models import get_model

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))
    data_dir = "data/processed/test"
    batch_size = 128  # Larger batch size for speed

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    models = {
        "vgg16": "best_VGG.pth",
        "resnet50": "best_ResNet.pth"
    }

    for model_name, weight_file in models.items():
        print(f"\nEvaluating {model_name}...")
        model = get_model(model_name, num_classes=100, pretrained=False).to(device)
        checkpoint_path = f"data/best_models/{weight_file}"
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            continue
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        acc = evaluate_model(model, dataloader, device)
        print(f"{model_name} Top-1 Accuracy: {acc:.4%}")

if __name__ == "__main__":
    main() 