import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import transforms
from src.adversarialAttacks.attacks.fgsm import FGSM
import argparse

def load_image(image_path):
    """Load and preprocess a single image."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

def tensor_to_image(tensor):
    """Convert a tensor to a numpy image."""
    img = tensor.squeeze().detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    img = np.clip(img, 0, 1)
    return img

def plot_attack_progression(original_image, model, target_label, output_path, epsilon_range=(0, 0.19), epsilon_step=0.01):
    """Plot the progression of FGSM attacks with increasing epsilon."""
    epsilons = np.arange(epsilon_range[0], epsilon_range[1] + epsilon_step, epsilon_step)
    num_attacks = len(epsilons)
    
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_attacks)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    fig.suptitle('FGSM Attack Progression (Epsilon Values)', fontsize=16)
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    # Generate and plot adversarial examples for each epsilon
    for idx, epsilon in enumerate(epsilons):
        # Create FGSM attack with current epsilon
        attack = FGSM(model, epsilon=epsilon)
        
        # Generate adversarial example
        adv_image = attack.generate(original_image, target_label)
        
        # Plot the result
        ax = axes_flat[idx]
        ax.imshow(tensor_to_image(adv_image))
        ax.set_title(f'ε = {epsilon:.2f}')
        ax.axis('off')
    
    # Remove empty subplots
    for idx in range(num_attacks, len(axes_flat)):
        fig.delaxes(axes_flat[idx])
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize FGSM attack progression')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--target-label', type=int, default=0, help='Target label for the attack')
    parser.add_argument('--output', type=str, default='fgsm_progression.png', help='Output path for the visualization')
    parser.add_argument('--min-epsilon', type=float, default=0.0, help='Minimum epsilon value')
    parser.add_argument('--max-epsilon', type=float, default=0.19, help='Maximum epsilon value')
    parser.add_argument('--epsilon-step', type=float, default=0.01, help='Epsilon step size')
    args = parser.parse_args()

    # Load pretrained ResNet model
    model = models.resnet50(pretrained=True)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load and preprocess input image
    image = load_image(args.image_path)
    image = image.to(device)
    
    # Create target label tensor
    target_label = torch.tensor([args.target_label]).to(device)

    # Generate and plot attack progression
    plot_attack_progression(
        image, 
        model, 
        target_label,
        args.output,
        epsilon_range=(args.min_epsilon, args.max_epsilon),
        epsilon_step=args.epsilon_step
    )

    print(f"Attack progression visualization saved to {args.output}")

if __name__ == '__main__':
    main() 