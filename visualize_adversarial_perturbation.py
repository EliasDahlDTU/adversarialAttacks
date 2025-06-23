import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
from src.adversarialAttacks.models import get_model
from src.adversarialAttacks.attacks import CW

# --- CONFIGURATION ---
csv_path = "data/perturbation_analysis/resnet50_cw_20.000.csv"
test_root = "data/processed/test"
model_name = "resnet50"
num_classes = 100
c_param = 20.0
max_iter = 100  # For speed; increase for higher quality
lr = 0.01
kappa = 0.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. Find the best sample in the CSV ---
df = pd.read_csv(csv_path)
df['conf_drop'] = df['clean_true_prob'] - df['adv_true_prob']
# Filter for low SSIM drop (high SSIM)
df['ssim_drop'] = 1 - df['ssim']
filtered = df[(df['ssim'] > 0.98) & (df['clean_correct_classification']) & (~df['adv_correct_classification'])]
best = filtered.sort_values(by=['conf_drop'], ascending=False).iloc[0]
row_idx = best.name  # This is the row index in the CSV

# --- 2. Map the row to the correct image file ---
# Get sorted class folders and image files
class_folders = sorted([d for d in os.listdir(test_root) if os.path.isdir(os.path.join(test_root, d))])
img_paths = []
for cls in class_folders:
    class_dir = os.path.join(test_root, cls)
    imgs = sorted([f for f in os.listdir(class_dir) if f.lower().endswith('.jpeg') or f.lower().endswith('.jpg') or f.lower().endswith('.png')])
    img_paths.extend([os.path.join(class_dir, img) for img in imgs])
clean_img_path = img_paths[row_idx]

# --- 3. Load the clean image ---
def load_img(path):
    img = Image.open(path).convert("RGB")
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # shape: (1, C, H, W)

clean_img = load_img(clean_img_path).to(device)
label = class_folders.index(os.path.basename(os.path.dirname(clean_img_path)))
y = torch.tensor([label], dtype=torch.long, device=device)

# --- 4. Load model and regenerate adversarial image ---
model = get_model(model_name, num_classes=num_classes, pretrained=False).to(device)
model.eval()
weight_file = f"data/best_models/best_{model_name.upper()}.pth"
checkpoint = torch.load(weight_file, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

cw_attack = CW(model, c=c_param, max_iter=max_iter, lr=lr, device=device)
adv_img = cw_attack.generate(clean_img, y)
perturbation = (adv_img - clean_img).detach().cpu().squeeze().numpy()

# --- 5. Get predictions and certainties ---
def get_pred_and_certainty(img):
    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).item()
        certainty = probs.max().item()
    return pred, certainty

pred_clean, cert_clean = get_pred_and_certainty(clean_img)
pred_adv, cert_adv = get_pred_and_certainty(adv_img)

# --- 6. Class labels (use class folder names) ---
class_labels = class_folders

# --- 7. Plot ---
fig, axs = plt.subplots(1, 3, figsize=(12, 4))

# Clean image
axs[0].imshow(clean_img.cpu().squeeze().permute(1, 2, 0).numpy())
axs[0].set_title(f"Clean\n{class_labels[pred_clean]} ({cert_clean:.2%})")
axs[0].axis('off')

# Perturbation (amplified for visibility)
amp = 10
pert_vis = np.clip(perturbation * amp + 0.5, 0, 1)
axs[1].imshow(pert_vis)
axs[1].set_title("Perturbation (x10)")
axs[1].axis('off')

# Adversarial image
axs[2].imshow(adv_img.cpu().squeeze().permute(1, 2, 0).numpy())
axs[2].set_title(f"Adversarial\n{class_labels[pred_adv]} ({cert_adv:.2%})")
axs[2].axis('off')

plt.tight_layout()
plt.savefig("frontpage_figure.png", dpi=300)
plt.show()
