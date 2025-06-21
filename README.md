# Adversarial Attacks for Image Protection

This project investigates the use of adversarial attacks as a tool for protecting intellectual property, particularly focusing on defending creative works from unauthorized AI scraping and model training. We explore how different adversarial attack methods can disrupt AI model training while maintaining visual integrity of the original images.

## Research Questions

1. Which adversarial attack (FGSM, PGD, CW) causes the most significant performance drop in Convolutional Neural Networks?
2. How generalizable are these attacks when trained on a specific model to other models trained on the same subset of data?
3. How effective is adversarial training as a defense mechanism against these attacks?

## Project Structure

```
.
├── data/
│   ├── raw/                                        # Original ImageNet100 dataset (unprocessed)
│   ├── processed/                                  # Preprocessed dataset (train/val/test splits)
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── best_models/                                # Best VGG and ResNet model weights after fine-tuning
│   ├── adversarial_examples/                       # Generated adversarial examples by attack type
│   │   ├── FGSM/
│   │   ├── CW/
│   │   └── PGD/
│   ├── perturbation_analysis/                      # CSVs with per-image attack/perturbation results
│   ├── training_history/                           # (If used) Training logs or histories
│   └── .gitkeep
├── results/                                        # Plots and figures (e.g., TCRR plots)
│   
├── src/
│   ├── analyze_results.py                          # Analysis and plotting of results
│   └── adversarialAttacks/
│       ├── __init__.py
│       ├── models.py                               # Model architectures and loading utilities
│       ├── attacks/                                # Implementation of the three adversarial attacks
│       │   ├── __init__.py                         
│       │   ├── base_attack.py                      
│       │   ├── fgsm.py                             
│       │   ├── pgd.py                     
│       │   └── cw.py
│       ├── data/                                   # Scripts used for preparing and moving large amounts of image data
│       │   ├── __init__.py
│       │   ├── preprocess_data.py
│       │   └── data_loader.py
│       ├── evaluation/                             # Scripts used for measuring and quantifying model degradation due to attacks
│       │   ├── __init__.py
│       │   ├── evaluate_perturbations.py
│       │   ├── evaluate_transferability.py
│       │   ├── save_image.py
│       │   ├── test_baseline_accuracy.py
│       │   └── hpc.lsf
│       ├── training/                               # for fine-tuning model weights of VGG16 and ResNet50 for the data
│       │   ├── __init__.py
│       │   ├── train.py
│       │   └── fast_train.py
│       └── __pycache__/
│           └── ... (compiled Python files)
├── requirements.txt                                # Python dependencies for the project
├── README.md                                       # You are here! :D
└── .gitignore
```

## Setup

1. Clone the repository:
```bash
git clone https://github.com/EliashDahlDTU/adversarialAttacks.git
cd adversarialAttacks
```

2. Create a virtual environment and activate it:
```bash
python -m venv adversarial_attacks
# On Windows:
adversarial_attacks\Scripts\activate
# On Linux/Mac:
source adversarial_attacks/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Data Pipeline

The project uses ImageNet-100, a subset of ImageNet-1000 containing 100 classes with 1,350 images per class. Since this project focuses on fine-tuning pre-trained models (VGG-16 and ResNet-50) rather than training from scratch, we don't need the full ImageNet dataset. The models are already pre-trained on the ImageNet-1000 dataset, and we're only adapting their final layers to work with our subset of 100 classes.

### Dataset Structure
- Total images per class: 1,350 images
- Image size: Preprocessed to 224×224 pixels
- Data split:
  - Training set: 80% of data (~1,080 images per class)
  - Validation set: 10% of data (~135 images per class)
  - Test set: 10% of data (~135 images per class)

Our data processing approach:
1. **Download and Organize**: First, we download all ImageNet-100 data
2. **Pool All Data**: We combine all images from all original folders into a single pool (135,000 images total)
3. **Split for Experiments**: From this pooled data, we create:
   - **Training Set**: 80% of the data (108,000 images)
   - **Validation Set**: 10% of the data (13,500 images)
   - **Test Set**: 10% of the data (13,500 images)
4. **Class Balance**: We maintain equal class representation in all splits

The data splitting is handled in the src/adversarialAttacks/data/preproces_data.py script, which by seeding the split ensures consistent and reproducible results across experiments.

### Data Download and Preprocessing
1. Go to https://www.kaggle.com/datasets/ambityga/imagenet100 and download the zip-file containing the dataset.
2. Place the zip-file in the data\raw folder and unzip it.

The unpacked file-structure should look as follows:
```
data/
├── raw/
│   ├── train.X1/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   └── ...
│   ├── train.X2/
│   │   ├── class_26/
│   │   ├── class_27/
│   │   └── ...
│  ...
│   ├── train.X4/
│   │   ├── class_76/
│   │   ├── class_77/
│   │   ├── ...
│   │   └── class_100/
│   └── val.X/
│       ├── class_1/
│       ├── class_2/
│       ├── ...
│       └── class_100/
└── processed/
```

NB: The zip-file will sometimes unpack into a folder named "archive". If this happens to you, simply move the subfolders directly into the raw data folder.

3. Now we preprocess the raw data with the following script:
   ```bash
   python src/adversarialAttacks/data/preprocess_data.py
   ```
This script:
- Resizes all images to 224×224 pixels
- Applies standardization to images
- Splits data into train/val/test sets (80/10/10)
- Saves preprocessed images to `data/processed/`

The processed file-structure will look as follows:
```
data/
├── raw/
└── processed/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   ├── ...
    │   └── class_100/
    ├── val/
    │   ├── class_1/
    │   ├── class_2/
    │   ├── ...
    │   └── class_100/
    └── test/
        ├── class_1/
        ├── class_2/
        ├── ...
        └── class_100/
```

## Training the Models

This project leverages two pretrained deep learning models, specifically VGG-16 and ResNet50, both of which are widely used convolutional neural networks originally trained on the full ImageNet-1000 dataset. Instead of training these models from scratch, we adapt (fine-tune) their final layers to work with our custom ImageNet-100 subset.

### Workflow

1. **Load Pretrained Weights:**  
   Both VGG-16 and ResNet-50 are initialized with weights pretrained on ImageNet-1000, ensuring strong feature extraction from the start.

2. **Freeze Most Layers:**  
   To preserve the general visual features learned from ImageNet-1000, we freeze all convolutional layers and only fine-tune the final classification layer. This approach speeds up training and reduces the risk of overfitting on the smaller dataset.

3. **Modify Output Layers:**  
   The final classification layers are replaced to output predictions for 100 classes (matching our dataset).

4. **Fine-tune on ImageNet-100:**  
   The models are trained further (fine-tuned) using our preprocessed training set (108,000 images), allowing them to adapt to the specific distribution and classes of ImageNet-100.  
   - Validation and test sets are used to monitor performance and prevent overfitting. (Although in practice, we saw little to no overfitting and solely used the validation to pick the best model.)

5. **Saving Best Models:**  
   The best-performing model weights (based on validation accuracy) are saved in `data/best_models/` for later use in adversarial attack and evaluation experiments.

### Running the Training

To train and fine-tune the models, use the scripts provided in `src/adversarialAttacks/training/`:

```bash
python src/adversarialAttacks/training/train.py
```

or, for a faster training routine (e.g., for debugging or quick experiments):

```bash
python src/adversarialAttacks/training/fast_train.py
```

These scripts will:
- Load the preprocessed data
- Initialize the chosen model with pretrained weights
- Freeze all layers except the final classification layer
- Replace the final layer for 100-class classification
- Train and validate the model, saving the best weights

## Adversarial Attacks and Benchmarking Results

### Baseline Accuracy

To measure the baseline (clean) accuracy of the models (i.e., accuracy on unperturbed test images), use the script:

```bash
python src/adversarialAttacks/evaluation/test_baseline_accuracy.py
```

This script evaluates the trained models on the clean test set and reports their standard classification accuracy, providing a baseline result.

To evaluate the robustness of our models, we generate adversarial examples (AEs) using three well-known attack methods: FGSM, PGD, and CW. These attacks are implemented in the `src/adversarialAttacks/attacks/` folder, where each attack is defined as a class inheriting from a common base (`base_attack.py`). This modular structure allows for easy extension and consistent usage across experiments.

### How Attacks Are Performed

1. **Attack Definition:**  
   - The attacks (FGSM, PGD, CW) are implemented in their respective files in `src/adversarialAttacks/attacks/`.
   - Each attack class provides a `generate` method, which takes a batch of images and labels and returns adversarially perturbed images designed to fool the model.

2. **Generating Adversarial Examples:**  
   - The main script for generating and evaluating adversarial examples is `src/adversarialAttacks/evaluation/evaluate_perturbations.py`.
   - This script loads a trained model, selects an attack and its perturbation parameter ($\epsilon$ for FGSM, PGD, or $c$ for CW), and iterates over the test dataset.
   - For each batch, it:
     - Computes the model's predictions on the clean images.
     - Generates adversarial examples using the selected attack.
     - Computes the model's predictions on the adversarial images.
     - Records various metrics, including whether the adversarial image was classified correctly, the true class confidence scores, and the size of the perturbation in regards to $L_2$ euclidian distance and a metric called SSIM (which we use as a proxy for how much humans see the difference).

3. **Saving Results:**  
   - All results are saved as CSV files in `data/perturbation_analysis/`, with each row corresponding to a specific image in the test set.
   - These CSVs include information such as:
     - Whether the adversarial image was classified correctly
     - The model's confidence before and after the attack
     - The L2 norm and SSIM of the perturbation

4. **Benchmarking:**  
   - The results can be further analyzed and visualized using scripts like `src/analyze_results.py`, which compute robust accuracy and other metrics across different attack strengths.

### Running the Attacks

To benchmark a model against adversarial attacks, run:

```bash
python src/adversarialAttacks/evaluation/evaluate_perturbations.py
```

You can specify the model, attack type, and parameters within the script or by modifying its arguments.


