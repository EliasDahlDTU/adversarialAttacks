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
│   ├── raw/              # ImageNet dataset
│   ├── processed/        # Processed and adversarial examples
│   └── models/           # Saved model checkpoints
├── src/
│   └── adversarialAttacks/
│       ├── data/         # Dataset loading and processing
│       ├── models/       # Model architectures (CNNs)
│       ├── attacks/      # FGSM, PGD, and CW implementations
│       ├── training/     # Training utilities
│       ├── evaluation/   # Evaluation metrics
│       └── visualization/# Visualization tools
└── notebooks/           # Experiment notebooks
```

## Setup

### Prerequisites
- Python 3.8 or higher
- Git
- A package manager (pip)

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/your-username/adversarial-attacks.git
cd adversarial-attacks
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

3. Install the project and all dependencies:
```bash
pip install -e .
```

This command will read the `pyproject.toml` file and install all required packages, including:
- Core ML: torch, torchvision, numpy, scikit-learn
- Data processing: pillow, opencv-python
- Visualization: matplotlib, seaborn
- Progress tracking: tqdm, tensorboard
- Adversarial attacks: torchattacks, foolbox
- Dataset download: kaggle
- Development tools: jupyter

The installation uses the project's `pyproject.toml` file, which ensures that all dependencies are installed with compatible versions. You don't need to manually install any additional packages.

### Kaggle Authentication Setup

To download the dataset, you'll need to set up Kaggle authentication:

1. Create a Kaggle account at https://www.kaggle.com if you don't have one
2. Go to your account settings: https://www.kaggle.com/settings
3. Click on "Create New API Token" to download `kaggle.json`
4. Create the Kaggle config directory:
   ```bash
   # On Windows:
   mkdir %USERPROFILE%\.kaggle
   # On Linux/Mac:
   mkdir -p ~/.kaggle
   ```
5. Move the downloaded `kaggle.json` to:
   - Windows: `C:\Users\<username>\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json`
6. Set appropriate permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

### Verify Installation

To verify that everything is installed correctly:

```bash
python -c "import adversarialAttacks"
```

If no error appears, the installation was successful.

### Download and Prepare Dataset

After setting up Kaggle authentication:

```bash
# Download and organize the dataset
python src/adversarialAttacks/data/download_data.py

# Preprocess the images (resize to 224×224 and standardize)
python src/adversarialAttacks/data/preprocess_data.py
```

## Data Pipeline

The project uses ImageNet-100, a subset of ImageNet containing 100 classes with 1,300 images per class. The dataset is structured as follows:

### Dataset Structure
- Training data: 100 classes × 1,300 images
- Validation data: 100 classes with corresponding validation images
- Image size: Preprocessed to 224×224 pixels

### Data Requirements and Splitting

Since this project focuses on fine-tuning pre-trained models (VGG-16 and ResNet-50) rather than training from scratch, we don't need the full ImageNet dataset. The models are already pre-trained on the full ImageNet-1000 dataset, and we're only adapting their final layers to work with our subset of 100 classes.

Our data processing approach:
1. **Download and Organize**: First, we download all ImageNet-100 data (both training and validation sets)
2. **Pool All Data**: We combine all images from both sets into a single pool (~130,000 images total)
3. **Split for Experiments**: From this pooled data, we create:
   - **Training Set**: ~80% of the data (approximately 104,000 images)
   - **Testing Set**: ~20% of the data (approximately 26,000 images)
4. **Class Balance**: We maintain class balance in both sets (approximately 1,040 training images and 260 test images per class)

This split provides sufficient data to:
1. Fine-tune the models to establish a good baseline performance on our 100 classes
2. Have enough test samples to generate adversarial examples and measure their impact

The data splitting is handled in the preprocessing script, which ensures consistent splits across experiments.

### Data Processing Scripts

1. **Download Script** (`src/adversarialAttacks/data/download_data.py`):
   ```bash
   python src/adversarialAttacks/data/download_data.py
   ```
   - Downloads ImageNet-100 dataset from Kaggle
   - Organizes the training data (merges 4 training folders into one structure)
   - Creates a clean directory structure in `data/raw/`

2. **Preprocessing Script** (`src/adversarialAttacks/data/preprocess_data.py`):
   ```bash
   python src/adversarialAttacks/data/preprocess_data.py
   ```
   - Resizes all images to 224×224 pixels
   - Applies standardization
   - Saves processed images to `data/processed/`

### Directory Structure After Processing
```
data/
├── raw/
│   ├── train/
│   │   ├── class_1/
│   │   ├── class_2/
│   │   └── ...
│   └── val/
│       ├── class_1/
│       ├── class_2/
│       └── ...
└── processed/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   └── ...
    └── val/
        ├── class_1/
        ├── class_2/
        └── ...
```

## Key Components

1. **Dataset**: ImageNet-100 subset with 100 diverse image categories
2. **Pre-trained Models**: 
   - VGG-16: Fine-tuned from ImageNet-1000 to our 100 classes
   - ResNet-50: Fine-tuned from ImageNet-1000 to our 100 classes
3. **Attacks**: 
   - Fast Gradient Sign Method (FGSM)
   - Projected Gradient Descent (PGD)
   - Carlini & Wagner (CW)
4. **Evaluation Metrics**:
   - Model accuracy drop under attack
   - Confidence scores
   - Attack transferability
   - Visual quality metrics

## Usage

See the `notebooks/` directory for example usage and experiments:
- `01_dataset_exploration.ipynb`: ImageNet-100 dataset analysis
- `02_model_finetuning.ipynb`: Fine-tuning VGG-16 and ResNet-50
- `03_adversarial_attacks.ipynb`: Implementation of attacks
- `04_transferability_study.ipynb`: Attack transferability analysis
- `05_adversarial_training.ipynb`: Defense mechanisms study