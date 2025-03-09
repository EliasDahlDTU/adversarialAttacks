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

1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the project in development mode:
```bash
pip install -e .
```

3. Download ImageNet subset:
The project uses a subset of ImageNet for experiments. Follow the data preparation script in `src/adversarialAttacks/data/prepare_imagenet.py` to download and prepare the dataset.

## Key Components

1. **Dataset**: ImageNet subset with diverse image categories
2. **Models**: Various CNN architectures (ResNet, VGG, etc.)
3. **Attacks**: 
   - Fast Gradient Sign Method (FGSM)
   - Projected Gradient Descent (PGD)
   - Carlini & Wagner (CW)
4. **Evaluation Metrics**:
   - Model accuracy
   - Confidence scores
   - Attack transferability
   - Visual quality metrics

## Usage

See the `notebooks/` directory for example usage and experiments:
- `01_dataset_exploration.ipynb`: ImageNet dataset analysis
- `02_model_training.ipynb`: CNN training and evaluation
- `03_adversarial_attacks.ipynb`: Implementation of attacks
- `04_transferability_study.ipynb`: Attack transferability analysis
- `05_adversarial_training.ipynb`: Defense mechanisms study