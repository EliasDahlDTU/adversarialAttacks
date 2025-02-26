# Adversarial Attacks Study

This project aims to study the 'adversarialness' of different attacks on machine learning models by comparing how different adversarial attacks affect model performance.

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

## Project Structure

```
.
├── data/
│   ├── raw/         # Raw datasets (MNIST, CIFAR-10)
│   └── processed/   # Processed data and adversarial examples
├── src/
│   └── adversarialAttacks/
│       ├── datasets/     # Dataset loading utilities
│       ├── models/       # Model architectures
│       └── attacks/      # Adversarial attack implementations
└── README.md
```

## Usage

To test the dataset loading:

```bash
python src/adversarialAttacks/examples/test_datasets.py
```

This will:
1. Download MNIST and CIFAR-10 datasets (if not already present)
2. Display sample images from both datasets
3. Print dataset statistics

## Datasets

Currently implemented datasets:
- MNIST: 28x28 grayscale handwritten digits (10 classes)
- CIFAR-10: 32x32 RGB images (10 classes)

Each dataset is automatically downloaded when first accessed and stored in the `data/raw` directory.