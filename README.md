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
│   ├── raw/                      # ImageNet100 dataset
│   ├── processed/                # Preprocessed dataset
│   ├── best_models/              # Best VGG and ResNet model after finetuning
│   └── adversarial_examples/     # Generated adversarial examples from different methods
│       ├── FGSM/                 
│       ├── CW/                   
│       └── PGD/                  
├── src/
│   └── adversarialAttacks/
│       ├── preprocess_data.py   # Dataset preprocessing
│       ├── models.py            # Model architecture and pretrained weight implementation
│       ├── attacks.py           # FGSM, PGD, and CW implementations and generation of adversarial examples
│       ├── training.py          # Fine-tuning script for VGG and ResNet
│       ├── evaluation.py        # Evaluation metrics
│       └── visualization.py  # Visualization tools
└── notebooks/           # Experiment notebooks
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

The project uses ImageNet-100, a subset of ImageNet containing 100 classes with 1,300 images per class. Since this project focuses on fine-tuning pre-trained models (VGG-16 and ResNet-50) rather than training from scratch, we don't need the full ImageNet dataset. The models are already pre-trained on the ImageNet-1000 dataset, and we're only adapting their final layers to work with our subset of 100 classes. The dataset is structured as follows:

### Dataset Structure
- Training data: 100 classes × 1,300 images
- Validation data: 100 × 50 images
- Image size: Preprocessed to 224×224 pixels


Our data processing approach:
1. **Download and Organize**: First, we download all ImageNet-100 data (both training and validation sets)
2. **Pool All Data**: We combine all images from both sets into a single pool (135,000 images total)
3. **Split for Experiments**: From this pooled data, we create:
   - **Training Set**: 80% of the data (108,000 images)
   - **Testing Set**: 20% of the data (27,000 images)
4. **Class Balance**: We maintain class balance in both sets (approximately 1,080 training images and 270 test images per class)

The data splitting is handled in the preprocessing script, which ensures consistent splits across experiments.

### Data Download and Preprocessing
1. Go to https://www.kaggle.com/datasets/ambityga/imagenet100 and download the zip-file containing the dataset.
2. Place the zip-file in the data\raw folder and unzip it.

The unpacked file-structure should look as follows (This is just how the creator of the sub-dataset decided to structure it, we will improve):
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
This:
- Resizes all images to 224×224 pixels
- Applies standardization to images
- Saves preprocessed images to `data/processed/`

The processed file-structure should look as follows :
```
data/
├── raw/
└── processed/
    ├── train/
    │   ├── class_1/
    │   ├── class_2/
    │   ├── ...
    │   └── class_100/
    └── val/
        ├── class_1/
        ├── class_2/
        ├── ...
        └── class_100/
```

## Pretrained Models

We have two models bla bla bla ... Pytorch implementation ... ...

## Fine-tuning / training
we fine-tune the models with 108k 


## Adversarial 
