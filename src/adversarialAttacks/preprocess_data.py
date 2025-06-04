import os
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import shutil
import random

def setup_directories(clear_existing: bool = True) -> Tuple[Path, Path, Path, Path]:
    """Create necessary directories if they don't exist and optionally clear them."""
    base_dir = Path("data")
    raw_dir = base_dir / "raw" / "archive"
    processed_dir = base_dir / "processed"
    
    # Create processed directories
    processed_train_dir = processed_dir / "train"
    processed_val_dir = processed_dir / "val"
    
    # Clear existing processed directories if requested
    if clear_existing and processed_dir.exists():
        print("Clearing existing processed directories...")
        if processed_train_dir.exists():
            shutil.rmtree(processed_train_dir)
        if processed_val_dir.exists():
            shutil.rmtree(processed_val_dir)
    
    # Create directories
    for dir_path in [processed_dir, processed_train_dir, processed_val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return raw_dir, processed_dir, processed_train_dir, processed_val_dir

def resize_and_standardize(img_path: Path, output_path: Path, target_size: Tuple[int, int] = (224, 224)):
    """Resize image to target size and apply standardization."""
    try:
        # Open and convert image to RGB (in case it's grayscale)
        img = Image.open(img_path).convert('RGB')
        
        # Resize image
        img = img.resize(target_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array and standardize
        img_array = np.array(img).astype(np.float32)
        
        # Standardize to [0, 1] range
        img_array = img_array / 255.0
        
        # Convert back to uint8 for saving
        img_array = (img_array * 255).astype(np.uint8)
        
        # Save processed image
        Image.fromarray(img_array).save(output_path, quality=95)
        return True
    except Exception as e:
        print(f"Error processing {img_path}: {str(e)}")
        return False

def collect_all_images(raw_dir: Path) -> Dict[str, List[Path]]:
    """Collect all images from both train and val directories, organized by class."""
    print("\nCollecting all images from train and val directories...")
    
    # Dictionary to store all images by class
    all_images_by_class = {}
    
    # Process training directories (train.X1, train.X2, train.X3, train.X4)
    train_dirs = [d for d in raw_dir.glob("train.X*") if d.is_dir()]
    
    if not train_dirs:
        print("No training directories found! Expected train.X1, train.X2, etc.")
        return {}
    
    for train_dir in train_dirs:
        print(f"Processing {train_dir.name}...")
        for class_dir in train_dir.glob("*"):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in all_images_by_class:
                    all_images_by_class[class_name] = []
                
                # Add all images from this class directory
                all_images_by_class[class_name].extend(list(class_dir.glob("*.JPEG")))
    
    # Process validation directory (val.X)
    val_dir = raw_dir / "val.X"
    if val_dir.exists():
        print(f"Processing {val_dir.name}...")
        for class_dir in val_dir.glob("*"):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in all_images_by_class:
                    all_images_by_class[class_name] = []
                
                # Add all images from this class directory
                all_images_by_class[class_name].extend(list(class_dir.glob("*.JPEG")))
    
    # Print statistics
    total_images = sum(len(images) for images in all_images_by_class.values())
    print(f"Found {total_images} images across {len(all_images_by_class)} classes")
    
    return all_images_by_class

def get_clean_filename(img_path: Path) -> str:
    """Remove 'val_' prefix from filenames to avoid confusion."""
    filename = img_path.name
    if filename.startswith("val_"):
        return filename[4:]  # Remove 'val_' prefix
    return filename

def split_and_process_images(
    all_images_by_class: Dict[str, List[Path]],
    processed_train_dir: Path,
    processed_val_dir: Path,
    target_size: Tuple[int, int],
    train_ratio: float = 0.8,
    seed: int = 42
):
    """Split images into train and test sets, then process them."""
    random.seed(seed)
    
    # Create class directories in processed folders
    for class_name in all_images_by_class.keys():
        (processed_train_dir / class_name).mkdir(exist_ok=True)
        (processed_val_dir / class_name).mkdir(exist_ok=True)
    
    # Collect all processing tasks
    train_tasks = []
    val_tasks = []
    
    print("\nSplitting data into train and test sets...")
    for class_name, images in all_images_by_class.items():
        # Shuffle images
        random.shuffle(images)
        
        # Split into train and test
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Create processing tasks
        for img_path in train_images:
            # Use clean filename (remove 'val_' prefix if present)
            clean_filename = get_clean_filename(img_path)
            output_path = processed_train_dir / class_name / clean_filename
            train_tasks.append((img_path, output_path, target_size))
        
        for img_path in val_images:
            # Use clean filename (remove 'val_' prefix if present)
            clean_filename = get_clean_filename(img_path)
            output_path = processed_val_dir / class_name / clean_filename
            val_tasks.append((img_path, output_path, target_size))
        
        print(f"Class {class_name}: {len(train_images)} train, {len(val_images)} test")
    
    # Process images in parallel
    num_workers = mp.cpu_count()
    
    # Process training images
    print(f"\nProcessing {len(train_tasks)} training images using {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda x: resize_and_standardize(*x), train_tasks),
            total=len(train_tasks)
        ))
    
    train_successful = sum(1 for r in results if r)
    train_failed = len(results) - train_successful
    print(f"Processed {train_successful} training images successfully, {train_failed} failed")
    
    # Process validation/test images
    print(f"\nProcessing {len(val_tasks)} test images using {num_workers} workers...")
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(
            executor.map(lambda x: resize_and_standardize(*x), val_tasks),
            total=len(val_tasks)
        ))
    
    val_successful = sum(1 for r in results if r)
    val_failed = len(results) - val_successful
    print(f"Processed {val_successful} test images successfully, {val_failed} failed")
    
    return train_successful, val_successful

def main():
    parser = argparse.ArgumentParser(description="Preprocess ImageNet-100 dataset")
    parser.add_argument("--size", type=int, default=224,
                      help="Target size for image resizing (default: 224)")
    parser.add_argument("--train-ratio", type=float, default=0.8,
                      help="Ratio of data to use for training (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-clear", action="store_true",
                      help="Don't clear existing processed directories")
    args = parser.parse_args()
    
    target_size = (args.size, args.size)
    
    print(f"Setting up preprocessing pipeline:")
    print(f"- Target image size: {target_size}")
    print(f"- Train/test split: {args.train_ratio:.0%}/{1-args.train_ratio:.0%}")
    print(f"- Random seed: {args.seed}")
    print(f"- Clear existing processed directories: {not args.no_clear}")
    
    raw_dir, processed_dir, processed_train_dir, processed_val_dir = setup_directories(clear_existing=not args.no_clear)
    
    # Check if raw directories exist
    train_dirs = list(raw_dir.glob("train.X*"))
    val_dir = raw_dir / "val.X"
    
    if not train_dirs and not val_dir.exists():
        print("No training or validation directories found! Expected train.X1, train.X2, etc. and val.X")
        exit(1)
    
    # Collect all images from both train and val directories
    all_images_by_class = collect_all_images(raw_dir)
    
    if not all_images_by_class:
        print("No images found in the expected directories!")
        exit(1)
    
    # Split and process images
    train_count, val_count = split_and_process_images(
        all_images_by_class,
        processed_train_dir,
        processed_val_dir,
        target_size,
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    print("\nPreprocessing completed!")
    print(f"Processed data location: {processed_dir}")
    print(f"Training images: {train_count}")
    print(f"Test images: {val_count}")

if __name__ == "__main__":
    main() 