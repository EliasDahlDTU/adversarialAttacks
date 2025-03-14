import os
import shutil
import argparse
from pathlib import Path
import kaggle
from tqdm import tqdm

def setup_directories():
    """Create necessary directories if they don't exist."""
    base_dir = Path("data")
    raw_dir = base_dir / "raw"
    raw_train_dir = raw_dir / "train"
    raw_val_dir = raw_dir / "val"

    for dir_path in [raw_dir, raw_train_dir, raw_val_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    return raw_dir, raw_train_dir, raw_val_dir

def download_dataset():
    """Download ImageNet-100 dataset from Kaggle."""
    print("Downloading ImageNet-100 dataset from Kaggle...")
    try:
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            "ambityga/imagenet100",
            path="data/raw",
            unzip=True,
            quiet=False
        )
        print("Download completed successfully!")
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("\nMake sure you have:")
        print("1. Installed kaggle package: pip install kaggle")
        print("2. Created a Kaggle account")
        print("3. Downloaded your Kaggle API token from:")
        print("   https://www.kaggle.com/settings")
        print("4. Placed the kaggle.json file in ~/.kaggle/")
        exit(1)

def organize_training_data(raw_dir: Path, raw_train_dir: Path):
    """Merge the 4 training folders into a single organized structure."""
    print("\nOrganizing training data...")
    
    # Expected structure: 4 training folders with 25 classes each
    train_folders = [f for f in raw_dir.glob("train*") if f.is_dir()]
    
    for train_folder in train_folders:
        print(f"Processing {train_folder.name}...")
        class_folders = [f for f in train_folder.glob("*") if f.is_dir()]
        
        for class_folder in tqdm(class_folders):
            # Create destination folder
            dest_folder = raw_train_dir / class_folder.name
            dest_folder.mkdir(exist_ok=True)
            
            # Move all images
            for img_file in class_folder.glob("*.JPEG"):
                shutil.move(str(img_file), str(dest_folder / img_file.name))
        
        # Remove the original train folder after moving its contents
        shutil.rmtree(train_folder)

def organize_validation_data(raw_dir: Path, raw_val_dir: Path):
    """Organize validation data into the proper structure."""
    print("\nOrganizing validation data...")
    
    val_source = raw_dir / "val"
    if val_source.exists():
        # Move all validation class folders to the final location
        for class_folder in tqdm(list(val_source.glob("*"))):
            if class_folder.is_dir():
                dest_folder = raw_val_dir / class_folder.name
                if not dest_folder.exists():
                    shutil.move(str(class_folder), str(dest_folder))
        
        # Remove the original val folder if it's empty
        if not any(val_source.iterdir()):
            val_source.rmdir()

def cleanup(raw_dir: Path):
    """Clean up any remaining temporary files."""
    print("\nCleaning up temporary files...")
    
    # Remove any zip files
    for zip_file in raw_dir.glob("*.zip"):
        zip_file.unlink()
    
    # Remove any other temporary files or empty directories
    for item in raw_dir.glob("*"):
        if item.name not in ["train", "val"]:
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

def main():
    parser = argparse.ArgumentParser(description="Download and organize ImageNet-100 dataset")
    args = parser.parse_args()

    print("Setting up data pipeline for ImageNet-100...")
    raw_dir, raw_train_dir, raw_val_dir = setup_directories()
    
    download_dataset()
    organize_training_data(raw_dir, raw_train_dir)
    organize_validation_data(raw_dir, raw_val_dir)
    cleanup(raw_dir)
    
    print("\nDataset organization completed!")
    print(f"Training data location: {raw_train_dir}")
    print(f"Validation data location: {raw_val_dir}")

if __name__ == "__main__":
    main() 