import os
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Source and destination directories
DATASET_PATH = 'The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset'
OUTPUT_DIR = 'processed_dataset'

# Target classification directories
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
VAL_DIR = os.path.join(OUTPUT_DIR, 'val')
TEST_DIR = os.path.join(OUTPUT_DIR, 'test')

def create_directories():
    """Create necessary directories for train/val/test sets with class subdirectories."""
    # Main directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)
    
    # Class subdirectories
    classes = ['Benign', 'Malignant', 'Normal']
    for cls in classes:
        os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(VAL_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(TEST_DIR, cls), exist_ok=True)
    
    return classes

def get_files_by_class():
    """Get files from original dataset organized by class."""
    class_folders = {
        'Benign': os.path.join(DATASET_PATH, 'Bengin cases'),  # Note the typo in original dataset folder
        'Malignant': os.path.join(DATASET_PATH, 'Malignant cases'),
        'Normal': os.path.join(DATASET_PATH, 'Normal cases')
    }
    
    files_by_class = {}
    for cls, folder in class_folders.items():
        files_by_class[cls] = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.jpg')]
        print(f"Found {len(files_by_class[cls])} images for class {cls}")
    
    return files_by_class

def split_and_copy_files(files_by_class, classes):
    """Split files into train/val/test and copy to destination folders."""
    # Split ratios
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    for cls in classes:
        lstFiles = files_by_class[cls]
        # Split files
        lstTrain, lstTemp = train_test_split(lstFiles, test_size=(VAL_RATIO + TEST_RATIO), random_state=42)
        lstVal, lstTest = train_test_split(lstTemp, test_size=TEST_RATIO/(VAL_RATIO + TEST_RATIO), random_state=42)
        
        print(f"Class {cls}: {len(lstTrain)} train, {len(lstVal)} validation, {len(lstTest)} test")
        
        # Copy files
        for src in tqdm(lstTrain, desc=f"Copying {cls} train files"):
            dst = os.path.join(TRAIN_DIR, cls, os.path.basename(src))
            shutil.copy(src, dst)
            
        for src in tqdm(lstVal, desc=f"Copying {cls} validation files"):
            dst = os.path.join(VAL_DIR, cls, os.path.basename(src))
            shutil.copy(src, dst)
            
        for src in tqdm(lstTest, desc=f"Copying {cls} test files"):
            dst = os.path.join(TEST_DIR, cls, os.path.basename(src))
            shutil.copy(src, dst)

def main():
    """Main function to prepare the dataset."""
    print("Starting dataset preparation...")
    classes = create_directories()
    files_by_class = get_files_by_class()
    split_and_copy_files(files_by_class, classes)
    print("Dataset preparation completed!")

if __name__ == "__main__":
    main() 