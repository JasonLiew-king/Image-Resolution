import argparse
import os
import shutil
import glob 
import random

def split_dataset(input_dir, output_dir, split_ratio = 0.8):
    #create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok = True)
    os.makedirs(val_dir, exist_ok = True)
    
    # Get all image files from input directory
    image_paths = glob.glob(os.path.join(input_dir, "*.*")) #supports all file format
    random.shuffle(image_paths) # Shuffle the dataset for randomness
    
    #Split dataset to 80-20
    split_index = int(len(image_paths) * split_ratio)
    train_images = image_paths[:split_index] #80%
    val_images = image_paths[split_index:] #20%
    
    #Copy images to respective folders
    for img_path in train_images:
        shutil.copy(img_path, os.path.join(train_dir, os.path.basename(img_path)))

    for img_path in val_images:
        shutil.copy(img_path, os.path.join(val_dir, os.path.basename(img_path)))
    
    print(f"Dataset split completed!")
    print(f"Training images: {len(train_images)} → {train_dir}")
    print(f"Validation images: {len(val_images)} → {val_dir}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into training (80%) and validation (20%)")
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing images")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory for the split dataset")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Training split ratio (default: 80%)")

    args = parser.parse_args()

    split_dataset(args.input_dir, args.output_dir, args.split_ratio)
        
        
        