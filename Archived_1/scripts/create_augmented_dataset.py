# create_augmented_dataset.py
# Create an augmented training dataset by applying random augmentations to images
# Structure: Same as training_data_small_sample

import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import argparse

def apply_random_augmentations(image):
    """
    Apply random augmentations to an image:
    - Random downsizing (70-95% of original size)
    - Random slight Gaussian blur (kernel size 3-5)
    - Random mild noise (Gaussian noise with small std)
    """
    # Make a copy to avoid modifying original
    augmented = image.copy()
    
    # 1. Random downsizing (40-80% of original)
    scale_factor = random.uniform(0.30, 0.80)
    h, w = augmented.shape[:2]
    new_h = int(h * scale_factor)
    new_w = int(w * scale_factor)
    
    # Resize down
    augmented = cv2.resize(augmented, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # Resize back up to original size (simulates lower quality/resolution)
    augmented = cv2.resize(augmented, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # 2. Random slight Gaussian blur (small kernel, subtle effect)
    kernel_size = random.choice([3, 5])  # Small kernel sizes
    sigma_x = random.uniform(0.5, 1.5)  # Small sigma for subtle blur
    augmented = cv2.GaussianBlur(augmented, (kernel_size, kernel_size), sigma_x)
    
    # 3. Random mild noise (Gaussian noise)
    noise_std = random.uniform(3, 8)  # Small noise amount
    noise = np.random.normal(0, noise_std, augmented.shape).astype(np.float32)
    augmented = augmented.astype(np.float32) + noise
    # Clip to valid range [0, 255]
    augmented = np.clip(augmented, 0, 255).astype(np.uint8)
    
    return augmented

def create_augmented_dataset(source_dir, output_dir, seed=None):
    """
    Create augmented dataset with same structure as source
    
    Args:
        source_dir: Source directory (e.g., training_data_small_sample)
        output_dir: Output directory for augmented dataset
        seed: Random seed for reproducibility (optional)
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(list(source_path.rglob(f'*{ext}')))
    
    print(f"Found {len(image_files)} images to process")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    print("=" * 70)
    
    # Process each image
    processed = 0
    errors = 0
    
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Calculate relative path from source
            rel_path = img_path.relative_to(source_path)
            
            # Create output path (same structure)
            output_file = output_path / rel_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load {img_path}")
                errors += 1
                continue
            
            # Apply augmentations
            augmented_img = apply_random_augmentations(img)
            
            # Save augmented image
            # Preserve original extension
            ext = img_path.suffix
            cv2.imwrite(str(output_file), augmented_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            processed += 1
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            errors += 1
            continue
    
    print("=" * 70)
    print(f"Processing complete!")
    print(f"  Processed: {processed} images")
    print(f"  Errors: {errors} images")
    print(f"  Output directory: {output_path}")
    
    return processed, errors

def main():
    parser = argparse.ArgumentParser(
        description='Create augmented training dataset with random augmentations'
    )
    parser.add_argument(
        '--source_dir',
        type=str,
        default='../images/training_data/training_data_small_sample',
        help='Source directory (default: ./training_data_small_sample)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='../images/training_data/training_data_augmented',
        help='Output directory (default: ./training_data_augmented)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility (optional)'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite output directory if it exists'
    )
    
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    
    # Check if output directory exists
    if output_path.exists() and not args.overwrite:
        response = input(f"Output directory {output_path} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            return
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create augmented dataset
    processed, errors = create_augmented_dataset(
        args.source_dir,
        args.output_dir,
        seed=args.seed
    )
    
    if errors == 0:
        print("\n✓ All images processed successfully!")
    else:
        print(f"\n⚠ {errors} images had errors during processing")

if __name__ == '__main__':
    main()

