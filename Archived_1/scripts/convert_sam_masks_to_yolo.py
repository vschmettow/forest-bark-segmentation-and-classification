# convert_sam_masks_to_yolo.py
# Convert SAM-generated masks to YOLO segmentation format for training

import cv2
import numpy as np
from pathlib import Path
import json
import argparse

def sam_mask_to_yolo_format(mask_array, image_width, image_height):
    """
    Convert SAM mask array to YOLO segmentation format
    
    Args:
        mask_array: Binary mask array (numpy array)
        image_width: Original image width
        image_height: Original image height
    
    Returns:
        YOLO format string: "class_id x1 y1 x2 y2 x3 y3 ..." (normalized coordinates)
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask_array.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return None
    
    # Get largest contour (main object)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Simplify contour (reduce points)
    epsilon = 0.001 * cv2.arcLength(largest_contour, True)
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Normalize coordinates to [0, 1]
    yolo_points = []
    for point in simplified:
        x = float(point[0][0]) / image_width
        y = float(point[0][1]) / image_height
        # Ensure within bounds
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        yolo_points.append(f"{x:.6f} {y:.6f}")
    
    # YOLO format: class_id x1 y1 x2 y2 ...
    # Class 0 = "bark" (only one class for segmentation)
    return f"0 {' '.join(yolo_points)}"

def convert_sam_result_to_yolo(sam_result, image_path, output_label_path):
    """
    Convert a single SAM result to YOLO format
    
    Args:
        sam_result: SAM mask dictionary with 'segmentation' key
        image_path: Path to original image
        output_label_path: Path to save YOLO label file
    """
    # Load image to get dimensions
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Warning: Could not load image {image_path}")
        return False
    
    h, w = image.shape[:2]
    
    # Get mask array
    mask_array = sam_result['segmentation']
    
    # Convert to YOLO format
    yolo_label = sam_mask_to_yolo_format(mask_array, w, h)
    
    if yolo_label is None:
        print(f"Warning: No valid mask found for {image_path}")
        return False
    
    # Save label file
    output_label_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_label_path, 'w') as f:
        f.write(yolo_label + '\n')
    
    return True

def batch_convert_sam_results(sam_results_dir, images_dir, output_dir):
    """
    Batch convert SAM results to YOLO format
    
    Args:
        sam_results_dir: Directory containing SAM results (JSON files with masks)
        images_dir: Directory containing original images
        output_dir: Output directory for YOLO labels
    """
    sam_results_dir = Path(sam_results_dir)
    images_dir = Path(images_dir)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all SAM result files
    sam_files = list(sam_results_dir.glob('*.json'))
    
    print(f"Found {len(sam_files)} SAM result files")
    
    converted = 0
    for sam_file in sam_files:
        # Load SAM results
        with open(sam_file, 'r') as f:
            sam_data = json.load(f)
        
        # Get corresponding image
        image_name = sam_file.stem  # Assuming same name
        image_path = images_dir / f"{image_name}.jpg"
        
        if not image_path.exists():
            # Try other extensions
            for ext in ['.png', '.jpeg', '.JPG', '.PNG']:
                alt_path = images_dir / f"{image_name}{ext}"
                if alt_path.exists():
                    image_path = alt_path
                    break
        
        if not image_path.exists():
            print(f"Warning: Image not found for {sam_file}")
            continue
        
        # Output label path
        label_path = output_dir / f"{image_name}.txt"
        
        # Convert each mask in the SAM result
        if isinstance(sam_data, list):
            # Multiple masks
            all_labels = []
            for mask_result in sam_data:
                yolo_label = sam_mask_to_yolo_format(
                    np.array(mask_result['segmentation']),
                    # Need image dimensions - load image
                )
                if yolo_label:
                    all_labels.append(yolo_label)
            
            if all_labels:
                with open(label_path, 'w') as f:
                    f.write('\n'.join(all_labels) + '\n')
                converted += 1
        else:
            # Single mask
            if convert_sam_result_to_yolo(sam_data, image_path, label_path):
                converted += 1
    
    print(f"\nConverted {converted} files to YOLO format")
    print(f"Labels saved to: {output_dir}")

def create_dataset_yaml(output_dir, train_ratio=0.8):
    """
    Create YOLO dataset YAML file and split train/val
    
    Args:
        output_dir: Directory containing images and labels
        train_ratio: Ratio of training data (default: 0.8)
    """
    output_dir = Path(output_dir)
    
    # Create train/val directories
    (output_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'train' / 'labels').mkdir(parents=True, exist_ok=True)
    (output_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
    (output_dir / 'val' / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Find all images
    images = list((output_dir / 'images').glob('*.*'))
    images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    # Shuffle and split
    import random
    random.shuffle(images)
    split_idx = int(len(images) * train_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Move files
    for img in train_images:
        label = output_dir / 'labels' / f"{img.stem}.txt"
        if label.exists():
            img.rename(output_dir / 'train' / 'images' / img.name)
            label.rename(output_dir / 'train' / 'labels' / label.name)
    
    for img in val_images:
        label = output_dir / 'labels' / f"{img.stem}.txt"
        if label.exists():
            img.rename(output_dir / 'val' / 'images' / img.name)
            label.rename(output_dir / 'val' / 'labels' / label.name)
    
    # Create dataset.yaml
    yaml_content = f"""# Bark Segmentation Dataset
path: {output_dir.absolute()}
train: train/images
val: val/images

# Classes
names:
  0: bark
"""
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"Dataset YAML created: {yaml_path}")
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert SAM masks to YOLO segmentation format'
    )
    parser.add_argument(
        '--sam_results',
        type=str,
        help='Directory containing SAM results (JSON files)'
    )
    parser.add_argument(
        '--images',
        type=str,
        help='Directory containing original images'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for YOLO format dataset'
    )
    parser.add_argument(
        '--create_yaml',
        action='store_true',
        help='Create dataset.yaml and split train/val'
    )
    parser.add_argument(
        '--train_ratio',
        type=float,
        default=0.8,
        help='Training data ratio (default: 0.8)'
    )
    
    args = parser.parse_args()
    
    if args.sam_results and args.images:
        batch_convert_sam_results(
            args.sam_results,
            args.images,
            args.output
        )
    
    if args.create_yaml:
        create_dataset_yaml(args.output, args.train_ratio)

if __name__ == '__main__':
    main()
