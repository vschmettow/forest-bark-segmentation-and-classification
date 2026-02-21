# train_yolov8_classifier.py
# Train YOLOv8 classification model for bark images (Picea, Pinus, Other)
# Uses YOLOv8 size 's' model

import os
from pathlib import Path
from ultralytics import YOLO
import shutil
import argparse
import json
import pandas as pd
import time

def prepare_yolo_dataset(source_dir, output_dir):
    """
    Reorganize data from:
    source_dir/
      Picea/training/, Picea/validation/
      Pinus/training/, Pinus/validation/
    
    To YOLOv8 format:
    output_dir/
      train/Picea/, train/Pinus/
      val/Picea/, val/Pinus/
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    (output_path / "train" / "Picea").mkdir(parents=True, exist_ok=True)
    (output_path / "train" / "Pinus").mkdir(parents=True, exist_ok=True)
    (output_path / "val" / "Picea").mkdir(parents=True, exist_ok=True)
    (output_path / "val" / "Pinus").mkdir(parents=True, exist_ok=True)
    
    # Copy training images
    print("Preparing training dataset...")
    for class_name in ["Picea", "Pinus"]:
        train_source = source_path / class_name / "training"
        if train_source.exists():
            image_files = list(train_source.glob("*.jpg")) + list(train_source.glob("*.JPEG")) + list(train_source.glob("*.png")) + list(train_source.glob("*.PNG"))
            for img_file in image_files:
                shutil.copy2(img_file, output_path / "train" / class_name / img_file.name)
            print(f"  {class_name}/training: {len(image_files)} images")
    
    # Copy validation images
    print("Preparing validation dataset...")
    for class_name in ["Picea", "Pinus"]:
        val_source = source_path / class_name / "validation"
        if val_source.exists():
            image_files = list(val_source.glob("*.jpg")) + list(val_source.glob("*.JPEG")) + list(val_source.glob("*.png")) + list(val_source.glob("*.PNG"))
            for img_file in image_files:
                shutil.copy2(img_file, output_path / "val" / class_name / img_file.name)
            print(f"  {class_name}/validation: {len(image_files)} images")
    
    print(f"\nDataset prepared in: {output_path}")
    return output_path

def extract_metrics_from_results(results_dir):
    """
    Extract metrics from YOLOv8 training results CSV files
    YOLOv8 saves results in results.csv
    """
    results_dir = Path(results_dir)
    csv_file = results_dir / "results.csv"
    
    if not csv_file.exists():
        return None
    
    try:
        df = pd.read_csv(csv_file)
        
        # YOLOv8 classification CSV columns typically include:
        # epoch, train/loss, metrics/accuracy_top1, metrics/accuracy_top5, etc.
        metrics = {
            'epochs': [],
            'train_loss': [],
            'val_loss': [],
            'precision': [],
            'recall': [],
            'map50': [],  # mAP@0.5
            'map': [],    # mAP@0.5:0.95
            'train_acc': [],
            'val_acc': []
        }
        
        # Extract metrics from CSV
        for idx, row in df.iterrows():
            epoch = int(row.get('epoch', idx + 1))
            metrics['epochs'].append(epoch)
            
            # Training loss
            train_loss = row.get('train/loss', row.get('train_loss', None))
            if train_loss is not None and not pd.isna(train_loss):
                metrics['train_loss'].append(float(train_loss))
            else:
                metrics['train_loss'].append(None)
            
            # Validation metrics - YOLOv8-cls uses metrics/accuracy_top1 for validation accuracy
            val_acc = row.get('metrics/accuracy_top1', row.get('val/accuracy_top1', None))
            if val_acc is not None and not pd.isna(val_acc):
                # Convert to percentage
                metrics['val_acc'].append(float(val_acc) * 100)
            else:
                metrics['val_acc'].append(None)
            
            # Training accuracy (if available)
            train_acc = row.get('metrics/accuracy_top1', None)
            if train_acc is not None and not pd.isna(train_acc):
                metrics['train_acc'].append(float(train_acc) * 100)
            else:
                metrics['train_acc'].append(None)
            
            # Precision, Recall, mAP (for classification, these might be in different columns)
            precision = row.get('metrics/precision(B)', row.get('precision', None))
            recall = row.get('metrics/recall(B)', row.get('recall', None))
            map50 = row.get('metrics/mAP50(B)', row.get('map50', None))
            map_val = row.get('metrics/mAP50-95(B)', row.get('map', None))
            
            metrics['precision'].append(float(precision) if precision is not None and not pd.isna(precision) else None)
            metrics['recall'].append(float(recall) if recall is not None and not pd.isna(recall) else None)
            metrics['map50'].append(float(map50) if map50 is not None and not pd.isna(map50) else None)
            metrics['map'].append(float(map_val) if map_val is not None and not pd.isna(map_val) else None)
            
            # Validation loss (might not be directly available in classification)
            val_loss = row.get('val/loss', None)
            if val_loss is not None and not pd.isna(val_loss):
                metrics['val_loss'].append(float(val_loss))
            else:
                # Use train loss as approximation if val loss not available
                metrics['val_loss'].append(metrics['train_loss'][-1] if metrics['train_loss'] else None)
        
        return metrics
    except Exception as e:
        print(f"Error reading results CSV: {e}")
        return None

def save_metrics_json(metrics, output_path):
    """Save metrics to JSON file for dashboard"""
    if metrics is None:
        return
    
    # Convert to list format (remove None values for cleaner JSON)
    history = {
        'train_loss': [x for x in metrics['train_loss'] if x is not None],
        'val_loss': [x for x in metrics['val_loss'] if x is not None],
        'train_acc': [x for x in metrics['train_acc'] if x is not None],
        'val_acc': [x for x in metrics['val_acc'] if x is not None],
        'precision': [x for x in metrics['precision'] if x is not None],
        'recall': [x for x in metrics['recall'] if x is not None],
        'map50': [x for x in metrics['map50'] if x is not None],
        'map': [x for x in metrics['map'] if x is not None],
    }
    
    with open(output_path, 'w') as f:
        json.dump(history, f, indent=2)

def train_yolov8(
    data_dir,
    epochs=60,
    batch_size=4,
    img_size=(600, 600),
    model_size='s',
    output_dir='../data/models/yolov8_results'
):
    """
    Train YOLOv8 classification model
    
    Args:
        data_dir: Path to dataset in YOLOv8 format (train/val subdirectories)
        epochs: Number of training epochs
        batch_size: Batch size for training
        img_size: Image size (width, height)
        model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
        output_dir: Directory to save results
    """
    print("=" * 70)
    print("YOLOV8 CLASSIFICATION TRAINING")
    print("=" * 70)
    print(f"Model: YOLOv8-{model_size} (classification)")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Image size: {img_size[0]}x{img_size[1]}")
    print(f"Data directory: {data_dir}")
    print("=" * 70)
    
    # Initialize YOLOv8 classification model
    model_name = f'yolov8{model_size}-cls.pt'
    print(f"\nInitializing model: {model_name}")
    model = YOLO(model_name)
    
    # Train the model
    print("\nStarting training...")
    print("-" * 70)
    
    # YOLOv8-cls uses square images
    # Use 600x600 for training (resizing from 600x800 original images)
    imgsz = 600
    
    print(f"Using image size: {imgsz}x{imgsz}")
    
    # Determine results directory
    results_base_dir = Path(output_dir) / 'bark_classifier'
    
    results = model.train(
        data=str(data_dir),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        project=output_dir,
        name='bark_classifier',
        exist_ok=True,
        verbose=True,
        plots=True,
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Results saved to: {results_base_dir}")
    print(f"Best model: {results_base_dir / 'weights' / 'best.pt'}")
    
    # Extract metrics and save to JSON for dashboard
    print("\nExtracting metrics for dashboard...")
    metrics = extract_metrics_from_results(results_base_dir)
    if metrics:
        metrics_json_path = results_base_dir / 'training_history.json'
        save_metrics_json(metrics, metrics_json_path)
        print(f"✓ Metrics saved to: {metrics_json_path}")
        print("  You can now use the dashboard to view training metrics!")
    else:
        print("⚠ Could not extract metrics from results.csv")
        print(f"  Results directory: {results_base_dir}")
    
    print("=" * 70)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Train YOLOv8 classification model for bark images')
    parser.add_argument('--data_dir', type=str, 
                        default='../images/training_data/training_data_small_sample',
                        help='Directory containing Picea/ and Pinus/ folders with training/ and validation/ subfolders')
    parser.add_argument('--yolo_data_dir', type=str, default=None,
                        help='Pre-organized YOLOv8 format dataset (if None, will reorganize from data_dir)')
    parser.add_argument('--epochs', type=int, default=60,
                        help='Number of training epochs (default: 60)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--img_size', type=int, nargs=2, default=[600, 600],
                        metavar=('WIDTH', 'HEIGHT'),
                        help='Image size width height - will be converted to square (default: 600 600)')
    parser.add_argument('--model_size', type=str, default='s',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size: n=nano, s=small, m=medium, l=large, x=xlarge (default: s)')
    parser.add_argument('--output_dir', type=str, default='../data/models/yolov8_results',
                        help='Output directory for training results (default: ./yolov8_results)')
    parser.add_argument('--keep_temp_data', action='store_true',
                        help='Keep temporary reorganized dataset (default: False)')
    
    args = parser.parse_args()
    
    # Prepare dataset if needed
    if args.yolo_data_dir is None:
        print("Reorganizing dataset into YOLOv8 format...")
        yolo_data_dir = Path(args.data_dir).parent / "yolo_dataset_temp"
        prepare_yolo_dataset(args.data_dir, yolo_data_dir)
        data_dir_to_use = yolo_data_dir
    else:
        data_dir_to_use = Path(args.yolo_data_dir)
        print(f"Using existing YOLOv8 dataset: {data_dir_to_use}")
    
    # Train model
    results = train_yolov8(
        data_dir=data_dir_to_use,
        epochs=args.epochs,
        batch_size=args.batch_size,
        img_size=tuple(args.img_size),
        model_size=args.model_size,
        output_dir=args.output_dir
    )
    
    # Clean up temporary dataset if not keeping it
    if args.yolo_data_dir is None and not args.keep_temp_data:
        print(f"\nCleaning up temporary dataset: {yolo_data_dir}")
        shutil.rmtree(yolo_data_dir, ignore_errors=True)
    
    print("\nTraining completed successfully!")

if __name__ == '__main__':
    main()
