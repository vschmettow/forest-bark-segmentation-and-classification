# train_bark_segmentation.py
# Train YOLOv8 instance segmentation model for bark detection
# This replaces SAM1/SAM2 with a custom-trained segmentation model

from ultralytics import YOLO
from pathlib import Path
import argparse

def train_bark_segmentation(
    data_yaml='dataset.yaml',
    model_size='n',  # n=nano (default), s, m, l, x
    epochs=100,
    imgsz=640,
    batch=16,
    project='../segmentation_results',
    name='bark_segmentation'
):
    """
    Train YOLOv8 instance segmentation model for bark detection
    
    Args:
        data_yaml: Path to dataset YAML file
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        epochs: Number of training epochs
        imgsz: Image size (square)
        batch: Batch size
        project: Project directory for results
        name: Experiment name
    """
    print("=" * 70)
    print("YOLOV8 INSTANCE SEGMENTATION TRAINING")
    print("=" * 70)
    print(f"Model: YOLOv8-{model_size}-seg (instance segmentation)")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Image size: {imgsz}x{imgsz}")
    print(f"Dataset: {data_yaml}")
    print("=" * 70)
    
    # Load YOLOv8 segmentation model
    model_name = f'yolov8{model_size}-seg.pt'
    print(f"\nLoading model: {model_name}")
    model = YOLO(model_name)
    
    # Train
    print("\nStarting training...")
    print("-" * 70)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name=name,
        project=project,
        save=True,
        plots=True,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    
    results_dir = Path(project) / name
    print(f"Results saved to: {results_dir}")
    print(f"Best model: {results_dir / 'weights' / 'best.pt'}")
    print(f"Last model: {results_dir / 'weights' / 'last.pt'}")
    
    return results

def main():
    parser = argparse.ArgumentParser(
        description='Train YOLOv8 instance segmentation model for bark detection'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default='dataset.yaml',
        help='Path to dataset YAML file (default: dataset.yaml)'
    )
    parser.add_argument(
        '--model_size', 
        type=str, 
        default='n',
        choices=['n', 's', 'm', 'l', 'x'],
        help='Model size: n=nano, s=small, m=medium, l=large, x=xlarge (default: n)'
    )
    parser.add_argument(
        '--epochs', 
        type=int, 
        default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--batch', 
        type=int, 
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--imgsz', 
        type=int, 
        default=640,
        help='Image size (square, default: 640)'
    )
    parser.add_argument(
        '--project', 
        type=str, 
        default='../segmentation_results',
        help='Project directory for results (default: ../segmentation_results)'
    )
    parser.add_argument(
        '--name', 
        type=str, 
        default='bark_segmentation',
        help='Experiment name (default: bark_segmentation)'
    )
    
    args = parser.parse_args()
    
    train_bark_segmentation(
        data_yaml=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name
    )

if __name__ == '__main__':
    main()
