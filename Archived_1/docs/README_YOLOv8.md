# YOLOv8 Bark Classification

This directory contains scripts for training and using a YOLOv8 classification model to classify bark images as **Picea**, **Pinus**, or **Other**.

## Model Details

- **Model**: YOLOv8 size 's' (small) for classification
- **Classes**: Picea, Pinus, Other
- **Input Size**: 600x600 (images will be resized to square format)
- **Other Class Logic**: Images with confidence below threshold (< 0.95 / 95% by default) for Picea/Pinus are classified as "Other"

## Installation

Make sure you have the ultralytics library installed:

```bash
pip install ultralytics
```

## Data Structure

Your training data should be organized as:

```
training_data_small_sample/
  ├── Picea/
  │   ├── training/
  │   │   └── [images]
  │   └── validation/
  │       └── [images]
  └── Pinus/
      ├── training/
      │   └── [images]
      └── validation/
          └── [images]
```

The script will automatically reorganize this into YOLOv8 format during training.

## Training

Train the model with default settings (60 epochs, batch size 4):

```bash
python train_yolov8_classifier.py --data_dir ./training_data_small_sample --epochs 60 --batch_size 4
```

Full command with all options:

```bash
python train_yolov8_classifier.py \
    --data_dir ./training_data_small_sample \
    --epochs 60 \
    --batch_size 4 \
    --img_size 600 600 \
    --model_size s \
    --output_dir ./yolov8_results
```

### Training Parameters

- `--data_dir`: Directory containing Picea/ and Pinus/ folders with training/validation subfolders
- `--epochs`: Number of training epochs (default: 60)
- `--batch_size`: Batch size (default: 4)
- `--img_size`: Image size width height (default: 600 800)
- `--model_size`: Model size - n/nano, s/small, m/medium, l/large, x/xlarge (default: s)
- `--output_dir`: Output directory for results (default: ./yolov8_results)
- `--keep_temp_data`: Keep temporary reorganized dataset (optional)

## Prediction

### Single Image

```bash
python predict_yolov8_bark.py \
    --model ./yolov8_results/bark_classifier/weights/best.pt \
    --image path/to/image.jpg \
    --confidence_threshold 0.95
```

### Directory of Images

```bash
python predict_yolov8_bark.py \
    --model ./yolov8_results/bark_classifier/weights/best.pt \
    --dir path/to/images \
    --confidence_threshold 0.95 \
    --output results.json
```

### Prediction Parameters

- `--model`: Path to trained model (.pt file)
- `--image`: Path to single image (use either --image or --dir)
- `--dir`: Directory containing images (use either --image or --dir)
- `--confidence_threshold`: Minimum confidence for Picea/Pinus (default: 0.95 / 95%). Images below this threshold are classified as "Other"
- `--output`: Output JSON file for batch predictions (optional)

## Results

Training results will be saved to:
- `{output_dir}/bark_classifier/weights/best.pt` - Best model during training
- `{output_dir}/bark_classifier/weights/last.pt` - Final model
- `{output_dir}/bark_classifier/` - Training plots, metrics, etc.

## Notes

- The model is trained on Picea and Pinus only
- "Other" class is determined post-prediction based on confidence threshold (default: 95%)
- Images are resized to square format (600x600) during training
- Training progress is displayed automatically by YOLOv8

