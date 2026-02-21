# Bark Image Classification Project

YOLOv8-based image classification system for classifying bark images as Picea, Pinus, or Other.

## Project Structure

### Essential Files (Current Directory)

**Training & Prediction:**
- `train_yolov8_classifier.py` - Training script for YOLOv8 model
- `predict_yolov8_bark.py` - Command-line prediction script
- `bark_classifier_web.py` - **Web interface for classifying images (drag & drop!)**
- `yolov8_dashboard_web.py` - Web dashboard for monitoring training

**Configuration:**
- `requirements.txt` - Python dependencies
- `README_YOLOv8.md` - Detailed documentation
- `yolov8s-cls.pt` - Pre-trained YOLOv8 model weights

**Data:**
- `training_data_small_sample/` - Training dataset (Picea & Pinus with train/val splits)
- `OriginalBark/` - Original unprocessed images
- `Picea-BarkNet-Part-1of4-modified/` - Processed Picea images
- `Pinus-Bark-KR-modified/` - Processed Pinus images

**Results:**
- `yolov8_results/` - Training results and saved models
  - `bark_classifier/weights/best.pt` - Best trained model (use for predictions)
  - `bark_classifier/weights/last.pt` - Final trained model
  - `bark_classifier/results.csv` - Training metrics
  - `bark_classifier/results.png` - Training plots

### Archived Files

All preprocessing scripts, old training scripts, and documentation files have been moved to `Archive/`:
- Image preprocessing scripts (segmentation, cropping, resizing)
- Old PyTorch-based training scripts
- Old dashboard implementations
- Setup and installation scripts
- Test images and temporary files

## Quick Start

### Training

```bash
python3 train_yolov8_classifier.py --data_dir ./training_data_small_sample --epochs 60 --batch_size 4
```

### Prediction

**Single image:**
```bash
python3 predict_yolov8_bark.py --model ./yolov8_results/bark_classifier/weights/best.pt --image path/to/image.jpg
```

**Directory of images:**
```bash
python3 predict_yolov8_bark.py --model ./yolov8_results/bark_classifier/weights/best.pt --dir path/to/images --output results.json
```

### Web Classifier Interface

Start the interactive web classifier:
```bash
python3 -m pip install flask
python3 bark_classifier_web.py --port 5001
```

Then open: http://localhost:5001

Drag & drop or upload bark images to classify them!

### Dashboard

Start the training dashboard (requires Flask):
```bash
python3 -m pip install flask pandas
python3 yolov8_dashboard_web.py --port 5000
```

Then open: http://localhost:5000

## Model Details

- **Model**: YOLOv8-s (small) classification
- **Classes**: Picea, Pinus, Other
- **Input Size**: 600x600
- **Confidence Threshold**: 95% (images below this threshold are classified as "Other")
- **Training**: 60 epochs, batch size 4

## Installation

```bash
python3 -m pip install -r requirements.txt
```

Or install ultralytics directly:
```bash
python3 -m pip install ultralytics
```

For dashboard:
```bash
python3 -m pip install flask pandas
```

## Documentation

See `README_YOLOv8.md` for detailed documentation.

