# Forest Bark Analysis System

A complete machine learning system for automatically segmenting and classifying bark objects in forest images. The system identifies bark regions using state-of-the-art segmentation models (SAM2/SAM1) and classifies them using a trained YOLOv8 classifier.

## 🎯 Project Overview

This project implements a **two-stage pipeline** for analyzing forest images:

1. **Segmentation Stage**: Uses Segment Anything Model (SAM2 or SAM1) to automatically detect and segment all objects in a forest image
2. **Classification Stage**: Uses a trained YOLOv8 classification model to identify each segmented object as **Picea**, **Pinus**, or **Other**

The system provides both web interfaces and command-line tools for easy use.

## 🏗️ System Architecture

### High-Level Architecture

```
Forest Image
    ↓
[SAM2/SAM1 Segmentation] → Finds all objects → Generates masks
    ↓
[Object Extraction] → Crops each masked region
    ↓
[YOLOv8 Classification] → Classifies each object → Picea/Pinus/Other
    ↓
[Visualization] → Colored outlines + statistics
```

### Component Breakdown

#### 1. **Segmentation Module** (SAM2/SAM1)
- **Purpose**: Automatically finds and segments all objects in an image
- **Technology**: 
  - Primary: SAM2 (Segment Anything Model 2) - attempts to use first
  - Fallback: SAM1 (Segment Anything Model v1) - used if SAM2 unavailable
- **How it works**: 
  - Uses automatic mask generation to detect all objects
  - Filters out very small masks (< 1% of image area)
  - Generates segmentation masks for each detected object
- **Location**: Handled in `forest_bark_analyzer.py` and `forest_bark_analyzer_augmented.py`

#### 2. **Classification Module** (YOLOv8)
- **Purpose**: Classifies individual bark objects
- **Technology**: YOLOv8-s (small) classification model
- **Classes**: 
  - **Picea** (Spruce trees)
  - **Pinus** (Pine trees)
  - **Other** (if confidence < 95% threshold)
- **Model Variants**:
  - **Non-augmented**: Trained on original training data
  - **Augmented**: Trained on augmented (transformed) training data
- **Input**: 600x600 pixel images (automatically resized)
- **Confidence Threshold**: 95% (objects below threshold classified as "Other")
- **Location**: `data/models/yolov8_results/` (non-augmented) or `data/models/yolov8_results_augmented/` (augmented)

#### 3. **Integration Layer**
- **Web Applications**: Flask-based interfaces for easy interaction
- **Command-line Tools**: Scripts for batch processing and automation

## 📁 Project Structure

```
Bark/
├── apps/                          # Main applications
│   ├── forest_bark_analyzer.py           # Full pipeline (segmentation + classification) - non-augmented model
│   ├── forest_bark_analyzer_augmented.py # Full pipeline - augmented model
│   └── bark_classifier_web.py            # Simple classifier (classification only)
│
├── scripts/                       # Training and utility scripts
│   ├── train_yolov8_classifier.py        # Train YOLOv8 classifier
│   ├── create_augmented_dataset.py       # Create augmented training dataset
│   ├── predict_yolov8_bark.py           # Command-line prediction
│   ├── yolov8_dashboard_web.py          # Training dashboard
│   └── yolov8_dashboard_augmented.py    # Augmented training dashboard
│
├── tests/                         # Test files
│   ├── test_sam2.py                      # SAM2 installation test
│   └── test_sam2_detailed.py             # Detailed SAM2 API test
│
├── docs/                          # Documentation
│   ├── README.md                         # Legacy README (see root README.md)
│   ├── README_YOLOv8.md                 # YOLOv8 detailed docs
│   ├── FOREST_ANALYZER_README.md        # Forest analyzer guide
│   ├── WEB_CLASSIFIER_README.md         # Web classifier guide
│   ├── SAM2_INSTALLATION.md             # SAM2 installation guide
│   └── requirements.txt                  # Python dependencies
│
├── data/                          # Model files only
│   └── models/                           # Trained models
│       ├── yolov8_results/              # Non-augmented model
│       └── yolov8_results_augmented/    # Augmented model
│
├── images/                        # All image data
│   ├── training_data/                    # Training datasets
│   │   ├── training_data_augmented/      # Augmented dataset
│   │   └── training_data_small_sample/   # Original dataset
│   ├── test_images/                      # Test images
│   ├── OriginalBark/                     # Original bark images
│   ├── Picea-BarkNet-Part-1of4-modified/ # Modified Picea images
│   └── Pinus-Bark-KR-modified/          # Modified Pinus images
│
├── Archive/                       # Old/unused files
│   └── Preprocessing scripts/            # Old scripts and SAM weights
│
└── uploads/                       # Temporary upload directory (created by apps)
```

See `PROJECT_STRUCTURE.md` for detailed directory structure.

## ✅ What Has Been Done

### Phase 1: Data Collection & Preprocessing
- ✅ Collected bark image datasets for Picea and Pinus species
- ✅ Preprocessed images (segmentation, cropping, resizing)
- ✅ Created training/validation splits
- ✅ Prepared both original and modified image sets

### Phase 2: Model Training
- ✅ Trained YOLOv8 classification model on original data
  - Model: YOLOv8-s (small)
  - Image size: 600x600
  - Classes: Picea, Pinus
  - Confidence threshold: 95%
- ✅ Created augmented training dataset (rotations, flips, brightness adjustments, etc.)
- ✅ Trained second YOLOv8 model on augmented data for improved robustness

### Phase 3: Integration & Applications
- ✅ Integrated SAM1/SAM2 for automatic segmentation
- ✅ Built full pipeline combining segmentation + classification
- ✅ Created web interfaces:
  - Full analyzer (segmentation + classification)
  - Simple classifier (classification only)
- ✅ Implemented command-line prediction tools
- ✅ Added training dashboards for monitoring

### Phase 4: Optimization & Organization
- ✅ Implemented SAM2 support with fallback to SAM1
- ✅ Organized project into logical folder structure
- ✅ Created comprehensive documentation
- ✅ Set up proper path management for different model variants

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Install dependencies**:
   ```bash
   pip install -r docs/requirements.txt
   ```
3. **SAM Models** (choose one):
   - **SAM1** (recommended for simplicity): `pip install git+https://github.com/facebookresearch/segment-anything.git`
   - **SAM2** (optional, better performance): Clone repository and install (see `docs/SAM2_INSTALLATION.md`)
4. **Model Weights**: 
   - SAM weights: Place in `Archive/Preprocessing scripts/`
   - YOLOv8 models: Already in `data/models/`

### Running the Applications

#### Full Analyzer (Segmentation + Classification)

**Non-augmented model:**
```bash
cd apps
python3 forest_bark_analyzer.py
# Open http://localhost:5002
```

**Augmented model:**
```bash
cd apps
python3 forest_bark_analyzer_augmented.py
# Open http://localhost:5004
```

#### Simple Classifier (Classification Only)

```bash
cd apps
python3 bark_classifier_web.py
# Open http://localhost:5000
```

#### Command-Line Prediction

```bash
cd scripts
python3 predict_yolov8_bark.py --image path/to/image.jpg
```

### Training New Models

See the [Complete Training Guide](docs/TRAINING_GUIDE.md) for detailed instructions and model weight download links.

**Quick start:**
```bash
cd scripts
python3 train_yolov8_classifier.py --data_dir ../images/training_data/training_data_small_sample
```

## 🔧 Technical Details

### Model Specifications

**YOLOv8 Classifier:**
- Architecture: YOLOv8-s (small)
- Input size: 600x600 pixels
- Output classes: Picea, Pinus, Other
- Confidence threshold: 95%
- Training: 60 epochs, batch size 4

**SAM Segmentation:**
- Primary: SAM2 (SAM2.1 Hiera Large) - 856MB checkpoint
- Fallback: SAM1 (ViT-H or ViT-B)
- Method: Automatic mask generation
- Filter: Masks < 1% image area removed

### Data Pipeline

1. **Input**: Forest/terrestrial image (JPEG/PNG, max 32MB)
2. **Segmentation**: SAM generates masks for all objects
3. **Filtering**: Small masks removed (< 1% of image)
4. **Extraction**: Each mask region cropped tightly
5. **Classification**: Each crop classified by YOLOv8
6. **Output**: Image with colored outlines + statistics

### Performance Considerations

- **First request**: Models load (~30-60 seconds)
- **Processing time**: Depends on image size and number of objects
  - Small images (< 2MP): 10-30 seconds
  - Large images (> 10MP): 1-5 minutes
- **GPU acceleration**: Significantly faster if available

## 📊 Current Status

### ✅ Completed Features
- Two-stage segmentation + classification pipeline
- SAM2 integration with SAM1 fallback
- Two trained model variants (original & augmented data)
- Web interfaces for easy use
- Command-line tools for batch processing
- Training dashboards
- Comprehensive documentation
- Organized project structure

### 🎯 Model Variants

1. **Non-Augmented Model**: 
   - Trained on original training data
   - Location: `data/models/yolov8_results/`
   - Used by: `forest_bark_analyzer.py`

2. **Augmented Model**: 
   - Trained on augmented (transformed) data
   - Location: `data/models/yolov8_results_augmented/`
   - Used by: `forest_bark_analyzer_augmented.py`

### 📝 Notes on SAM2 vs SAM1

- **SAM2** is preferred but requires more setup (repository clone + config files)
- **SAM1** is simpler (pip install) and works as reliable fallback
- Both use automatic mask generation (not text prompts)
- For text-prompt based segmentation, you'd need Grounding DINO + SAM2
- Current approach (automatic mask generation + YOLOv8 filtering) works well

## 📚 Additional Documentation

- `PROJECT_STRUCTURE.md` - Detailed directory organization
- `docs/TRAINING_GUIDE.md` - **Complete training guide with model weight download links**
- `docs/INSTANCE_SEGMENTATION_GUIDE.md` - **Guide for training your own segmentation model**
- `docs/FOREST_ANALYZER_README.md` - Full analyzer documentation
- `docs/WEB_CLASSIFIER_README.md` - Simple classifier documentation
- `docs/README_YOLOv8.md` - YOLOv8 training details
- `docs/SAM2_INSTALLATION.md` - SAM2 setup guide
- `docs/IMAGE_QUALITY_REQUIREMENTS.md` - Image quality guidelines

## 🛠️ Development Notes

- All paths in scripts/apps are relative to their parent directories
- Run apps from `apps/` directory
- Run scripts from `scripts/` directory
- Models stored in `data/models/`
- Training data in `images/training_data/`

## 📄 License

This project is part of a university project (Projektarbeit).

## 🙏 Acknowledgments

- **SAM2/SAM1**: Facebook Research (Segment Anything Model)
- **YOLOv8**: Ultralytics
- **Bark Datasets**: Picea-BarkNet and Pinus-Bark-KR
# forest-bark-segmentation-and-classification
