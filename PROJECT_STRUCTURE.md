# Project Structure

This document explains the organization of the Bark Classification Project.

## Directory Structure

```
Bark/
├── apps/                          # Main application files
│   ├── forest_bark_analyzer.py   # Full analyzer (segmentation + classification) - non-augmented model
│   ├── forest_bark_analyzer_augmented.py  # Full analyzer - augmented model
│   └── bark_classifier_web.py    # Simple classifier web interface (classification only)
│
├── scripts/                       # Training and utility scripts
│   ├── train_yolov8_classifier.py        # Train YOLOv8 classifier
│   ├── create_augmented_dataset.py       # Create augmented training dataset
│   ├── predict_yolov8_bark.py           # Command-line prediction script
│   ├── yolov8_dashboard_web.py          # Training dashboard (non-augmented)
│   └── yolov8_dashboard_augmented.py    # Training dashboard (augmented)
│
├── tests/                         # Test files
│   ├── test_sam2.py              # SAM2 installation test
│   └── test_sam2_detailed.py     # Detailed SAM2 API test
│
├── docs/                          # Documentation
│   ├── README.md                 # Main project README
│   ├── README_YOLOv8.md         # YOLOv8 specific documentation
│   ├── FOREST_ANALYZER_README.md # Forest analyzer documentation
│   ├── WEB_CLASSIFIER_README.md  # Web classifier documentation
│   ├── SAM2_INSTALLATION.md     # SAM2 installation guide
│   ├── SAM2_TEXT_PROMPTS_NOTE.md # SAM2 text prompts information
│   ├── IMAGE_QUALITY_REQUIREMENTS.md # Image quality guidelines
│   └── requirements.txt          # Python dependencies
│
├── data/                          # All data files
│   ├── models/                   # Trained models and results
│   │   ├── yolov8_results/      # Non-augmented model results
│   │   └── yolov8_results_augmented/  # Augmented model results
│   ├── training_data/            # Training datasets
│   │   ├── training_data_augmented/    # Augmented training data
│   │   └── training_data_small_sample/ # Small sample training data
│   ├── test_images/              # Test images
│   │   └── testforest/          # Forest test images
│   ├── OriginalBark/            # Original bark image datasets
│   ├── Picea-BarkNet-Part-1of4-modified/  # Modified Picea images
│   └── Pinus-Bark-KR-modified/  # Modified Pinus images
│
├── Archive/                       # Old/unused files
│   └── Preprocessing scripts/    # Old preprocessing scripts and SAM weights
│
└── uploads/                       # Temporary upload directory (created by apps)

```

## Quick Start

### Running the Full Analyzer (Segmentation + Classification)

**Non-augmented model:**
```bash
cd apps
python3 forest_bark_analyzer.py
# Access at http://localhost:5002
```

**Augmented model:**
```bash
cd apps
python3 forest_bark_analyzer_augmented.py
# Access at http://localhost:5004
```

### Running Simple Classifier (Classification Only)

```bash
cd apps
python3 bark_classifier_web.py
# Access at http://localhost:5000
```

### Training a New Model

```bash
cd scripts
python3 train_yolov8_classifier.py --data_dir ../data/training_data/training_data_small_sample
```

### Creating Augmented Dataset

```bash
cd scripts
python3 create_augmented_dataset.py --source_dir ../data/training_data/training_data_small_sample
```

## Notes

- All paths in scripts/apps are relative to their parent directories
- Models are stored in `data/models/`
- Training data is stored in `data/training_data/`
- Old/unused files are archived in `Archive/`
