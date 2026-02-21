# Project Structure

This document explains the organization of the Bark Classification Project and how it relates to GitHub.

## Two Tracks: Original Work vs New Approach

- **Original work (on GitHub):** The existing bark classifier pipeline lives at repo root: `apps/`, `scripts/`, `data/models/`, `docs/`, `tests/`. This is the SAM + YOLOv8 classification flow; all of this is tracked and pushed to GitHub.
- **New approach (separate folder):** The new forest analysis pipeline using **instance segmentation** and the **FinnWoodlands** dataset lives in `forest_instance_segmentation/`. Code and configs there are tracked; large datasets are not (see below).

## What Goes to GitHub vs What Stays Local

| Location | On GitHub? | Notes |
|----------|------------|--------|
| `apps/`, `scripts/`, `docs/`, `tests/`, `data/models/` | Yes | Original pipeline code and trained model metadata |
| `forest_instance_segmentation/` (code, configs, small files) | Yes | New instance-segmentation approach |
| `data/datasets/README.md` | Yes | Explains where to put large datasets |
| `FinnWoodlands/` | **No** | Large image/annotation dataset; listed in `.gitignore` |
| `data/datasets/*/` (any subfolder) | **No** | All dataset subfolders are gitignored |
| `images/` | **No** | Training/test images; already in `.gitignore` |
| `uploads/`, `.venv/`, `__pycache__/` | No | Local/temporary |

Large datasets (FinnWoodlands, etc.) are intentionally **not** pushed so that repo size stays small and cloning stays fast.

## Directory Structure

```
Bark/
├── .gitignore                    # Excludes FinnWoodlands/, data/datasets/*/, images/, etc.
├── README.md                     # Main project README
├── PROJECT_STRUCTURE.md         # This file
│
├── apps/                         # [Original] Main application files
│   ├── forest_bark_analyzer.py
│   ├── forest_bark_analyzer_augmented.py
│   └── bark_classifier_web.py
│
├── scripts/                      # [Original] Training and utility scripts
│   ├── train_yolov8_classifier.py
│   ├── create_augmented_dataset.py
│   ├── predict_yolov8_bark.py
│   ├── yolov8_dashboard_web.py
│   ├── yolov8_dashboard_augmented.py
│   ├── convert_sam_masks_to_yolo.py
│   ├── integrate_segmentation_with_classifier.py
│   └── train_bark_segmentation.py
│
├── tests/                        # [Original] Test files
│   ├── test_sam2.py
│   └── test_sam2_detailed.py
│
├── docs/                         # [Original] Documentation
│   ├── README.md
│   ├── README_YOLOv8.md
│   ├── FOREST_ANALYZER_README.md
│   ├── WEB_CLASSIFIER_README.md
│   ├── SAM2_INSTALLATION.md
│   ├── IMAGE_QUALITY_REQUIREMENTS.md
│   ├── INSTANCE_SEGMENTATION_GUIDE.md
│   ├── TRAINING_GUIDE.md
│   └── requirements.txt
│
├── data/                         # Shared data layout
│   ├── models/                   # [Tracked] Trained models and results
│   │   ├── yolov8_results/
│   │   └── yolov8_results_augmented/
│   └── datasets/                 # [Not tracked] Large datasets only
│       ├── README.md             # Instructions (this file is tracked)
│       └── (FinnWoodlands/ etc. — gitignored)
│
├── forest_instance_segmentation/  # [New] Instance segmentation approach (FinnWoodlands)
│   └── README.md                 # Overview and dataset paths
│
├── FinnWoodlands/                # [Not tracked] Dataset for new model (gitignored)
│   ├── rgb/
│   ├── annotations/
│   ├── train.json, val.json, ...
│
├── images/                       # [Not tracked] Original training/test images
│   ├── training_data/
│   ├── test_images/
│   └── ...
│
├── Archive/                      # Old/unused files
└── uploads/                      # Temporary (created by apps)
```

## Quick Start

### Running the Full Analyzer (Segmentation + Classification)

**Non-augmented model:**
```bash
cd apps
python3 forest_bark_analyzer.py
# Access at http://localhost:5002
```

**Augmented images model:**
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
python3 train_yolov8_classifier.py --data_dir ../images/training_data/training_data_small_sample
```

### Creating Augmented Dataset

```bash
cd scripts
python3 create_augmented_dataset.py --source_dir ../images/training_data/training_data_small_sample
```

## Notes

- All paths in scripts/apps are relative to their parent directories.
- Models are stored in `data/models/`.
- Training data for the original pipeline is in `images/training_data/` (gitignored).
- FinnWoodlands is used by the new instance-segmentation approach; it can stay at `FinnWoodlands/` or be moved/symlinked to `data/datasets/FinnWoodlands/` for a single shared location (both are gitignored).
- Old/unused files are archived in `Archive/`.

## Optional: Move FinnWoodlands Under data/datasets

To keep all large datasets in one place:

1. Move or symlink: `FinnWoodlands/` → `data/datasets/FinnWoodlands/`.
2. In `forest_instance_segmentation/` scripts, point to `../data/datasets/FinnWoodlands/` (or `../../data/datasets/FinnWoodlands/` from subfolders).

FinnWoodlands at repo root is already gitignored; `data/datasets/*/` is also gitignored, so either location stays off GitHub.
