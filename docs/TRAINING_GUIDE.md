# Complete Training Guide

This guide covers everything you need to train the models and set up the complete forest bark analysis system.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Downloading Model Weights](#downloading-model-weights)
3. [Training the YOLOv8 Classifier](#training-the-yolov8-classifier)
4. [Setting Up SAM Models](#setting-up-sam-models)
5. [Complete Setup Checklist](#complete-setup-checklist)

## Prerequisites

### Python Environment

```bash
# Install Python 3.10 or higher
python3 --version

# Install dependencies
pip install -r docs/requirements.txt
```

### Required Libraries

```bash
pip install ultralytics torch torchvision opencv-python pillow numpy flask
```

## Downloading Model Weights

### SAM1 (Segment Anything Model v1) Weights

**Required for segmentation (fallback if SAM2 not available)**

Download one of the following SAM1 model weights:

#### Option 1: ViT-H (Huge) - Recommended for Best Quality
- **File**: `sam_vit_h_4b8939.pth`
- **Size**: ~2.4 GB
- **Download**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
- **Place in**: `Archive/Preprocessing scripts/sam_vit_h_4b8939.pth`

#### Option 2: ViT-B (Base) - Faster, Smaller
- **File**: `sam_vit_b_01ec64.pth`
- **Size**: ~375 MB
- **Download**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
- **Place in**: `Archive/Preprocessing scripts/sam_vit_b_01ec64.pth`

**Download Command:**
```bash
# ViT-H (recommended)
cd "Archive/Preprocessing scripts"
curl -L -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# OR ViT-B (faster)
curl -L -o sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### SAM2 (Segment Anything Model 2) Weights

**Optional - Better performance, requires more setup**

#### SAM2.1 Hiera Large - Recommended
- **File**: `sam2.1_hiera_large.pt`
- **Size**: ~856 MB
- **Download**: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
- **Place in**: `Archive/Preprocessing scripts/sam2.1_hiera_large.pt`

**Download Command:**
```bash
cd "Archive/Preprocessing scripts"
curl -L -o sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

**Note**: SAM2 also requires the repository to be cloned. See [SAM2 Installation Guide](SAM2_INSTALLATION.md) for complete setup.

#### Alternative SAM2 Models

If you prefer a smaller/faster model:

- **SAM2.1 Hiera Base Plus**: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt (~300 MB)
- **SAM2.1 Hiera Small**: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt (~150 MB)
- **SAM2.1 Hiera Tiny**: https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt (~50 MB)

### YOLOv8 Base Models

**Automatically downloaded during training** - No manual download needed!

YOLOv8 will automatically download the base model (`yolov8s-cls.pt`) when you start training. However, if you want to download it manually:

- **YOLOv8-s Classification**: Automatically downloaded by Ultralytics
- **Manual download** (if needed): The model will be cached in your Ultralytics cache directory

## Training the YOLOv8 Classifier

### Step 1: Prepare Training Data

Your training data should be organized as:

```
images/training_data/training_data_small_sample/
├── Picea/
│   ├── training/
│   │   └── [bark images]
│   └── validation/
│       └── [bark images]
└── Pinus/
    ├── training/
    │   └── [bark images]
    └── validation/
        └── [bark images]
```

### Step 2: Train the Model

**Basic Training:**
```bash
cd scripts
python3 train_yolov8_classifier.py \
    --data_dir ../images/training_data/training_data_small_sample \
    --epochs 60 \
    --batch_size 4
```

**Full Training with All Options:**
```bash
cd scripts
python3 train_yolov8_classifier.py \
    --data_dir ../images/training_data/training_data_small_sample \
    --epochs 60 \
    --batch_size 4 \
    --img_size 600 600 \
    --model_size s \
    --output_dir ../data/models/yolov8_results
```

### Step 3: Train Augmented Model (Optional)

First, create augmented dataset:
```bash
cd scripts
python3 create_augmented_dataset.py \
    --source_dir ../images/training_data/training_data_small_sample \
    --output_dir ../images/training_data/training_data_augmented
```

Then train on augmented data:
```bash
cd scripts
python3 train_yolov8_classifier.py \
    --data_dir ../images/training_data/training_data_augmented \
    --output_dir ../data/models/yolov8_results_augmented \
    --epochs 60 \
    --batch_size 4
```

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | Required | Path to training data directory |
| `--epochs` | 60 | Number of training epochs |
| `--batch_size` | 4 | Batch size for training |
| `--img_size` | 600 600 | Image dimensions (width height) |
| `--model_size` | s | Model size: n/nano, s/small, m/medium, l/large, x/xlarge |
| `--output_dir` | `../data/models/yolov8_results` | Where to save training results |

### Training Output

After training, you'll find:

- **Best model**: `{output_dir}/bark_classifier/weights/best.pt`
- **Last model**: `{output_dir}/bark_classifier/weights/last.pt`
- **Training plots**: `{output_dir}/bark_classifier/results.png`
- **Metrics**: `{output_dir}/bark_classifier/results.csv`
- **Confusion matrix**: `{output_dir}/bark_classifier/confusion_matrix.png`

## Setting Up SAM Models

### Option 1: SAM1 (Simpler Setup)

1. **Install SAM1:**
   ```bash
   pip install git+https://github.com/facebookresearch/segment-anything.git
   ```

2. **Download weights** (see [Downloading Model Weights](#downloading-model-weights) above)

3. **Place weights in**: `Archive/Preprocessing scripts/`

4. **Verify installation:**
   ```bash
   python3 -c "from segment_anything import sam_model_registry; print('SAM1 installed successfully!')"
   ```

### Option 2: SAM2 (Better Performance)

1. **Clone SAM2 repository:**
   ```bash
   cd ..
   git clone https://github.com/facebookresearch/segment-anything-2.git
   cd segment-anything-2
   pip install -e .
   ```

2. **Download SAM2 weights** (see [Downloading Model Weights](#downloading-model-weights) above)

3. **Place weights in**: `Archive/Preprocessing scripts/`

4. **Verify installation:**
   ```bash
   cd tests
   python3 test_sam2.py
   ```

See [SAM2_INSTALLATION.md](SAM2_INSTALLATION.md) for detailed SAM2 setup instructions.

## Complete Setup Checklist

### ✅ Before Training

- [ ] Python 3.10+ installed
- [ ] Dependencies installed (`pip install -r docs/requirements.txt`)
- [ ] Training data prepared in `images/training_data/`
- [ ] Training data organized with Picea/ and Pinus/ folders
- [ ] Training/validation splits created

### ✅ For Segmentation (Choose One)

**SAM1 Setup:**
- [ ] SAM1 installed (`pip install git+https://github.com/facebookresearch/segment-anything.git`)
- [ ] SAM1 weights downloaded (ViT-H or ViT-B)
- [ ] Weights placed in `Archive/Preprocessing scripts/`

**OR SAM2 Setup:**
- [ ] SAM2 repository cloned
- [ ] SAM2 installed (`pip install -e .` in segment-anything-2 directory)
- [ ] SAM2 weights downloaded
- [ ] Weights placed in `Archive/Preprocessing scripts/`
- [ ] SAM2 config files available (in cloned repository)

### ✅ After Training

- [ ] YOLOv8 model trained (`best.pt` file exists)
- [ ] Model saved in `data/models/yolov8_results/` or `data/models/yolov8_results_augmented/`
- [ ] Training metrics reviewed
- [ ] Model tested with sample images

### ✅ For Full System

- [ ] Both SAM and YOLOv8 models ready
- [ ] Paths in `apps/forest_bark_analyzer.py` point to correct model locations
- [ ] Test the full pipeline with a sample forest image

## Model Weight Summary

| Model | File | Size | Location | Download |
|-------|------|------|----------|----------|
| SAM1 ViT-H | `sam_vit_h_4b8939.pth` | 2.4 GB | `Archive/Preprocessing scripts/` | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |
| SAM1 ViT-B | `sam_vit_b_01ec64.pth` | 375 MB | `Archive/Preprocessing scripts/` | [Download](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |
| SAM2.1 Large | `sam2.1_hiera_large.pt` | 856 MB | `Archive/Preprocessing scripts/` | [Download](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt) |
| YOLOv8-s | Auto-downloaded | ~20 MB | Auto-cached | Auto-downloaded during training |

## Troubleshooting

### "Model weights not found" Error

- Verify weights are in `Archive/Preprocessing scripts/`
- Check file names match exactly (case-sensitive)
- For SAM2, ensure repository is cloned and config files are accessible

### "Out of memory" During Training

- Reduce `--batch_size` (try 2 or 1)
- Use smaller model size (`--model_size n` for nano)
- Reduce image size (`--img_size 400 400`)

### Slow Training

- Use GPU if available (CUDA)
- Reduce batch size if using CPU
- Use smaller model size for faster training

## Additional Resources

- [YOLOv8 Training Details](README_YOLOv8.md)
- [SAM2 Installation Guide](SAM2_INSTALLATION.md)
- [Forest Analyzer Documentation](FOREST_ANALYZER_README.md)
- [Main Project README](../README.md)

## Quick Reference Commands

```bash
# Download SAM1 ViT-H (recommended)
cd "Archive/Preprocessing scripts"
curl -L -o sam_vit_h_4b8939.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

# Download SAM2.1 Large
curl -L -o sam2.1_hiera_large.pt https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

# Train YOLOv8 classifier
cd scripts
python3 train_yolov8_classifier.py --data_dir ../images/training_data/training_data_small_sample

# Create augmented dataset
python3 create_augmented_dataset.py --source_dir ../images/training_data/training_data_small_sample
```
