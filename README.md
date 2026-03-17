# Forest Bark Instance Segmentation

Species-level detection of *Picea abies* (Norway spruce) and *Pinus sylvestris* (Scots pine) from bark images using YOLOv8s-seg.

## Project Structure

```
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── code/                  # Application and training code
│   ├── apps/              # Web applications (Flask)
│   ├── scripts/           # Training, evaluation, data conversion scripts
│   ├── notebooks/         # Jupyter notebooks (Colab training)
│   └── tests/             # Test files
├── dataset/               # Dataset documentation (data not in repo)
│   └── README.md          # Dataset description and setup
├── configuration/         # Config files and examples
├── models/                # Trained model weights (best_models)
├── utils/                 # Utility scripts and helpers
└── results/               # Evaluation reports and outputs
```

## Quick Start

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download pretrained weights**
   - YOLOv8s-seg base: `yolov8s-seg.pt` (from Ultralytics)
   - Trained models: see `models/` (in repo)

3. **Train** (see `configuration/docs/COLAB_YOLO_ALL_STAGES.md` for Colab workflow)
   ```bash
   python code/scripts/train_barknet_yolov8s_seg.py --data BarkNetYOLO/stage1_80/dataset.yaml
   ```

4. **Evaluate**
   ```bash
   python code/scripts/evaluate_and_visualize_models.py
   ```

## Dataset

PlantNet RGB bark dataset (n=1,931 images), balanced *Picea* / *Pinus*. See `dataset/README.md` for details.

## Models

Best trained weights in `models/`:
- `yolov8s_seg_stage1_80_best.pt` … `stage4_363_best.pt` (staged experiments)
- `yolov8s_seg_final_dataset_best.pt` (full dataset, best overall)

## Documentation

- `README_Data_Methods_Results.md` – Data, methods, and results for the research paper
- `configuration/docs/COLAB_YOLO_ALL_STAGES.md` – Colab training notebook (all stages)
- `configuration/docs/TRAINING_BARKNET.md` – Training guide

## License

[Add your license]
