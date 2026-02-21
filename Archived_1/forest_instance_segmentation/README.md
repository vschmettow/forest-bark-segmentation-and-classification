# Forest Analysis — Instance Segmentation

This folder contains the **new approach** to forest analysis using **instance segmentation** (e.g. YOLO segmentation or similar) on the **FinnWoodlands** dataset.

## Relation to the rest of the repo

- **Root-level** `apps/`, `scripts/`, `data/models/`, `docs/`, `tests/`: existing bark classifier pipeline (SAM + YOLOv8 classification). This is what is currently on GitHub and considered the “legacy” / original work.
- **This folder** (`forest_instance_segmentation/`): new pipeline and experiments. FinnWoodlands is used here; large dataset files are **not** committed to GitHub (see repo root `.gitignore` and `PROJECT_STRUCTURE.md`).

## Dataset: FinnWoodlands

- **Location**: `../FinnWoodlands/` (repo root) or, if you use the shared data layout, `../data/datasets/FinnWoodlands/`.
- **Content**: RGB images, annotations, COCO/panoptic-style JSONs (e.g. `train.json`, `val.json`, `forest_coco_panoptic_val.json`).
- **Git**: The `FinnWoodlands/` directory is listed in `.gitignore`; only code and small configs in this folder are pushed to GitHub.

## YOLOv8-seg (Spruce + Pine) with FinnWoodlands

- **Download YOLOv8n-seg**:  
  `python forest_instance_segmentation/scripts/download_yolov8_seg.py`  
  (optionally `--size s` for small).

- **Convert FinnWoodlands to YOLO-seg format**:  
  `python forest_instance_segmentation/scripts/convert_finnwoodlands_to_yolo_seg.py`  
  Optional: `--val-pine-min 100` so val has at least 100 Pine (moves some train images to val).  
  Produces `data/yolo_finnwoodlands/` (images + labels + `dataset.yaml`) for Spruce (0) and Pine (1).

- **Training dashboard** (progress + ETA on localhost):  
  `python forest_instance_segmentation/scripts/training_dashboard.py --results-dir <run_folder> --epochs 100 --port 5010`  
  Open http://127.0.0.1:5010/ while training runs.

- **Data format, pitfalls, and inference**:  
  See **`docs/DATA_AND_TRAINING.md`** for:
  - What YOLOv8-seg needs from FinnWoodlands (COCO polygons + category_id → masks + labels)
  - Potential issues (small dataset, class imbalance, polygon complexity, etc.)
  - Using the trained model on new forest images (instance segmentation for all trees).

- **After training (you have best.pt)**:  
  Put **best.pt** in `forest_instance_segmentation/models/best.pt`, then run inference with **`scripts/predict_forest_seg.py`** on a forest image. Full steps: **`docs/AFTER_TRAINING.md`**.

- **Web app** (drag & drop, original vs overlay, Spruce/Pine counts):  
  `python forest_instance_segmentation/segmentation_web.py --port 5020`  
  Open http://127.0.0.1:5020/

## Suggested layout

```
forest_instance_segmentation/
├── README.md                 # this file
├── segmentation_web.py          # web app: drag & drop, original vs overlay, counts
├── scripts/
│   ├── download_yolov8_seg.py
│   ├── convert_finnwoodlands_to_yolo_seg.py
│   └── predict_forest_seg.py   # run trained model on a forest image
├── data/
│   └── yolo_finnwoodlands/   # converted dataset (dataset.yaml, images/, labels/)
├── docs/
│   └── DATA_AND_TRAINING.md
├── configs/                  # optional
└── models/                   # optional; trained weights
```

Paths in scripts assume FinnWoodlands at repo root (`../FinnWoodlands` from `scripts/`). Override with `--finnwoodlands` and `--output` if needed.

**No local GPU?** See **`docs/COLAB_STEP_BY_STEP.md`** for a full Colab walkthrough (what to upload, each cell to run). See **`docs/PUBLIC_GPU_OPTIONS.md`** for other free GPU options.
