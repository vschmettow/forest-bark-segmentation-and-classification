# Training BarkNet with YOLOv8s-seg and ConvexMask R50-FPN

Guide for training on **BarkNetYOLO/data0** and **BarkNetCOCO/data0** (smallest datasets: ~35 train images).

---

## Laptop vs Google Colab

| Factor | YOLOv8s-seg | ConvexMask R50-FPN |
|--------|-------------|---------------------|
| **GPU memory** | ~4–6 GB | ~6–8 GB |
| **Laptop with GPU** | ✅ 20–40 min | ✅ 1–2 hours |
| **Laptop CPU only** | ⚠️ 3–8 hours | ❌ Not practical |
| **Colab free (T4)** | ✅ ~15–20 min | ✅ ~45–90 min |
| **Colab Pro (A100)** | ✅ ~5 min | ✅ ~15–30 min |

### Recommendation

- **YOLOv8s-seg**: Fine on a laptop with a decent GPU (e.g. RTX 3060). On CPU, expect several hours.
- **ConvexMask**: Prefer Colab or a laptop with a strong GPU. CPU training is not realistic.

**Colab free** (T4, 16 GB) is enough for both models on data0.

---

## 1. YOLOv8s-seg on BarkNetYOLO/data0

### Local

```bash
# From project root
python scripts/train_barknet_yolov8s_seg.py
```

Options:
```bash
python scripts/train_barknet_yolov8s_seg.py --epochs 100 --batch 8 --imgsz 640
# Reduce batch to 4 if you get OOM
```

### Colab

```python
# Cell 1: Setup
!pip install ultralytics
from google.colab import files
# Upload your Bark folder or clone from git

# Cell 2: Train
from ultralytics import YOLO
model = YOLO("yolov8s-seg.pt")
model.train(
    data="/content/Bark/BarkNetYOLO/data0/dataset.yaml",
    epochs=100,
    batch=8,
    imgsz=640,
    project="/content/results",
    name="barknet_data0_yolov8s_seg",
)
```

---

## 2. ConvexMask R50-FPN on BarkNetCOCO/data0

### Prerequisites

1. **Clone ConvexMask**
   ```bash
   cd /path/to/Bark
   git clone https://github.com/rcondat/convexmask.git convexmask_repo
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision opencv-python pycocotools
   ```

3. **ResNet50 backbone**
   - Create `convexmask_repo/weights/` and add `resnet50-19c8e357.pth`
   - Download from [PyTorch model zoo](https://download.pytorch.org/models/resnet50-19c8e357.pth) or use:
     ```python
     import torch
     from torchvision.models import resnet50, ResNet50_Weights
     m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
     torch.save(m.state_dict(), "convexmask_repo/weights/resnet50-19c8e357.pth")
     ```

### Local training

```bash
# From project root (patches ConvexMask config and runs train)
python scripts/train_barknet_convexmask.py
```

Or manually from ConvexMask repo:
```bash
cd convexmask_repo
# Add BarkNet config to data/config.py (see script)
python train.py --config convex_barknet_r50 --dataset barknet_dataset
```

### Colab

1. Clone ConvexMask and your Bark repo.
2. Run `train_barknet_convexmask.py` or the manual `train.py` commands above.
3. Enable GPU: Runtime → Change runtime type → T4 GPU.

---

## Output locations

| Model | Output path |
|-------|-------------|
| YOLOv8s-seg | `segmentation_results/barknet_data0_yolov8s_seg/weights/best.pt` |
| ConvexMask | `convexmask_repo/weights/convex_barknet_r50/best_checkpoint.pth` |

---

## Tips for small datasets (data0)

- **Overfitting**: With ~35 images, expect overfitting. Use more data (data_all) for better generalization.
- **Augmentation**: Both frameworks use augmentation by default.
- **Epochs**: 50–100 for YOLOv8; 50 for ConvexMask on data0.
- **Batch size**: Reduce to 4 if you run out of GPU memory.
