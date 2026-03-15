# Best trained models

Central folder for the best checkpoints from YOLOv8s-seg and ConvexMask R50-FPN.

## Expected files

| File | Model | Dataset | Status |
|------|-------|--------|--------|
| `yolov8s_seg_data0_best.pt` | YOLOv8s-seg | BarkNetYOLO/data0 | ✓ present |
| `yolov8s_seg_data_all_best.pt` | YOLOv8s-seg | BarkNetYOLO/data_all | *(upload after Colab training)* |
| `convexmask_r50_data0_best.pth` | ConvexMask R50-FPN | BarkNetCOCO/data0 | *(download from Colab when ready)* |

## Usage

**YOLOv8:**
```python
from ultralytics import YOLO
model = YOLO("best_models/yolov8s_seg_data0_best.pt")
```

**ConvexMask:** Load via ConvexMask repo `eval.py` or `load_weights()` with the `.pth` file.
