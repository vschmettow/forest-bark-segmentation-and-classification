# After Training: What to Do With best.pt

You've downloaded **best.pt** from Colab. Here’s what to do next.

---

## 1. Put the model in your project

- **Suggested path:** `forest_instance_segmentation/models/finnwoodlands_seg_best.pt`
- From repo root: create `forest_instance_segmentation/models/` if it doesn’t exist, then put **best.pt** there and (optionally) rename it to **finnwoodlands_seg_best.pt**.

---

## 2. Run inference on a forest image

Use the trained model to segment **Spruce** and **Pine** trees in any forest image. From repo root:

```bash
python forest_instance_segmentation/scripts/predict_forest_seg.py \
  --image /path/to/your/forest_photo.jpg \
  --model forest_instance_segmentation/models/finnwoodlands_seg_best.pt \
  --save forest_instance_segmentation/output/overlay.jpg
```

- **--image:** Path to a forest image (e.g. from FinnWoodlands val set or your own photo).
- **--model:** Path to **best.pt** (default is `forest_instance_segmentation/models/finnwoodlands_seg_best.pt`).
- **--save:** Optional; saves an overlay image with colored masks (Spruce / Pine) to the given path.
- **--conf:** Optional; confidence threshold (default 0.25). Lower = more detections, more false positives.

The script prints how many Spruce and Pine instances were found and, if you used **--save**, writes an overlay image.

---

## 3. Use the model in your own code

```python
from ultralytics import YOLO

model = YOLO("forest_instance_segmentation/models/finnwoodlands_seg_best.pt")
results = model("path/to/forest.jpg", conf=0.25)

for result in results:
    if result.masks is not None:
        for i, mask in enumerate(result.masks.data):
            cls_id = int(result.boxes.cls[i])   # 0 = Spruce, 1 = Pine
            conf = float(result.boxes.conf[i])
            # mask: numpy array (H, W) for this instance
```

---

## Summary

| Step | Action |
|------|--------|
| 1 | Put **best.pt** in `forest_instance_segmentation/models/finnwoodlands_seg_best.pt` (or any path you like). |
| 2 | Run **predict_forest_seg.py** on a forest image to get Spruce/Pine instance segmentation and (optionally) an overlay. |
| 3 | Use **YOLO(path_to_best.pt)** in your own scripts for batch processing or integration. |
