# FinnWoodlands → YOLOv8-seg: Data, Inputs, and Training

## YOLOv8-seg-n (default)

- **Model**: We use **yolov8n-seg** (nano) by default for training.
- **Download**: Run `scripts/download_yolov8_seg.py` (default is `n`; use `--size s` for small). Ultralytics downloads the weight file on first use.
- **Cached location**: Usually `~/.config/Ultralytics/` or next to the script; the script just loads `YOLO('yolov8n-seg.pt')`.

---

## What YOLOv8-seg Needs from FinnWoodlands

YOLOv8-seg expects **one image folder** and **one label folder** per split, with matching filenames (same stem, image = jpg, label = txt).

### Inputs used

| Source | What we use |
|--------|-------------|
| **FinnWoodlands/rgb/train/** | RGB images (e.g. `00183.jpg`) |
| **FinnWoodlands/rgb/val/** | RGB images for validation |
| **FinnWoodlands/train.json** | COCO format: `images`, `annotations` with `segmentation` (polygon) and `category_id` |
| **FinnWoodlands/val.json** | Same for val split |

### Masks and labels (same object)

- **Masks**: COCO `annotations[].segmentation` — list of polygons per instance. Each polygon is `[x,y,x,y,...]` in **pixel** coordinates. By default we use the **first** polygon; with `--merge-multipolygon` we union all parts into one polygon (see below).
- **Labels**: COCO `annotations[].category_id`. We keep only **Spruce (6)** and **Pine (8)** and map them to YOLO class **0** and **1**.

So each instance has **one polygon (mask)** and **one class (Spruce or Pine)**; no separate mask file is needed because the polygon defines the mask.

### Multipolygon: one instance, multiple parts

In FinnWoodlands there are **no multipolygons**: all Spruce/Pine annotations use a single polygon (train: 1092 single, 0 multipolygon; val: 712 single, 0 multipolygon). So **no merging is needed** for this dataset.

COCO can have **multiple polygons per annotation** (e.g. one tree with several disjoint mask parts). YOLO-seg expects **one polygon per instance**. You can:

- **Default**: Use the first polygon only (other parts are ignored).
- **Merge into one polygon**: Run conversion with `--merge-multipolygon`. The script uses **shapely** to union all parts into a single polygon (exterior boundary). If the result is a MultiPolygon (e.g. disjoint islands), we take the **largest** polygon by area. Requires `pip install shapely`.

```bash
python forest_instance_segmentation/scripts/convert_finnwoodlands_to_yolo_seg.py --merge-multipolygon
```

### Conversion output (YOLO-seg format)

- **images/train**, **images/val**: Same images (symlinked or copied).
- **labels/train**, **labels/val**: One `.txt` per image. Each **line** = one instance:  
  `class_id x1 y1 x2 y2 ...`  
  Coordinates are **normalized** (0–1) by image width/height.
- **dataset.yaml**: Paths, `train`/`val`, and `names: {0: Spruce, 1: Pine}`.

Run the conversion:

```bash
python forest_instance_segmentation/scripts/convert_finnwoodlands_to_yolo_seg.py
# Optional: --finnwoodlands /path/to/FinnWoodlands --output /path/to/output --copy-images
```

---

## Potential Problems

1. **Small dataset**  
   - 250 train + 50 val images, ~1.1k train instances.  
   - Risk of overfitting. Use augmentation (Ultralytics does by default), consider more data or a smaller model (e.g. `yolov8n-seg`).

2. **Class imbalance**  
   - **Train**: Spruce **662**, Pine **430** → ratio Spruce/Pine ≈ **1.54** (moderate imbalance; Spruce ~60%, Pine ~40%).  
   - **Val**: Spruce **712**, Pine **0** → **no Pine in validation**. You cannot measure Pine performance on val; consider re-splitting the dataset so both classes appear in val, or accept that val metrics reflect Spruce only.

3. **Polygon complexity**  
   - COCO polygons can be very detailed (many points). With `--merge-multipolygon`, multiple parts are merged into one; otherwise only the first polygon is used.  
   - Very long polygons can make training slower; Ultralytics may simplify; if you hit issues, simplify polygons (e.g. fewer points) before conversion.

4. **Train/val alignment**  
   - Conversion uses `train.json` ↔ `rgb/train` and `val.json` ↔ `rgb/val`.  
   - If your JSON and folder structure differ (e.g. different split or naming), update the script or paths.

5. **Image ID vs file name**  
   - Annotations use `image_id`; we map via `images[].id` and `images[].file_name`.  
   - Ensure every annotated image has a matching file in `rgb/train` or `rgb/val`.

6. **Only Spruce and Pine**  
   - Tree (5) and Birch (7) are **dropped** in the default conversion.  
   - To add classes, extend `CAT_ID_TO_YOLO` in `convert_finnwoodlands_to_yolo_seg.py` and `dataset.yaml` `names`/`nc`.

7. **Path in dataset.yaml**  
   - The generated YAML uses an absolute `path` to the dataset directory.  
   - If you move the dataset, re-run the conversion or edit `path` in `dataset.yaml`.

---

## Epochs and batch size (for this data and YOLOv8n-seg)

With **~230 train images** and **~965 instances** (after re-split with `--val-pine-min 100`), and **YOLOv8n-seg** (nano):

### Epochs

- **100 epochs** is a good default: enough to converge, not so many that overfitting is likely with augmentation.
- **80–120** is a reasonable range. Fewer than ~50 often underfits; more than ~150 on 230 images increases overfitting risk unless you use early stopping.
- If you use **early stopping** (e.g. Ultralytics `patience=30`), you can set epochs to 150 and let training stop when val mAP stops improving.

**Recommendation:** **100 epochs** (default). Optionally 80 for a quicker run, or 120–150 with early stopping.

### Batch size

- **16** is a solid default: ~14 steps per epoch (230÷16), stable gradients, and YOLOv8n-seg is small enough that batch 16 usually fits in 4–6 GB VRAM at 640×640.
- **8** is better if you have limited VRAM or want more gradient updates per epoch (~29 steps/epoch), which can help a bit on small datasets.
- **32** gives only ~7 steps/epoch; with 230 images that’s a bit coarse and not necessary for this model size.

**Recommendation:** **batch 16** (default). Use **batch 8** if you hit OOM or want more steps per epoch.

### Summary

| Setting   | Recommended | Alternative   |
|----------|-------------|----------------|
| **Epochs** | 100         | 80 (faster) or 120–150 (with early stopping) |
| **Batch**  | 16          | 8 (less VRAM / more steps per epoch)         |

Script defaults (`train_bark_segmentation.py`: `epochs=100`, `batch=16`) are appropriate for this dataset and model.

---

## Using the Model on New Forest Images (Inference)

Yes. After training, you can run the model on **any forest image** and get **instance segmentation** for Spruce and Pine:

- Each detected instance has: **mask** (pixel-level) and **class** (Spruce or Pine).
- You can count trees, overlay masks, or feed crops to another pipeline.

Example:

```python
from ultralytics import YOLO

model = YOLO("path/to/best.pt")  # your trained YOLOv8-seg
results = model("forest_photo.jpg")

for r in results:
    if r.masks is not None:
        for i, mask in enumerate(r.masks.data):
            cls_id = int(r.boxes.cls[i])
            # cls_id 0 = Spruce, 1 = Pine
            # mask: tensor, shape (H, W)
    # r.boxes: xyxy, conf, cls
```

So: **input = forest image → output = all detected trees (Spruce/Pine) with masks and labels.**

---

## Quick Training

From repo root, after conversion:

```bash
cd forest_instance_segmentation
# Optional: ensure yolov8n-seg is downloaded
python scripts/download_yolov8_seg.py

# Train (pass absolute path to dataset.yaml so path: . resolves)
python -c "
from pathlib import Path
from ultralytics import YOLO
data_yaml = Path('data/yolo_finnwoodlands/dataset.yaml').resolve()
model = YOLO('yolov8n-seg.pt')
model.train(data=str(data_yaml), epochs=100, imgsz=640, batch=8, name='finnwoodlands_seg', project='.')
"
```

Or use the existing `scripts/train_bark_segmentation.py` with `--data` pointing to `forest_instance_segmentation/data/yolo_finnwoodlands/dataset.yaml` (and correct paths). Default model is **yolov8n-seg** (`--model_size n`).

---

## Before training can start (checklist)

You do **not** need anything else beyond the following to start training:

1. **YOLOv8n-seg weights**  
   - Run once: `python forest_instance_segmentation/scripts/download_yolov8_seg.py` (or load `yolov8n-seg.pt` in code; Ultralytics will download).

2. **Converted dataset**  
   - Run: `python forest_instance_segmentation/scripts/convert_finnwoodlands_to_yolo_seg.py`  
   - Optional: `--merge-multipolygon` (needs `shapely`), `--copy-images` if you prefer copies over symlinks.  
   - Output: `forest_instance_segmentation/data/yolo_finnwoodlands/` with `dataset.yaml`, `images/train`, `images/val`, `labels/train`, `labels/val`.

3. **Environment**  
   - Python with `ultralytics` (and `shapely` only if you use `--merge-multipolygon`).  
   - FinnWoodlands at the path expected by the convert script (default: repo root `FinnWoodlands/`).

4. **Training command**  
   - From repo root:  
     `python scripts/train_bark_segmentation.py --data <absolute_path_to_dataset.yaml> --project <output_dir> --name <run_name>`  
   - Or call `model.train(...)` with the dataset YAML path. Default model is **yolov8n-seg** (`--model_size n`).

No extra config or data is required. Optionally: fix the val split so Pine appears in validation, or add class weights if you want to compensate for train imbalance.

---

## Re-split: 100 Pine in val

Original val has **0 Pine**. Run conversion with `--val-pine-min 100` so val gets at least 100 Pine instances (by moving some train images into val):

```bash
python forest_instance_segmentation/scripts/convert_finnwoodlands_to_yolo_seg.py --val-pine-min 100
```

Example outcome: **230 train images** (965 instances), **70 val images** (839 instances, ~102 Pine). Original FinnWoodlands files are not modified.

---

## Training dashboard (localhost)

Dashboard shows **epoch progress**, **elapsed time**, and **ETA** while training runs.

```bash
python forest_instance_segmentation/scripts/training_dashboard.py --results-dir <path_to_run_folder> --epochs 100 --port 5010
```

- **--results-dir**: Folder containing `results.csv` (e.g. `segmentation_results/bark_segmentation` if you use default project/name). From repo root: `--results-dir segmentation_results/bark_segmentation`.
- **--epochs**: Same as training (default 100) so ETA is correct.
- Open **http://127.0.0.1:5010/**; page auto-refreshes every 2s.

Shows: current epoch / total, progress bar, elapsed, remaining (ETA), and latest metrics.

---

## Expected training time

**YOLOv8n-seg**, 100 epochs, imgsz 640, batch 8, ~230 train images:

| Hardware | Per epoch | Total (100 epochs) |
|----------|-----------|--------------------|
| **GPU** (mid-range) | ~1–2 min | **~1.5–3.5 hours** |
| **GPU** (high-end) | ~30–60 s | **~50–100 min** |
| **CPU only** | ~10–30 min | **~17–50 hours** (not recommended) |

Expect **about 1.5–3 hours** on a typical GPU. Dashboard ETA refines as training runs.
