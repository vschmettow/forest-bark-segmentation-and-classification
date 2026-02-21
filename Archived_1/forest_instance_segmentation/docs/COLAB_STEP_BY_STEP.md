# Google Colab: Step-by-Step Walkthrough

You have a new notebook and GPU enabled. Follow these steps in order.

---

## Pre-training checklist (is everything prepped?)

| Item | Status |
|------|--------|
| **Dataset converted** | Done: YOLO format with Spruce + Pine, `--val-pine-min 100`, `--copy-images`. |
| **Zip created** | Done: `forest_instance_segmentation/data/yolo_finnwoodlands.zip` (~98 MB). |
| **Colab notebook** | New notebook, runtime set to **GPU** (Runtime → Change runtime type → GPU). |
| **Upload** | You will upload the zip in Cell 2 (file picker). |

You’re ready to run the cells in order. No other prep needed.

---

## Progress during training (do I get a dashboard?)

**Yes – in the notebook itself.** When you run the training cell (Cell 4), Ultralytics prints **live progress** in that cell’s output:

- **Epoch** (e.g. `Epoch 1/100`)
- **Time** per epoch and elapsed
- **Losses** (train/box_loss, train/seg_loss, val/box_loss, etc.)
- **mAP** (e.g. mAP50, mAP50-95) as validation runs

You don’t need a separate dashboard: the training cell output is your progress view. Scroll in that cell to see the latest.

**Optional – progress summary in another cell:** You can add the cell below and run it **while training is running** (e.g. in a separate cell, run it every few minutes) to see epoch, elapsed time, and a rough ETA from `results.csv`:

```python
# Optional: run this cell while training runs to see progress summary
import pandas as pd
import os
csv_path = "/content/finnwoodlands_seg/results.csv"
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    last = df.iloc[-1]
    epoch = int(last.get("epoch", len(df)))
    elapsed = float(last.get("time", 0))
    remaining_epochs = 100 - epoch
    eta_min = (elapsed / epoch * remaining_epochs / 60) if epoch > 0 else 0
    print(f"Epoch {epoch}/100 | Elapsed: {elapsed/60:.1f} min | ETA: ~{eta_min:.0f} min left")
    print(last[["train/box_loss", "train/seg_loss", "metrics/mAP50(B)"]].to_string())
else:
    print("Training not started yet or results.csv not found.")
```

The **localhost dashboard** (`training_dashboard.py`) only works when training runs on your MacBook (it reads `results.csv` from a folder on your machine). In Colab, the notebook output + optional cell above are your “dashboard.”

---

## When do I stop training?

**You don’t stop it yourself.** Training is set to **100 epochs** and will **stop automatically** when epoch 100 finishes. Let it run to the end (about 1.5–3 hours on a T4).

**What you get when it finishes:**

- Ultralytics saves **`best.pt`** (the checkpoint with the **best validation mAP** so far) and **`last.pt`** (epoch 100). You use **`best.pt`** for inference; that’s the one to download.
- So even if the last few epochs don’t improve (or get worse), you still have the best model in **`best.pt`**.

**When you might stop the cell early (optional):**

- If you see **clear overfitting**: validation loss or mAP gets **worse for many epochs in a row** (e.g. 20+ epochs) while training loss keeps improving, you can stop the cell (Runtime → Interrupt execution) and use the **already-saved** `best.pt` (it was saved earlier when val was best). You don’t have to; you can also let it run to 100.
- If Colab is about to disconnect or you need to leave, stop the cell and download **`best.pt`** (or copy the run folder to Drive) so you don’t lose the best checkpoint.

**Summary:** Let training run until it **stops by itself** at epoch 100. Use **`best.pt`** as your final model. Only stop the cell yourself if you want to save time when things are clearly getting worse, or to save your work before a disconnect.

---

## There’s no single “val loss” – what should I watch?

For **segmentation**, Ultralytics does **not** print one combined “val loss”. It prints **separate validation metrics**, for example:

- **val/box_loss** – validation box loss  
- **val/seg_loss** – validation segmentation (mask) loss  
- **val/cls_loss** – validation classification loss  

So if you’re looking for “val loss”, you won’t see that exact label; you’ll see **val/box_loss**, **val/seg_loss**, **val/cls_loss** (and possibly **val/dfl_loss**). Some Colab outputs or tables may show only train loss clearly; the val components are still computed and saved in `results.csv`.

**What actually tells you “training is working” (and not worthless):**

1. **mAP50** (and **mAP50-95**) – These are the main metrics to watch. If **mAP50** goes up over epochs (e.g. from 0.1 → 0.3 → 0.5), training is learning. If it stays near zero for 30+ epochs, something may be wrong (e.g. data path, labels).
2. **Train loss going down** – Train loss (or train/box_loss, train/seg_loss) should generally decrease. If it never goes down, training might not be learning.
3. **best.pt is saved** – Ultralytics saves **best.pt** when validation mAP improves. So even if the printed table doesn’t highlight “val loss”, the best checkpoint is chosen using validation performance.

**When to worry:**

- **mAP50 stays 0 or very low** (e.g. &lt; 0.05) after 20–30 epochs → Check that the dataset and paths are correct (right images, right labels).
- **Train loss never decreases** → Possible bug in data or labels, or learning rate issue.
- **Val losses (val/seg_loss, etc.) go up a lot while mAP goes down** for many epochs → Possible overfitting; you can stop early and use **best.pt** (already saved from when mAP was best).

**When not to worry:**

- **Val/seg_loss or other val losses fluctuate or go up a bit** while **mAP50 still improves** → This can happen with segmentation; focus on mAP.
- **You don’t see a column literally named “val loss”** → Normal; use val/box_loss, val/seg_loss, val/cls_loss and, above all, **mAP50** as the main “is training worthwhile?” signal.

**Short rule:** Use **mAP50** (and mAP50-95) as your main “is training going well?” metric. If mAP50 improves over epochs, training is not worthless; **best.pt** will then be a usable model.

---

## Part 1: On your MacBook (before Colab)

### What you need to upload

You need the **YOLO-format dataset** that the conversion script created:

- **Folder:** `forest_instance_segmentation/data/yolo_finnwoodlands/`
- **Contents:** `dataset.yaml`, `images/train/`, `images/val/`, `labels/train/`, `labels/val/`

Colab needs **real image files** inside the zip. If you created the dataset **without** `--copy-images`, the `images/` folder may contain **symlinks**. Symlinks don’t travel well in a zip.

**Option A – You already have the folder:**

1. If you used `--copy-images` when you ran the conversion, you can zip as-is.
2. If you’re not sure or used symlinks, re-run the conversion with `--copy-images`, then zip:

```bash
cd /Users/vicky/Documents/Uni/Projektarbeit/Bark
python3 forest_instance_segmentation/scripts/convert_finnwoodlands_to_yolo_seg.py --val-pine-min 100 --copy-images
```

Then create the zip:

```bash
cd forest_instance_segmentation/data
zip -r yolo_finnwoodlands.zip yolo_finnwoodlands
```

You should get **`yolo_finnwoodlands.zip`** (a few hundred MB) in `forest_instance_segmentation/data/`.

**Option B – Zip from Finder:**

1. Open `Bark/forest_instance_segmentation/data/`.
2. Right‑click the **`yolo_finnwoodlands`** folder → **Compress "yolo_finnwoodlands"**.
3. You get `yolo_finnwoodlands.zip`. Use that file in the next part.

---

## Part 2: In Google Colab

Run **one cell at a time** in order. Wait for each to finish before running the next.

---

### Cell 1: Install Ultralytics and check GPU

```python
!pip install -q ultralytics
import torch
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")
```

**Run:** Shift+Enter.  
You should see something like `GPU: Tesla T4`.

---

### Cell 2: Upload your zip file

```python
from google.colab import files
uploaded = files.upload()
```

**Run:** Shift+Enter.  
A **file picker** opens. Choose **`yolo_finnwoodlands.zip`** (from your MacBook). Wait until the upload finishes (progress in the cell output).

---

### Cell 3: Unzip and fix the dataset path

```python
!unzip -q -o yolo_finnwoodlands.zip -d /content
import os
dataset_dir = "/content/yolo_finnwoodlands"
assert os.path.isdir(dataset_dir), "Folder not found. Check the zip contained 'yolo_finnwoodlands'."
# Write dataset.yaml with Colab path (overwrites the one from your laptop)
yaml_content = f"""# YOLOv8-seg dataset: FinnWoodlands (path fixed for Colab)
path: {dataset_dir}
train: images/train
val: images/val
names:
  0: Spruce
  1: Pine
nc: 2
"""
with open(f"{dataset_dir}/dataset.yaml", "w") as f:
    f.write(yaml_content)
print("Dataset ready at", dataset_dir)
!ls -la {dataset_dir}
```

**Run:** Shift+Enter.  
You should see the folder contents (e.g. `dataset.yaml`, `images`, `labels`). If you get an error like "Folder not found", your zip might have a different structure (e.g. an extra parent folder). Adjust the path in the next cells accordingly.

---

### Cell 4: Train the model

```python
from ultralytics import YOLO

data_yaml = "/content/yolo_finnwoodlands/dataset.yaml"
model = YOLO("yolov8n-seg.pt")
results = model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    project="/content",
    name="finnwoodlands_seg",
    exist_ok=True,
)
print("Training finished.")
```

**Run:** Shift+Enter.  
Training will run (about 1.5–3 hours on a T4). Don’t close the tab; you can scroll or click occasionally so Colab doesn’t think you’re idle. When it finishes, you’ll see "Training finished."

---

### Cell 5: Save the trained model so you don’t lose it

**Option A – Download to your laptop**

```python
from google.colab import files
files.download("/content/finnwoodlands_seg/weights/best.pt")
```

**Run:** Shift+Enter.  
`best.pt` will download to your MacBook (e.g. into Downloads). Put it wherever you keep your project weights.

**Option B – Copy to Google Drive (so you can download later)**

Run this once to mount Drive and copy the whole run folder:

```python
from google.colab import drive
drive.mount("/content/drive")
!cp -r /content/finnwoodlands_seg "/content/drive/MyDrive/finnwoodlands_seg"
print("Copied to Google Drive: MyDrive/finnwoodlands_seg")
```

Then you can download `best.pt` from Drive anytime (Right‑click `best.pt` → Download).

---

## Quick reference

| Step | What you do |
|------|-------------|
| **On laptop** | Zip `yolo_finnwoodlands` (use `--copy-images` if needed) → get `yolo_finnwoodlands.zip` |
| **Colab 1** | Install Ultralytics, check GPU |
| **Colab 2** | Upload `yolo_finnwoodlands.zip` with the file picker |
| **Colab 3** | Unzip to `/content`, fix `dataset.yaml` path |
| **Colab 4** | Run training (100 epochs, ~1.5–3 h) |
| **Colab 5** | Download `best.pt` or copy folder to Drive |

---

## If something goes wrong

- **"Folder not found" in Cell 3:** Your zip might unzip to a different name (e.g. `yolo_finnwoodlands/yolo_finnwoodlands/`). Run `!ls /content` and adjust `dataset_dir` and `data_yaml` to the path that actually contains `dataset.yaml`, `images/`, and `labels/`.
- **Out of memory:** In Cell 4, change `batch=16` to `batch=8` and run again.
- **Session disconnected:** If training was saving to `/content`, it’s lost. Next time use Option B (Drive) and set `project="/content/drive/MyDrive"` so checkpoints are saved to Drive; you can resume later with `model.train(resume=True)`.

Once you’ve done this once, you can reuse the same notebook and only re-upload the zip (or load from Drive) for future runs.
