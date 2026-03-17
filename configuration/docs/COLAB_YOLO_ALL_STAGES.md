# Colab – YOLOv8s-seg All Stages + stage2 Comparison (One Notebook)

**GPU**: Runtime → Change runtime type → T4 GPU (do this *before* running any cells)  
**Data**: Upload `BarkNetYOLO` (stage1_80 … stage4_363, **test_set**, **final_dataset**) to `My Drive` / `Colab Notebooks` / `Bark` / `BarkNetYOLO/`

**Mount Drive once** at the start of each Colab session. Run cells in order – stages and stage2 batch/epoch experiment save to Drive.

---

## Cell 1: Mount Drive (once per session)

```python
from google.colab import drive
drive.mount('/content/drive')

BARK = "/content/drive/MyDrive/Colab Notebooks/Bark"
!ls "{BARK}/BarkNetYOLO/stage1_80/images/train/" | head -3
!ls "{BARK}/BarkNetYOLO/stage2_160/images/train/" | head -3
```

---

## Cell 2: Install (once)

```python
!pip install -q ultralytics
```

---

## Cell 3: Check GPU + setup

```python
import torch
from pathlib import Path

if not torch.cuda.is_available():
    print("⚠️  GPU not available! Go to: Runtime → Change runtime type → T4 GPU")
    print("   Then re-run from Cell 1.")
else:
    print(f"✓ GPU: {torch.cuda.get_device_name(0)}")

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
MODELS_DIR = BARK / "best_models"
MODELS_DIR.mkdir(exist_ok=True)
```

---

## Cell 4: Train stage1_80

```python
from pathlib import Path
from ultralytics import YOLO
import shutil

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
MODELS_DIR = BARK / "best_models"
STAGE = "stage1_80"

DATA_DIR = BARK / "BarkNetYOLO" / STAGE
YAML = Path("/content/barknet_stage1_80.yaml")
YAML.write_text(f"""path: {DATA_DIR}
train: images/train
val: images/val
nc: 2
names: ['Picea', 'Pinus']
""")

model = YOLO("yolov8s-seg.pt")
model.train(data=str(YAML), epochs=80, batch=4, imgsz=640,
            project="/content/yolo_stages", name=STAGE, save=True, plots=True)

src = Path("/content/yolo_stages/stage1_80/weights/best.pt")
if src.exists():
    shutil.copy2(src, MODELS_DIR / "yolov8s_seg_stage1_80_best.pt")
    print("Saved to Drive: best_models/yolov8s_seg_stage1_80_best.pt")
```

---

## Cell 5: Train stage2_160

```python
from pathlib import Path
from ultralytics import YOLO
import shutil

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
MODELS_DIR = BARK / "best_models"
STAGE = "stage2_160"

DATA_DIR = BARK / "BarkNetYOLO" / STAGE
YAML = Path("/content/barknet_stage2_160.yaml")
YAML.write_text(f"""path: {DATA_DIR}
train: images/train
val: images/val
nc: 2
names: ['Picea', 'Pinus']
""")

model = YOLO("yolov8s-seg.pt")
model.train(data=str(YAML), epochs=80, batch=4, imgsz=640,
            project="/content/yolo_stages", name=STAGE, save=True, plots=True)

src = Path("/content/yolo_stages/stage2_160/weights/best.pt")
if src.exists():
    shutil.copy2(src, MODELS_DIR / "yolov8s_seg_stage2_160_best.pt")
    print("Saved to Drive: best_models/yolov8s_seg_stage2_160_best.pt")
```

---

## Cell 6: Train stage3_280

```python
from pathlib import Path
from ultralytics import YOLO
import shutil

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
MODELS_DIR = BARK / "best_models"
STAGE = "stage3_280"

DATA_DIR = BARK / "BarkNetYOLO" / STAGE
YAML = Path("/content/barknet_stage3_280.yaml")
YAML.write_text(f"""path: {DATA_DIR}
train: images/train
val: images/val
nc: 2
names: ['Picea', 'Pinus']
""")

model = YOLO("yolov8s-seg.pt")
model.train(data=str(YAML), epochs=80, batch=4, imgsz=640,
            project="/content/yolo_stages", name=STAGE, save=True, plots=True)

src = Path("/content/yolo_stages/stage3_280/weights/best.pt")
if src.exists():
    shutil.copy2(src, MODELS_DIR / "yolov8s_seg_stage3_280_best.pt")
    print("Saved to Drive: best_models/yolov8s_seg_stage3_280_best.pt")
```

---

## Cell 7: Train stage4_363

```python
from pathlib import Path
from ultralytics import YOLO
import shutil

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
MODELS_DIR = BARK / "best_models"
STAGE = "stage4_363"

DATA_DIR = BARK / "BarkNetYOLO" / STAGE
YAML = Path("/content/barknet_stage4_363.yaml")
YAML.write_text(f"""path: {DATA_DIR}
train: images/train
val: images/val
nc: 2
names: ['Picea', 'Pinus']
""")

model = YOLO("yolov8s-seg.pt")
model.train(data=str(YAML), epochs=80, batch=4, imgsz=640,
            project="/content/yolo_stages", name=STAGE, save=True, plots=True)

src = Path("/content/yolo_stages/stage4_363/weights/best.pt")
if src.exists():
    shutil.copy2(src, MODELS_DIR / "yolov8s_seg_stage4_363_best.pt")
    print("Saved to Drive: best_models/yolov8s_seg_stage4_363_best.pt")

print("Done. All 4 models in Drive: best_models/")
```

---

## Cell 7b: Train final_dataset (full dataset, 1537 train / 330 val)

Uses `BarkNetYOLO/final_dataset` – full dataset (932 Picea, 935 Pinus) with test_set excluded. Epochs=100, batch=32.

```python
from pathlib import Path
from ultralytics import YOLO
import shutil

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
MODELS_DIR = BARK / "best_models"
STAGE = "final_dataset"

DATA_DIR = BARK / "BarkNetYOLO" / STAGE
YAML = Path("/content/barknet_final_dataset.yaml")
YAML.write_text(f"""path: {DATA_DIR}
train: images/train
val: images/val
nc: 2
names: ['Picea', 'Pinus']
""")

model = YOLO("yolov8s-seg.pt")
model.train(data=str(YAML), epochs=100, batch=32, imgsz=640,
            project="/content/yolo_stages", name=STAGE, save=True, plots=True)

src = Path("/content/yolo_stages/final_dataset/weights/best.pt")
if src.exists():
    shutil.copy2(src, MODELS_DIR / "yolov8s_seg_final_dataset_best.pt")
    print("Saved to Drive: best_models/yolov8s_seg_final_dataset_best.pt")

print("Done. final_dataset model in Drive: best_models/")
```

---

## Cell 8: Train stage2_160 new (batch=16, epochs=100)

Same dataset as stage2_160, different batch and epochs.

```python
from pathlib import Path
from ultralytics import YOLO
import shutil

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
MODELS_DIR = BARK / "best_models"
DATA_DIR = BARK / "BarkNetYOLO" / "stage2_160"

YAML = Path("/content/barknet_stage2_160_exp.yaml")
YAML.write_text(f"""path: {DATA_DIR}
train: images/train
val: images/val
nc: 2
names: ['Picea', 'Pinus']
""")

model = YOLO("yolov8s-seg.pt")
model.train(data=str(YAML), epochs=100, batch=16, imgsz=640,
            project="/content/yolo_stage2_exp", name="stage2_b16_e100", save=True, plots=True)

src = Path("/content/yolo_stage2_exp/stage2_b16_e100/weights/best.pt")
if src.exists():
    shutil.copy2(src, MODELS_DIR / "yolov8s_seg_stage2_160_b16_e100_best.pt")
    print("Saved to Drive: best_models/yolov8s_seg_stage2_160_b16_e100_best.pt")

print("Done. stage2 comparison models in Drive: best_models/")
```

---

## Cell 9: Train stage2_160 new (batch=32, epochs=100)

Same dataset as stage2_160, batch=32, epochs=100.

```python
from pathlib import Path
from ultralytics import YOLO
import shutil

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
MODELS_DIR = BARK / "best_models"
DATA_DIR = BARK / "BarkNetYOLO" / "stage2_160"

YAML = Path("/content/barknet_stage2_160_exp.yaml")
YAML.write_text(f"""path: {DATA_DIR}
train: images/train
val: images/val
nc: 2
names: ['Picea', 'Pinus']
""")

model = YOLO("yolov8s-seg.pt")
model.train(data=str(YAML), epochs=100, batch=32, imgsz=640,
            project="/content/yolo_stage2_exp", name="stage2_b32_e100", save=True, plots=True)

src = Path("/content/yolo_stage2_exp/stage2_b32_e100/weights/best.pt")
if src.exists():
    shutil.copy2(src, MODELS_DIR / "yolov8s_seg_stage2_160_b32_e100_best.pt")
    print("Saved to Drive: best_models/yolov8s_seg_stage2_160_b32_e100_best.pt")

print("Done. stage2 b32/e100 model in Drive: best_models/")
```

---

## Cell 10: Download stage2_160 b32/e100 model only (optional)

```python
from google.colab import files
from pathlib import Path

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
MODELS_DIR = BARK / "best_models"
model_name = "yolov8s_seg_stage2_160_b32_e100_best.pt"
p = MODELS_DIR / model_name

if p.exists():
    files.download(str(p))
    print(f"Downloaded: {model_name}")
else:
    print(f"Not found: {p}")
```

---

## Cell 10b: Download final_dataset (alldata) best model only (optional)

```python
from google.colab import files
from pathlib import Path

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
MODELS_DIR = BARK / "best_models"
model_name = "yolov8s_seg_final_dataset_best.pt"
p = MODELS_DIR / model_name

if p.exists():
    files.download(str(p))
    print(f"Downloaded: {model_name}")
else:
    print(f"Not found: {p}. Run Cell 7b first.")
```

---

## Cell 11: Accuracy table (stages + stage2 comparison)

```python
import pandas as pd
from pathlib import Path

def get_stage_metrics(stage):
    csv = Path(f"/content/yolo_stages/{stage}/results.csv")
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    r = df.dropna(how="all").iloc[-1]
    cols = df.columns.tolist()
    m50 = next((c for c in cols if "mAP50" in c and "(M)" in c), next((c for c in cols if "mAP50" in c), None))
    m95 = next((c for c in cols if "mAP50-95" in c and "(M)" in c), next((c for c in cols if "mAP50-95" in c), None))
    p = next((c for c in cols if "precision" in c.lower() and "(M)" in c), next((c for c in cols if "precision" in c.lower()), None))
    rec = next((c for c in cols if "recall" in c.lower() and "(M)" in c), next((c for c in cols if "recall" in c.lower()), None))
    def f(x): v = r.get(x); return f"{float(v):.4f}" if v is not None and v==v else "N/A"
    prec_val, rec_val = r.get(p) if p else None, r.get(rec) if rec else None
    try:
        pv, rv = float(prec_val), float(rec_val)
        f1 = f"{2*pv*rv/(pv+rv):.4f}" if (pv+rv) > 0 else "N/A"
    except (TypeError, ValueError):
        f1 = "N/A"
    return {"Model": stage, "mAP50": f(m50), "mAP50-95": f(m95), "Precision": f(p), "Recall": f(rec), "F1": f1}

def get_stage2_exp_metrics(run_name, label):
    csv = Path(f"/content/yolo_stage2_exp/{run_name}/results.csv")
    if not csv.exists():
        return None
    df = pd.read_csv(csv)
    r = df.dropna(how="all").iloc[-1]
    cols = df.columns.tolist()
    m50 = next((c for c in cols if "mAP50" in c and "(M)" in c), next((c for c in cols if "mAP50" in c), None))
    m95 = next((c for c in cols if "mAP50-95" in c and "(M)" in c), next((c for c in cols if "mAP50-95" in c), None))
    p = next((c for c in cols if "precision" in c.lower() and "(M)" in c), next((c for c in cols if "precision" in c.lower()), None))
    rec = next((c for c in cols if "recall" in c.lower() and "(M)" in c), next((c for c in cols if "recall" in c.lower()), None))
    def f(x): v = r.get(x); return f"{float(v):.4f}" if v is not None and v==v else "N/A"
    prec_val, rec_val = r.get(p) if p else None, r.get(rec) if rec else None
    try:
        pv, rv = float(prec_val), float(rec_val)
        f1 = f"{2*pv*rv/(pv+rv):.4f}" if (pv+rv) > 0 else "N/A"
    except (TypeError, ValueError):
        f1 = "N/A"
    return {"Model": label, "mAP50": f(m50), "mAP50-95": f(m95), "Precision": f(p), "Recall": f(rec), "F1": f1}

print("Stages (batch=4, epochs=80):")
rows_stages = [get_stage_metrics(s) for s in ["stage1_80", "stage2_160", "stage3_280", "stage4_363"]]
rows_stages = [r for r in rows_stages if r]
if rows_stages:
    display(pd.DataFrame(rows_stages))
else:
    print("  Run Cells 4–7 first.")

print("\nstage2_160 comparison (batch/epoch experiment):")
base = get_stage_metrics("stage2_160")
if base:
    base["Model"] = "stage2_160 (batch=4, epochs=80)"
rows_stage2 = [
    base,
    get_stage2_exp_metrics("stage2_b16_e100", "stage2_160 (batch=16, epochs=100)"),
    get_stage2_exp_metrics("stage2_b32_e100", "stage2_160 (batch=32, epochs=100)"),
]
rows_stage2 = [r for r in rows_stage2 if r]
if rows_stage2:
    display(pd.DataFrame(rows_stage2))
else:
    print("  Run Cells 5, 8 and 9 first.")
```

---

## Cell 11b: Accuracy table for final_dataset – val on test_set

Runs `model.val()` on the test_set for the final_dataset model and displays mAP50, mAP50-95, Precision, Recall, F1.

```python
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
BEST_MODELS = BARK / "best_models"
TEST_SET = BARK / "BarkNetYOLO" / "test_set"

# Create test_set dataset.yaml for val
test_yaml = TEST_SET / "dataset_colab.yaml"
test_yaml.write_text(f"""path: {TEST_SET}
train: images
val: images
nc: 2
names: ['Picea', 'Pinus']
""")

def get_val_metrics(model_path, label):
    """Run val on test_set, return dict with mAP50, mAP50-95, Precision, Recall, F1."""
    if not model_path.exists():
        return {"Model": label, "mAP50": "N/A", "mAP50-95": "N/A", "Precision": "N/A", "Recall": "N/A", "F1": "N/A"}
    try:
        model = YOLO(str(model_path))
        res = model.val(data=str(test_yaml), verbose=False)
        seg = getattr(res, "seg", None) or getattr(res, "mask", None)
        m = res
        map50 = getattr(seg, "map50", None) or getattr(m.box, "map50", None)
        map95 = getattr(seg, "map", None) or getattr(m.box, "map", None)
        prec = getattr(m.box, "mp", None)
        rec = getattr(m.box, "mr", None)
        p_val = float(prec) if prec is not None else None
        r_val = float(rec) if rec is not None else None
        f1 = (2 * p_val * r_val / (p_val + r_val)) if (p_val and r_val and (p_val + r_val) > 0) else None
        def fmt(v): return f"{v:.4f}" if v is not None and v == v else "N/A"
        return {"Model": label, "mAP50": fmt(map50), "mAP50-95": fmt(map95), "Precision": fmt(p_val), "Recall": fmt(r_val), "F1": fmt(f1)}
    except Exception as e:
        return {"Model": label, "mAP50": "N/A", "mAP50-95": "N/A", "Precision": "N/A", "Recall": "N/A", "F1": f"Error: {e}"}

# final_dataset only
model_path = BEST_MODELS / "yolov8s_seg_final_dataset_best.pt"
row = get_val_metrics(model_path, "final_dataset")

print("Accuracy table – final_dataset, validation on test_set (64 images):")
display(pd.DataFrame([row]))
```

---

## Cell 11c: Generate HTML report for final_dataset

Creates a self-contained HTML with metrics (mAP50, mAP50-95, Precision, Recall, F1) and predictions on 4 test images. Saves to Drive and downloads.

```python
import cv2
import base64
from pathlib import Path
from ultralytics import YOLO
from google.colab import files

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
BEST_MODELS = BARK / "best_models"
TEST_SET = BARK / "BarkNetYOLO" / "test_set"
OUTPUT_DIR = BARK / "final_dataset_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "predictions").mkdir(exist_ok=True)

MODEL_PATH = BEST_MODELS / "yolov8s_seg_final_dataset_best.pt"
if not MODEL_PATH.exists():
    print(f"Model not found: {MODEL_PATH}. Upload it to best_models/ first.")
else:
    # 1. Fixed test images (2 Picea + 2 Pinus)
    FIXED_STEMS = ["4a085a9c75f0e68b994a2371475a1a7dc29bfff2", "7d43abfaff3b2d12c67567cdeedc28e0d43e9f1b", "02ba18373a3dbbec4dd430bae821f9e39b89d9f0", "6f7642441ad37b1be215d81bb8b1499807acbcb7"]
    labels = ["Picea", "Picea", "Pinus", "Pinus"]
    img_dir = TEST_SET / "images"
    test_images = []
    for stem in FIXED_STEMS:
        for ext in [".jpg", ".jpeg", ".png"]:
            p = img_dir / f"{stem}{ext}"
            if p.exists():
                test_images.append((stem, str(p)))
                break

    # Copy originals
    for i, ((stem, img_path), lbl) in enumerate(zip(test_images, labels)):
        import shutil
        shutil.copy2(img_path, OUTPUT_DIR / "predictions" / f"original_{lbl}_{i}{Path(img_path).suffix}")

    # 2. Create test_set dataset.yaml
    test_yaml = TEST_SET / "dataset_colab.yaml"
    test_yaml.write_text(f"""path: {TEST_SET}
train: images
val: images
nc: 2
names: ['Picea', 'Pinus']
""")

    # 3. Run val + inference
    model = YOLO(str(MODEL_PATH))
    res = model.val(data=str(test_yaml), verbose=False)
    seg = getattr(res, "seg", None) or getattr(res, "mask", None)
    m = res
    map50 = getattr(seg, "map50", None) or getattr(m.box, "map50", None)
    map95 = getattr(seg, "map", None) or getattr(m.box, "map", None)
    prec = getattr(m.box, "mp", None)
    rec = getattr(m.box, "mr", None)
    p_val = float(prec) if prec is not None else None
    r_val = float(rec) if rec is not None else None
    f1 = (2 * p_val * r_val / (p_val + r_val)) if (p_val and r_val and (p_val + r_val) > 0) else None

    pred_paths = []
    for (stem, img_path), lbl in zip(test_images, labels):
        pred = model.predict(img_path, save=False, verbose=False)[0]
        out_path = OUTPUT_DIR / "predictions" / f"final_dataset_{lbl}_{stem}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(pred.plot(), cv2.COLOR_RGB2BGR))
        pred_paths.append((lbl, f"predictions/final_dataset_{lbl}_{stem}.jpg"))

    # 4. Build HTML (base64 images for self-contained file)
    def img_data(path):
        p = Path(path)
        mime = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
        return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
    def fmt(v): return f"{v:.4f}" if v is not None and v == v else "N/A"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BarkNet final_dataset – Results</title>
<style>body{{font-family:system-ui;max-width:1400px;margin:2rem auto}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:.5rem}} th{{background:#f5f5f5}} img{{max-width:100%}}</style></head><body>
<h1>BarkNet YOLOv8s-seg – final_dataset</h1>
<p><strong>Training:</strong> Batch size = 32, Epochs = 100</p>
<p><strong>Test set:</strong> 33 Picea, 31 Pinus (64 images total). Shown: 2 Picea + 2 Pinus.</p>
<h2>1. Metrics (validation on test_set)</h2>
<table><tr><th>Model</th><th>mAP50</th><th>mAP50-95</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
<tr><td><strong>final_dataset</strong></td><td>{fmt(map50)}</td><td>{fmt(map95)}</td><td>{fmt(p_val)}</td><td>{fmt(r_val)}</td><td>{fmt(f1)}</td></tr>
</table>
<h2>2. Predictions on 4 Test Images</h2>
<table><tr><th></th><th>Picea</th><th>Picea</th><th>Pinus</th><th>Pinus</th></tr>
<tr><td><strong>Original</strong></td>"""
    for i, lbl in enumerate(labels):
        p = OUTPUT_DIR / "predictions" / f"original_{lbl}_{i}{Path(test_images[i][1]).suffix}"
        html += f'<td><img src="{img_data(p)}" alt="Original"><br><small>Ground truth</small></td>'
    html += "</tr>\n<tr><td><strong>final_dataset</strong></td>"
    for lbl, rel in pred_paths:
        html += f'<td><img src="{img_data(OUTPUT_DIR/rel)}" alt="final_dataset"><br><small>Prediction</small></td>'
    html += "</tr>\n</table></body></html>"

    report_path = OUTPUT_DIR / "final_dataset_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"Report saved to Drive: {report_path}")
    files.download(str(report_path))
    print("Downloaded: final_dataset_report.html (self-contained, open in browser)")
```

---

## Cell 12: Download all models (optional)

```python
from google.colab import files
from pathlib import Path

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
MODELS_DIR = BARK / "best_models"

for name in ["yolov8s_seg_stage1_80_best.pt", "yolov8s_seg_stage2_160_best.pt",
             "yolov8s_seg_stage3_280_best.pt", "yolov8s_seg_stage4_363_best.pt",
             "yolov8s_seg_final_dataset_best.pt",
             "yolov8s_seg_stage2_160_b16_e100_best.pt", "yolov8s_seg_stage2_160_b32_e100_best.pt"]:
    p = MODELS_DIR / name
    if p.exists():
        files.download(str(p))
        print(f"Downloaded: {name}")
    else:
        print(f"Not found: {name}")
```

---

## Cell 13: Generate HTML report (stages, run after Cells 4–7)

Creates the combined report with metrics + predictions on 4 test images. Saves to Drive and downloads.

```python
import cv2
from pathlib import Path
from ultralytics import YOLO
from google.colab import files

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
BEST_MODELS = BARK / "best_models"
TEST_SET = BARK / "BarkNetYOLO" / "test_set"
OUTPUT_DIR = BARK / "stages_combined_report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "predictions").mkdir(exist_ok=True)

STAGES = ["stage1_80", "stage2_160", "stage3_280", "stage4_363"]
MODEL_NAMES = {s: f"yolov8s_seg_{s}_best.pt" for s in STAGES}

# 1. Fixed test images (2 Picea + 2 Pinus)
FIXED_STEMS = ["4a085a9c75f0e68b994a2371475a1a7dc29bfff2", "7d43abfaff3b2d12c67567cdeedc28e0d43e9f1b", "02ba18373a3dbbec4dd430bae821f9e39b89d9f0", "6f7642441ad37b1be215d81bb8b1499807acbcb7"]
labels = ["Picea", "Picea", "Pinus", "Pinus"]
img_dir = TEST_SET / "images"
test_images = []
for stem in FIXED_STEMS:
    for ext in [".jpg", ".jpeg", ".png"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            test_images.append((stem, str(p)))
            break

# Copy originals
for i, ((stem, img_path), lbl) in enumerate(zip(test_images, labels)):
    import shutil
    shutil.copy2(img_path, OUTPUT_DIR / "predictions" / f"original_{lbl}_{i}{Path(img_path).suffix}")

# 2. Create test_set dataset.yaml for Colab
test_yaml = TEST_SET / "dataset_colab.yaml"
test_yaml.write_text(f"""path: {TEST_SET}
train: images
val: images
nc: 2
names: ['Picea', 'Pinus']
""")

# 3. Run inference + collect metrics per stage
all_metrics, pred_paths = [], {}
for stage in STAGES:
    model_path = BEST_MODELS / MODEL_NAMES[stage]
    if not model_path.exists():
        print(f"Skip {stage}: model not found"); all_metrics.append({"stage": stage, "mAP50": None, "mAP50-95": None, "precision": None, "recall": None, "F1": None}); pred_paths[stage] = []; continue
    model = YOLO(str(model_path))
    try:
        res = model.val(data=str(test_yaml), verbose=False)
        seg = getattr(res, "seg", None) or getattr(res, "mask", None)
        m = res
        map50 = getattr(seg, "map50", None) or getattr(m.box, "map50", None)
        map95 = getattr(seg, "map", None) or getattr(m.box, "map", None)
        prec = getattr(m.box, "mp", None); rec = getattr(m.box, "mr", None)
        p_val = float(prec) if prec is not None else None
        r_val = float(rec) if rec is not None else None
        f1 = (2 * p_val * r_val / (p_val + r_val)) if (p_val and r_val and (p_val + r_val) > 0) else None
        all_metrics.append({"stage": stage, "mAP50": float(map50) if map50 else None, "mAP50-95": float(map95) if map95 else None, "precision": p_val, "recall": r_val, "F1": f1})
    except Exception as e:
        print(f"Val error {stage}: {e}")
        all_metrics.append({"stage": stage, "mAP50": None, "mAP50-95": None, "precision": None, "recall": None, "F1": None})
    stage_preds = []
    for (stem, img_path), lbl in zip(test_images, labels):
        pred = model.predict(img_path, save=False, verbose=False)[0]
        out_path = OUTPUT_DIR / "predictions" / f"{stage}_{lbl}_{stem}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(pred.plot(), cv2.COLOR_RGB2BGR))
        stage_preds.append((lbl, f"predictions/{stage}_{lbl}_{stem}.jpg"))
    pred_paths[stage] = stage_preds

# 4. Build HTML (embed images as base64 so the downloaded file is self-contained)
import base64
def img_data(path):
    p = Path(path)
    mime = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()
def fmt(v): return f"{v:.4f}" if v is not None and v == v else "N/A"
html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BarkNet Stages</title>
<style>body{{font-family:system-ui;max-width:1400px;margin:2rem auto}} table{{border-collapse:collapse}} th,td{{border:1px solid #ddd;padding:.5rem}} th{{background:#f5f5f5}} img{{max-width:100%}}</style></head><body>
<h1>BarkNet YOLOv8s-seg – All Stages Combined</h1>
<p><strong>Training:</strong> Batch size = 4, Epochs = 80</p>
<p><strong>Test set:</strong> 33 Picea, 31 Pinus (64 images total). Shown: 2 Picea + 2 Pinus.</p>
<h2>1. Metrics Comparison</h2>
<table><tr><th>Stage</th><th>mAP50</th><th>mAP50-95</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
"""
for m in all_metrics:
    html += f"<tr><td><strong>{m['stage']}</strong></td><td>{fmt(m.get('mAP50'))}</td><td>{fmt(m.get('mAP50-95'))}</td><td>{fmt(m.get('precision'))}</td><td>{fmt(m.get('recall'))}</td><td>{fmt(m.get('F1'))}</td></tr>\n"
html += "</table><h2>2. Predictions on Same 4 Test Images</h2><table><tr><th>Stage</th><th>Picea</th><th>Picea</th><th>Pinus</th><th>Pinus</th></tr>\n"
html += "<tr><td><strong>Original</strong></td>"
for i, lbl in enumerate(labels):
    p = OUTPUT_DIR / "predictions" / f"original_{lbl}_{i}{Path(test_images[i][1]).suffix}"
    html += f'<td><img src="{img_data(p)}" alt="Original"><br><small>Ground truth</small></td>'
html += "</tr>\n"
for stage in STAGES:
    preds = pred_paths.get(stage, [])
    html += f"<tr><td><strong>{stage}</strong></td>"
    for lbl, rel in preds:
        html += f'<td><img src="{img_data(OUTPUT_DIR/rel)}" alt="{stage}"><br><small>{stage}</small></td>'
    for _ in range(4 - len(preds)): html += "<td>—</td>"
    html += "</tr>\n"
html += "</table></body></html>"

report_path = OUTPUT_DIR / "stages_combined_report.html"
report_path.write_text(html, encoding="utf-8")
print(f"Report saved to Drive: {report_path}")
files.download(str(report_path))
print("Downloaded: stages_combined_report.html (self-contained, open in browser)")
```
