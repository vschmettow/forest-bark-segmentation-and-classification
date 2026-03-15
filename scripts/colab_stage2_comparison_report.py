#!/usr/bin/env python3
"""
Colab cell: Generate stage2_160 comparison HTML with training results.
Run after Cells 5 and 8. Saves to Drive and downloads.
Paste this into a Colab cell.
"""
# === PASTE INTO COLAB ===

import cv2
import shutil
from pathlib import Path
from ultralytics import YOLO
from google.colab import files

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
BEST_MODELS = BARK / "best_models"
TEST_SET = BARK / "BarkNetYOLO" / "test_set"
OUTPUT_DIR = BARK / "Results" / "stage2_comparison"
PRED_DIR = OUTPUT_DIR / "stage2_predictions"

# Training results paths (Colab)
TRAIN_CSV = [
    ("stage2 (batch=4, 80 epochs)", "/content/yolo_stages/stage2_160/results.csv", "yolov8s_seg_stage2_160_best.pt", 4, 80),
    ("stage2 (batch=16, 100 epochs)", "/content/yolo_stage2_exp/stage2_b16_e100/results.csv", "yolov8s_seg_stage2_160_b16_e100_best.pt", 16, 100),
    ("stage2 (batch=32, 100 epochs)", "/content/yolo_stage2_exp/stage2_b32_e100/results.csv", "yolov8s_seg_stage2_160_b32_e100_best.pt", 32, 100),
]

FIXED_STEMS = ["4a085a9c75f0e68b994a2371475a1a7dc29bfff2", "7d43abfaff3b2d12c67567cdeedc28e0d43e9f1b", "02ba18373a3dbbec4dd430bae821f9e39b89d9f0", "6f7642441ad37b1be215d81bb8b1499807acbcb7"]
labels = ["Picea", "Picea", "Pinus", "Pinus"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(exist_ok=True)

def parse_csv(csv_path):
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        r = df.dropna(how="all").iloc[-1]
        cols = df.columns.tolist()
        m50 = next((c for c in cols if "mAP50" in c and "(M)" in c), next((c for c in cols if "mAP50" in c), None))
        m95 = next((c for c in cols if "mAP50-95" in c and "(M)" in c), next((c for c in cols if "mAP50-95" in c), None))
        p = next((c for c in cols if "precision" in c.lower() and "(M)" in c), next((c for c in cols if "precision" in c.lower()), None))
        rec = next((c for c in cols if "recall" in c.lower() and "(M)" in c), next((c for c in cols if "recall" in c.lower()), None))
        def f(c): v = r.get(c); return float(v) if c and v is not None and v == v else None
        return {"mAP50": f(m50), "mAP50-95": f(m95), "precision": f(p), "recall": f(rec)}
    except Exception:
        return None

def fmt(v): return f"{v:.4f}" if v is not None and v == v else "N/A"

# Test images
img_dir = TEST_SET / "images"
test_images = []
for stem in FIXED_STEMS:
    for ext in [".jpg", ".jpeg", ".png"]:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            test_images.append((stem, str(p)))
            break

for i, ((stem, img_path), lbl) in enumerate(zip(test_images, labels)):
    shutil.copy2(img_path, PRED_DIR / f"original_{lbl}_{i}{Path(img_path).suffix}")

test_yaml = TEST_SET / "dataset_colab.yaml"
test_yaml.write_text(f"path: {TEST_SET}\ntrain: images\nval: images\nnc: 2\nnames: ['Picea', 'Pinus']\n")

all_metrics, pred_paths, train_results = [], {}, []

for model_label, csv_path, model_name, batch, epochs in TRAIN_CSV:
    train_res = parse_csv(csv_path) if Path(csv_path).exists() else None
    train_results.append({"label": model_label, "batch": batch, "epochs": epochs, "train": train_res})

    model_path = BEST_MODELS / model_name
    short_name = model_name.replace("yolov8s_seg_", "").replace("_best.pt", "")

    if not model_path.exists():
        print(f"Skip {model_label}: {model_name} not found")
        all_metrics.append({"model": model_label, "short_name": short_name, "mAP50": None, "mAP50-95": None, "precision": None, "recall": None, "F1": None})
        pred_paths[short_name] = []
        continue

    model = YOLO(str(model_path))
    try:
        res = model.val(data=str(test_yaml), verbose=False)
        seg = getattr(res, "seg", None) or getattr(res, "mask", None)
        m = res
        map50 = getattr(seg, "map50", None) or getattr(m.box, "map50", None)
        map95 = getattr(seg, "map", None) or getattr(m.box, "map", None)
        prec, rec = getattr(m.box, "mp", None), getattr(m.box, "mr", None)
        p_val = float(prec) if prec is not None else None
        r_val = float(rec) if rec is not None else None
        f1 = (2 * p_val * r_val / (p_val + r_val)) if (p_val and r_val and (p_val + r_val) > 0) else None
        all_metrics.append({"model": model_label, "short_name": short_name, "mAP50": float(map50) if map50 else None, "mAP50-95": float(map95) if map95 else None, "precision": p_val, "recall": r_val, "F1": f1})
    except Exception as e:
        print(f"Val error {model_label}: {e}")
        all_metrics.append({"model": model_label, "short_name": short_name, "mAP50": None, "mAP50-95": None, "precision": None, "recall": None, "F1": None})

    stage_preds = []
    for (stem, img_path), lbl in zip(test_images, labels):
        pred = model.predict(img_path, save=False, verbose=False)[0]
        out_path = PRED_DIR / f"{short_name}_{lbl}_{stem}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(pred.plot(), cv2.COLOR_RGB2BGR))
        stage_preds.append((lbl, f"stage2_predictions/{short_name}_{lbl}_{stem}.jpg"))
    pred_paths[short_name] = stage_preds

has_training = any(tr.get("train") for tr in train_results)

html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>BarkNet stage2_160 – Model Comparison</title>
<style>body{font-family:system-ui;max-width:1400px;margin:2rem auto;padding:0 1rem} h1{color:#333} h2{margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.5rem} table{border-collapse:collapse;width:100%;margin:1rem 0} th,td{border:1px solid #ddd;padding:.75rem;text-align:left} th{background:#f5f5f5}</style></head><body>
<h1>BarkNet YOLOv8s-seg – stage2_160 Model Comparison</h1>
<p><strong>Dataset:</strong> BarkNetYOLO/stage2_160 (same for both models)</p>
<p><strong>Test set:</strong> 33 Picea, 31 Pinus (64 images). Shown: 2 Picea + 2 Pinus.</p>

<h2>1. Training Config</h2>
<table><tr><th>Model</th><th>Batch size</th><th>Epochs</th></tr>
"""
for tr in train_results:
    html += f"<tr><td><strong>{tr['label']}</strong></td><td>{tr['batch']}</td><td>{tr['epochs']}</td></tr>\n"
html += "</table>\n"

if has_training:
    html += "<h2>2. Training Results (final epoch, val split)</h2><table><tr><th>Model</th><th>mAP50</th><th>mAP50-95</th><th>Precision</th><th>Recall</th></tr>\n"
    for tr in train_results:
        t = tr.get("train") or {}
        html += f"<tr><td><strong>{tr['label']}</strong></td><td>{fmt(t.get('mAP50'))}</td><td>{fmt(t.get('mAP50-95'))}</td><td>{fmt(t.get('precision'))}</td><td>{fmt(t.get('recall'))}</td></tr>\n"
    html += "</table>\n"

sn = "3" if has_training else "2"
html += f"<h2>{sn}. Test Set Metrics</h2><table><tr><th>Model</th><th>mAP50</th><th>mAP50-95</th><th>Precision</th><th>Recall</th><th>F1</th></tr>\n"
for m in all_metrics:
    html += f"<tr><td><strong>{m['model']}</strong></td><td>{fmt(m.get('mAP50'))}</td><td>{fmt(m.get('mAP50-95'))}</td><td>{fmt(m.get('precision'))}</td><td>{fmt(m.get('recall'))}</td><td>{fmt(m.get('F1'))}</td></tr>\n"
html += "</table>\n"

import base64
def img_data(path):
    p = Path(path)
    mime = "image/jpeg" if p.suffix.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64," + base64.b64encode(p.read_bytes()).decode()

pn = "4" if has_training else "3"
html += f"<h2>{pn}. Predictions on Same 4 Test Images</h2><table><tr><th>Model</th><th>Picea</th><th>Picea</th><th>Pinus</th><th>Pinus</th></tr>\n"
html += "<tr><td><strong>Original</strong></td>"
for i, lbl in enumerate(labels):
    p = PRED_DIR / f"original_{lbl}_{i}{Path(test_images[i][1]).suffix}"
    html += f'<td><img src="{img_data(p)}" alt="Original" style="max-width:100%"><br><small>Ground truth</small></td>'
html += "</tr>\n"
for m in all_metrics:
    preds = pred_paths.get(m.get("short_name", ""), [])
    html += f"<tr><td><strong>{m['model']}</strong></td>"
    for lbl, rel in preds:
        full_path = PRED_DIR / rel.replace("stage2_predictions/", "")
        html += f'<td><img src="{img_data(full_path)}" alt="{m["short_name"]}" style="max-width:100%"><br><small>{m["short_name"]}</small></td>'
    for _ in range(4 - len(preds)):
        html += "<td>—</td>"
    html += "</tr>\n"
html += "</table></body></html>"

report_path = OUTPUT_DIR / "stage2_comparison.html"
report_path.write_text(html, encoding="utf-8")
print(f"Report saved to Drive: {report_path}")
files.download(str(report_path))
print("Downloaded: stage2_comparison.html")
