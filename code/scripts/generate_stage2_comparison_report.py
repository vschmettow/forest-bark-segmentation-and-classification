#!/usr/bin/env python3
"""
Generate HTML comparison report for stage2_160 models:
- Smaller batch: batch=4, epochs=80
- Larger batch: batch=16, epochs=100
Same dataset (BarkNetYOLO/stage2_160). Includes training results if available.
Output: Results/stage2_comparison.html
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BEST_MODELS = PROJECT_ROOT / "models"
TEST_SET = PROJECT_ROOT / "BarkNetYOLO" / "test_set"
OUTPUT_DIR = PROJECT_ROOT  / "results"
PRED_DIR = OUTPUT_DIR / "stage2_predictions"
TRAINING_RESULTS_DIR = PROJECT_ROOT  / "results" / "training_results"

# (label, model_file, batch, epochs, results_csv_relative_path)
MODELS = [
    ("stage2 (batch=4, 80 epochs)", "yolov8s_seg_stage2_160_best.pt", 4, 80, "stage2_160/results.csv"),
    ("stage2 (batch=16, 100 epochs)", "yolov8s_seg_stage2_160_b16_e100_best.pt", 16, 100, "stage2_160_b16_e100/results.csv"),
    ("stage2 (batch=32, 100 epochs)", "yolov8s_seg_stage2_160_b32_e100_best.pt", 32, 100, "stage2_160_b32_e100/results.csv"),
]

FIXED_TEST_STEMS = [
    "4a085a9c75f0e68b994a2371475a1a7dc29bfff2",  # Picea
    "7d43abfaff3b2d12c67567cdeedc28e0d43e9f1b",  # Picea
    "02ba18373a3dbbec4dd430bae821f9e39b89d9f0",  # Pinus
    "6f7642441ad37b1be215d81bb8b1499807acbcb7",  # Pinus
]


def get_test_images():
    img_dir = TEST_SET / "images"
    selected = []
    for stem in FIXED_TEST_STEMS:
        for ext in [".jpg", ".jpeg", ".png"]:
            img = img_dir / f"{stem}{ext}"
            if img.exists():
                selected.append((stem, str(img)))
                break
    return selected


def parse_training_results(csv_path: Path):
    """Parse last row of YOLOv8 results.csv for segmentation metrics (M)."""
    if not csv_path.exists():
        return None
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        r = df.dropna(how="all").iloc[-1]
        cols = df.columns.tolist()
        m50 = next((c for c in cols if "mAP50" in c and "(M)" in c), next((c for c in cols if "mAP50" in c), None))
        m95 = next((c for c in cols if "mAP50-95" in c and "(M)" in c), next((c for c in cols if "mAP50-95" in c), None))
        p = next((c for c in cols if "precision" in c.lower() and "(M)" in c), next((c for c in cols if "precision" in c.lower()), None))
        rec = next((c for c in cols if "recall" in c.lower() and "(M)" in c), next((c for c in cols if "recall" in c.lower()), None))

        def safe_float(col):
            if col is None:
                return None
            v = r.get(col)
            if v is None or (isinstance(v, float) and v != v):
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        return {
            "mAP50": safe_float(m50),
            "mAP50-95": safe_float(m95),
            "precision": safe_float(p),
            "recall": safe_float(rec),
        }
    except Exception:
        return None


def main():
    from ultralytics import YOLO
    import cv2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(exist_ok=True)

    test_images = get_test_images()
    labels = ["Picea", "Picea", "Pinus", "Pinus"]

    # Copy originals
    for i, ((stem, img_path), lbl) in enumerate(zip(test_images, labels)):
        import shutil
        src = Path(img_path)
        shutil.copy2(src, PRED_DIR / f"original_{lbl}_{i}{src.suffix}")

    all_metrics = []
    pred_paths = {}
    training_results = []

    for model_label, model_name, batch, epochs, csv_rel in MODELS:
        model_path = BEST_MODELS / model_name
        short_name = model_name.replace("yolov8s_seg_", "").replace("_best.pt", "")

        # Training results from results.csv (if exported from Colab)
        train_res = parse_training_results(TRAINING_RESULTS_DIR / csv_rel)
        training_results.append({
            "label": model_label,
            "batch": batch,
            "epochs": epochs,
            "train": train_res,
        })

        if not model_path.exists():
            print(f"  Skip {model_label}: {model_name} not found")
            all_metrics.append({
                "model": model_label,
                "short_name": short_name,
                "mAP50": None, "mAP50-95": None, "precision": None, "recall": None, "F1": None,
            })
            pred_paths[short_name] = []
            continue

        model = YOLO(str(model_path))

        try:
            res = model.val(data=str(TEST_SET / "dataset.yaml"), verbose=False)
            m = res
            seg = getattr(m, "seg", None) or getattr(m, "mask", None)
            map50 = getattr(seg, "map50", None) or getattr(m.box, "map50", None)
            map95 = getattr(seg, "map", None) or getattr(m.box, "map", None)
            prec = getattr(m.box, "mp", None)
            rec = getattr(m.box, "mr", None)
            p_val = float(prec) if prec is not None else None
            r_val = float(rec) if rec is not None else None
            f1 = (2 * p_val * r_val / (p_val + r_val)) if (p_val is not None and r_val is not None and (p_val + r_val) > 0) else None
            all_metrics.append({
                "model": model_label,
                "short_name": short_name,
                "mAP50": float(map50) if map50 is not None else None,
                "mAP50-95": float(map95) if map95 is not None else None,
                "precision": p_val,
                "recall": r_val,
                "F1": f1,
            })
        except Exception as e:
            print(f"  Val error {model_label}: {e}")
            all_metrics.append({
                "model": model_label,
                "short_name": short_name,
                "mAP50": None, "mAP50-95": None, "precision": None, "recall": None, "F1": None,
            })

        stage_preds = []
        for (stem, img_path), lbl in zip(test_images, labels):
            pred = model.predict(img_path, save=False, verbose=False)[0]
            plot_img = pred.plot()
            if plot_img is not None:
                out_name = f"{short_name}_{lbl}_{stem}.jpg"
                out_path = PRED_DIR / out_name
                cv2.imwrite(str(out_path), cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
                stage_preds.append((lbl, f"stage2_predictions/{out_name}"))
        pred_paths[short_name] = stage_preds

    # Build HTML
    def fmt(v):
        return f"{v:.4f}" if v is not None and v == v else "N/A"

    has_training = any(tr.get("train") for tr in training_results)

    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>BarkNet stage2_160 – Model Comparison</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 1400px; margin: 2rem auto; padding: 0 1rem; }
    h1 { color: #333; }
    h2 { margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.5rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
    th, td { border: 1px solid #ddd; padding: 0.75rem; text-align: left; }
    th { background: #f5f5f5; }
    .config { background: #f9f9f9; padding: 1rem; border-radius: 6px; margin: 1rem 0; }
  </style>
</head>
<body>
  <h1>BarkNet YOLOv8s-seg – stage2_160 Model Comparison</h1>
  <p><strong>Dataset:</strong> BarkNetYOLO/stage2_160 (same for all three models)</p>
  <p><strong>Test set:</strong> 33 Picea, 31 Pinus (64 images). Shown: 2 Picea + 2 Pinus.</p>

  <h2>1. Training Config</h2>
  <table>
    <tr><th>Model</th><th>Batch size</th><th>Epochs</th></tr>
"""
    for tr in training_results:
        html += f"    <tr><td><strong>{tr['label']}</strong></td><td>{tr['batch']}</td><td>{tr['epochs']}</td></tr>\n"
    html += "  </table>\n"

    if has_training:
        html += """
  <h2>2. Training Results (final epoch, val split)</h2>
  <p><small>From results.csv. Place in <code>Results/training_results/stage2_160/</code>, <code>stage2_160_b16_e100/</code>, <code>stage2_160_b32_e100/</code> after exporting from Colab.</small></p>
  <table>
    <tr><th>Model</th><th>mAP50</th><th>mAP50-95</th><th>Precision</th><th>Recall</th></tr>
"""
        for tr in training_results:
            t = tr.get("train") or {}
            html += f"    <tr><td><strong>{tr['label']}</strong></td><td>{fmt(t.get('mAP50'))}</td><td>{fmt(t.get('mAP50-95'))}</td><td>{fmt(t.get('precision'))}</td><td>{fmt(t.get('recall'))}</td></tr>\n"
        html += "  </table>\n"

    html += """
  <h2>""" + ("3" if has_training else "2") + """. Test Set Metrics</h2>
  <table>
    <tr>
      <th>Model</th>
      <th>mAP50</th>
      <th>mAP50-95</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
"""

    for m in all_metrics:
        html += f"""    <tr>
      <td><strong>{m['model']}</strong></td>
      <td>{fmt(m.get('mAP50'))}</td>
      <td>{fmt(m.get('mAP50-95'))}</td>
      <td>{fmt(m.get('precision'))}</td>
      <td>{fmt(m.get('recall'))}</td>
      <td>{fmt(m.get('F1'))}</td>
    </tr>
"""

    pred_section_num = "4" if has_training else "3"
    html += f"""  </table>

  <h2>{pred_section_num}. Predictions on Same 4 Test Images</h2>
  <table style="margin-top:1rem; table-layout:fixed;"><tr><th>Model</th><th>Picea</th><th>Picea</th><th>Pinus</th><th>Pinus</th></tr>
"""

    html += "  <tr><td><strong>Original</strong></td>"
    for i, lbl in enumerate(labels):
        ext = Path(test_images[i][1]).suffix
        html += f'<td style="vertical-align:top;"><img src="stage2_predictions/original_{lbl}_{i}{ext}" alt="Original" style="max-width:100%;"><br><small>Ground truth</small></td>'
    html += "</tr>\n"

    for m in all_metrics:
        short_name = m.get("short_name", "")
        preds = pred_paths.get(short_name, [])
        html += f"  <tr><td><strong>{m['model']}</strong></td>"
        for lbl, rel_path in preds:
            html += f'<td style="vertical-align:top;"><img src="{rel_path}" alt="{short_name}" style="max-width:100%;"><br><small>{short_name}</small></td>'
        for _ in range(4 - len(preds)):
            html += "<td>—</td>"
        html += "</tr>\n"

    html += """  </table>

  <p style="margin-top: 2rem; color: #666; font-size: 0.9rem;">
    Generated by generate_stage2_comparison_report.py
  </p>
</body>
</html>
"""

    report_path = OUTPUT_DIR / "stage2_comparison.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"Report: {report_path}")

    try:
        import webbrowser
        webbrowser.open(f"file://{report_path.resolve()}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    exit(main())
