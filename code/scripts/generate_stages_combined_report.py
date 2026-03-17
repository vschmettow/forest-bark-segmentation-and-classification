#!/usr/bin/env python3
"""
Generate combined HTML report for all 4 YOLOv8s-seg stages.
- Uses 2 Picea + 2 Pinus from test_set (fixed stems, see FIXED_TEST_STEMS)
- Loads each stage model, runs inference on the 4 images
- Collects metrics from results.csv or runs val() on test_set
- Outputs stages_combined_report.html
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BEST_MODELS = PROJECT_ROOT / "models"
TEST_SET = PROJECT_ROOT / "BarkNetYOLO" / "test_set"
OUTPUT_DIR = PROJECT_ROOT  / "results"

STAGES = ["stage1_80", "stage2_160", "stage3_280", "stage4_363"]
MODEL_NAMES = {
    "stage1_80": "yolov8s_seg_stage1_80_best.pt",
    "stage2_160": "yolov8s_seg_stage2_160_best.pt",
    "stage3_280": "yolov8s_seg_stage3_280_best.pt",
    "stage4_363": "yolov8s_seg_stage4_363_best.pt",
}


# Fixed test images: 2 Picea + 2 Pinus
FIXED_TEST_STEMS = [
    "4a085a9c75f0e68b994a2371475a1a7dc29bfff2",  # Picea
    "7d43abfaff3b2d12c67567cdeedc28e0d43e9f1b",  # Picea
    "02ba18373a3dbbec4dd430bae821f9e39b89d9f0",  # Pinus
    "6f7642441ad37b1be215d81bb8b1499807acbcb7",  # Pinus
]


def get_test_images():
    """Select 2 Picea + 2 Pinus from test_set (fixed stems)."""
    img_dir = TEST_SET / "images"
    selected = []
    for stem in FIXED_TEST_STEMS:
        for ext in [".jpg", ".jpeg", ".png"]:
            img = img_dir / f"{stem}{ext}"
            if img.exists():
                selected.append((stem, str(img)))
                break
    return selected


def main():
    from ultralytics import YOLO
    import cv2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(exist_ok=True)

    # 1. Select 4 fixed test images
    test_images = get_test_images()
    labels = ["Picea", "Picea", "Pinus", "Pinus"]

    # Copy original 4 test images for reference
    orig_dir = OUTPUT_DIR / "predictions"
    for i, ((stem, img_path), lbl) in enumerate(zip(test_images, labels)):
        import shutil
        src = Path(img_path)
        ext = src.suffix
        dst = orig_dir / f"original_{lbl}_{i}{ext}"
        shutil.copy2(src, dst)

    # 2. Collect metrics and run inference for each stage
    all_metrics = []
    pred_paths = {}  # stage -> [(img_label, path), ...]

    for stage in STAGES:
        model_path = BEST_MODELS / MODEL_NAMES[stage]
        if not model_path.exists():
            print(f"  Skip {stage}: {model_path.name} not found")
            all_metrics.append({"stage": stage, "mAP50": None, "mAP50-95": None, "precision": None, "recall": None, "F1": None})
            pred_paths[stage] = []
            continue

        model = YOLO(str(model_path))

        # Validate on test_set for metrics
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
                "stage": stage,
                "mAP50": float(map50) if map50 is not None else None,
                "mAP50-95": float(map95) if map95 is not None else None,
                "precision": p_val,
                "recall": r_val,
                "F1": f1,
            })
        except Exception as e:
            print(f"  Val error {stage}: {e}")
            all_metrics.append({"stage": stage, "mAP50": None, "mAP50-95": None, "precision": None, "recall": None, "F1": None})

        # Inference on 4 images
        stage_preds = []
        for (stem, img_path), lbl in zip(test_images, labels):
            pred = model.predict(img_path, save=False, verbose=False)[0]
            plot_img = pred.plot()
            if plot_img is not None:
                out_name = f"{stage}_{lbl}_{stem}.jpg"
                out_path = OUTPUT_DIR / "predictions" / out_name
                cv2.imwrite(str(out_path), cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
                stage_preds.append((lbl, f"predictions/{out_name}"))
        pred_paths[stage] = stage_preds

    # 3. Generate HTML
    html = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>BarkNet Stages – Combined Results</title>
  <style>
    body { font-family: system-ui, sans-serif; max-width: 1400px; margin: 2rem auto; padding: 0 1rem; }
    h1 { color: #333; }
    h2 { margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.5rem; }
    table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
    th, td { border: 1px solid #ddd; padding: 0.75rem; text-align: left; }
    th { background: #f5f5f5; }
    .pred-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1rem 0; }
    .pred-cell { border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }
    .pred-cell img { width: 100%; display: block; }
    .pred-cell p { margin: 0; padding: 0.5rem; font-size: 0.85rem; background: #f9f9f9; }
    .stage-section { margin-bottom: 2rem; }
  </style>
</head>
<body>
  <h1>BarkNet YOLOv8s-seg – All Stages Combined</h1>
  <p><strong>Training:</strong> Batch size = 4, Epochs = 80</p>
  <p><strong>Test set:</strong> 33 Picea, 31 Pinus (64 images total). Shown below: 2 Picea + 2 Pinus (fixed random selection).</p>

  <h2>1. Metrics Comparison</h2>
  <table>
    <tr>
      <th>Stage</th>
      <th>mAP50</th>
      <th>mAP50-95</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1</th>
    </tr>
"""

    for m in all_metrics:
        def fmt(v):
            return f"{v:.4f}" if v is not None and v == v else "N/A"
        html += f"""    <tr>
      <td><strong>{m['stage']}</strong></td>
      <td>{fmt(m.get('mAP50'))}</td>
      <td>{fmt(m.get('mAP50-95'))}</td>
      <td>{fmt(m.get('precision'))}</td>
      <td>{fmt(m.get('recall'))}</td>
      <td>{fmt(m.get('F1'))}</td>
    </tr>
"""

    html += """  </table>

  <h2>2. Predictions on Same 4 Test Images (per stage)</h2>
"""

    # Layout: table with 4 images as columns (Picea, Picea, Pinus, Pinus), stages as rows
    html += '  <table style="margin-top:1rem; table-layout:fixed;"><tr><th>Stage</th>'
    for lbl in labels:
        html += f'<th>{lbl}</th>'
    html += "</tr>\n"
    # Original images row
    html += '  <tr><td><strong>Original</strong></td>'
    for i, lbl in enumerate(labels):
        ext = Path(test_images[i][1]).suffix
        html += f'<td style="vertical-align:top;"><img src="predictions/original_{lbl}_{i}{ext}" alt="Original {lbl}" style="max-width:100%;"><br><small>Ground truth</small></td>'
    html += "</tr>\n"
    for stage in STAGES:
        preds = pred_paths.get(stage, [])
        html += f"  <tr><td><strong>{stage}</strong></td>"
        for lbl, rel_path in preds:
            html += f'<td style="vertical-align:top;"><img src="{rel_path}" alt="{stage} {lbl}" style="max-width:100%;"><br><small>{stage}</small></td>'
        for _ in range(4 - len(preds)):
            html += "<td>—</td>"
        html += "</tr>\n"
    html += "</table>\n"

    html += """
  <p style="margin-top: 2rem; color: #666; font-size: 0.9rem;">
    Generated by generate_stages_combined_report.py
  </p>
</body>
</html>
"""

    report_path = OUTPUT_DIR / "stages_combined_report.html"
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
