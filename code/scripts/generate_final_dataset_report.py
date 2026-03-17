#!/usr/bin/env python3
"""
Generate HTML report for the final_dataset YOLOv8s-seg model.
- Runs val() on test_set for metrics (mAP50, mAP50-95, Precision, Recall, F1)
- Runs inference on 4 fixed test images (2 Picea + 2 Pinus)
- Outputs Results/final_dataset_report.html
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BEST_MODELS = PROJECT_ROOT / "models"
TEST_SET = PROJECT_ROOT / "BarkNetYOLO" / "test_set"
OUTPUT_DIR = PROJECT_ROOT  / "results"

MODEL_NAME = "yolov8s_seg_final_dataset_best.pt"

FIXED_STEMS = [
    "4a085a9c75f0e68b994a2371475a1a7dc29bfff2",  # Picea
    "7d43abfaff3b2d12c67567cdeedc28e0d43e9f1b",  # Picea
    "02ba18373a3dbbec4dd430bae821f9e39b89d9f0",  # Pinus
    "6f7642441ad37b1be215d81bb8b1499807acbcb7",  # Pinus
]
LABELS = ["Picea", "Picea", "Pinus", "Pinus"]


def main():
    import shutil
    import cv2
    from ultralytics import YOLO

    model_path = BEST_MODELS / MODEL_NAME
    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        print("  Place yolov8s_seg_final_dataset_best.pt in models/")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(exist_ok=True)

    # 1. Get test images
    test_images = []
    for stem in FIXED_STEMS:
        for ext in [".jpg", ".jpeg", ".png"]:
            p = TEST_SET / "images" / f"{stem}{ext}"
            if p.exists():
                test_images.append((stem, str(p)))
                break

    if len(test_images) != 4:
        print(f"Warning: Expected 4 test images, found {len(test_images)}")

    # Copy originals
    for i, ((stem, img_path), lbl) in enumerate(zip(test_images, LABELS)):
        shutil.copy2(img_path, OUTPUT_DIR / "predictions" / f"original_{lbl}_{i}{Path(img_path).suffix}")

    # 2. Val + inference
    model = YOLO(str(model_path))
    test_yaml = TEST_SET / "dataset.yaml"
    if not test_yaml.exists():
        test_yaml = TEST_SET / "dataset_colab.yaml"
        test_yaml.write_text(f"""path: {TEST_SET}
train: images
val: images
nc: 2
names: ['Picea', 'Pinus']
""")

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
    for (stem, img_path), lbl in zip(test_images, LABELS):
        pred = model.predict(img_path, save=False, verbose=False)[0]
        out_path = OUTPUT_DIR / "predictions" / f"final_dataset_{lbl}_{stem}.jpg"
        cv2.imwrite(str(out_path), cv2.cvtColor(pred.plot(), cv2.COLOR_RGB2BGR))
        pred_paths.append((lbl, f"predictions/final_dataset_{lbl}_{stem}.jpg"))

    # 3. Build HTML (relative image paths for local viewing)
    def fmt(v):
        return f"{v:.4f}" if v is not None and v == v else "N/A"

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>BarkNet final_dataset – Results</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1400px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ color: #333; }}
    h2 {{ margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 0.75rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    img {{ max-width: 100%; }}
  </style>
</head>
<body>
  <h1>BarkNet YOLOv8s-seg – final_dataset</h1>
  <p><strong>Training:</strong> Batch size = 32, Epochs = 100</p>
  <p><strong>Test set:</strong> 33 Picea, 31 Pinus (64 images total). Shown: 2 Picea + 2 Pinus.</p>

  <h2>1. Metrics (validation on test_set)</h2>
  <table>
    <tr><th>Model</th><th>mAP50</th><th>mAP50-95</th><th>Precision</th><th>Recall</th><th>F1</th></tr>
    <tr><td><strong>final_dataset</strong></td><td>{fmt(map50)}</td><td>{fmt(map95)}</td><td>{fmt(p_val)}</td><td>{fmt(r_val)}</td><td>{fmt(f1)}</td></tr>
  </table>

  <h2>2. Predictions on 4 Test Images</h2>
  <table style="table-layout: fixed;">
    <tr><th></th><th>Picea</th><th>Picea</th><th>Pinus</th><th>Pinus</th></tr>
    <tr><td><strong>Original</strong></td>"""
    for i, lbl in enumerate(LABELS):
        ext = Path(test_images[i][1]).suffix
        html += f'<td><img src="predictions/original_{lbl}_{i}{ext}" alt="Original"><br><small>Ground truth</small></td>'
    html += "</tr>\n    <tr><td><strong>final_dataset</strong></td>"
    for lbl, rel in pred_paths:
        html += f'<td><img src="{rel}" alt="Prediction"><br><small>Prediction</small></td>'
    html += """</tr>
  </table>

  <p style="margin-top: 2rem; color: #666; font-size: 0.9rem;">
    Generated by generate_final_dataset_report.py
  </p>
</body>
</html>
"""

    report_path = OUTPUT_DIR / "final_dataset_report.html"
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
