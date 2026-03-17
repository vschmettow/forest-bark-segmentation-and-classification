#!/usr/bin/env python3
"""
Evaluate both YOLOv8s-seg models (data0 and data_all) on their test sets
and create visualizations: metrics comparison, sample predictions.

Usage:
  python scripts/evaluate_and_visualize_models.py

Output:
  - evaluation_report/  (metrics, charts, sample predictions)
  - Opens evaluation_report/report.html in browser
"""

from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BEST_MODELS = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "results/evaluation_report"

# Model -> (weights path, dataset.yaml for test set)
MODELS = [
    ("YOLOv8s-seg (data0)", "yolov8s_seg_data0_best.pt", "BarkNetYOLO/data0/dataset.yaml"),
    ("YOLOv8s-seg (data_all)", "yolov8s_seg_dataall_best.pt", "BarkNetYOLO/data_all/dataset.yaml"),
]


def find_model_path(name: str) -> Path | None:
    """Find model file (try data_all and dataall variants)."""
    candidates = [
        BEST_MODELS / name,
        BEST_MODELS / name.replace("data_all", "dataall"),
        BEST_MODELS / name.replace("dataall", "data_all"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def main():
    from ultralytics import YOLO

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "predictions").mkdir(exist_ok=True)

    results = []
    all_metrics = []

    for label, model_name, data_yaml in MODELS:
        model_path = find_model_path(model_name)
        if not model_path:
            print(f"  Skip {label}: {model_name} not found")
            continue

        data_path = PROJECT_ROOT / data_yaml
        if not data_path.exists():
            print(f"  Skip {label}: dataset {data_path} not found")
            continue

        print(f"\nEvaluating {label}...")
        model = YOLO(str(model_path))

        # 1. Validation on test set
        val_results = model.val(data=str(data_path), split="test", verbose=True)
        # val_results is SegmentMetrics: has .box and .seg (not .mask)
        m = val_results

        # SegmentMetrics: seg.map50, seg.map for segmentation; box has mp, mr for precision/recall
        map50 = map5095 = None
        seg = getattr(m, "seg", None) or getattr(m, "mask", None)
        if seg is not None:
            map50 = getattr(seg, "map50", None)
            map5095 = getattr(seg, "map", None)
        if map50 is None and hasattr(m, "box"):
            map50 = getattr(m.box, "map50", None)
            map5095 = getattr(m.box, "map", None)
        map50 = float(map50) if map50 is not None else 0.0
        map5095 = float(map5095) if map5095 is not None else 0.0

        prec = getattr(m.box, "mp", None) if hasattr(m, "box") else None
        rec = getattr(m.box, "mr", None) if hasattr(m, "box") else None

        metrics = {
            "model": label,
            "mAP50": map50,
            "mAP50-95": map5095,
            "precision": float(prec) if prec is not None else None,
            "recall": float(rec) if rec is not None else None,
        }
        results.append(metrics)
        all_metrics.append(metrics)
        print(f"  mAP50: {map50:.4f}, mAP50-95: {map5095:.4f}")

        # 2. Predict on sample images and save
        test_images = PROJECT_ROOT / data_path.parent / "images" / "test"
        if test_images.exists():
            imgs = sorted(test_images.glob("*.jpg"))[:6]
            if not imgs:
                imgs = list(test_images.glob("*.png"))[:6]
            if not imgs:
                imgs = list(test_images.glob("*.jpeg"))[:6]

            import cv2
            safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
            for i, img_path in enumerate(imgs):
                pred = model.predict(str(img_path), save=False, verbose=False)[0]
                out_name = f"{safe_label}_sample{i}.jpg"
                out_path = OUTPUT_DIR / "predictions" / out_name
                plot_img = pred.plot()
                if plot_img is not None:
                    cv2.imwrite(str(out_path), cv2.cvtColor(plot_img, cv2.COLOR_RGB2BGR))
                    print(f"  Saved: {out_path.name}")

    if not results:
        print("No models evaluated. Check models/ and dataset paths.")
        return 1

    # 3. Generate HTML report
    report_path = OUTPUT_DIR / "report.html"
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>BarkNet Model Evaluation</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 0 1rem; }}
    h1 {{ color: #333; }}
    h2 {{ margin-top: 2rem; border-bottom: 1px solid #ddd; padding-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
    th, td {{ border: 1px solid #ddd; padding: 0.75rem; text-align: left; }}
    th {{ background: #f5f5f5; }}
    .metric {{ font-weight: 600; }}
    .bar-container {{ background: #eee; border-radius: 4px; height: 24px; margin: 4px 0; overflow: hidden; }}
    .bar {{ height: 100%; background: linear-gradient(90deg, #4CAF50, #8BC34A); border-radius: 4px; }}
    .pred-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 1rem; margin: 1rem 0; }}
    .pred-item {{ border: 1px solid #ddd; border-radius: 8px; overflow: hidden; }}
    .pred-item img {{ width: 100%; display: block; }}
    .pred-item p {{ margin: 0; padding: 0.5rem; font-size: 0.9rem; background: #f9f9f9; }}
  </style>
</head>
<body>
  <h1>BarkNet Model Evaluation Report</h1>
  <p>Evaluation of YOLOv8s-seg models on their respective test sets.</p>

  <h2>1. Metrics Comparison</h2>
  <table>
    <tr>
      <th>Model</th>
      <th>mAP50</th>
      <th>mAP50-95</th>
      <th>Precision</th>
      <th>Recall</th>
    </tr>
"""

    for r in results:
        prec = f"{r['precision']:.4f}" if r.get('precision') is not None else "N/A"
        rec = f"{r['recall']:.4f}" if r.get('recall') is not None else "N/A"
        html += f"""    <tr>
      <td class="metric">{r['model']}</td>
      <td>{r['mAP50']:.4f}</td>
      <td>{r['mAP50-95']:.4f}</td>
      <td>{prec}</td>
      <td>{rec}</td>
    </tr>
"""

    html += """  </table>

  <h2>2. mAP50 Comparison (bar chart)</h2>
  <div style="max-width: 500px;">
"""

    max_map = max(r["mAP50"] for r in results) or 1
    for r in results:
        pct = (r["mAP50"] / max_map * 100) if max_map > 0 else 0
        html += f"""    <p><strong>{r['model']}</strong></p>
    <div class="bar-container"><div class="bar" style="width: {pct}%"></div></div>
    <p style="margin-top: 0; font-size: 0.85rem;">{r['mAP50']:.4f}</p>
"""

    html += """  </div>

  <h2>3. Sample Predictions</h2>
  <p>Predictions on test images (masks overlaid).</p>
  <div class="pred-grid">
"""

    pred_dir = OUTPUT_DIR / "predictions"
    for f in sorted(pred_dir.glob("*.jpg")):
        rel = f.relative_to(OUTPUT_DIR)
        html += f"""    <div class="pred-item">
      <img src="{rel}" alt="{f.stem}" style="max-width: 100%;">
      <p>{f.stem}</p>
    </div>
"""

    html += """  </div>

  <p style="margin-top: 2rem; color: #666; font-size: 0.9rem;">
    Generated by evaluate_and_visualize_models.py
  </p>
</body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    print(f"\nReport saved: {report_path}")

    # 4. Save metrics JSON
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # 5. Open in browser
    try:
        import webbrowser
        webbrowser.open(f"file://{report_path.resolve()}")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    exit(main())
