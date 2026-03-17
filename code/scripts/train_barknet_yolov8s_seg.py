#!/usr/bin/env python3
"""
Train YOLOv8s-seg on BarkNetYOLO/data0 (smallest dataset).

Usage:
  From project root:
    python scripts/train_barknet_yolov8s_seg.py

  Or with custom paths:
    python scripts/train_barknet_yolov8s_seg.py --data BarkNetYOLO/data1/dataset.yaml
"""

from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA = PROJECT_ROOT / "BarkNetYOLO" / "data0" / "dataset.yaml"
DEFAULT_PROJECT = PROJECT_ROOT / "segmentation_results"


def main():
    from ultralytics import YOLO

    parser = argparse.ArgumentParser(description="Train YOLOv8s-seg on BarkNet")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="Path to dataset.yaml (default: BarkNetYOLO/data0/dataset.yaml)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=80,
        help="Number of epochs (default: 80)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(DEFAULT_PROJECT),
        help="Output project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="barknet_data0_yolov8s_seg",
        help="Experiment name",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset not found: {data_path}")
        print("Run: python scripts/convert_barknet_to_coco_and_yolo.py")
        return 1

    print("=" * 70)
    print("YOLOv8s-seg on BarkNetYOLO/data0")
    print("=" * 70)
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}, Batch: {args.batch}, Img size: {args.imgsz}")
    print("=" * 70)

    model = YOLO("yolov8s-seg.pt")
    model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        save=True,
        plots=True,
        verbose=True,
    )

    results_dir = Path(args.project) / args.name
    print("\nDone. Best model:", results_dir / "weights" / "best.pt")

    # Print metrics table
    csv_path = results_dir / "results.csv"
    if csv_path.exists():
        import pandas as pd
        df = pd.read_csv(csv_path)
        last = df.dropna(how="all").iloc[-1]
        cols = df.columns.tolist()
        map50 = next((c for c in cols if "mAP50" in c), None)
        map5095 = next((c for c in cols if "mAP50-95" in c), None)
        prec = next((c for c in cols if "precision" in c.lower()), None)
        rec = next((c for c in cols if "recall" in c.lower()), None)

        def fmt(x):
            v = last.get(x)
            if v is None or (isinstance(v, float) and (v != v)):
                return "N/A"
            return f"{float(v):.4f}"

        tbl = pd.DataFrame({
            "Metric": ["mAP50", "mAP50-95", "Precision", "Recall"],
            "Value": [fmt(map50), fmt(map5095), fmt(prec), fmt(rec)],
        })
        print("\n" + "=" * 50)
        print("YOLOv8s-seg (BarkNetYOLO data0) – Accuracy Results")
        print("=" * 50)
        print(tbl.to_string(index=False))

    return 0


if __name__ == "__main__":
    exit(main())
