#!/usr/bin/env python3
"""
Download YOLOv8-seg-n (nano) so it's available for training.
Ultralytics downloads the weight file on first use; this script triggers that
and optionally saves it to a known location.
"""
from pathlib import Path
import argparse

def download_yolov8_seg(model_size: str = "n"):
    """Load YOLOv8-seg model; Ultralytics downloads weights on first use."""
    from ultralytics import YOLO

    name = f"yolov8{model_size}-seg.pt"
    print(f"Loading {name} (will download from Ultralytics if not cached)...")
    model = YOLO(name)
    print(f"OK: {name} is ready.")
    return model

def main():
    parser = argparse.ArgumentParser(description="Download YOLOv8-seg weights (n/s/m/l/x).")
    parser.add_argument("--size", "-s", default="n", choices=["n", "s", "m", "l", "x"],
                        help="Model size: n=nano, s=small, m=medium, l=large, x=xlarge (default: n)")
    args = parser.parse_args()
    download_yolov8_seg(args.size)

if __name__ == "__main__":
    main()
