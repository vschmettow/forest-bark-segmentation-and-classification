#!/usr/bin/env python3
"""
Run the trained YOLOv8-seg model (FinnWoodlands Spruce + Pine) on a forest image.
Output: instance masks + class (Spruce/Pine) per detection.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def predict(image_path: str, model_path: str, conf_threshold: float = 0.25, save_overlay: str | None = None):
    """
    Run segmentation on a forest image. Returns results and optional overlay image.
    """
    model = YOLO(model_path)
    results = model(image_path, conf=conf_threshold, verbose=False)

    detections = []  # list of (class_name, confidence, mask)
    for result in results:
        if result.masks is None:
            continue
        names = result.names or {}
        for i, mask in enumerate(result.masks.data):
            cls_id = int(result.boxes.cls[i]) if result.boxes is not None else 0
            conf = float(result.boxes.conf[i]) if result.boxes is not None else 1.0
            class_name = names.get(cls_id, f"class_{cls_id}")
            detections.append((class_name, conf, mask.cpu().numpy()))

    if save_overlay and detections:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w = img.shape[:2]
        overlay = img_rgb.copy()
        # Bright colors (RGB 0-255) so overlay is clearly visible
        colors_rgb = {
            "Spruce": (50, 205, 80),   # green
            "Pine": (30, 144, 255),   # blue
        }
        for class_name, conf, mask in detections:
            mask_np = mask if isinstance(mask, np.ndarray) else mask.cpu().numpy()
            mask_resized = cv2.resize(
                mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
            mask_bool = mask_resized > 0.5
            color = np.array(colors_rgb.get(class_name, (200, 200, 200)), dtype=np.float32)
            # Blend: where mask, use 50% image + 50% color so overlay is obvious
            for c in range(3):
                overlay[:, :, c] = np.where(
                    mask_bool,
                    0.5 * overlay[:, :, c] + 0.5 * color[c],
                    overlay[:, :, c],
                )
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        Path(save_overlay).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(save_overlay, overlay_bgr)
        print(f"Saved overlay to {save_overlay}")

    return detections


def main():
    parser = argparse.ArgumentParser(
        description="Run FinnWoodlands YOLOv8-seg (Spruce + Pine) on a forest image"
    )
    parser.add_argument("--image", "-i", required=True, help="Path to forest image")
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Path to best.pt (default: ../models/finnwoodlands_seg_best.pt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--save",
        "-o",
        type=str,
        default=None,
        help="Save overlay image to this path",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    model_path = args.model or (script_dir.parent / "models" / "finnwoodlands_seg_best.pt")
    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        print("Put your downloaded best.pt there, or pass --model /path/to/best.pt")
        return 1

    detections = predict(args.image, str(model_path), conf_threshold=args.conf, save_overlay=args.save)

    # Summary
    from collections import Counter
    counts = Counter(d[0] for d in detections)
    print(f"Image: {args.image}")
    print(f"Detections: {len(detections)}")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
