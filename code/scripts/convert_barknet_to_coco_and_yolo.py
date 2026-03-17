#!/usr/bin/env python3
"""
Convert rawBarkNet (LabelMe format) to BarkNetCOCO and BarkNetYOLO.

Creates:
- BarkNetCOCO/data0, data1, data2, data3, data_all (COCO format, train/val/test 70/15/15)
- BarkNetYOLO/data0, data1, data2, data3, data_all (YOLO format, train/val/test 70/15/15)

Classes: Picea=0, Pinus=1 (COCO uses 1-indexed: Picea=1, Pinus=2)
Duplicates (same filename across data folders): keep first occurrence only in data_all.
"""

import json
import shutil
import random
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_ROOT = PROJECT_ROOT / "rawBarkNet"
COCO_OUT = PROJECT_ROOT / "BarkNetCOCO"
YOLO_OUT = PROJECT_ROOT / "BarkNetYOLO"

DATA_FOLDERS = ["data0", "data1", "data2", "data3"]
SPECIES_FOLDERS = ["Picea abies L. H.Karst", "Pinus sylvestris L"]
LABEL_TO_CLASS = {"Picea": 0, "Pinus": 1}
CLASS_NAMES = ["Picea", "Pinus"]

# COCO uses 1-indexed category_id
COCO_CATEGORIES = [
    {"id": 1, "name": "Picea"},
    {"id": 2, "name": "Pinus"},
]

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


def load_labelme_annotations(json_path: Path) -> Optional[dict]:
    """Load LabelMe JSON and return parsed data."""
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  Warning: Could not load {json_path}: {e}")
        return None


def labelme_to_coco_annotation(shape: dict, image_id: int, ann_id: int, img_w: int, img_h: int) -> Optional[dict]:
    """Convert one LabelMe shape to COCO annotation."""
    label = shape.get("label")
    if label not in LABEL_TO_CLASS:
        return None
    points = shape.get("points", [])
    if len(points) < 3:
        return None

    # Flatten polygon: [[x,y],[x,y],...] -> [x,y,x,y,...]
    seg_flat = []
    for p in points:
        x, y = float(p[0]), float(p[1])
        x = max(0, min(img_w, x))
        y = max(0, min(img_h, y))
        seg_flat.extend([x, y])

    # COCO category_id is 1-indexed
    category_id = LABEL_TO_CLASS[label] + 1

    # Bbox: [x, y, width, height]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_min = max(0, min(img_w, x_min))
    y_min = max(0, min(img_h, y_min))
    x_max = max(0, min(img_w, x_max))
    y_max = max(0, min(img_h, y_max))
    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
    area = (x_max - x_min) * (y_max - y_min)

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": category_id,
        "segmentation": [seg_flat],
        "area": area,
        "bbox": bbox,
        "iscrowd": 0,
    }


def labelme_to_yolo_line(shape: dict, img_w: int, img_h: int) -> Optional[str]:
    """Convert one LabelMe shape to YOLO segmentation line."""
    label = shape.get("label")
    if label not in LABEL_TO_CLASS:
        return None
    points = shape.get("points", [])
    if len(points) < 3:
        return None

    class_id = LABEL_TO_CLASS[label]
    parts = [str(class_id)]
    for p in points:
        x = float(p[0]) / img_w
        y = float(p[1]) / img_h
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        parts.append(f"{x:.6f}")
        parts.append(f"{y:.6f}")
    return " ".join(parts)


def collect_images_from_data_folder(data_folder: Path) -> list[tuple[Path, Path, str]]:
    """
    Collect (image_path, json_path, stem) for all annotated images in a data folder.
    Returns list of (img_path, json_path, stem).
    """
    collected = []
    for species in SPECIES_FOLDERS:
        species_dir = data_folder / species
        if not species_dir.exists():
            continue
        for json_path in species_dir.glob("*.json"):
            stem = json_path.stem
            img_path = json_path.with_suffix(".jpg")
            if not img_path.exists():
                img_path = json_path.with_suffix(".jpeg")
            if not img_path.exists():
                img_path = json_path.with_suffix(".png")
            if img_path.exists():
                collected.append((img_path, json_path, stem))
    return collected


def process_split(
    items: list[tuple[Path, Path, str]],
    split_name: str,
    coco_base: Path,
    yolo_base: Path,
    prefix: str = "",
) -> None:
    """Process one split (train/val/test) for both COCO and YOLO."""
    coco_images_dir = coco_base / "images" / split_name
    coco_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_images_dir = yolo_base / "images" / split_name
    yolo_labels_dir = yolo_base / "labels" / split_name
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)

    coco_images = []
    coco_annotations = []
    image_id = 1
    ann_id = 1

    for img_path, json_path, stem in items:
        data = load_labelme_annotations(json_path)
        if data is None:
            continue
        img_w = data.get("imageWidth", 0)
        img_h = data.get("imageHeight", 0)
        if img_w <= 0 or img_h <= 0:
            continue

        # Unique filename (prefix avoids collisions when merging)
        out_stem = f"{prefix}{stem}" if prefix else stem
        out_img_name = f"{out_stem}{img_path.suffix}"

        # Copy image
        coco_dst = coco_images_dir / out_img_name
        yolo_dst_img = yolo_images_dir / out_img_name
        shutil.copy2(img_path, coco_dst)
        shutil.copy2(img_path, yolo_dst_img)

        coco_images.append({
            "id": image_id,
            "file_name": out_img_name,
            "width": img_w,
            "height": img_h,
        })

        # YOLO labels
        yolo_lines = []
        for shape in data.get("shapes", []):
            line = labelme_to_yolo_line(shape, img_w, img_h)
            if line:
                yolo_lines.append(line)

        yolo_label_path = yolo_labels_dir / f"{out_stem}.txt"
        with open(yolo_label_path, "w") as f:
            f.write("\n".join(yolo_lines))

        # COCO annotations
        for shape in data.get("shapes", []):
            ann = labelme_to_coco_annotation(shape, image_id, ann_id, img_w, img_h)
            if ann:
                coco_annotations.append(ann)
                ann_id += 1

        image_id += 1

    coco_data = {
        "info": {"description": "BarkNet converted from LabelMe"},
        "licenses": [],
        "images": coco_images,
        "categories": COCO_CATEGORIES,
        "annotations": coco_annotations,
    }
    coco_ann_path = coco_base / f"annotations_{split_name}.json"
    with open(coco_ann_path, "w") as f:
        json.dump(coco_data, f, indent=2)


def create_yolo_yaml(yolo_base: Path, dataset_name: str) -> None:
    """Create dataset.yaml for YOLO format."""
    # Use absolute path so it works regardless of cwd
    abs_path = str(yolo_base.resolve()).replace("\\", "/")
    yaml_content = f"""# BarkNet {dataset_name} - YOLO segmentation
# Classes: Picea=0, Pinus=1

path: {abs_path}
train: images/train
val: images/val
test: images/test

nc: 2
names: ['Picea', 'Pinus']
"""
    with open(yolo_base / "dataset.yaml", "w") as f:
        f.write(yaml_content)


def convert_data_folder(
    data_name: str,
    items: list[tuple[Path, Path, str]],
    coco_out: Path,
    yolo_out: Path,
) -> None:
    """Convert one data folder (data0, data1, etc.) to COCO and YOLO."""
    coco_base = coco_out / data_name
    yolo_base = yolo_out / data_name
    coco_base.mkdir(parents=True, exist_ok=True)
    yolo_base.mkdir(parents=True, exist_ok=True)

    random.shuffle(items)
    n = len(items)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val = int(n * SPLIT_RATIOS["val"])
    n_test = n - n_train - n_val

    train_items = items[:n_train]
    val_items = items[n_train : n_train + n_val]
    test_items = items[n_train + n_val :]

    print(f"  {data_name}: {len(train_items)} train, {len(val_items)} val, {len(test_items)} test")

    for split_name, split_items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        process_split(split_items, split_name, coco_base, yolo_base)

    create_yolo_yaml(yolo_base, data_name)


def main():
    random.seed(RANDOM_SEED)

    if not SOURCE_ROOT.exists():
        print(f"Error: Source not found: {SOURCE_ROOT}")
        return 1

    COCO_OUT.mkdir(parents=True, exist_ok=True)
    YOLO_OUT.mkdir(parents=True, exist_ok=True)

    # Per-data conversions (data0, data1, data2, data3)
    print("Converting data0, data1, data2, data3...")
    for data_name in DATA_FOLDERS:
        data_path = SOURCE_ROOT / data_name
        if not data_path.exists():
            print(f"  Skipping {data_name}: not found")
            continue
        items = collect_images_from_data_folder(data_path)
        if not items:
            print(f"  Skipping {data_name}: no images found")
            continue
        convert_data_folder(data_name, items, COCO_OUT, YOLO_OUT)

    # data_all: merge all, deduplicate by stem (keep first)
    print("\nConverting data_all (merged, duplicates removed)...")
    seen_stems = set()
    all_items = []
    for data_name in DATA_FOLDERS:
        data_path = SOURCE_ROOT / data_name
        if not data_path.exists():
            continue
        for img_path, json_path, stem in collect_images_from_data_folder(data_path):
            if stem not in seen_stems:
                seen_stems.add(stem)
                all_items.append((img_path, json_path, stem))

    if all_items:
        convert_data_folder("data_all", all_items, COCO_OUT, YOLO_OUT)
        print(f"  data_all: {len(all_items)} unique images (duplicates removed)")
    else:
        print("  data_all: no images found")

    print("\nDone.")
    print(f"  BarkNetCOCO: {COCO_OUT}")
    print(f"  BarkNetYOLO: {YOLO_OUT}")
    return 0


if __name__ == "__main__":
    exit(main())
