#!/usr/bin/env python3
"""
1. Convert data4 from rawBarkNet to YOLO format (BarkNetYOLO/data4)
2. Add data4 to data_all in BarkNetYOLO
3. Create test_set (64 images: 33 Picea, 31 Pinus) - randomly selected
4. Merge test folders from data0,1,2,3 into train (single test_set from now on)
"""

import json
import shutil
import random
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_ROOT = PROJECT_ROOT / "rawBarkNet"
YOLO_OUT = PROJECT_ROOT / "BarkNetYOLO"

SPECIES_FOLDERS = ["Picea abies L. H.Karst", "Pinus sylvestris L"]
LABEL_TO_CLASS = {"Picea": 0, "Pinus": 1}
SPLIT_RATIOS = {"train": 0.70, "val": 0.30}  # No test - we have test_set
TEST_SET_COUNTS = {"Picea": 33, "Pinus": 31}
RANDOM_SEED = 42


def load_labelme_annotations(json_path: Path) -> Optional[dict]:
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def labelme_to_yolo_line(shape: dict, img_w: int, img_h: int) -> Optional[str]:
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


def collect_images_from_data_folder(data_folder: Path) -> list[tuple[Path, Path, str, str]]:
    """Returns (img_path, json_path, stem, species_label)."""
    collected = []
    for species_dir in SPECIES_FOLDERS:
        label = "Picea" if "Picea" in species_dir else "Pinus"
        full_path = data_folder / species_dir
        if not full_path.exists():
            continue
        for json_path in full_path.glob("*.json"):
            stem = json_path.stem
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = json_path.with_suffix(ext)
                if img_path.exists():
                    collected.append((img_path, json_path, stem, label))
                    break
    return collected


def process_yolo_split(
    items: list[tuple[Path, Path, str, str]],
    split_name: str,
    yolo_base: Path,
    prefix: str = "",
) -> None:
    """Write YOLO images and labels for one split."""
    yolo_images_dir = yolo_base / "images" / split_name
    yolo_labels_dir = yolo_base / "labels" / split_name
    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)

    for img_path, json_path, stem, _ in items:
        data = load_labelme_annotations(json_path)
        if data is None:
            continue
        img_w = data.get("imageWidth", 0)
        img_h = data.get("imageHeight", 0)
        if img_w <= 0 or img_h <= 0:
            continue

        out_stem = f"{prefix}{stem}" if prefix else stem
        out_img_name = f"{out_stem}{img_path.suffix}"

        shutil.copy2(img_path, yolo_images_dir / out_img_name)

        yolo_lines = []
        for shape in data.get("shapes", []):
            line = labelme_to_yolo_line(shape, img_w, img_h)
            if line:
                yolo_lines.append(line)
        (yolo_labels_dir / f"{out_stem}.txt").write_text("\n".join(yolo_lines))


def create_yolo_yaml(yolo_base: Path, dataset_name: str, has_test: bool = True) -> None:
    abs_path = str(yolo_base.resolve()).replace("\\", "/")
    test_line = "test: images/test\n" if has_test else ""
    yaml_content = f"""# BarkNet {dataset_name} - YOLO segmentation
# Classes: Picea=0, Pinus=1

path: {abs_path}
train: images/train
val: images/val
{test_line}
nc: 2
names: ['Picea', 'Pinus']
"""
    (yolo_base / "dataset.yaml").write_text(yaml_content)


def main():
    random.seed(RANDOM_SEED)

    if not SOURCE_ROOT.exists():
        print(f"Error: Source not found: {SOURCE_ROOT}")
        return 1

    # Collect all images with species, dedupe by stem (first occurrence)
    all_by_species = {"Picea": [], "Pinus": []}
    seen_stems = set()

    for data_name in ["data0", "data1", "data2", "data3", "data4"]:
        data_path = SOURCE_ROOT / data_name
        if not data_path.exists():
            continue
        for img_path, json_path, stem, label in collect_images_from_data_folder(data_path):
            if stem not in seen_stems:
                seen_stems.add(stem)
                all_by_species[label].append((img_path, json_path, stem, label))

    print(f"Total unique: {len(seen_stems)} (Picea: {len(all_by_species['Picea'])}, Pinus: {len(all_by_species['Pinus'])})")

    # 1. Select 64 images for test_set (33 Picea, 31 Pinus)
    test_set_stems = set()
    picea_pool = list(all_by_species["Picea"])
    pinus_pool = list(all_by_species["Pinus"])
    random.shuffle(picea_pool)
    random.shuffle(pinus_pool)

    test_items = []
    test_items.extend(picea_pool[: TEST_SET_COUNTS["Picea"]])
    test_items.extend(pinus_pool[: TEST_SET_COUNTS["Pinus"]])
    test_set_stems = {item[2] for item in test_items}

    print(f"Test set: {len(test_items)} images (33 Picea, 31 Pinus)")

    # Create test_set folder
    test_set_dir = YOLO_OUT / "test_set"
    test_set_dir.mkdir(parents=True, exist_ok=True)
    img_dir = test_set_dir / "images"
    lbl_dir = test_set_dir / "labels"
    img_dir.mkdir(exist_ok=True)
    lbl_dir.mkdir(exist_ok=True)
    for img_path, json_path, stem, _ in test_items:
        data = load_labelme_annotations(json_path)
        if data is None:
            continue
        img_w, img_h = data.get("imageWidth", 0), data.get("imageHeight", 0)
        if img_w <= 0 or img_h <= 0:
            continue
        out_name = f"{stem}{img_path.suffix}"
        shutil.copy2(img_path, img_dir / out_name)
        yolo_lines = []
        for shape in data.get("shapes", []):
            line = labelme_to_yolo_line(shape, img_w, img_h)
            if line:
                yolo_lines.append(line)
        (lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))

    abs_path = str(test_set_dir.resolve()).replace("\\", "/")
    (test_set_dir / "dataset.yaml").write_text(f"""# BarkNet test_set - 64 images (33 Picea, 31 Pinus)
path: {abs_path}
train: images
val: images
nc: 2
names: ['Picea', 'Pinus']
""")
    print(f"  Created {test_set_dir}")

    # 2. Convert data4 to YOLO (exclude test_set)
    data4_items = [
        (img_path, json_path, stem, label)
        for img_path, json_path, stem, label in collect_images_from_data_folder(SOURCE_ROOT / "data4")
        if stem not in test_set_stems
    ]
    if data4_items:
        random.shuffle(data4_items)
        n = len(data4_items)
        n_train = int(n * SPLIT_RATIOS["train"])
        train_4, val_4 = data4_items[:n_train], data4_items[n_train:]
        yolo_data4 = YOLO_OUT / "data4"
        yolo_data4.mkdir(parents=True, exist_ok=True)
        process_yolo_split(train_4, "train", yolo_data4)
        process_yolo_split(val_4, "val", yolo_data4)
        (yolo_data4 / "images" / "test").mkdir(exist_ok=True)
        (yolo_data4 / "labels" / "test").mkdir(exist_ok=True)
        create_yolo_yaml(yolo_data4, "data4", has_test=True)
        print(f"  Created BarkNetYOLO/data4: {len(train_4)} train, {len(val_4)} val (test_set excluded)")

    # 3. Merge test from data0,1,2,3 into train
    for data_name in ["data0", "data1", "data2", "data3"]:
        yolo_base = YOLO_OUT / data_name
        test_img = yolo_base / "images" / "test"
        test_lbl = yolo_base / "labels" / "test"
        train_img = yolo_base / "images" / "train"
        train_lbl = yolo_base / "labels" / "train"
        if test_img.exists():
            for f in test_img.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(train_img / f.name))
            for f in test_lbl.iterdir():
                if f.is_file():
                    shutil.move(str(f), str(train_lbl / f.name))
            test_img.rmdir()
            test_lbl.rmdir()
            print(f"  Merged {data_name}/test into train")

    # 4. Rebuild data_all: data0,1,2,3,4 excluding test_set
    seen_all = set()
    all_items = []
    for data_name in ["data0", "data1", "data2", "data3", "data4"]:
        data_path = SOURCE_ROOT / data_name
        if not data_path.exists():
            continue
        for item in collect_images_from_data_folder(data_path):
            stem = item[2]
            if stem not in seen_all and stem not in test_set_stems:
                seen_all.add(stem)
                all_items.append(item)

    random.shuffle(all_items)
    n = len(all_items)
    n_train = int(n * SPLIT_RATIOS["train"])
    train_items, val_items = all_items[:n_train], all_items[n_train:]

    yolo_all = YOLO_OUT / "data_all"
    yolo_all.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            d = yolo_all / sub / split
            if d.exists():
                for f in d.iterdir():
                    if f.is_file():
                        f.unlink()
            d.mkdir(parents=True, exist_ok=True)
    process_yolo_split(train_items, "train", yolo_all)
    process_yolo_split(val_items, "val", yolo_all)
    (yolo_all / "images" / "test").mkdir(exist_ok=True)
    (yolo_all / "labels" / "test").mkdir(exist_ok=True)
    create_yolo_yaml(yolo_all, "data_all", has_test=True)
    print(f"  Rebuilt data_all: {len(train_items)} train, {len(val_items)} val (test_set excluded)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    exit(main())
