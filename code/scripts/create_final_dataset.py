#!/usr/bin/env python3
"""
Create BarkNetYOLO/final_dataset from full rawBarkNet (data0-5 + root species folders).

- Same split as other folders: 70% train, 15% val (15% test excluded)
- Excludes test_set images (64 images) from final_dataset
- YOLO format: images/train, images/val, labels/train, labels/val
- No test folder in final_dataset (test images stay in test_set)
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

# Same as original convert: 70 train, 15 val, 15 test (test excluded from final_dataset)
# Of non-test data: train = 70/85, val = 15/85
SPLIT_RATIOS = {"train": 70 / 85, "val": 15 / 85}
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


def get_test_set_stems() -> set[str]:
    """Read stems from existing test_set."""
    test_img_dir = YOLO_OUT / "test_set" / "images"
    stems = set()
    if test_img_dir.exists():
        for f in test_img_dir.iterdir():
            if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                stems.add(f.stem)
    return stems


def process_yolo_split(
    items: list[tuple[Path, Path, str, str]],
    split_name: str,
    yolo_base: Path,
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

        out_img_name = f"{stem}{img_path.suffix}"
        shutil.copy2(img_path, yolo_images_dir / out_img_name)

        yolo_lines = []
        for shape in data.get("shapes", []):
            line = labelme_to_yolo_line(shape, img_w, img_h)
            if line:
                yolo_lines.append(line)
        (yolo_labels_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))


def main():
    random.seed(RANDOM_SEED)

    if not SOURCE_ROOT.exists():
        print(f"Error: Source not found: {SOURCE_ROOT}")
        return 1

    test_set_stems = get_test_set_stems()
    print(f"Excluding {len(test_set_stems)} test_set images from final_dataset")

    # Collect all images from data0-5 and root species folders; dedupe by stem (first occurrence)
    all_by_species: dict[str, list] = {"Picea": [], "Pinus": []}
    seen_stems = set()

    # Data folders: data0, data1, data2, data3, data4, data5
    data_folders = ["data0", "data1", "data2", "data3", "data4", "data5"]
    # Root-level species folders (rawBarkNet/Picea..., rawBarkNet/Pinus...)
    for data_name in data_folders + [""]:
        data_path = SOURCE_ROOT / data_name if data_name else SOURCE_ROOT
        if not data_path.exists():
            continue
        for img_path, json_path, stem, label in collect_images_from_data_folder(data_path):
            if stem not in seen_stems:
                seen_stems.add(stem)
                all_by_species[label].append((img_path, json_path, stem, label))

    # Exclude test_set
    trainable = []
    for label in ["Picea", "Pinus"]:
        for item in all_by_species[label]:
            if item[2] not in test_set_stems:
                trainable.append(item)

    n_picea = sum(1 for _, _, _, l in trainable if l == "Picea")
    n_pinus = sum(1 for _, _, _, l in trainable if l == "Pinus")
    print(f"Trainable (excluding test_set): {len(trainable)} total (Picea: {n_picea}, Pinus: {n_pinus})")

    if not trainable:
        print("Error: No trainable images after excluding test_set")
        return 1

    # Stratified split: 70/15 ratio (train/val)
    by_species = {"Picea": [x for x in trainable if x[3] == "Picea"], "Pinus": [x for x in trainable if x[3] == "Pinus"]}
    random.shuffle(by_species["Picea"])
    random.shuffle(by_species["Pinus"])

    n_train_picea = int(len(by_species["Picea"]) * SPLIT_RATIOS["train"])
    n_train_pinus = int(len(by_species["Pinus"]) * SPLIT_RATIOS["train"])
    train_items = (
        by_species["Picea"][:n_train_picea]
        + by_species["Pinus"][:n_train_pinus]
    )
    val_items = (
        by_species["Picea"][n_train_picea:]
        + by_species["Pinus"][n_train_pinus:]
    )
    random.shuffle(train_items)
    random.shuffle(val_items)

    # Create final_dataset
    final_dir = YOLO_OUT / "final_dataset"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    final_dir.mkdir(parents=True, exist_ok=True)

    process_yolo_split(train_items, "train", final_dir)
    process_yolo_split(val_items, "val", final_dir)

    # dataset.yaml (no test - only train/val)
    abs_path = str(final_dir.resolve()).replace("\\", "/")
    (final_dir / "dataset.yaml").write_text(f"""# BarkNet final_dataset - full dataset (test_set excluded)
# Classes: Picea=0, Pinus=1
# Split: 70% train, 15% val (15% test in test_set)

path: {abs_path}
train: images/train
val: images/val
nc: 2
names: ['Picea', 'Pinus']
""")

    print(f"\nCreated BarkNetYOLO/final_dataset:")
    print(f"  Train: {len(train_items)} images")
    print(f"  Val:   {len(val_items)} images")
    print(f"  Picea total (train+val): {n_picea}")
    print(f"  Pinus total (train+val): {n_pinus}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    exit(main())
