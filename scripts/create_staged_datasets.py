#!/usr/bin/env python3
"""
Create stage1_80, stage2_160, stage3_280, stage4_363 from data_all.
- Remove data0, data1, data2, data3, data4
- Sample 363 images from data_all (stratified Picea/Pinus)
- Create cumulative stages: stage1(80) ⊂ stage2(160) ⊂ stage3(280) ⊂ stage4(363)
- Rebuild data_all with 363 images (same as stage4)
"""

import shutil
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
YOLO_OUT = PROJECT_ROOT / "BarkNetYOLO"
RANDOM_SEED = 42

STAGE_SIZES = [80, 160, 280, 363]
TARGET_TOTAL = 363


def get_species_from_label(lbl_path: Path) -> str:
    """Class 0=Picea, 1=Pinus. Return species based on first annotation."""
    try:
        line = lbl_path.read_text().strip().split("\n")[0]
        cls = line.split()[0]
        return "Picea" if cls == "0" else "Pinus"
    except Exception:
        return "Picea"  # default


def collect_data_all_items(yolo_all: Path) -> list[tuple[Path, Path, str]]:
    """Collect (img_path, lbl_path, stem) from data_all train+val."""
    items = []
    seen = set()
    for split in ["train", "val"]:
        img_dir = yolo_all / "images" / split
        lbl_dir = yolo_all / "labels" / split
        if not img_dir.exists():
            continue
        for img_path in img_dir.iterdir():
            if not img_path.is_file() or img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            stem = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            if not lbl_path.exists():
                continue
            if stem in seen:
                continue
            seen.add(stem)
            items.append((img_path, lbl_path, stem))
    return items


def main():
    random.seed(RANDOM_SEED)

    yolo_all = YOLO_OUT / "data_all"
    if not yolo_all.exists():
        print("Error: data_all not found")
        return 1

    # Collect all items with species
    all_items = collect_data_all_items(yolo_all)
    by_species = {"Picea": [], "Pinus": []}
    for img_path, lbl_path, stem in all_items:
        species = get_species_from_label(lbl_path)
        by_species[species].append((img_path, lbl_path, stem))

    n_picea, n_pinus = len(by_species["Picea"]), len(by_species["Pinus"])
    print(f"data_all: {len(all_items)} images (Picea: {n_picea}, Pinus: {n_pinus})")

    # Stratified sample of 363
    total = n_picea + n_pinus
    if total < TARGET_TOTAL:
        print(f"Warning: only {total} images, using all")
        pool = all_items
    else:
        ratio_picea = n_picea / total
        n_picea_sample = int(TARGET_TOTAL * ratio_picea)
        n_pinus_sample = TARGET_TOTAL - n_picea_sample
        random.shuffle(by_species["Picea"])
        random.shuffle(by_species["Pinus"])
        pool = []
        for img_path, lbl_path, stem in by_species["Picea"][:n_picea_sample]:
            pool.append((img_path, lbl_path, stem))
        for img_path, lbl_path, stem in by_species["Pinus"][:n_pinus_sample]:
            pool.append((img_path, lbl_path, stem))
        random.shuffle(pool)

    # Cumulative stages: first 80, first 160, first 280, all 363
    stages = {
        "stage1_80": pool[:80],
        "stage2_160": pool[:160],
        "stage3_280": pool[:280],
        "stage4_363": pool[:363],
    }

    # Remove data0, data1, data2, data3, data4
    for old in ["data0", "data1", "data2", "data3", "data4"]:
        old_path = YOLO_OUT / old
        if old_path.exists():
            shutil.rmtree(old_path)
            print(f"  Removed {old}")

    # Create each stage
    for stage_name, stage_items in stages.items():
        stage_dir = YOLO_OUT / stage_name
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        stage_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ["train", "val"]:
            (stage_dir / "images" / split_name).mkdir(parents=True, exist_ok=True)
            (stage_dir / "labels" / split_name).mkdir(parents=True, exist_ok=True)

        # 80/20 train/val split within stage
        random.shuffle(stage_items)
        n = len(stage_items)
        n_train = int(n * 0.80)
        train_items = stage_items[:n_train]
        val_items = stage_items[n_train:]

        for img_path, lbl_path, stem in train_items:
            shutil.copy2(img_path, stage_dir / "images" / "train" / img_path.name)
            shutil.copy2(lbl_path, stage_dir / "labels" / "train" / lbl_path.name)
        for img_path, lbl_path, stem in val_items:
            shutil.copy2(img_path, stage_dir / "images" / "val" / img_path.name)
            shutil.copy2(lbl_path, stage_dir / "labels" / "val" / lbl_path.name)

        # dataset.yaml
        abs_path = str(stage_dir.resolve()).replace("\\", "/")
        (stage_dir / "dataset.yaml").write_text(f"""# BarkNet {stage_name}
path: {abs_path}
train: images/train
val: images/val
nc: 2
names: ['Picea', 'Pinus']
""")
        print(f"  Created {stage_name}: {len(train_items)} train, {len(val_items)} val")

    # Rebuild data_all with 363 (copy from stage4_363 which we just created)
    stage4_dir = YOLO_OUT / "stage4_363"
    for split in ["train", "val"]:
        src_img = stage4_dir / "images" / split
        src_lbl = stage4_dir / "labels" / split
        dst_img = yolo_all / "images" / split
        dst_lbl = yolo_all / "labels" / split
        # Remove and recreate to ensure clean state
        if dst_img.exists():
            shutil.rmtree(dst_img)
        if dst_lbl.exists():
            shutil.rmtree(dst_lbl)
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        # Copy from stage4
        for f in src_img.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_img / f.name)
        for f in src_lbl.iterdir():
            if f.is_file():
                shutil.copy2(f, dst_lbl / f.name)
    n_train = sum(1 for f in (yolo_all / "images" / "train").iterdir() if f.is_file())
    n_val = sum(1 for f in (yolo_all / "images" / "val").iterdir() if f.is_file())
    print(f"  Rebuilt data_all: {n_train} train, {n_val} val ({n_train + n_val} total)")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    exit(main())
