#!/usr/bin/env python3
"""
Unflatten FinnWoodlands Forest Images: split the flat FinnWoodlands_forest_images_all
back into subfolders by original source (using folder_mapping.csv).

Creates one subfolder per source (e.g. images_left, images_left 2, ...) and moves
each image into the folder it came from.

Usage:
  python scripts/unflatten_finnwoodlands_forest_images.py [--copy] [--flat-dir DIR]

  --copy     Copy instead of move (default: move).
  --flat-dir Directory containing flattened images and folder_mapping.csv
             (default: FinnWoodlands_forest_images_all)
"""

import argparse
import csv
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FLAT_DIR = PROJECT_ROOT / "FinnWoodlands_forest_images_all"
MAPPING_CSV = "folder_mapping.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Split FinnWoodlands_forest_images_all back into subfolders by source."
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    parser.add_argument(
        "--flat-dir",
        type=Path,
        default=DEFAULT_FLAT_DIR,
        help=f"Flat directory with images and {MAPPING_CSV}",
    )
    args = parser.parse_args()

    flat_dir = args.flat_dir if args.flat_dir.is_absolute() else PROJECT_ROOT / args.flat_dir
    mapping_path = flat_dir / MAPPING_CSV

    if not flat_dir.is_dir():
        print(f"Flat directory not found: {flat_dir}")
        return 1
    if not mapping_path.is_file():
        print(f"Mapping CSV not found: {mapping_path}")
        return 1

    # output_filename -> source_folder
    file_to_folder = {}
    with open(mapping_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_to_folder[row["output_filename"]] = row["source_folder"].strip()

    # Create subfolders and collect moves
    subdirs = {}
    for source_folder in set(file_to_folder.values()):
        subdir = flat_dir / source_folder
        subdir.mkdir(parents=True, exist_ok=True)
        subdirs[source_folder] = subdir

    mode = "Copy" if args.copy else "Move"
    print(f"Flat dir: {flat_dir}")
    print(f"Mode: {mode}")
    print(f"Subfolders: {list(subdirs.keys())}")
    print()

    counts = {name: 0 for name in subdirs}
    for filename, source_folder in file_to_folder.items():
        src = flat_dir / filename
        if not src.is_file():
            continue
        dst = subdirs[source_folder] / filename
        if args.copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))
        counts[source_folder] += 1

    for name in sorted(subdirs.keys()):
        print(f"  {name}: {counts[name]} images")
    print(f"\nTotal: {sum(counts.values())} images")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
