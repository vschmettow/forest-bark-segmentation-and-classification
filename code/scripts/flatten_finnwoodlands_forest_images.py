#!/usr/bin/env python3
"""
Flatten FinnWoodlands Forest Images: move every image from all subfolders
into a single folder with unique IDs that encode the source folder.

ID format: folder{idx:02d}_{image_num:05d}.{ext}
  e.g. folder01_00001.jpg, folder01_00002.jpg, ..., folder16_01234.jpg

Usage:
  python scripts/flatten_finnwoodlands_forest_images.py [--copy] [--output DIR]

  --copy   Copy instead of move (default: move).
  --output Output directory (default: FinnWoodlands_forest_images_all)
"""

import argparse
import shutil
import csv
from pathlib import Path

# Default paths (Bark project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCE_ROOT = PROJECT_ROOT / "Archived_2" / "FinnWoodlands_forest_images"
DEFAULT_OUTPUT = PROJECT_ROOT / "Archived_2" / "FinnWoodlands_forest_images_all"
MAPPING_CSV = "folder_mapping.csv"  # name only; written inside output dir

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def get_subfolders(root: Path):
    """Return direct subdirs of root, sorted for stable ordering."""
    if not root.is_dir():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def collect_images_by_folder(source_root: Path):
    """Return list of (folder_path, list of image paths) for each subfolder."""
    subfolders = get_subfolders(source_root)
    result = []
    for folder in subfolders:
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(folder.glob(f"*{ext}"))
        images = sorted(images)
        if images:
            result.append((folder, images))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Flatten FinnWoodlands forest images into one folder with unique IDs."
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy files instead of moving them.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory (default: {DEFAULT_OUTPUT.name})",
    )
    args = parser.parse_args()

    source_root = SOURCE_ROOT
    if not source_root.exists():
        print(f"Source not found: {source_root}")
        return 1

    by_folder = collect_images_by_folder(source_root)
    if not by_folder:
        print("No subfolders with images found.")
        return 1

    out_dir = args.output
    if out_dir.is_relative_to(PROJECT_ROOT):
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mode = "Copy" if args.copy else "Move"
    print(f"Source: {source_root}")
    print(f"Output: {out_dir}")
    print(f"Mode: {mode}")
    print(f"Folders: {len(by_folder)}")
    total = sum(len(imgs) for _, imgs in by_folder)
    print(f"Total images: {total}")
    print()

    # Build mapping: new_filename -> (folder_name, original_path)
    mapping_rows = [["output_filename", "source_folder", "source_path"]]
    folder_id_width = 2
    image_num_width = 5

    for folder_idx, (folder_path, image_list) in enumerate(by_folder, start=1):
        folder_name = folder_path.name
        for image_idx, img_path in enumerate(image_list, start=1):
            ext = img_path.suffix
            new_name = f"folder{folder_idx:0{folder_id_width}d}_{image_idx:0{image_num_width}d}{ext}"
            dest = out_dir / new_name

            if args.copy:
                shutil.copy2(img_path, dest)
            else:
                shutil.move(str(img_path), str(dest))

            mapping_rows.append([new_name, folder_name, str(img_path)])

        print(f"  folder{folder_idx:02d} ({folder_name}): {len(image_list)} images")

    mapping_path = out_dir / MAPPING_CSV
    with open(mapping_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(mapping_rows)
    print(f"\nMapping saved to: {mapping_path}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
