#!/usr/bin/env python3
"""Remove all images from 'right' folders from FinnWoodlands_forest_images_all.
Keep only images from folders whose name contains 'left'. Update mapping CSV."""
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FLAT_DIR = PROJECT_ROOT / "Archived_2" / "FinnWoodlands_forest_images_all"
MAPPING_CSV = FLAT_DIR / "folder_mapping.csv"


def main():
    if not MAPPING_CSV.exists():
        print("Mapping CSV not found:", MAPPING_CSV)
        return 1

    rows = []
    with open(MAPPING_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            rows.append(row)

    to_remove = [r for r in rows if "right" in r["source_folder"].lower()]
    to_keep = [r for r in rows if "right" not in r["source_folder"].lower()]

    print(f"Total rows: {len(rows)}")
    print(f"Removing (right): {len(to_remove)} images")
    print(f"Keeping (left):   {len(to_keep)} images")

    removed = 0
    for row in to_remove:
        p = FLAT_DIR / row["output_filename"]
        if p.exists():
            p.unlink()
            removed += 1
    print(f"Deleted {removed} files.")

    with open(MAPPING_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(to_keep)
    print("Updated folder_mapping.csv to contain only left-folder images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
