#!/usr/bin/env python3
"""
Create archive structure: Archived_1 (pushable), Archived_2 (large files), labelling_tool.
Run from Bark project root.
"""
import os
import shutil
import subprocess
from pathlib import Path

BARK = Path(__file__).resolve().parent
SIZE_LIMIT = 50 * 1024 * 1024  # 50MB

EXCLUDE_DIRS = {
    ".venv", ".git", "__pycache__", "Archived_1", "Archived_2", "labelling_tool",
    # Large dirs from .gitignore (not pushable)
    "images", "zz.Archive", "zz.archive", "FinnWoodlands", "FinnWoodlands_forest_images_all",
    "FinnWoodlands_masks", "plantnet_bark_images",
}
EXCLUDE_FILES = {".DS_Store", "create_archive.py"}


def should_skip_file(rel_path: Path, filename: str) -> bool:
    """Skip files that are in .gitignore (large model weights, etc.)."""
    # data/models/*/bark_classifier/weights/epoch*.pt
    if filename.startswith("epoch") and filename.endswith(".pt"):
        if "data" in rel_path.parts and "models" in rel_path.parts and "weights" in rel_path.parts:
            return True
    return False


def get_large_files():
    """Find all files > 50MB, excluding Labeling Tool (goes to labelling_tool)."""
    large = []
    for root, dirs, files in os.walk(BARK):
        # Don't descend into excluded dirs or Labeling Tool
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and d != "Labeling Tool"]
        rel = Path(root).relative_to(BARK)
        if "Labeling Tool" in rel.parts:
            continue
        for f in files:
            if f in EXCLUDE_FILES:
                continue
            path = Path(root) / f
            try:
                if path.stat().st_size > SIZE_LIMIT:
                    large.append(path)
            except OSError:
                pass
    return large


def main():
    archived_1 = BARK / "Archived_1"
    archived_2 = BARK / "Archived_2"
    labelling_tool = BARK / "labelling_tool"

    for d in (archived_1, archived_2, labelling_tool):
        d.mkdir(exist_ok=True)

    print("Finding large files (>50MB)...")
    large_files = get_large_files()
    print(f"Found {len(large_files)} files > 50MB")

    # 1. Archived_1: full project structure, small files only, exclude Labeling Tool
    print("\nCreating Archived_1 (small files, pushable)...")
    for root, dirs, files in os.walk(BARK):
        rel = Path(root).relative_to(BARK)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS and d != "Labeling Tool"]
        # Don't descend into data/datasets subdirs (gitignore: data/datasets/*/)
        if rel.parts == ("data", "datasets"):
            dirs[:] = []
        if any(p in rel.parts for p in EXCLUDE_DIRS):
            continue
        if "Labeling Tool" in rel.parts:
            continue
        if rel.parts and rel.parts[0] in ("Archived_1", "Archived_2", "labelling_tool"):
            continue

        dest_dir = archived_1 / rel
        dest_dir.mkdir(parents=True, exist_ok=True)

        for f in files:
            if f in EXCLUDE_FILES:
                continue
            if should_skip_file(rel, f):
                continue
            src = Path(root) / f
            try:
                size = src.stat().st_size
            except OSError:
                continue
            if size > SIZE_LIMIT:
                continue
            dest = dest_dir / f
            if not dest.exists() or dest.stat().st_size != size:
                shutil.copy2(src, dest)

    # 2. Archived_2: large files only
    print("Creating Archived_2 (large files)...")
    for src in large_files:
        rel = src.relative_to(BARK)
        dest = archived_2 / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)

    # 3. labelling_tool: open_labeling_tool.py + Labeling Tool
    print("Creating labelling_tool...")
    shutil.copy2(BARK / "open_labeling_tool.py", labelling_tool / "open_labeling_tool.py")
    lt_src = BARK / "Labeling Tool"
    lt_dest = labelling_tool / "Labeling Tool"
    if lt_src.exists() and not lt_dest.exists():
        print("  Copying Labeling Tool (this may take a while)...")
        shutil.copytree(lt_src, lt_dest, symlinks=False, ignore=shutil.ignore_patterns(".DS_Store"))

    # README for Archived_1
    readme_1 = archived_1 / "ARCHIVE_README.md"
    readme_1.write_text("""# Archived_1 – Pushable to GitHub

This folder contains the Bark project with all files ≤50MB.
Large files are stored in `../Archived_2/`.

## Restoring large files

Copy contents from `Archived_2` into the corresponding paths in this folder to restore the full project.
""", encoding="utf-8")

    # README for Archived_2
    readme_2 = archived_2 / "ARCHIVE_README.md"
    readme_2.write_text("""# Archived_2 – Large Files (not pushed to GitHub)

Files >50MB from the Bark project.
Place these in the corresponding paths under `Archived_1/` to restore the full project.
""", encoding="utf-8")

    # README for labelling_tool
    readme_lt = labelling_tool / "README.md"
    readme_lt.write_text("""# Labelling Tool

Contains the labeling tool launcher and AutoLabeling-GPU (Windows).

## Usage

- **macOS/Linux**: `pip3 install anylabeling PyQt5` then `python3 open_labeling_tool.py`
- **Windows**: Run `Labeling Tool/AutoLabeling-GPU/AutoLabeling-GPU.exe` or use the launcher
""", encoding="utf-8")

    print("\nDone. Archive structure created.")
    print(f"  Archived_1: {archived_1}")
    print(f"  Archived_2: {archived_2}")
    print(f"  labelling_tool: {labelling_tool}")


if __name__ == "__main__":
    main()
