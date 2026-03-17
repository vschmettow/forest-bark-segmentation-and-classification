#!/usr/bin/env python3
"""
Export one FinnWoodlands example: image + mask overlay to show labeling format.
Output: Results/finnwoodlands_example/
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINN_MASKS = PROJECT_ROOT / "Archived_2" / "FinnWoodlands_masks"
OUTPUT_DIR = PROJECT_ROOT  / "results" / "finnwoodlands_example"

CAT_NAMES = {5: "Tree", 6: "Spruce", 7: "Birch", 8: "Pine"}


def main():
    import numpy as np
    from PIL import Image
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    train_json = FINN_MASKS / "train.json"
    rgb_dir = FINN_MASKS / "rgb" / "train"
    if not train_json.exists() or not rgb_dir.exists():
        print(f"FinnWoodlands not found at {FINN_MASKS}")
        return 1

    with open(train_json) as f:
        data = json.load(f)

    # Find image with Spruce or Pine
    img_id = None
    for ann in data["annotations"]:
        if ann["category_id"] in [6, 8]:
            img_id = ann["image_id"]
            break
    if img_id is None:
        print("No Spruce/Pine annotations found")
        return 1

    img_info = next(i for i in data["images"] if i["id"] == img_id)
    img_path = rgb_dir / img_info["file_name"]
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        return 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Copy original image
    import shutil
    shutil.copy2(img_path, OUTPUT_DIR / "example_image.jpg")
    print(f"Copied: {OUTPUT_DIR / 'example_image.jpg'}")

    # Load image and draw polygons
    img = np.array(Image.open(img_path).convert("RGB"))
    h, w = img.shape[:2]

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.imshow(img)

    colors = {"Spruce": "lime", "Pine": "orange", "Tree": "cyan", "Birch": "yellow"}
    anns = [a for a in data["annotations"] if a["image_id"] == img_id and a["category_id"] in [5, 6, 7, 8]]
    for ann in anns:
        seg = ann.get("segmentation")
        if not seg:
            continue
        cat_name = CAT_NAMES.get(ann["category_id"], "?")
        color = colors.get(cat_name, "white")
        for poly in seg if isinstance(seg[0], (list, tuple)) else [seg]:
            if len(poly) < 6:
                continue
            xs = [poly[i] for i in range(0, len(poly) - 1, 2)]
            ys = [poly[i + 1] for i in range(0, len(poly) - 1, 2)]
            patch = mpatches.Polygon(list(zip(xs, ys)), fill=False, edgecolor=color, linewidth=2, label=cat_name)
            ax.add_patch(patch)

    ax.set_title(f"FinnWoodlands example: {img_info['file_name']} (polygon masks)")
    ax.axis("off")
    # Dedupe legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    plt.tight_layout()
    out_path = OUTPUT_DIR / "example_with_masks.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    print(f"\nExample in: {OUTPUT_DIR}")
    return 0


if __name__ == "__main__":
    exit(main())
