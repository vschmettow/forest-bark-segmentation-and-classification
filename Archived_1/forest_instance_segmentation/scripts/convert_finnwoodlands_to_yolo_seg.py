#!/usr/bin/env python3
"""
Convert FinnWoodlands COCO JSON + RGB images to YOLOv8-seg format.
Uses train.json / val.json and rgb/train, rgb/val.
Produces: images (symlinks or copies), labels/*.txt (class_id + normalized polygon),
and dataset.yaml for Spruce + Pine instance segmentation.
"""
from pathlib import Path
import json
import argparse
import shutil

# FinnWoodlands COCO category_id -> YOLO class index (Spruce and Pine only)
CAT_ID_TO_YOLO = {
    6: 0,   # Spruce -> 0
    8: 1,   # Pine  -> 1
}
# Optional: include Tree(5) or Birch(7) as extra classes; here we only use Spruce and Pine.

def _poly_to_xy_pairs(poly):
    """Flatten [x,y,x,y,...] into list of (x,y) pairs."""
    return [(poly[i], poly[i + 1]) for i in range(0, len(poly) - 1, 2)]


def _merge_multipolygon_shapely(segmentation, width: int, height: int):
    """
    Merge COCO multipolygon (list of polygons) into a single polygon via shapely union.
    Returns flat list [x,y,x,y,...] for the exterior, or None if merge fails.
    """
    try:
        from shapely.geometry import Polygon
        from shapely.ops import unary_union
    except ImportError:
        return None
    polys = []
    for p in segmentation:
        if isinstance(p, (list, tuple)):
            if len(p) < 6:
                continue
            try:
                poly = Polygon(_poly_to_xy_pairs(p))
                if not poly.is_empty and not poly.is_valid:
                    poly = poly.buffer(0)
                if not poly.is_empty:
                    polys.append(poly)
            except Exception:
                continue
    if not polys:
        return None
    merged = unary_union(polys)
    if merged.is_empty:
        return None
    if merged.geom_type == "Polygon":
        coords = list(merged.exterior.coords)
    elif merged.geom_type == "MultiPolygon":
        # Take largest polygon by area
        largest = max(merged.geoms, key=lambda g: g.area)
        coords = list(largest.exterior.coords)
    else:
        return None
    if len(coords) < 3:
        return None
    out = []
    for x, y in coords:
        out.append(max(0, min(width, x)))
        out.append(max(0, min(height, y)))
    return out


def coco_polygon_to_yolo_line(
    segmentation, width: int, height: int, class_id: int, merge_multipolygon: bool = False
) -> str | None:
    """
    Convert COCO segmentation to one YOLO-seg line.
    COCO can have multiple polygons per object (multipolygon). If merge_multipolygon is True,
    we union them into a single polygon (requires shapely). Otherwise we use the first polygon.
    Coordinates are normalized to [0, 1].
    """
    if not segmentation or not width or not height:
        return None
    if merge_multipolygon and len(segmentation) > 1:
        poly_flat = _merge_multipolygon_shapely(segmentation, width, height)
        if poly_flat is not None:
            poly = poly_flat
        else:
            poly = segmentation[0] if isinstance(segmentation[0], (list, tuple)) else segmentation
    else:
        poly = segmentation[0] if isinstance(segmentation[0], (list, tuple)) else segmentation
    if len(poly) < 6:  # at least 3 points (x,y,x,y,x,y)
        return None
    parts = [str(class_id)]
    for i in range(0, len(poly), 2):
        if i + 1 >= len(poly):
            break
        x = poly[i] / width
        y = poly[i + 1] / height
        x = max(0, min(1, x))
        y = max(0, min(1, y))
        parts.append(f"{x:.6f}")
        parts.append(f"{y:.6f}")
    return " ".join(parts)

def _build_by_image(data, merge_multipolygon: bool):
    """From COCO data, build {image_id: [yolo_line, ...]} for Spruce/Pine only."""
    images_by_id = {img["id"]: img for img in data["images"]}
    by_image = {}
    for ann in data["annotations"]:
        cid = ann["category_id"]
        if cid not in CAT_ID_TO_YOLO:
            continue
        yolo_cls = CAT_ID_TO_YOLO[cid]
        img_id = ann["image_id"]
        w = images_by_id[img_id]["width"]
        h = images_by_id[img_id]["height"]
        line = coco_polygon_to_yolo_line(
            ann["segmentation"], w, h, yolo_cls, merge_multipolygon=merge_multipolygon
        )
        if line is None:
            continue
        by_image.setdefault(img_id, []).append(line)
    return images_by_id, by_image


def _write_split(
    image_ids_to_include: set,
    images_by_id: dict,
    by_image: dict,
    rgb_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    copy_images: bool,
):
    """Write images and labels for the given image IDs (from one COCO source)."""
    out_images_dir.mkdir(parents=True, exist_ok=True)
    out_labels_dir.mkdir(parents=True, exist_ok=True)
    n_img, n_obj = 0, 0
    for img_id in image_ids_to_include:
        img = images_by_id.get(img_id)
        if not img:
            continue
        fn = img["file_name"]
        stem = Path(fn).stem
        src = rgb_dir / fn
        if not src.exists():
            alt = rgb_dir.parent / fn
            if alt.exists():
                src = alt
        if not src.exists():
            continue
        dst_img = out_images_dir / fn
        if dst_img.exists():
            dst_img.unlink()
        if copy_images:
            shutil.copy2(src, dst_img)
        else:
            try:
                dst_img.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst_img)
        lines = by_image.get(img_id, [])
        (out_labels_dir / f"{stem}.txt").write_text("\n".join(lines))
        n_img += 1
        n_obj += len(lines)
    return n_img, n_obj


def convert_split(
    coco_path: Path,
    rgb_dir: Path,
    out_images_dir: Path,
    out_labels_dir: Path,
    copy_images: bool,
    merge_multipolygon: bool = False,
    image_ids_filter: set | None = None,
):
    """Convert one split (train or val): COCO JSON + rgb -> images + labels. If image_ids_filter is set, only include those image IDs."""
    with open(coco_path) as f:
        data = json.load(f)
    images_by_id, by_image = _build_by_image(data, merge_multipolygon)
    ids = set(images_by_id.keys())
    if image_ids_filter is not None:
        ids &= image_ids_filter
    return _write_split(ids, images_by_id, by_image, rgb_dir, out_images_dir, out_labels_dir, copy_images)

def main():
    parser = argparse.ArgumentParser(description="Convert FinnWoodlands to YOLOv8-seg format (Spruce + Pine).")
    parser.add_argument("--finnwoodlands", type=Path, default=None,
                        help="FinnWoodlands root (default: ../../FinnWoodlands from this script)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output dataset root (default: ../data/yolo_finnwoodlands under this script)")
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking")
    parser.add_argument(
        "--merge-multipolygon",
        action="store_true",
        help="Merge COCO multipolygons (multiple parts per instance) into one polygon via shapely union (requires shapely)",
    )
    parser.add_argument(
        "--val-pine-min",
        type=int,
        default=None,
        metavar="N",
        help="Move train images to val until val has at least N Pine instances (e.g. 100). Original FinnWoodlands val has 0 Pine.",
    )
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    fw = args.finnwoodlands or (script_dir / ".." / ".." / "FinnWoodlands").resolve()
    out = args.output or (script_dir / ".." / "data" / "yolo_finnwoodlands").resolve()
    if not fw.exists():
        print(f"FinnWoodlands root not found: {fw}")
        return 1
    rgb_train = fw / "rgb" / "train"
    rgb_val = fw / "rgb" / "val"
    train_json = fw / "train.json"
    val_json = fw / "val.json"
    for p in (rgb_train, rgb_val, train_json, val_json):
        if not p.exists():
            print(f"Missing: {p}")
            return 1
    out.mkdir(parents=True, exist_ok=True)

    with open(train_json) as f:
        train_data = json.load(f)
    with open(val_json) as f:
        val_data = json.load(f)
    train_images_by_id = {img["id"]: img for img in train_data["images"]}
    val_images_by_id = {img["id"]: img for img in val_data["images"]}
    train_image_ids = set(train_images_by_id.keys())
    val_image_ids = set(val_images_by_id.keys())

    to_val_from_train: set = set()  # train image IDs to treat as val (for Pine rebalance)
    if args.val_pine_min is not None and args.val_pine_min > 0:
        pine_per_train_image = {}
        for ann in train_data["annotations"]:
            if ann["category_id"] != 8:
                continue
            img_id = ann["image_id"]
            pine_per_train_image[img_id] = pine_per_train_image.get(img_id, 0) + 1
        # Greedily pick train images with most Pine until we have >= val_pine_min Pine in val
        sorted_by_pine = sorted(pine_per_train_image.items(), key=lambda x: -x[1])
        pine_count = 0
        for img_id, count in sorted_by_pine:
            if pine_count >= args.val_pine_min:
                break
            to_val_from_train.add(img_id)
            pine_count += count
        print(f"Re-split: moving {len(to_val_from_train)} train images to val so val has >= {args.val_pine_min} Pine (total Pine in moved: {pine_count})")

    train_ids_final = train_image_ids - to_val_from_train
    val_ids_final = val_image_ids | to_val_from_train

    train_images_by_id, train_by_image = _build_by_image(train_data, args.merge_multipolygon)
    val_images_by_id, val_by_image = _build_by_image(val_data, args.merge_multipolygon)

    # Train output: only images still in train
    n_img_tr, n_obj_tr = _write_split(
        train_ids_final,
        train_images_by_id,
        train_by_image,
        rgb_train,
        out / "images" / "train",
        out / "labels" / "train",
        args.copy_images,
    )
    # Val output: original val images + moved train images (write in two passes to same dirs)
    out_val_img = out / "images" / "val"
    out_val_lbl = out / "labels" / "val"
    n_va1, n_va1_obj = _write_split(
        val_image_ids,
        val_images_by_id,
        val_by_image,
        rgb_val,
        out_val_img,
        out_val_lbl,
        args.copy_images,
    )
    n_va2, n_va2_obj = _write_split(
        to_val_from_train,
        train_images_by_id,
        train_by_image,
        rgb_train,
        out_val_img,
        out_val_lbl,
        args.copy_images,
    )
    n_img_va = n_va1 + n_va2
    n_obj_va = n_va1_obj + n_va2_obj
    # dataset.yaml
    yaml_path = out / "dataset.yaml"
    yaml_content = f"""# YOLOv8-seg dataset: FinnWoodlands Spruce + Pine
# Generated by convert_finnwoodlands_to_yolo_seg.py
path: {out.as_posix()}
train: images/train
val: images/val

names:
  0: Spruce
  1: Pine
nc: 2
"""
    yaml_path.write_text(yaml_content)
    print(f"Train: {n_img_tr} images, {n_obj_tr} instances | Val: {n_img_va} images, {n_obj_va} instances")
    print(f"Dataset YAML: {yaml_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
