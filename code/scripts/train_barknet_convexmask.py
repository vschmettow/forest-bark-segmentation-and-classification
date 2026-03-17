#!/usr/bin/env python3
"""
Train ConvexMask R50-FPN on BarkNetCOCO/data0.

Prerequisites:
  1. Clone ConvexMask: git clone https://github.com/rcondat/convexmask.git convexmask_repo
  2. Install deps: pip install torch torchvision opencv-python pycocotools
  3. Download ResNet50 backbone: resnet50-19c8e357.pth (from PyTorch model zoo)
     Place in convexmask_repo/../weights/ or set init_weights_folder

This script patches the ConvexMask config with BarkNet dataset and runs training.

Usage:
  From project root:
    python scripts/train_barknet_convexmask.py

  Or specify ConvexMask repo:
    python scripts/train_barknet_convexmask.py --convexmask_repo path/to/convexmask
"""

from pathlib import Path
import sys
import subprocess
import argparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONVEXMASK_REPO = PROJECT_ROOT / "convexmask_repo"
BARKNET_COCO_DATA0 = PROJECT_ROOT / "BarkNetCOCO" / "data0"


def create_barknet_dataset_patch():
    """Return Python code to add BarkNet dataset to ConvexMask config."""
    train_images = str(BARKNET_COCO_DATA0 / "images" / "train")
    train_info = str(BARKNET_COCO_DATA0 / "annotations_train.json")
    valid_images = str(BARKNET_COCO_DATA0 / "images" / "val")
    valid_info = str(BARKNET_COCO_DATA0 / "annotations_val.json")

    return f'''
# --- BarkNet dataset (added by train_barknet_convexmask.py) ---
barknet_dataset = dataset_base.copy({{
    'name': 'BarkNet_data0',
    'train_images': '{train_images}',
    'train_info': '{train_info}',
    'valid_images': '{valid_images}',
    'valid_info': '{valid_info}',
    'has_gt': True,
    'class_names': ('Picea', 'Pinus'),
    'label_map': {{1: 1, 2: 2}},
}})

convex_barknet_r50 = convex_synthtree_focal.copy({{
    'name': 'convex_barknet_r50',
    'dataset': barknet_dataset,
    'num_classes': 3,
    'max_size': (1280, 720),
    'fixed_size': True,
    'epochs': 50,
    'batch_size': 4,
    'lr_steps': (35, 45),
    'init_weights_folder': './weights/',
}})
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--convexmask_repo",
        type=Path,
        default=DEFAULT_CONVEXMASK_REPO,
        help="Path to ConvexMask repo",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (reduce if OOM)",
    )
    args = parser.parse_args()

    if not BARKNET_COCO_DATA0.exists():
        print(f"Error: BarkNetCOCO/data0 not found at {BARKNET_COCO_DATA0}")
        print("Run: python scripts/convert_barknet_to_coco_and_yolo.py")
        return 1

    repo = Path(args.convexmask_repo)
    if not repo.exists():
        print(f"ConvexMask repo not found: {repo}")
        print("Clone it: git clone https://github.com/rcondat/convexmask.git convexmask_repo")
        return 1

    config_py = repo / "data" / "config.py"
    if not config_py.exists():
        print(f"Config not found: {config_py}")
        return 1

    # Append BarkNet config to config.py
    patch = create_barknet_dataset_patch()
    with open(config_py, "r") as f:
        content = f.read()
    if "barknet_dataset" not in content:
        with open(config_py, "a") as f:
            f.write(patch)
        print("Added BarkNet config to ConvexMask config.py")
    else:
        print("BarkNet config already present")

    # Run training
    cmd = [
        sys.executable,
        str(repo / "train.py"),
        "--config", "convex_barknet_r50",
        "--dataset", "barknet_dataset",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(repo))


if __name__ == "__main__":
    exit(main())
