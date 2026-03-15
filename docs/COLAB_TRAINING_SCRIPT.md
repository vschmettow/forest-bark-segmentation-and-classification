# Google Colab – ConvexMask Training (plug-in script)

**Enable GPU**: Runtime → Change runtime type → T4 GPU

**Settings**: batch=4, epochs=50 | Data: `My Drive` / `Colab Notebooks` / `Bark` / `BarkNetCOCO/data0`

**Path in Colab**: `/content/drive/MyDrive/Colab Notebooks/Bark`

> **If you see `CalledProcessError`**: Your Colab cell still uses `subprocess.run`. Replace the training block with the version below (uses `!python` so the real error prints inline).

---

## Cell 1: Mount Drive + verify

```python
from google.colab import drive
drive.mount('/content/drive')
BARK = "/content/drive/MyDrive/Colab Notebooks/Bark"
!ls "{BARK}/BarkNetCOCO/data0/images/train/" | head -5
```

---

## Cell 2: Install

```python
!pip install -q torch torchvision opencv-python pycocotools pandas
```

---

## Cell 3: ConvexMask setup + train (batch=4, epochs=50)

```python
import sys
import subprocess
from pathlib import Path

BARK = Path("/content/drive/MyDrive/Colab Notebooks/Bark")
REPO = Path("/content/convexmask_repo")

if not REPO.exists():
    !git clone -q https://github.com/rcondat/convexmask.git {REPO}

# Fix ConvexMask train.py compatibility issues
TRAIN_PY = REPO / "train.py"
if TRAIN_PY.exists():
    txt = TRAIN_PY.read_text()
    changed = False
    if "from tensorboard import SummaryWriter" in txt:
        txt = txt.replace("from tensorboard import SummaryWriter", "from torch.utils.tensorboard import SummaryWriter")
        changed = True
    if "if args.kfold is not None:" in txt:
        txt = txt.replace("if args.kfold is not None:", "if getattr(args, 'kfold', None) is not None:")
        changed = True
    if "num_workers=8" in txt:
        txt = txt.replace("num_workers=8", "num_workers=2")  # Colab: 8 workers can cause "worker killed" errors
        changed = True
    if "tb_log.close()" in txt and "if epoch == num_epochs - 1:" not in txt:
        txt = txt.replace("tb_log.close()", "tb_log.close() if epoch == num_epochs - 1 else None")
        changed = True
    if changed:
        TRAIN_PY.write_text(txt)
        print("Patched train.py")

WEIGHTS = REPO / "weights"
WEIGHTS.mkdir(parents=True, exist_ok=True)
BACKBONE = WEIGHTS / "resnet50-19c8e357.pth"
if not BACKBONE.exists():
    import torch
    from torchvision.models import resnet50, ResNet50_Weights
    m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    torch.save(m.state_dict(), str(BACKBONE))
    print("Saved ResNet50 backbone")

CONFIG = REPO / "data" / "config.py"
PATCH = f'''
barknet_dataset = dataset_base.copy({{
    'name': 'BarkNet_data0',
    'train_images': '{str(BARK / "BarkNetCOCO/data0/images/train")}',
    'train_info': '{str(BARK / "BarkNetCOCO/data0/annotations_train.json")}',
    'valid_images': '{str(BARK / "BarkNetCOCO/data0/images/val")}',
    'valid_info': '{str(BARK / "BarkNetCOCO/data0/annotations_val.json")}',
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
if "barknet_dataset" not in CONFIG.read_text():
    with open(CONFIG, "a") as f:
        f.write(PATCH)
    print("Patched config")

# Check GPU (ConvexMask requires GPU)
import torch
if not torch.cuda.is_available():
    print("ERROR: No GPU detected! Go to Runtime → Change runtime type → T4 GPU")
else:
    print(f"GPU: {torch.cuda.get_device_name(0)}")

print("Starting ConvexMask (batch=4, epochs=50)...")
# Use ! to run - output prints inline, no CalledProcessError
%cd /content/convexmask_repo
!python train.py --config convex_barknet_r50 --dataset barknet_dataset --home_dir ./ 2>&1
```

---

## Cell 4: Accuracy table

```python
import subprocess
import re
import pandas as pd
from pathlib import Path

REPO = Path("/content/convexmask_repo")
print("Running evaluation...")
result = subprocess.run(
    [sys.executable, str(REPO / "eval.py"),
     "--config", "convex_barknet_r50",
     "--dataset", "barknet_dataset",
     "--home_dir", str(REPO) + "/",
     "--no_bar", "--max_images", "100"],
    cwd=str(REPO), capture_output=True, text=True,
)
eval_out = result.stdout + result.stderr

cm_map50 = cm_map5095 = None
for line in eval_out.split("\n"):
    m = re.search(r"mask[_\s]*50[:\s]*([\d.]+)", line, re.I)
    if m: cm_map50 = float(m.group(1))
    m = re.search(r"mask[_\s]*all[:\s]*([\d.]+)", line, re.I)
    if m: cm_map5095 = float(m.group(1))
sf = REPO / "mAP" / "convex_barknet_r50.txt"
if (cm_map50 is None or cm_map5095 is None) and sf.exists():
    for line in sf.read_text().split("\n"):
        if "mask_50" in line: cm_map50 = float(line.split(":")[-1].strip())
        if "mask_all" in line: cm_map5095 = float(line.split(":")[-1].strip())

def fmt(x):
    return f"{x:.4f}" if x is not None and not (isinstance(x, float) and (x != x)) else "N/A"

print("\n" + "="*50)
print("CONVEXMASK R50-FPN – Accuracy Results")
print("="*50)
display(pd.DataFrame({
    "Metric": ["mAP50", "mAP50-95", "Precision", "Recall"],
    "Value": [fmt(cm_map50), fmt(cm_map5095), "N/A", "N/A"],
}))
```

---

## Cell 5: Download best checkpoint to your computer

```python
from google.colab import files
import shutil
from pathlib import Path

src = Path("/content/convexmask_repo/weights/convex_barknet_r50/best_checkpoint.pth")
if src.exists():
    dst = Path("/content/convexmask_r50_data0_best.pth")
    shutil.copy2(src, dst)
    files.download(str(dst))
    print("Download started: convexmask_r50_data0_best.pth → save to best_models/")
else:
    print("best_checkpoint.pth not found. Run Cell 3 first and wait for training to finish.")
```

---

## Things to watch out for

| Issue | What to do |
|------|------------|
| **Session timeout** | Colab free tier disconnects after ~90 min of inactivity. Keep the tab active or use Colab Pro for long runs. |
| **Drive unmount** | If paths fail, re-run Cell 1 to remount Drive. |
| **cuFFT/cuDNN/cuBLAS warnings** | Harmless TensorFlow messages; ignore them. |
| **DataLoader worker killed** | Already patched (num_workers=2). If it still happens, the script can be changed to `num_workers=0`. |
| **Path mismatch** | Ensure your Drive folder is exactly `Colab Notebooks/Bark` with `BarkNetCOCO/data0/images/train`, `annotations_train.json`, etc. |
| **Empty dataset** | If training fails with "no images", verify Cell 1 lists files and that `annotations_train.json` exists. |
