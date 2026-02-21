# Tree / Bark Instance Segmentation — Web App

Drag-and-drop web app for tree instance segmentation. Supports two backends:

- **PercepTree (default):** [PercepTreeV1](https://github.com/norlab-ulaval/PercepTreeV1) — Detectron2 Mask R-CNN with **pretrained models** (SynthTree43k, CanaTree100).
- **ConvexMask (optional):** [ConvexMask](https://github.com/rcondat/convexmask) — train your own; no pretrained weights in repo.

The UI shows **original** and **overlay with masks**, plus **per-class counts** (e.g. “Tree” for PercepTree).

---

## Option A: PercepTree (recommended — has pretrained weights)

### 1. Install dependencies

```bash
pip install torch torchvision opencv-python numpy flask
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

(See [Detectron2 install](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) if you need CUDA.)

### 2. Download a pretrained PercepTree model

From [PercepTreeV1 pre-trained models](https://github.com/norlab-ulaval/PercepTreeV1#pre-trained-models):

| Backbone   | Modality | mask AP50 | Download |
|-----------|----------|-----------|----------|
| **X-101-FPN** (best) | RGB | 71.07 | [model](https://drive.google.com/file/d/1Q5KV5beWVZXK_vlIED1jgpf4XJgN71ky/view?usp=sharing) |
| R-101-FPN | RGB | 70.53 | [model](https://drive.google.com/file/d/1ApKm914PuKm24kPl0sP7-XgG_Ottx5tJ/view?usp=sharing) |
| R-50-FPN  | RGB | 69.36 | [model](https://drive.google.com/file/d/1pnJZ3Vc0SVTn_J8l_pwR4w1LMYnFHzhV/view?usp=sharing) |

Download the `.pth` file and save it in:

```text
ConvexMask_Bark_segmentation/models/
```

Example: `models/X-101_RGB_60k.pth`. The app picks the first `.pth` in `models/` if you don’t pass `--weights`.

### 3. Run the app

```bash
cd ConvexMask_Bark_segmentation
python app.py
```

Or explicitly:

```bash
python app.py --model perceptree --weights models/X-101_RGB_60k.pth --port 5030
```

Open **http://127.0.0.1:5030/** and drag & drop an image.

---

## Option B: ConvexMask (optional)

Use this if you have (or will train) ConvexMask weights.

### 1. Clone ConvexMask

```bash
cd ConvexMask_Bark_segmentation
git clone https://github.com/rcondat/convexmask.git convexmask_repo
```

### 2. Install ConvexMask deps

ConvexMask needs its own env (see [convexmask/environment.yml](https://github.com/rcondat/convexmask/blob/main/environment.yml)). You also need Flask and opencv for the web app:

```bash
pip install torch torchvision opencv-python numpy flask pycocotools
```

### 3. Place ConvexMask weights

Put your trained ConvexMask `.pth` in `models/`. Set the config to match (e.g. SynthTree → `convex_synthtree_focal`):

```bash
export CONVEXMASK_CONFIG=convex_synthtree_focal
```

### 4. Run with ConvexMask backend

```bash
python app.py --model convexmask --weights models/your_convexmask.pth
```

---

## Model choice and CLI

- **Default backend:** PercepTree (`--model perceptree` or env `SEGMENTATION_MODEL=perceptree`).
- **ConvexMask:** `--model convexmask` or `SEGMENTATION_MODEL=convexmask`.

```bash
python app.py --model perceptree --weights models/X-101_RGB_60k.pth
python app.py --model convexmask --weights models/best.pth
```

---

## Folder layout

```text
ConvexMask_Bark_segmentation/
├── README.md
├── app.py                 # Web app (PercepTree + ConvexMask)
├── requirements.txt
├── models/                # Put PercepTree or ConvexMask .pth here
├── convexmask_repo/       # Only for ConvexMask backend (git clone)
└── uploads/               # Temporary uploads (gitignored)
```

---

## Notes

- **PercepTree:** One class, “Tree”. Pretrained on SynthTree43k (synthetic) or CanaTree100 (real); best mask AP is X-101 RGB (71.07).
- **ConvexMask:** No pretrained weights in the repo; you must train or obtain `.pth` elsewhere. Class names come from the config (e.g. “tree” or COCO 80).
- **GPU:** Both backends run faster with CUDA; Detectron2 and ConvexMask support GPU.
