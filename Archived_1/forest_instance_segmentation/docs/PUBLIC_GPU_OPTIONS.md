# Public / Free GPU Options for Training (MacBook Alternative)

If your MacBook can’t run YOLOv8-seg training (or it would be too slow on CPU), you can use **free cloud GPUs**. Below are practical options, from no-approval to academic credits.

---

## 1. Google Colab (easiest, no approval)

**What you get:** Free GPU (T4 or similar) in a Jupyter notebook in the browser.

- **Free tier:** GPU when available, ~12 hours max per session, ~90 min idle timeout.
- **No credit card** required for the free tier.
- **Setup:** [colab.research.google.com](https://colab.research.google.com) → New notebook → **Runtime → Change runtime type → GPU** (e.g. T4).
- **Limitation:** GPU is not always available; usage limits can apply. For ~1.5–3 hours of training (100 epochs), one or two sessions are usually enough.

**How to use it for this project:** Upload your YOLO dataset (or mount Google Drive and put the dataset there), then run the same Ultralytics commands as on your laptop. See “Colab quick start” below.

---

## 2. Kaggle Notebooks (free GPU)

**What you get:** Jupyter notebooks with free GPU (P100 or T4, ~15.9 GB).

- **Free tier:** Up to **30 hours/week** GPU (P100), **20 hours/week** TPU; **9 hours** max per run.
- **Requirement:** Kaggle account; **phone number verification** for GPU.
- **Setup:** [kaggle.com](https://www.kaggle.com) → Code → New Notebook → **Settings → Accelerator → GPU**.
- **Storage:** 20 GB for datasets; you can upload your YOLO dataset or use a Kaggle Dataset.

Good if Colab is busy or you hit limits; workflow is similar (notebook + GPU).

---

## 3. Paid but cheap (if free tiers are full)

- **Google Colab Pro** (~$10/month): Better GPU availability and longer sessions.
- **Lambda Labs / RunPod / Vast.ai**: Rent GPU by the hour (e.g. ~$0.20–0.50/hr for a T4); pay only for the 1.5–3 hours of training.
- **Google Cloud / AWS / Azure**: Free tiers or trials; then pay-as-you-go. Only worth it if you already use them or need more control.

---

## 4. Academic / research credits (if you’re a student or researcher)

- **Google Cloud for Researchers:** Up to **$5,000** in credits (e.g. PhD students up to $1,000) on application. [Google Cloud for Researchers](https://cloud.google.com/edu/researchers)
- **AWS Cloud Credits for Research:** Up to **$5,000** for students, more for faculty; application required. [AWS Cloud Credit for Research](https://aws.amazon.com/government-education/research-and-technical-computing/cloud-credit-for-research/)
- **NVIDIA Academic Grant Program:** GPU credits via partners (e.g. Saturn Cloud) for research groups. [NVIDIA Academic Grant](https://www.nvidia.com/en-us/academic/)

Apply if you’re doing a university project (Projektarbeit); approval can take weeks.

---

## Colab quick start (run this project’s training there)

1. Open [Google Colab](https://colab.research.google.com), **New notebook**.
2. **Runtime → Change runtime type → GPU** (e.g. T4) → Save.
3. In the first cell, run:

```python
# Install Ultralytics
!pip install -q ultralytics
```

4. **Upload your dataset** (e.g. zip of `yolo_finnwoodlands` with `images/`, `labels/`, `dataset.yaml`) or use Google Drive:

```python
# Option A: Upload zip (run this cell, then use Files panel to upload your zip)
from google.colab import files
# uploaded = files.upload()  # then unzip: !unzip -q your_dataset.zip -d /content/data

# Option B: Mount Drive and use dataset already in Drive
from google.colab import drive
drive.mount("/content/drive")
# Then set data_yaml to e.g. "/content/drive/MyDrive/yolo_finnwoodlands/dataset.yaml"
```

5. **Train** (adjust paths to where your data is):

```python
from ultralytics import YOLO

# Path to your dataset.yaml (after upload or Drive mount)
data_yaml = "/content/data/yolo_finnwoodlands/dataset.yaml"  # or your path

model = YOLO("yolov8n-seg.pt")
results = model.train(
    data=data_yaml,
    epochs=100,
    imgsz=640,
    batch=16,
    project="/content/drive/MyDrive/seg_results",  # save to Drive so you don't lose it
    name="finnwoodlands_seg",
)
```

6. **Download weights** when done:

```python
# Download best.pt to your laptop
from google.colab import files
files.download("/content/drive/MyDrive/seg_results/finnwoodlands_seg/weights/best.pt")
```

**Tips:**

- Keep the notebook **active** (e.g. scroll or run a cell occasionally) to reduce idle disconnects.
- Save checkpoints to **Google Drive** so you don’t lose them if the session ends.
- If the session disconnects, you can resume training by loading `last.pt` and calling `model.train(resume=True)` (see Ultralytics docs).

---

## Summary

| Option              | Best for              | Approval     | Typical use for this project   |
|---------------------|------------------------|-------------|---------------------------------|
| **Google Colab**    | Quick start, no setup | None        | 1–2 sessions, ~1.5–3 h training |
| **Kaggle**          | When Colab is limited | Phone verify| Same as Colab                  |
| **Colab Pro**       | More reliable GPU     | Paid        | Same, fewer “no GPU” issues    |
| **Cloud credits**   | University project    | Application | Full control, longer experiments |

**Recommendation:** Start with **Google Colab** (free, GPU). Upload your YOLO dataset, run the snippet above, and save `best.pt` to Drive or download it. That’s enough to train the FinnWoodlands YOLOv8n-seg model without using your MacBook’s CPU.
