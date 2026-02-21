#!/usr/bin/env python3
"""
Tree / bark instance segmentation web app.
Supports two backends:
  - PercepTree (default): Detectron2 Mask R-CNN from https://github.com/norlab-ulaval/PercepTreeV1
  - ConvexMask: https://github.com/rcondat/convexmask
Drag & drop images; overlay and per-class counts use the model's classes.
"""
from pathlib import Path
import base64
import os
import sys
import uuid

APP_DIR = Path(__file__).resolve().parent

# Model type: "perceptree" (default) or "convexmask"
MODEL_TYPE = os.environ.get("SEGMENTATION_MODEL", "perceptree").lower().strip()
if MODEL_TYPE not in ("perceptree", "convexmask"):
    MODEL_TYPE = "perceptree"

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

# Backend-specific state (set in load_model)
_predictor = None  # PercepTree: Detectron2 DefaultPredictor
_convexmask_net = None
_FastBaseTransform = None
_postprocess = None
_cfg = None
_COLORS = None


def get_class_names():
    """Class names from the loaded model."""
    if MODEL_TYPE == "perceptree":
        return ("Tree",)
    if _cfg is not None:
        return getattr(_cfg.dataset, "class_names", ("tree",))
    return ("tree",)


def _load_perceptree(weights_path: Path):
    """Load PercepTree (Detectron2 Mask R-CNN) from PercepTreeV1 config + weights."""
    global _predictor
    try:
        from detectron2.utils.logger import setup_logger
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor
        from detectron2.config import get_cfg
    except ImportError as e:
        raise ImportError(
            "PercepTree backend needs Detectron2. Install with: pip install 'git+https://github.com/facebookresearch/detectron2.git'"
        ) from e
    setup_logger(name=__name__)
    cfg = get_cfg()
    cfg.INPUT.MASK_FORMAT = "bitmask"
    # Match PercepTree demo: keypoint R-CNN config (they use it for Mask R-CNN with 1 class)
    model_name = weights_path.name.lower()
    if "x-101" in model_name or "x_101" in model_name:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"))
    elif "r-101" in model_name or "r_101" in model_name:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
    else:
        cfg.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ()
    cfg.DATASETS.TEST = ()
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 1
    cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 5
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.WEIGHTS = str(weights_path)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    _predictor = DefaultPredictor(cfg)


def _run_perceptree(image_path: str, score_threshold: float):
    """Run PercepTree (Detectron2) on image. Returns (detections, overlay_bgr)."""
    global _predictor
    if _predictor is None:
        raise RuntimeError("PercepTree model not loaded")
    im = cv2.imread(image_path)
    if im is None:
        return [], None
    h, w = im.shape[:2]
    outputs = _predictor(im)
    instances = outputs["instances"].to("cpu")
    boxes = instances.pred_boxes.tensor.numpy()
    scores = instances.scores.numpy()
    pred_classes = instances.pred_classes.numpy()
    if not hasattr(instances, "pred_masks") or instances.pred_masks is None:
        return [], im
    masks = instances.pred_masks.numpy()  # (N, H, W)
    class_names = get_class_names()
    detections = []
    overlay = im.astype(np.float32)
    color_tree = (76, 175, 80)  # BGR green
    for i in range(len(scores)):
        if scores[i] < score_threshold:
            continue
        cls_id = int(pred_classes[i])
        name = class_names[cls_id] if cls_id < len(class_names) else "Tree"
        mask = masks[i]
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        detections.append((name, float(scores[i]), mask))
        mask_bool = mask > 0.5
        for c in range(3):
            overlay[:, :, c] = np.where(mask_bool, 0.5 * overlay[:, :, c] + 0.5 * color_tree[c], overlay[:, :, c])
    overlay_bgr = np.clip(overlay, 0, 255).astype(np.uint8)
    return detections, overlay_bgr


def _load_convexmask(weights_path: Path):
    """Load ConvexMask (requires convexmask_repo cloned)."""
    global _convexmask_net, _cfg, _COLORS
    CONVEXMASK_REPO = APP_DIR / "convexmask_repo"
    if not CONVEXMASK_REPO.is_dir():
        raise FileNotFoundError(
            "ConvexMask repo not found. Run: git clone https://github.com/rcondat/convexmask.git convexmask_repo"
        )
    sys.path.insert(0, str(CONVEXMASK_REPO))
    config_name = os.environ.get("CONVEXMASK_CONFIG", "convex_synthtree_focal")
    from data.config import set_cfg
    set_cfg(config_name)
    from data import cfg as cm_cfg
    from data.config import COLORS as cm_colors
    _cfg = cm_cfg
    _COLORS = cm_colors
    from convexmask import ConvexMask
    from utils.augmentations import FastBaseTransform
    from layers.output_utils import postprocess
    global _FastBaseTransform, _postprocess
    _FastBaseTransform = FastBaseTransform
    _postprocess = postprocess
    import torch
    net = ConvexMask()
    net.load_weights(str(weights_path))
    net.eval()
    if torch.cuda.is_available():
        net = net.cuda()
    _convexmask_net = net


def _run_convexmask(image_path: str, score_threshold: float):
    """Run ConvexMask on image. Returns (detections, overlay_bgr)."""
    global _convexmask_net, _FastBaseTransform, _postprocess, _cfg, _COLORS
    if _convexmask_net is None:
        raise RuntimeError("ConvexMask model not loaded")
    import torch
    net = _convexmask_net
    transform = _FastBaseTransform().cuda() if torch.cuda.is_available() else _FastBaseTransform()
    frame = cv2.imread(image_path)
    if frame is None:
        return [], None
    h, w = frame.shape[:2]
    frame_tensor = torch.from_numpy(frame).float()
    if torch.cuda.is_available():
        frame_tensor = frame_tensor.cuda()
    batch = transform(frame_tensor.unsqueeze(0))
    with torch.no_grad():
        preds = net(batch)
    result = _postprocess(preds, w, h, score_threshold=score_threshold, output_mask_poly=True, convex_poly=True)
    if len(result) < 6:
        return [], frame
    classes, scores, center_scores, boxes, masks, poly_coords = result[0], result[1], result[2], result[3], result[4], result[5]
    class_names = get_class_names()
    classes = classes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    masks = masks.cpu().numpy() if hasattr(masks, "cpu") else masks
    if masks.ndim != 3:
        return [], frame
    detections = []
    for i in range(len(classes)):
        cls_id = int(classes[i])
        name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        detections.append((name, float(scores_np[i]), masks[i]))
    overlay = frame.astype(np.float32)
    num_colors = len(_COLORS)
    for i, (class_name, conf, mask) in enumerate(detections):
        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        mask_bool = mask > 0.5
        color_idx = (i * 5) % num_colors
        r, g, b = _COLORS[color_idx]
        color_bgr = (b, g, r)
        for c in range(3):
            overlay[:, :, c] = np.where(mask_bool, 0.5 * overlay[:, :, c] + 0.5 * color_bgr[c], overlay[:, :, c])
    overlay_bgr = np.clip(overlay, 0, 255).astype(np.uint8)
    return detections, overlay_bgr


def load_model(weights_path: Path):
    """Load the backend specified by MODEL_TYPE."""
    if MODEL_TYPE == "perceptree":
        _load_perceptree(weights_path)
    else:
        _load_convexmask(weights_path)


def run_segmentation(image_path: str, score_threshold: float = 0.25):
    """Run the active backend. Returns (detections, overlay_bgr)."""
    if MODEL_TYPE == "perceptree":
        return _run_perceptree(image_path, score_threshold)
    return _run_convexmask(image_path, score_threshold)


def image_to_base64_bgr(bgr_array):
    if bgr_array is None:
        return None
    _, buf = cv2.imencode(".jpg", bgr_array)
    return base64.b64encode(buf).decode("utf-8")


# --- Flask app ---
UPLOAD_FOLDER = APP_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "PNG", "JPG", "JPEG"}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {e.lower() for e in ALLOWED_EXTENSIONS}


def _find_weights():
    """First .pth in models/ (works for both PercepTree and ConvexMask)."""
    models_dir = APP_DIR / "models"
    if not models_dir.is_dir():
        return None
    for p in sorted(models_dir.glob("*.pth")):
        return p
    return None


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/segment", methods=["POST"])
def segment():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Allowed types: JPG, JPEG, PNG"}), 400

    ext = file.filename.rsplit(".", 1)[1].lower()
    safe_name = f"{uuid.uuid4().hex}.{ext}"
    save_path = UPLOAD_FOLDER / safe_name
    file.save(str(save_path))

    try:
        score_threshold = float(request.form.get("conf", 0.5))
    except (TypeError, ValueError):
        score_threshold = 0.5

    try:
        detections, overlay_bgr = run_segmentation(str(save_path), score_threshold=score_threshold)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500

    img_bgr = cv2.imread(str(save_path))
    original_b64 = image_to_base64_bgr(img_bgr)
    overlay_b64 = image_to_base64_bgr(overlay_bgr) if overlay_bgr is not None else None
    save_path.unlink(missing_ok=True)

    from collections import Counter
    counts = Counter(d[0] for d in detections)
    class_names = list(get_class_names())
    counts_dict = {c: counts.get(c, 0) for c in class_names}
    total = len(detections)

    return jsonify({
        "original_b64": original_b64,
        "overlay_b64": overlay_b64,
        "class_names": class_names,
        "counts": counts_dict,
        "total": total,
    })


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tree instance segmentation</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a2526 0%, #2d3d3f 50%, #3d5155 100%);
            min-height: 100vh;
            padding: 24px;
            color: #e0f2f1;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 32px; }
        h1 { font-size: 2rem; color: #80cbc4; margin-bottom: 8px; }
        .subtitle { color: #b2dfdb; font-size: 1rem; }
        .drop-zone {
            border: 3px dashed #4db6ac;
            border-radius: 16px;
            padding: 48px 24px;
            text-align: center;
            background: rgba(77, 182, 172, 0.1);
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 24px;
        }
        .drop-zone:hover, .drop-zone.dragover {
            border-color: #26a69a;
            background: rgba(77, 182, 172, 0.2);
        }
        .drop-icon { font-size: 3em; margin-bottom: 12px; }
        .drop-text { font-size: 1.2em; margin-bottom: 4px; }
        .drop-subtext { font-size: 0.9em; color: #b2dfdb; }
        input[type="file"] { display: none; }
        .loading { display: none; text-align: center; padding: 24px; }
        .spinner {
            border: 4px solid rgba(255,255,255,0.2);
            border-top-color: #4db6ac;
            border-radius: 50%;
            width: 48px; height: 48px;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 16px;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
        .error { display: none; background: rgba(244,67,54,0.2); border: 1px solid #f44336; border-radius: 8px; padding: 12px; margin-bottom: 16px; color: #ffcdd2; }
        .comparison {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-top: 24px;
        }
        @media (max-width: 900px) { .comparison { grid-template-columns: 1fr; } }
        .panel {
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .panel h3 { font-size: 1rem; margin-bottom: 12px; color: #80cbc4; }
        .panel img { width: 100%; height: auto; border-radius: 8px; display: block; }
        .stats {
            display: flex;
            gap: 24px;
            flex-wrap: wrap;
            margin-top: 24px;
            padding: 20px;
            background: rgba(0,0,0,0.25);
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .stat { font-size: 1.25rem; }
        .stat span { font-weight: 700; color: #4db6ac; }
        .clear-btn {
            margin-top: 20px;
            padding: 12px 24px;
            background: rgba(244,67,54,0.3);
            color: #ffcdd2;
            border: 1px solid #f44336;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1rem;
        }
        .clear-btn:hover { background: rgba(244,67,54,0.5); }
        .results-wrap { display: none; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Tree instance segmentation</h1>
            <p class="subtitle">Drag & drop image — PercepTree or ConvexMask</p>
        </header>
        <div class="drop-zone" id="dropZone">
            <div class="drop-icon">🌲</div>
            <div class="drop-text">Drag & drop image here</div>
            <div class="drop-subtext">or click to choose (JPG, PNG)</div>
            <input type="file" id="fileInput" accept="image/*">
        </div>
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <p>Running segmentation...</p>
        </div>
        <div class="error" id="error"></div>
        <div class="results-wrap" id="resultsWrap">
            <div class="comparison">
                <div class="panel">
                    <h3>Original</h3>
                    <img id="imgOriginal" alt="Original">
                </div>
                <div class="panel">
                    <h3>With masks</h3>
                    <img id="imgOverlay" alt="Overlay">
                </div>
            </div>
            <div class="stats" id="stats"></div>
            <button class="clear-btn" onclick="clearResults()">Clear & upload another</button>
        </div>
    </div>
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const resultsWrap = document.getElementById('resultsWrap');
        const imgOriginal = document.getElementById('imgOriginal');
        const imgOverlay = document.getElementById('imgOverlay');
        const statsEl = document.getElementById('stats');

        dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });
        dropZone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', e => { if (e.target.files.length) handleFile(e.target.files[0]); });

        function showError(msg) {
            error.textContent = msg;
            error.style.display = 'block';
        }
        function hideError() {
            error.style.display = 'none';
        }

        function handleFile(file) {
            if (!file.type.startsWith('image/')) { showError('Please use an image (JPG, PNG).'); return; }
            hideError();
            loading.style.display = 'block';
            resultsWrap.style.display = 'none';
            const formData = new FormData();
            formData.append('file', file);
            formData.append('conf', '0.5');
            fetch('/segment', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    loading.style.display = 'none';
                    if (data.error) {
                        showError(data.error);
                    } else {
                        imgOriginal.src = 'data:image/jpeg;base64,' + data.original_b64;
                        imgOverlay.src = data.overlay_b64 ? 'data:image/jpeg;base64,' + data.overlay_b64 : imgOriginal.src;
                        statsEl.innerHTML = '';
                        (data.class_names || []).forEach(c => {
                            const div = document.createElement('div');
                            div.className = 'stat';
                            div.innerHTML = c + ': <span>' + (data.counts[c] || 0) + '</span>';
                            statsEl.appendChild(div);
                        });
                        const totalDiv = document.createElement('div');
                        totalDiv.className = 'stat';
                        totalDiv.innerHTML = 'Total: <span>' + (data.total || 0) + '</span>';
                        statsEl.appendChild(totalDiv);
                        resultsWrap.style.display = 'block';
                    }
                })
                .catch(err => {
                    loading.style.display = 'none';
                    showError('Error: ' + err.message);
                });
        }

        function clearResults() {
            resultsWrap.style.display = 'none';
            imgOriginal.src = '';
            imgOverlay.src = '';
            fileInput.value = '';
            hideError();
        }
    </script>
</body>
</html>
"""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Tree instance segmentation web app (PercepTree or ConvexMask)")
    parser.add_argument("--port", type=int, default=5030, help="Port (default 5030)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--weights", type=Path, default=None, help="Path to model weights (.pth)")
    parser.add_argument("--model", type=str, choices=("perceptree", "convexmask"), default=None,
                        help="Backend: perceptree (default) or convexmask")
    args = parser.parse_args()

    model_type = (args.model or os.environ.get("SEGMENTATION_MODEL", "perceptree")).lower()
    if model_type not in ("perceptree", "convexmask"):
        model_type = "perceptree"
    global MODEL_TYPE
    MODEL_TYPE = model_type

    weights_path = args.weights or _find_weights()
    if not weights_path or not Path(weights_path).exists():
        print("No model weights found.")
        if MODEL_TYPE == "perceptree":
            print("For PercepTree: download a .pth from https://github.com/norlab-ulaval/PercepTreeV1")
            print("  (e.g. X-101 RGB 71.07 mask AP) and put it in ConvexMask_Bark_segmentation/models/")
        else:
            print("Put a ConvexMask .pth in ConvexMask_Bark_segmentation/models/ or pass --weights path")
        return 1

    print(f"Backend: {MODEL_TYPE}")
    print(f"Classes: {get_class_names()}")
    print(f"Loading weights: {weights_path}")
    load_model(Path(weights_path))
    print(f"Open http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
