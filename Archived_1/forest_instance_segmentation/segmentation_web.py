#!/usr/bin/env python3
"""
Web app for FinnWoodlands YOLOv8-seg (Spruce + Pine) instance segmentation.
Drag & drop forest images; see original vs overlay and detection counts.
"""
from pathlib import Path
import base64
import io
import uuid

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string
from werkzeug.utils import secure_filename

from ultralytics import YOLO

# Paths relative to this file
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "best.pt"
UPLOAD_FOLDER = APP_DIR / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "PNG", "JPG", "JPEG"}
MAX_CONTENT_LENGTH = 32 * 1024 * 1024  # 32MB

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

model = None


def load_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
        print(f"Loading model from {MODEL_PATH}...")
        model = YOLO(str(MODEL_PATH))
        print("Model loaded.")
    return model


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {e.lower() for e in ALLOWED_EXTENSIONS}


def run_segmentation(image_path: str, conf_threshold: float = 0.25):
    """Run YOLOv8-seg and return detections + overlay image as numpy (BGR)."""
    mdl = load_model()
    results = mdl(image_path, conf=conf_threshold, verbose=False)

    detections = []
    for result in results:
        if result.masks is None:
            continue
        names = result.names or {}
        for i, mask in enumerate(result.masks.data):
            cls_id = int(result.boxes.cls[i]) if result.boxes is not None else 0
            conf = float(result.boxes.conf[i]) if result.boxes is not None else 1.0
            class_name = names.get(cls_id, f"class_{cls_id}")
            detections.append((class_name, conf, mask.cpu().numpy()))

    img = cv2.imread(image_path)
    if img is None:
        return detections, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    h, w = img.shape[:2]
    overlay = img_rgb.copy()
    colors_rgb = {"Spruce": (50, 205, 80), "Pine": (30, 144, 255)}
    for class_name, conf, mask in detections:
        mask_np = mask if isinstance(mask, np.ndarray) else mask
        mask_resized = cv2.resize(
            mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
        )
        mask_bool = mask_resized > 0.5
        color = np.array(colors_rgb.get(class_name, (200, 200, 200)), dtype=np.float32)
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask_bool,
                0.5 * overlay[:, :, c] + 0.5 * color[c],
                overlay[:, :, c],
            )
    overlay_uint8 = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR)
    return detections, overlay_bgr


def image_to_base64_bgr(bgr_array):
    if bgr_array is None:
        return None
    _, buf = cv2.imencode(".jpg", bgr_array)
    return base64.b64encode(buf).decode("utf-8")


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
        conf = float(request.form.get("conf", 0.25))
    except (TypeError, ValueError):
        conf = 0.25

    try:
        detections, overlay_bgr = run_segmentation(str(save_path), conf_threshold=conf)
    except Exception as e:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": str(e)}), 500

    # Counts
    from collections import Counter
    counts = Counter(d[0] for d in detections)
    spruce = counts.get("Spruce", 0)
    pine = counts.get("Pine", 0)
    total = len(detections)

    # Original image as base64 (read again to send)
    img_bgr = cv2.imread(str(save_path))
    original_b64 = image_to_base64_bgr(img_bgr)
    overlay_b64 = image_to_base64_bgr(overlay_bgr) if overlay_bgr is not None else None

    save_path.unlink(missing_ok=True)

    return jsonify({
        "original_b64": original_b64,
        "overlay_b64": overlay_b64,
        "spruce": spruce,
        "pine": pine,
        "total": total,
    })


HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forest Instance Segmentation — Spruce & Pine</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a2f1a 0%, #2d4a2d 50%, #3d6b3d 100%);
            min-height: 100vh;
            padding: 24px;
            color: #e8f5e9;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 32px; }
        h1 { font-size: 2rem; color: #c8e6c9; margin-bottom: 8px; }
        .subtitle { color: #a5d6a7; font-size: 1rem; }
        .drop-zone {
            border: 3px dashed #81c784;
            border-radius: 16px;
            padding: 48px 24px;
            text-align: center;
            background: rgba(129, 199, 132, 0.1);
            cursor: pointer;
            transition: all 0.2s;
            margin-bottom: 24px;
        }
        .drop-zone:hover, .drop-zone.dragover {
            border-color: #66bb6a;
            background: rgba(129, 199, 132, 0.2);
        }
        .drop-icon { font-size: 3em; margin-bottom: 12px; }
        .drop-text { font-size: 1.2em; margin-bottom: 4px; }
        .drop-subtext { font-size: 0.9em; color: #a5d6a7; }
        input[type="file"] { display: none; }
        .loading { display: none; text-align: center; padding: 24px; }
        .spinner {
            border: 4px solid rgba(255,255,255,0.2);
            border-top-color: #66bb6a;
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
        .panel h3 { font-size: 1rem; margin-bottom: 12px; color: #c8e6c9; }
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
        .stat span { font-weight: 700; color: #81c784; }
        .stat.spruce span { color: #66bb6a; }
        .stat.pine span { color: #42a5f5; }
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
            <h1>Forest instance segmentation</h1>
            <p class="subtitle">Spruce & Pine — drag & drop or select a forest image</p>
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
                    <h3>With masks (Spruce = green, Pine = blue)</h3>
                    <img id="imgOverlay" alt="Overlay">
                </div>
            </div>
            <div class="stats">
                <div class="stat spruce">Spruce: <span id="countSpruce">0</span></div>
                <div class="stat pine">Pine: <span id="countPine">0</span></div>
                <div class="stat">Total: <span id="countTotal">0</span></div>
            </div>
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
        const countSpruce = document.getElementById('countSpruce');
        const countPine = document.getElementById('countPine');
        const countTotal = document.getElementById('countTotal');

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
            formData.append('conf', '0.25');
            fetch('/segment', { method: 'POST', body: formData })
                .then(r => r.json())
                .then(data => {
                    loading.style.display = 'none';
                    if (data.error) {
                        showError(data.error);
                    } else {
                        imgOriginal.src = 'data:image/jpeg;base64,' + data.original_b64;
                        imgOverlay.src = data.overlay_b64 ? 'data:image/jpeg;base64,' + data.overlay_b64 : imgOriginal.src;
                        countSpruce.textContent = data.spruce;
                        countPine.textContent = data.pine;
                        countTotal.textContent = data.total;
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
    parser = argparse.ArgumentParser(description="Forest instance segmentation web app (Spruce & Pine)")
    parser.add_argument("--port", type=int, default=5020, help="Port (default 5020)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    args = parser.parse_args()
    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        print("Put best.pt in forest_instance_segmentation/models/best.pt")
        return 1
    print(f"Open http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
