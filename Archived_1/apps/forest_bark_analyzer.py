# forest_bark_analyzer.py
# Web app for analyzing forest images: segments bark objects and classifies them

from flask import Flask, render_template_string, request, jsonify
from werkzeug.utils import secure_filename
from pathlib import Path
import os
import cv2
import numpy as np
import torch
from PIL import Image
import base64
import io
from ultralytics import YOLO
import tempfile
import uuid
import threading

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'JPEG', 'JPG', 'PNG'}

Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Global models
sam_predictor = None
classifier_model = None
device = None

# Progress tracking
progress_dict = {}
progress_lock = threading.Lock()

def update_progress(task_id, step, message, percentage=None):
    """Update progress for a task"""
    with progress_lock:
        progress_dict[task_id] = {
            'step': step,
            'message': message,
            'percentage': percentage
        }

def get_progress(task_id):
    """Get current progress for a task"""
    with progress_lock:
        return progress_dict.get(task_id, {'step': 0, 'message': 'Starting...', 'percentage': 0})

def clear_progress(task_id):
    """Clear progress for a task"""
    with progress_lock:
        if task_id in progress_dict:
            del progress_dict[task_id]

# Progress tracking
import uuid
import threading
progress_dict = {}
progress_lock = threading.Lock()

def setup_sam2_model():
    """Setup SAM2 model for segmentation"""
    global sam_predictor, device
    
    if sam_predictor is None:
        try:
            # Try SAM2 first
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                
                print("Loading SAM2 model...")
                
                # SAM2 requires config file - check if SAM2 repo is available
                # SAM2 checkpoints directory structure
                sam2_checkpoint_dir = "../Archive/Preprocessing scripts/"
                sam2_repo_dir = None
                
                # Check if SAM2 repo exists nearby (common location)
                possible_repos = [
                    "../segment-anything-2",
                    "./segment-anything-2",
                    os.path.expanduser("~/segment-anything-2")
                ]
                
                for repo_path in possible_repos:
                    config_dir = os.path.join(repo_path, "configs")
                    if os.path.exists(config_dir):
                        sam2_repo_dir = repo_path
                        break
                
                # Check for SAM2 checkpoint files
                sam2_checkpoint = None
                model_cfg = None
                
                checkpoint_names = [
                    ("sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
                    ("sam2_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
                    ("sam2.1_hiera_base.pt", "configs/sam2.1/sam2.1_hiera_b.yaml"),
                    ("sam2_hiera_b.pt", "configs/sam2.1/sam2.1_hiera_b.yaml"),
                ]
                
                for ckpt_name, cfg_path in checkpoint_names:
                    ckpt_path = os.path.join(sam2_checkpoint_dir, ckpt_name)
                    if os.path.exists(ckpt_path):
                        sam2_checkpoint = ckpt_path
                        if sam2_repo_dir:
                            model_cfg = os.path.join(sam2_repo_dir, cfg_path)
                            if os.path.exists(model_cfg):
                                break
                        # If repo not found, try relative path
                        model_cfg = cfg_path
                        break
                
                if not sam2_checkpoint or not model_cfg:
                    print("SAM2 checkpoint not found. Falling back to SAM v1...")
                    raise ImportError("SAM2 not available, using SAM v1")
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")
                print(f"SAM2 checkpoint: {sam2_checkpoint}")
                print(f"SAM2 config: {model_cfg}")
                
                # Build SAM2 model (API: build_sam2(config_file, checkpoint_path))
                sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
                sam_predictor = SAM2ImagePredictor(sam2_model)
                
                print("✓ SAM2 model loaded successfully!")
                return sam_predictor
                
            except (ImportError, FileNotFoundError) as e:
                # Fallback to SAM v1
                print(f"SAM2 not available ({e}). Using SAM v1 instead...")
                from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
                
                # Check for SAM v1 model weights
                sam_checkpoint_base = "../Archive/Preprocessing scripts/sam_vit_b_01ec64.pth"
                sam_checkpoint_huge = "../Archive/Preprocessing scripts/sam_vit_h_4b8939.pth"
                
                if os.path.exists(sam_checkpoint_huge):
                    sam_checkpoint = sam_checkpoint_huge
                    model_type = "vit_h"
                elif os.path.exists(sam_checkpoint_base):
                    sam_checkpoint = sam_checkpoint_base
                    model_type = "vit_b"
                else:
                    raise FileNotFoundError("Neither SAM2 nor SAM v1 model checkpoints found. Please ensure model weights are available.")
                
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(f"Using device: {device}")
                print(f"Loading SAM v1 model: {model_type} from {sam_checkpoint}")
                
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(device=device)
                sam_predictor = SamAutomaticMaskGenerator(sam)
                
                print("✓ SAM v1 model loaded successfully!")
                
        except ImportError as e:
            raise ImportError(
                "Neither SAM2 nor SAM v1 libraries found. Install one of:\n"
                "  SAM2: git clone https://github.com/facebookresearch/segment-anything-2.git\n"
                "  SAM v1: pip install git+https://github.com/facebookresearch/segment-anything.git"
            )
        except Exception as e:
            raise Exception(f"Error loading segmentation model: {e}")
    
    return sam_predictor

def setup_classifier_model():
    """Setup YOLOv8 classifier model"""
    global classifier_model
    
    if classifier_model is None:
        model_path = '../data/models/yolov8_results/bark_classifier/weights/best.pt'
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Classifier model not found at {model_path}")
        
        print(f"Loading classifier model from {model_path}...")
        classifier_model = YOLO(model_path)
        print("✓ Classifier model loaded successfully!")
    
    return classifier_model

def update_progress(task_id, step, message, percentage=None):
    """Update progress for a task"""
    with progress_lock:
        progress_dict[task_id] = {
            'step': step,
            'message': message,
            'percentage': percentage
        }

def get_progress(task_id):
    """Get current progress for a task"""
    with progress_lock:
        return progress_dict.get(task_id, {'step': 0, 'message': 'Starting...', 'percentage': 0})

def segment_bark_objects(image_path, text_prompt="bark", task_id=None):
    """
    Segment bark objects from image using SAM2 (or SAM v1 fallback)
    
    Note: SAM2 (like SAM v1) uses point/box prompts, not text prompts directly.
    For text-prompt based segmentation (like "bark"), you'd need Grounding DINO
    to first detect regions, then SAM2 to segment them. For now, we use
    automatic mask generation which finds all objects.
    """
    sam_predictor = setup_sam2_model()
    
    # Load image
    if task_id:
        update_progress(task_id, 1, "Loading image...", 5)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Check if using SAM2 or SAM v1
    predictor_type = type(sam_predictor).__name__
    
    if task_id:
        update_progress(task_id, 1, "Segmenting objects with SAM...", 10)
    
    if "SAM2" in predictor_type:
        # SAM2 - use automatic mask generation
        # Note: SAM2 doesn't have built-in text prompts - would need Grounding DINO for that
        print("Using SAM2 automatic mask generation...")
        try:
            # SAM2 should have automatic mask generator similar to SAM v1
            # Check if predictor itself can generate, or if we need a separate generator
            if hasattr(sam_predictor, 'generate'):
                masks = sam_predictor.generate(image_rgb)
            else:
                # Try to create automatic mask generator from the model
                from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
                mask_generator = SAM2AutomaticMaskGenerator(sam_predictor.model)
                masks = mask_generator.generate(image_rgb)
        except Exception as e:
            print(f"SAM2 automatic mask generation failed ({e})")
            print("Note: SAM2 API may differ. Falling back to basic approach...")
            raise e
    else:
        # SAM v1 - automatic mask generation
        print("Using SAM v1 automatic mask generation...")
        masks = sam_predictor.generate(image_rgb)
    
    # Filter masks by area (remove very small masks)
    min_area = image_rgb.shape[0] * image_rgb.shape[1] * 0.01  # At least 1% of image
    filtered_masks = [m for m in masks if m['area'] > min_area]
    
    if task_id:
        update_progress(task_id, 2, f"Found {len(filtered_masks)} objects to classify...", 30)
    print(f"Found {len(masks)} total masks, {len(filtered_masks)} after filtering")
    
    return image_rgb, filtered_masks

def crop_mask_region(image, mask):
    """Crop image tightly to mask region - only bark visible"""
    mask_array = mask['segmentation']
    h_img, w_img = image.shape[:2]
    
    # Find tight bounding box from mask (not the SAM bbox which might be loose)
    # Get all non-zero pixels (the mask region)
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        # Fallback to original bbox if mask is empty
        bbox = mask['bbox']
        x, y, w, h = bbox
        x, y = int(x), int(y)
        w, h = int(w), int(h)
    else:
        # Get tight bounding box
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        
        # Convert to bbox format [x, y, w, h]
        x, y = x_min, y_min
        w = x_max - x_min + 1
        h = y_max - y_min + 1
    
    # Minimal padding (just 2-3 pixels to avoid edge artifacts)
    padding = 3
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(w_img, x + w + padding)
    y_end = min(h_img, y + h + padding)
    
    # Crop image to tight region
    cropped = image[y_start:y_end, x_start:x_end]
    
    # Crop mask to same region
    mask_cropped = mask_array[y_start:y_end, x_start:x_end]
    
    # Apply mask: keep only mask region, white background
    # This ensures the classifier only sees bark, not surrounding background
    result = np.ones_like(cropped) * 255
    mask_3d = np.stack([mask_cropped] * 3, axis=2).astype(bool)
    result[mask_3d] = cropped[mask_3d]
    
    # YOLOv8 classifier will automatically resize to 600x600 (training size)
    # We just need to ensure the crop is tight to show only bark
    
    return result, (x_start, y_start, x_end - x_start, y_end - y_start)

def classify_bark_image(image_array, confidence_threshold=0.95, task_id=None, obj_num=None, total_objs=None):
    """Classify a bark image (numpy array) using YOLOv8"""
    if task_id and obj_num is not None and total_objs is not None:
        # Update progress: classification phase starts at 30%, goes to 90%
        progress_pct = 30 + int((obj_num / total_objs) * 60)
        update_progress(task_id, 3, f"Classifying object {obj_num}/{total_objs}...", progress_pct)
    
    classifier_model = setup_classifier_model()
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(image_array)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        tmp_path = tmp_file.name
        pil_image.save(tmp_path, 'JPEG')
    
    try:
        # Run prediction
        results = classifier_model(tmp_path, verbose=False)
        result = results[0]
        
        if hasattr(result, 'probs'):
            probs = result.probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            class_names = result.names
            
            # Get all probabilities
            all_probs = {}
            for i, class_name in class_names.items():
                all_probs[class_name] = float(probs.data[i])
            
            # Determine final class based on confidence threshold
            predicted_class_name = class_names[top1_idx]
            confidence = top1_conf
            
            # If confidence is below threshold, classify as "Other"
            if confidence < confidence_threshold:
                predicted_class = "Other"
            else:
                predicted_class = predicted_class_name
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'probabilities': all_probs
            }
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    return None

def draw_mask_overlay(image, masks, classifications):
    """Draw mask outlines with distinct colors for each class (no labels)"""
    result = image.copy()
    
    # Distinct colors for each class (BGR format for OpenCV)
    colors = {
        'Picea': (80, 175, 76),      # Bright green (BGR)
        'Pinus': (33, 150, 243),     # Blue (BGR) - more distinct from green
        'Other': (0, 0, 255)         # Red (BGR)
    }
    
    for mask, classification in zip(masks, classifications):
        mask_array = mask['segmentation']
        
        # Get color based on classification
        class_name = classification['predicted_class']
        color = colors.get(class_name, (158, 158, 158))
        
        # Draw mask outline (thicker line for better visibility)
        contours, _ = cv2.findContours(mask_array.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 4)
    
    return result

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Forest Bark Analyzer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
            min-height: 100vh;
            padding: 20px;
            color: #f5f5f5;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
        }
        
        h1 {
            font-size: 2.5em;
            color: #e8f5e9;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.1em;
            color: #c8e6c9;
            opacity: 0.9;
        }
        
        .upload-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 2px dashed rgba(255, 255, 255, 0.3);
        }
        
        .drop-zone {
            border: 3px dashed #a5d6a7;
            border-radius: 12px;
            padding: 50px 20px;
            text-align: center;
            background: rgba(165, 214, 167, 0.1);
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .drop-zone:hover {
            border-color: #81c784;
            background: rgba(165, 214, 167, 0.2);
        }
        
        .drop-zone.dragover {
            border-color: #66bb6a;
            background: rgba(165, 214, 167, 0.3);
            transform: scale(1.02);
        }
        
        .drop-icon {
            font-size: 4em;
            margin-bottom: 15px;
        }
        
        .drop-text {
            font-size: 1.2em;
            color: #e8f5e9;
            margin-bottom: 8px;
        }
        
        .drop-subtext {
            font-size: 0.9em;
            color: #c8e6c9;
            opacity: 0.8;
        }
        
        input[type="file"] {
            display: none;
        }
        
        .file-input-label {
            display: inline-block;
            margin-top: 20px;
            padding: 12px 30px;
            background: #66bb6a;
            color: white;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
        }
        
        .file-input-label:hover {
            background: #4caf50;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .results-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            border: 2px solid rgba(255, 255, 255, 0.2);
            display: none;
        }
        
        .result-image {
            width: 100%;
            max-width: 100%;
            border-radius: 10px;
            margin-bottom: 20px;
            background: rgba(0, 0, 0, 0.2);
            padding: 10px;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        
        .stat-card {
            background: rgba(255, 255, 255, 0.1);
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #a5d6a7;
        }
        
        .stat-label {
            font-size: 0.9em;
            color: #c8e6c9;
            margin-top: 5px;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        #progressBar {
            min-width: 50px;
        }
        
        .spinner {
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top: 4px solid #66bb6a;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            display: none;
            background: rgba(244, 67, 54, 0.2);
            border: 2px solid #f44336;
            border-radius: 10px;
            padding: 15px;
            color: #ffcdd2;
            margin-top: 20px;
        }
        
        .info-box {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            padding: 15px;
            margin-top: 20px;
            font-size: 0.9em;
            color: #c8e6c9;
        }
        
        .clear-btn {
            display: none;
            margin-top: 20px;
            padding: 12px 30px;
            background: rgba(244, 67, 54, 0.3);
            color: #ffcdd2;
            border: 2px solid #f44336;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            transition: all 0.3s ease;
            width: 100%;
        }
        
        .clear-btn:hover {
            background: rgba(244, 67, 54, 0.5);
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🌲 Forest Bark Analyzer</h1>
            <p class="subtitle">Upload a forest image to segment and classify bark objects</p>
        </header>
        
        <div class="upload-section">
            <div class="drop-zone" id="dropZone">
                <div class="drop-icon">📸</div>
                <div class="drop-text">Drag & Drop Forest Image Here</div>
                <div class="drop-subtext">or click to select</div>
                <label for="fileInput" class="file-input-label">
                    Choose File
                </label>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p id="loadingMessage">Processing image... This may take a minute</p>
                <div style="margin: 20px auto; width: 80%; max-width: 500px;">
                    <div style="background: rgba(255,255,255,0.2); border-radius: 10px; padding: 3px;">
                        <div id="progressBar" style="background: linear-gradient(90deg, #66bb6a 0%, #4caf50 100%); height: 30px; border-radius: 8px; width: 0%; transition: width 0.3s ease; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 0.9em;">
                            <span id="progressText">0%</span>
                        </div>
                    </div>
                    <p id="progressMessage" style="font-size: 0.9em; opacity: 0.8; margin-top: 10px; text-align: center;">
                        Starting...
                    </p>
                </div>
            </div>
            
            <div class="error" id="error"></div>
            
            <div class="info-box">
                <strong>ℹ️ How it works:</strong><br>
                1. Upload a forest/terrestrial image<br>
                2. SAM segments all objects automatically<br>
                3. Each object is classified as Picea, Pinus, or Other<br>
                4. Results show colored mask outlines: <span style="color: #4caf50;">●</span> Picea (Green), <span style="color: #2196f3;">●</span> Pinus (Blue), <span style="color: #f44336;">●</span> Other (Red)<br><br>
                <strong>📏 Max size:</strong> 32MB | <strong>🎯 Confidence threshold:</strong> 95%
            </div>
        </div>
        
        <div class="results-section" id="resultsSection">
            <img id="resultImage" class="result-image" alt="Analysis result">
            
            <div class="stats" id="statsContainer"></div>
            
            <button class="clear-btn" onclick="clearResults()">Analyze New Image</button>
        </div>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const resultsSection = document.getElementById('resultsSection');
        const resultImage = document.getElementById('resultImage');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const statsContainer = document.getElementById('statsContainer');
        const progressBar = document.getElementById('progressBar');
        const progressText = document.getElementById('progressText');
        const progressMessage = document.getElementById('progressMessage');
        const loadingMessage = document.getElementById('loadingMessage');
        
        let currentTaskId = null;
        let progressInterval = null;
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('dragover');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('dragover');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFile(files[0]);
            }
        });
        
        dropZone.addEventListener('click', () => {
            fileInput.click();
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFile(e.target.files[0]);
            }
        });
        
        function handleFile(file) {
            if (!file.type.startsWith('image/')) {
                showError('Please upload an image file (JPG, PNG, etc.)');
                return;
            }
            
            if (file.size > 32 * 1024 * 1024) {
                showError('File size exceeds 32MB limit');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', file);
            
            resultsSection.style.display = 'none';
            loading.style.display = 'block';
            error.style.display = 'none';
            
            // Generate task ID for progress tracking
            currentTaskId = Math.random().toString(36).substring(7);
            formData.append('task_id', currentTaskId);
            
            // Reset progress
            progressBar.style.width = '0%';
            progressText.textContent = '0%';
            progressMessage.textContent = 'Starting...';
            loadingMessage.textContent = 'Processing image...';
            
            // Start polling for progress
            startProgressPolling(currentTaskId);
            
            fetch('/analyze', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                stopProgressPolling();
                loading.style.display = 'none';
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(err => {
                stopProgressPolling();
                loading.style.display = 'none';
                showError('Error processing image: ' + err.message);
            });
        }
        
        function displayResults(data) {
            resultImage.src = 'data:image/png;base64,' + data.result_image;
            
            // Display statistics
            const stats = data.stats;
            statsContainer.innerHTML = `
                <div class="stat-card">
                    <div class="stat-value">${stats.total_objects}</div>
                    <div class="stat-label">Bark Objects Found</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.picea_count}</div>
                    <div class="stat-label">Picea</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.pinus_count}</div>
                    <div class="stat-label">Pinus</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">${stats.other_count}</div>
                    <div class="stat-label">Other</div>
                </div>
            `;
            
            resultsSection.style.display = 'block';
            document.querySelector('.clear-btn').style.display = 'block';
        }
        
        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
        }
        
        function startProgressPolling(taskId) {
            // Poll every 500ms for progress updates
            progressInterval = setInterval(() => {
                fetch(`/progress/${taskId}`)
                    .then(response => response.json())
                    .then(data => {
                        if (data.percentage !== undefined) {
                            progressBar.style.width = data.percentage + '%';
                            progressText.textContent = data.percentage + '%';
                        }
                        if (data.message) {
                            progressMessage.textContent = data.message;
                        }
                    })
                    .catch(err => {
                        console.error('Progress polling error:', err);
                    });
            }, 500);
        }
        
        function stopProgressPolling() {
            if (progressInterval) {
                clearInterval(progressInterval);
                progressInterval = null;
            }
        }
        
        function clearResults() {
            resultsSection.style.display = 'none';
            fileInput.value = '';
            error.style.display = 'none';
            stopProgressPolling();
            currentTaskId = null;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/progress/<task_id>')
def get_progress_endpoint(task_id):
    """Get progress for a task"""
    progress = get_progress(task_id)
    return jsonify(progress)

@app.route('/analyze', methods=['POST'])
def analyze():
    """Handle image upload, segmentation, and classification"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    task_id = request.form.get('task_id', str(uuid.uuid4()))
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not (file and allowed_file(file.filename)):
        return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG'}), 400
    
    try:
        update_progress(task_id, 0, "Loading image...", 5)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Step 1: Segment bark objects using SAM2 with "bark" prompt
        print("Step 1: Segmenting bark objects with SAM2...")
        image_rgb, masks = segment_bark_objects(filepath, text_prompt="bark", task_id=task_id)
        print(f"Found {len(masks)} bark objects")
        
        if len(masks) == 0:
            os.remove(filepath)
            clear_progress(task_id)
            return jsonify({'error': 'No bark objects found in the image'}), 400
        
        # Step 2: Classify each bark object
        print("Step 2: Classifying each bark object...")
        classifications = []
        total_masks = len(masks)
        for i, mask in enumerate(masks):
            print(f"Classifying object {i+1}/{total_masks}...")
            cropped_image, bbox = crop_mask_region(image_rgb, mask)
            classification = classify_bark_image(
                cropped_image, 
                confidence_threshold=0.95,
                task_id=task_id,
                obj_num=i+1,
                total_objs=total_masks
            )
            if classification:
                classifications.append(classification)
            else:
                classifications.append({'predicted_class': 'Other', 'confidence': 0.0})
        
        # Step 3: Draw results
        update_progress(task_id, 4, "Drawing results...", 90)
        print("Step 3: Drawing results...")
        result_image = draw_mask_overlay(image_rgb, masks, classifications)
        
        # Convert to base64 for web display
        result_pil = Image.fromarray(result_image)
        buffer = io.BytesIO()
        result_pil.save(buffer, format='PNG')
        result_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Calculate statistics
        stats = {
            'total_objects': len(masks),
            'picea_count': sum(1 for c in classifications if c['predicted_class'] == 'Picea'),
            'pinus_count': sum(1 for c in classifications if c['predicted_class'] == 'Pinus'),
            'other_count': sum(1 for c in classifications if c['predicted_class'] == 'Other')
        }
        
        # Clean up
        os.remove(filepath)
        update_progress(task_id, 5, "Complete!", 100)
        clear_progress(task_id)
        
        return jsonify({
            'result_image': result_base64,
            'stats': stats
        })
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        traceback.print_exc()
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        if 'task_id' in locals():
            clear_progress(task_id)
        return jsonify({'error': f'Error processing image: {error_msg}'}), 500

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Forest Bark Analyzer Web Interface')
    parser.add_argument('--port', type=int, default=5002,
                        help='Port to run the server on (default: 5002)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🌲 Forest Bark Analyzer Web Interface")
    print("=" * 70)
    print(f"Server running on http://{args.host}:{args.port}")
    print("=" * 70)
    print("\nModels will be loaded on first request.")
    print("Open your browser and navigate to the URL above!")
    print("=" * 70)
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()

