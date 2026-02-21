# bark_classifier_web.py
# Web interface for bark image classification
# Drag & drop or upload images to classify as Picea, Pinus, or Other

from flask import Flask, render_template_string, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from pathlib import Path
import os
from ultralytics import YOLO
import base64
from PIL import Image
import io

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'JPEG', 'JPG', 'PNG'}

# Create uploads directory if it doesn't exist
Path(app.config['UPLOAD_FOLDER']).mkdir(exist_ok=True)

# Load model (will be loaded on first request)
MODEL_PATH = '../data/models/yolov8_results/bark_classifier/weights/best.pt'
model = None

def load_model():
    """Load the YOLOv8 model"""
    global model
    if model is None:
        print(f"Loading model from {MODEL_PATH}...")
        model = YOLO(MODEL_PATH)
        print("Model loaded successfully!")
    return model

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(image_path, confidence_threshold=0.95):
    """Predict class for an image"""
    model = load_model()
    
    # Run prediction
    results = model(str(image_path), verbose=False)
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
        
        # Add "Other" to probabilities for display
        picea_conf = all_probs.get("Picea", 0.0)
        pinus_conf = all_probs.get("Pinus", 0.0)
        max_conf = max(picea_conf, pinus_conf)
        
        if max_conf < confidence_threshold:
            all_probs["Other"] = 1.0 - max_conf
        else:
            all_probs["Other"] = 0.0
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': all_probs,
            'original_prediction': predicted_class_name
        }
    return None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bark Image Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #2d5016 0%, #3d6b1f 50%, #4a7c28 100%);
            min-height: 100vh;
            padding: 20px;
            color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        header {
            text-align: center;
            margin-bottom: 40px;
            padding: 20px;
        }
        
        h1 {
            font-size: 3em;
            color: #e8f5e9;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2em;
            color: #c8e6c9;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-top: 30px;
        }
        
        @media (max-width: 968px) {
            .main-content {
                grid-template-columns: 1fr;
            }
        }
        
        .upload-section {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 30px;
            border: 2px dashed rgba(255, 255, 255, 0.3);
        }
        
        .drop-zone {
            border: 3px dashed #a5d6a7;
            border-radius: 12px;
            padding: 60px 20px;
            text-align: center;
            background: rgba(165, 214, 167, 0.1);
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
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
            margin-bottom: 20px;
        }
        
        .drop-text {
            font-size: 1.3em;
            color: #e8f5e9;
            margin-bottom: 10px;
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
        }
        
        .preview-image {
            width: 100%;
            max-height: 400px;
            object-fit: contain;
            border-radius: 10px;
            margin-bottom: 20px;
            background: rgba(0, 0, 0, 0.2);
            padding: 10px;
        }
        
        .prediction-result {
            margin-top: 20px;
        }
        
        .predicted-class {
            font-size: 2em;
            font-weight: bold;
            text-align: center;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        
        .class-picea {
            background: linear-gradient(135deg, #8bc34a 0%, #7cb342 100%);
            color: #1b5e20;
        }
        
        .class-pinus {
            background: linear-gradient(135deg, #689f38 0%, #558b2f 100%);
            color: #e8f5e9;
        }
        
        .class-other {
            background: linear-gradient(135deg, #9e9e9e 0%, #757575 100%);
            color: #f5f5f5;
        }
        
        .confidence {
            text-align: center;
            font-size: 1.2em;
            color: #c8e6c9;
            margin-bottom: 20px;
        }
        
        .probabilities {
            margin-top: 20px;
        }
        
        .probability-bar {
            margin-bottom: 15px;
        }
        
        .probability-label {
            display: flex;
            justify-content: space-between;
            margin-bottom: 5px;
            font-size: 0.95em;
        }
        
        .probability-name {
            font-weight: 600;
            color: #e8f5e9;
        }
        
        .probability-value {
            color: #c8e6c9;
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 15px;
            overflow: hidden;
            position: relative;
        }
        
        .progress-fill {
            height: 100%;
            border-radius: 15px;
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .progress-picea {
            background: linear-gradient(90deg, #8bc34a 0%, #7cb342 100%);
        }
        
        .progress-pinus {
            background: linear-gradient(90deg, #689f38 0%, #558b2f 100%);
        }
        
        .progress-other {
            background: linear-gradient(90deg, #9e9e9e 0%, #757575 100%);
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
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
            padding: 10px 25px;
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
            <h1>🌲 Bark Image Classifier</h1>
            <p class="subtitle">Upload or drag & drop images to classify as Picea, Pinus, or Other</p>
        </header>
        
        <div class="main-content">
            <div class="upload-section">
                <div class="drop-zone" id="dropZone">
                    <div class="drop-icon">📸</div>
                    <div class="drop-text">Drag & Drop Image Here</div>
                    <div class="drop-subtext">or click to select</div>
                    <label for="fileInput" class="file-input-label">
                        Choose File
                    </label>
                    <input type="file" id="fileInput" accept="image/*">
                </div>
                
                <div class="loading" id="loading">
                    <div class="spinner"></div>
                    <p>Classifying image...</p>
                </div>
                
                <div class="error" id="error"></div>
                
                <div class="info-box">
                    <strong>ℹ️ Supported formats:</strong> JPG, JPEG, PNG<br>
                    <strong>📏 Max size:</strong> 16MB<br>
                    <strong>🎯 Confidence threshold:</strong> 95% (images below this will be classified as "Other")
                </div>
            </div>
            
            <div class="results-section" id="resultsSection" style="display: none;">
                <img id="previewImage" class="preview-image" alt="Preview">
                
                <div class="prediction-result">
                    <div id="predictedClass" class="predicted-class"></div>
                    <div id="confidence" class="confidence"></div>
                    
                    <div class="probabilities">
                        <div id="probabilitiesContainer"></div>
                    </div>
                </div>
                
                <button class="clear-btn" onclick="clearResults()">Clear & Upload New Image</button>
            </div>
        </div>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const resultsSection = document.getElementById('resultsSection');
        const previewImage = document.getElementById('previewImage');
        const loading = document.getElementById('loading');
        const error = document.getElementById('error');
        const predictedClassDiv = document.getElementById('predictedClass');
        const confidenceDiv = document.getElementById('confidence');
        const probabilitiesContainer = document.getElementById('probabilitiesContainer');
        
        // Drag and drop handlers
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
            
            if (file.size > 16 * 1024 * 1024) {
                showError('File size exceeds 16MB limit');
                return;
            }
            
            // Show preview
            const reader = new FileReader();
            reader.onload = (e) => {
                previewImage.src = e.target.result;
                resultsSection.style.display = 'block';
                loading.style.display = 'block';
                error.style.display = 'none';
                predictedClassDiv.innerHTML = '';
                confidenceDiv.innerHTML = '';
                probabilitiesContainer.innerHTML = '';
                document.querySelector('.clear-btn').style.display = 'none';
            };
            reader.readAsDataURL(file);
            
            // Upload and classify
            const formData = new FormData();
            formData.append('file', file);
            
            fetch('/classify', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                loading.style.display = 'none';
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data);
                }
            })
            .catch(err => {
                loading.style.display = 'none';
                showError('Error classifying image: ' + err.message);
            });
        }
        
        function displayResults(data) {
            const { predicted_class, confidence, probabilities } = data;
            
            // Display predicted class
            predictedClassDiv.className = 'predicted-class class-' + predicted_class.toLowerCase();
            predictedClassDiv.textContent = predicted_class.toUpperCase();
            
            // Display confidence
            confidenceDiv.textContent = `Confidence: ${(confidence * 100).toFixed(2)}%`;
            
            // Display probabilities
            probabilitiesContainer.innerHTML = '';
            const sortedProbs = Object.entries(probabilities).sort((a, b) => b[1] - a[1]);
            
            sortedProbs.forEach(([className, prob]) => {
                const probBar = document.createElement('div');
                probBar.className = 'probability-bar';
                
                const label = document.createElement('div');
                label.className = 'probability-label';
                label.innerHTML = `
                    <span class="probability-name">${className}</span>
                    <span class="probability-value">${(prob * 100).toFixed(2)}%</span>
                `;
                
                const progress = document.createElement('div');
                progress.className = 'progress-bar';
                const fill = document.createElement('div');
                fill.className = 'progress-fill progress-' + className.toLowerCase();
                fill.style.width = (prob * 100) + '%';
                fill.textContent = (prob * 100).toFixed(1) + '%';
                progress.appendChild(fill);
                
                probBar.appendChild(label);
                probBar.appendChild(progress);
                probabilitiesContainer.appendChild(probBar);
            });
            
            document.querySelector('.clear-btn').style.display = 'block';
        }
        
        function showError(message) {
            error.textContent = message;
            error.style.display = 'block';
        }
        
        function clearResults() {
            resultsSection.style.display = 'none';
            fileInput.value = '';
            error.style.display = 'none';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Render the main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/classify', methods=['POST'])
def classify():
    """Handle image upload and classification"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Classify image
            result = predict_image(filepath, confidence_threshold=0.95)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if result:
                return jsonify(result)
            else:
                return jsonify({'error': 'Failed to classify image'}), 500
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload JPG, JPEG, or PNG'}), 400

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Bark Image Classifier Web Interface')
    parser.add_argument('--port', type=int, default=5001,
                        help='Port to run the server on (default: 5001)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--model', type=str, default='./yolov8_results/bark_classifier/weights/best.pt',
                        help='Path to model file')
    
    args = parser.parse_args()
    
    global MODEL_PATH
    MODEL_PATH = args.model
    
    print("=" * 70)
    print("🌲 Bark Image Classifier Web Interface")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Server running on http://{args.host}:{args.port}")
    print("=" * 70)
    print("\nOpen your browser and navigate to the URL above")
    print("to start classifying bark images!")
    print("=" * 70)
    
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == '__main__':
    main()

