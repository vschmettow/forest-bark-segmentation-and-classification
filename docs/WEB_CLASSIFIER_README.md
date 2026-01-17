# Bark Image Classifier - Web Interface

A beautiful, forest-themed web interface for classifying bark images using the trained YOLOv8 model.

## Features

- 🎨 **Beautiful UI**: Green, forest-themed interface inspired by bark and nature
- 📸 **Drag & Drop**: Easy image upload with drag & drop support
- 🔍 **Real-time Classification**: Instant results showing predicted class and confidence
- 📊 **Visual Probabilities**: See probability bars for Picea, Pinus, and Other
- 🎯 **95% Confidence Threshold**: Images below 95% confidence are classified as "Other"

## Installation

Make sure Flask is installed:

```bash
python3 -m pip install flask
```

## Usage

### Start the Web Server

```bash
python3 bark_classifier_web.py
```

By default, it runs on port 5001. You can specify a different port:

```bash
python3 bark_classifier_web.py --port 8080
```

### Access the Interface

Open your browser and navigate to:
```
http://localhost:5001
```

### Classify Images

1. **Drag & Drop**: Drag an image file onto the upload area
2. **Click to Upload**: Click the upload area or "Choose File" button to select an image
3. **View Results**: See the classification results with:
   - Predicted class (Picea, Pinus, or Other)
   - Confidence percentage
   - Probability bars for each class

## Supported Formats

- JPG/JPEG
- PNG
- Maximum file size: 16MB

## Model Location

The web interface uses the trained model from:
```
./yolov8_results/bark_classifier/weights/best.pt
```

You can specify a different model:

```bash
python3 bark_classifier_web.py --model path/to/your/model.pt
```

## Command Line Options

```bash
python3 bark_classifier_web.py --help
```

**Options:**
- `--port`: Port to run the server on (default: 5001)
- `--host`: Host to bind to (default: 0.0.0.0 - accessible from network)
- `--model`: Path to model file (default: ./yolov8_results/bark_classifier/weights/best.pt)

## Troubleshooting

**"ModuleNotFoundError: No module named 'flask'"**
- Install Flask: `python3 -m pip install flask`

**"Address already in use"**
- Port 5001 is in use. Use a different port: `--port 5002`

**Model not found error**
- Make sure the model file exists at the specified path
- If you haven't trained yet, train first: `python3 train_yolov8_classifier.py`

**Classification fails**
- Check that the image file is valid and in a supported format
- Make sure the file size is under 16MB

## Example Workflow

```bash
# Terminal 1: Start the web server
python3 bark_classifier_web.py

# Browser: Open http://localhost:5001
# Then drag & drop or upload bark images to classify them!
```

## Interface Features

- **Green Theme**: Colors inspired by forests and bark
- **Responsive Design**: Works on desktop and tablet
- **Visual Feedback**: Loading spinner during classification
- **Error Handling**: Clear error messages for issues
- **Probability Visualization**: See confidence for all three classes

