# Forest Bark Analyzer

A web application that analyzes forest images by automatically segmenting bark objects and classifying each one.

## Features

- 🖼️ **Upload Forest Images**: Drag & drop or upload terrestrial forest images
- ✂️ **Automatic Segmentation**: Uses SAM (Segment Anything Model) to find all bark objects
- 🔍 **Individual Classification**: Each segmented bark object is classified as Picea, Pinus, or Other
- 🎨 **Visual Results**: Original image with colored mask outlines and labels for each object
- 📊 **Statistics**: Summary counts of detected objects by class

## How It Works

1. **Image Upload**: User uploads a forest/terrestrial image
2. **Segmentation**: SAM automatically generates masks for all objects in the image
3. **Object Extraction**: Each mask is cropped and extracted separately
4. **Classification**: Each extracted object is classified using the trained YOLOv8 model
5. **Visualization**: Results are displayed with:
   - Original image
   - Colored mask outlines (Green for Picea, Light Green for Pinus, Gray for Other)
   - Labels showing class name and confidence percentage
   - Statistics summary

## Installation

### Required Dependencies

**Option 1: SAM2 (Recommended - supports text prompts like "bark")**
```bash
# Install SAM2 (supports text prompts)
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2
pip install -e .

# Install other dependencies
pip install flask opencv-python pillow numpy torch ultralytics
```

**Option 2: SAM v1 (Fallback - no text prompts, uses automatic mask generation)**
```bash
# Install SAM v1
pip install git+https://github.com/facebookresearch/segment-anything.git

# Install other dependencies
pip install flask opencv-python pillow numpy torch ultralytics
```

### Model Weights

**For SAM2:**
- Download SAM2 checkpoints from: https://github.com/facebookresearch/segment-anything-2
- Place in `Archive/Preprocessing scripts/` (e.g., `sam2_hiera_large.pt`)

**For SAM v1 (fallback):**
The app will look for SAM model weights in:
- `Archive/Preprocessing scripts/sam_vit_h_4b8939.pth` (preferred - more accurate)
- `Archive/Preprocessing scripts/sam_vit_b_01ec64.pth` (faster - smaller model)

If these files don't exist, you can download them:
- **ViT-H (Huge)**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth (~2.4GB)
- **ViT-B (Base)**: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth (~375MB)

Place the downloaded file in `Archive/Preprocessing scripts/`

## Usage

### Start the Server

```bash
python3 forest_bark_analyzer.py
```

By default, it runs on port 5002. You can specify a different port:

```bash
python3 forest_bark_analyzer.py --port 8080
```

### Access the Interface

Open your browser and navigate to:
```
http://localhost:5002
```

### Process an Image

1. **Upload**: Drag & drop or click to upload a forest image
2. **Wait**: Processing may take 30 seconds to several minutes depending on:
   - Image size
   - Number of objects found
   - Hardware (CPU vs GPU)
3. **View Results**: See the annotated image with segmented objects and classifications

## Output

The result image shows:
- **Green outlines**: Picea objects
- **Light green outlines**: Pinus objects  
- **Gray outlines**: Other/unclassified objects
- **Labels**: Each object has a label showing:
  - Class name (Picea, Pinus, or Other)
  - Confidence percentage

Statistics panel shows:
- Total bark objects found
- Count by class (Picea, Pinus, Other)

## Technical Details

### Segmentation Method

The app tries to use **SAM2** first (which supports text prompts for "bark"), with automatic fallback to **SAM v1** if SAM2 is not available.

**SAM2 (Preferred):**
- Uses text prompt "bark" to segment only bark objects
- More targeted segmentation
- Better for filtering specific objects

**SAM v1 (Fallback):**
- Uses automatic mask generation
- Finds all objects in the image
- Filters out very small masks (< 1% of image area)
- Processes each mask separately

### Classification

- Uses the trained YOLOv8 classifier model
- Confidence threshold: **95%** (objects below this threshold are classified as "Other")
- Each segmented object is cropped, padded, and classified individually

### Performance

- **First request**: Models load (SAM + Classifier) - takes 30-60 seconds
- **Processing time**: Depends on image size and number of objects
  - Small images (< 2MP): 10-30 seconds
  - Large images (> 10MP): 1-5 minutes
  - More objects = longer processing time

## Troubleshooting

**"segment_anything library not found"**
- Install: `pip install git+https://github.com/facebookresearch/segment-anything.git`

**"SAM model checkpoint not found"**
- Download SAM weights and place in `Archive/Preprocessing scripts/`
- See Installation section above

**"Classifier model not found"**
- Make sure you've trained the model first: `python3 train_yolov8_classifier.py`

**"No bark objects found"**
- Try a different image with clearer bark objects
- Make sure the image is a forest/terrestrial scene with visible bark

**Processing is very slow**
- Use smaller images (< 5MP recommended)
- Consider using ViT-B model instead of ViT-H for faster processing
- GPU acceleration helps significantly if available

**Memory errors**
- Reduce image size before uploading
- Use ViT-B model instead of ViT-H (uses less memory)

## Command Line Options

```bash
python3 forest_bark_analyzer.py --help
```

**Options:**
- `--port`: Port to run the server on (default: 5002)
- `--host`: Host to bind to (default: 0.0.0.0)

## Example Workflow

```bash
# Terminal: Start the analyzer
python3 forest_bark_analyzer.py

# Browser: Open http://localhost:5002
# Then upload a forest image and wait for results!
```

## Limitations

- Uses SAM automatic mask generation (finds all objects, not just bark)
- Very large images may take a long time to process
- Requires significant RAM (especially with ViT-H model)
- Processing is CPU-intensive (GPU recommended for faster processing)

## Future Enhancements

Potential improvements:
- Integration with Grounding DINO for text-prompt based "bark" filtering
- SAM2 support (when available) for better text prompts
- Batch processing of multiple images
- Export results as JSON/CSV
- Interactive mask refinement

