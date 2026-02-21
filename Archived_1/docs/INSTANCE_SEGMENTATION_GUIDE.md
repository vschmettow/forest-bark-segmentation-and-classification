# Training Your Own Instance Segmentation Model

This guide explains how to train a custom instance segmentation model to replace SAM1/SAM2, and integrate it with your existing YOLOv8 classifier.

## 🎯 Overview

Instead of using SAM1/SAM2 for segmentation, you can train your own model that:
1. **Segments bark objects** in forest images (instance segmentation)
2. **Works with your existing classifier** - each segmented instance is classified by your YOLOv8 model

## 🏗️ Architecture Options

### Option 1: Two-Stage (Recommended)
```
Forest Image
    ↓
[Your Instance Segmentation Model] → Finds bark objects → Generates masks
    ↓
[Object Extraction] → Crops each masked region
    ↓
[YOLOv8 Classifier] → Classifies each object → Picea/Pinus/Other
```

**Advantages:**
- Reuse your existing trained classifier
- Segmentation model only needs to find "bark" (not classify)
- Easier to train (simpler task)

### Option 2: Single-Stage (End-to-End)
```
Forest Image
    ↓
[Instance Segmentation + Classification Model] → Segments AND classifies in one step
```

**Advantages:**
- Faster inference
- Single model to maintain

**Disadvantages:**
- Need to retrain everything
- More complex training data (masks + class labels)

## 🤖 Recommended Models

### 1. **YOLOv8 Instance Segmentation** (Recommended)

**Why YOLOv8-seg:**
- ✅ Same framework as your classifier (Ultralytics)
- ✅ Easy integration
- ✅ Fast inference
- ✅ Good performance
- ✅ Well-documented

**Model sizes:**
- `yolov8n-seg.pt` - Nano (fastest, smallest)
- `yolov8s-seg.pt` - Small (balanced)
- `yolov8m-seg.pt` - Medium
- `yolov8l-seg.pt` - Large
- `yolov8x-seg.pt` - XLarge (best accuracy)

### 2. **Mask R-CNN** (Alternative)

**Why Mask R-CNN:**
- ✅ Very accurate
- ✅ Well-established
- ✅ Good for complex shapes

**Disadvantages:**
- Slower than YOLOv8
- More complex setup
- Requires Detectron2 or mmdetection

### 3. **Detectron2** (Alternative)

**Why Detectron2:**
- ✅ Facebook Research framework
- ✅ Many model options
- ✅ Very flexible

**Disadvantages:**
- More complex setup
- Steeper learning curve

## 📊 Data Preparation

### What You Need

For training instance segmentation, you need:

1. **Images**: Forest images with bark objects
2. **Masks**: Segmentation masks for each bark object
3. **Annotations**: Can be in YOLO format or COCO format

### Option A: Use SAM to Generate Training Data

**Step 1: Generate masks using SAM**
```python
# Use SAM to segment bark objects in your training images
# Save masks for each image
```

**Step 2: Convert to training format**
- Convert SAM masks to YOLO segmentation format
- Or COCO format

### Option B: Manual Annotation

Use annotation tools:
- **LabelMe**: https://github.com/wkentaro/labelme
- **CVAT**: https://cvat.org/
- **Roboflow**: https://roboflow.com/

### Data Format: YOLO Segmentation

YOLOv8-seg expects:
```
images/
  ├── train/
  │   ├── image1.jpg
  │   └── image2.jpg
  └── val/
      ├── image3.jpg
      └── image4.jpg

labels/
  ├── train/
  │   ├── image1.txt  # Segmentation polygons
  │   └── image2.txt
  └── val/
      ├── image3.txt
      └── image4.txt
```

**Label format** (`image1.txt`):
```
0 0.5 0.3 0.6 0.3 0.6 0.7 0.5 0.7  # class_id x1 y1 x2 y2 x3 y3 ... (normalized)
```

**Note**: For bark segmentation, you only need **one class** (bark). Classification happens later.

## 🚀 Training YOLOv8 Instance Segmentation

### Step 1: Prepare Dataset

Create YOLO format dataset with segmentation masks:

```python
# scripts/prepare_segmentation_dataset.py
# Convert SAM masks or annotations to YOLO format
```

### Step 2: Train the Model

```python
from ultralytics import YOLO

# Load YOLOv8 segmentation model
model = YOLO('yolov8s-seg.pt')  # or yolov8n-seg.pt, yolov8m-seg.pt, etc.

# Train
results = model.train(
    data='path/to/dataset.yaml',  # Dataset config
    epochs=100,
    imgsz=640,  # Image size
    batch=16,
    name='bark_segmentation',
    project='./segmentation_results'
)
```

### Step 3: Dataset YAML File

Create `dataset.yaml`:
```yaml
path: ../images/segmentation_dataset
train: train/images
val: val/images

# Classes (only "bark" - classification happens later)
names:
  0: bark
```

## 🔗 Integration with Existing Classifier

### Modified Pipeline

```python
def segment_and_classify(image_path):
    # Step 1: Instance segmentation (your trained model)
    seg_model = YOLO('segmentation_results/bark_segmentation/weights/best.pt')
    seg_results = seg_model(image_path)
    
    # Step 2: Extract masks
    masks = []
    for result in seg_results:
        if result.masks is not None:
            for mask in result.masks.data:
                masks.append(mask.cpu().numpy())
    
    # Step 3: Classify each mask (your existing classifier)
    classifier = YOLO('../data/models/yolov8_results/bark_classifier/weights/best.pt')
    classifications = []
    
    for mask in masks:
        # Crop image to mask
        cropped = crop_mask_region(image, mask)
        
        # Classify
        class_result = classifier(cropped)
        classifications.append(class_result)
    
    return masks, classifications
```

## 📝 Complete Training Script

Here's a complete example script:

```python
# scripts/train_bark_segmentation.py
from ultralytics import YOLO
from pathlib import Path

def train_bark_segmentation(
    data_yaml='dataset.yaml',
    model_size='s',  # n, s, m, l, x
    epochs=100,
    imgsz=640,
    batch=16,
    project='./segmentation_results'
):
    """
    Train YOLOv8 instance segmentation model for bark detection
    """
    # Load model
    model_name = f'yolov8{model_size}-seg.pt'
    model = YOLO(model_name)
    
    print(f"Training YOLOv8-{model_size} instance segmentation model...")
    print(f"Dataset: {data_yaml}")
    print(f"Epochs: {epochs}, Batch: {batch}, Image size: {imgsz}")
    
    # Train
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        name='bark_segmentation',
        project=project,
        save=True,
        plots=True
    )
    
    print(f"\nTraining complete!")
    print(f"Best model: {project}/bark_segmentation/weights/best.pt")
    
    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='dataset.yaml', help='Dataset YAML file')
    parser.add_argument('--model_size', default='s', choices=['n', 's', 'm', 'l', 'x'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--imgsz', type=int, default=640)
    
    args = parser.parse_args()
    train_bark_segmentation(
        data_yaml=args.data,
        model_size=args.model_size,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz
    )
```

## 🔄 Converting SAM Masks to Training Data

If you want to use SAM-generated masks as training data:

```python
# scripts/convert_sam_masks_to_yolo.py
import cv2
import numpy as np
from pathlib import Path

def sam_mask_to_yolo_format(mask_array, image_width, image_height):
    """
    Convert SAM mask array to YOLO segmentation format
    """
    # Find contours
    contours, _ = cv2.findContours(
        mask_array.astype(np.uint8), 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    if len(contours) == 0:
        return None
    
    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Normalize coordinates
    yolo_points = []
    for point in largest_contour:
        x = point[0][0] / image_width
        y = point[0][1] / image_height
        yolo_points.append(f"{x:.6f} {y:.6f}")
    
    # YOLO format: class_id x1 y1 x2 y2 ...
    return f"0 {' '.join(yolo_points)}"

def convert_sam_results_to_yolo_dataset(sam_results_dir, output_dir):
    """
    Convert SAM segmentation results to YOLO training format
    """
    # Implementation here
    pass
```

## 📊 Comparison: SAM vs Custom Model

| Feature | SAM1/SAM2 | Custom YOLOv8-seg |
|---------|-----------|-------------------|
| **Training Required** | No (pre-trained) | Yes |
| **Data Needed** | None | Masks + images |
| **Speed** | Medium | Fast |
| **Accuracy** | Very high (general) | High (domain-specific) |
| **Customization** | Limited | Full control |
| **Size** | Large (2-3GB) | Small (20-50MB) |
| **Bark-specific** | No (general purpose) | Yes (trained on bark) |

## 🎯 Recommended Approach

**For your use case, I recommend:**

1. **Start with YOLOv8-seg** (small model)
2. **Use SAM to generate initial training data** (semi-supervised)
3. **Train on bark-specific images** (better than general SAM)
4. **Keep your existing classifier** (two-stage approach)

**Benefits:**
- Faster inference than SAM
- Smaller model size
- Better for bark-specific segmentation
- Reuses your existing classifier
- Easier to deploy

## 📚 Next Steps

1. **Generate training masks** using SAM on your forest images
2. **Convert to YOLO format** using conversion script
3. **Train YOLOv8-seg model** on bark masks
4. **Integrate with existing classifier** in `forest_bark_analyzer.py`
5. **Test and compare** with SAM-based approach

## 🔗 Resources

- **YOLOv8 Segmentation Docs**: https://docs.ultralytics.com/tasks/segment/
- **YOLOv8 Training Guide**: https://docs.ultralytics.com/modes/train/
- **LabelMe (Annotation Tool)**: https://github.com/wkentaro/labelme
- **Roboflow (Dataset Management)**: https://roboflow.com/
