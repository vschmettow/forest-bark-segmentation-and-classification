# integrate_segmentation_with_classifier.py
# Example: How to use trained YOLOv8-seg model with existing classifier

from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def segment_with_yolov8(image_path, seg_model_path):
    """
    Segment bark objects using trained YOLOv8-seg model
    
    Args:
        image_path: Path to forest image
        seg_model_path: Path to trained YOLOv8-seg model
    
    Returns:
        List of masks (numpy arrays)
    """
    # Load segmentation model
    seg_model = YOLO(seg_model_path)
    
    # Run inference
    results = seg_model(image_path, verbose=False)
    
    masks = []
    for result in results:
        if result.masks is not None:
            for mask in result.masks.data:
                # Convert to numpy array
                mask_np = mask.cpu().numpy()
                masks.append(mask_np)
    
    return masks

def crop_mask_region(image, mask_array):
    """
    Crop image to mask region (same as existing function)
    """
    h_img, w_img = image.shape[:2]
    
    # Resize mask to image size if needed
    if mask_array.shape != (h_img, w_img):
        mask_array = cv2.resize(mask_array.astype(np.uint8), (w_img, h_img))
        mask_array = mask_array > 0.5
    
    # Find bounding box
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return None, None
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add padding
    padding = 3
    x_start = max(0, x_min - padding)
    y_start = max(0, y_min - padding)
    x_end = min(w_img, x_max + 1 + padding)
    y_end = min(h_img, y_max + 1 + padding)
    
    # Crop
    cropped = image[y_start:y_end, x_start:x_end]
    mask_cropped = mask_array[y_start:y_end, x_start:x_end]
    
    # Apply mask
    result = np.ones_like(cropped) * 255
    mask_3d = np.stack([mask_cropped] * 3, axis=2).astype(bool)
    result[mask_3d] = cropped[mask_3d]
    
    return result, (x_start, y_start, x_end - x_start, y_end - y_start)

def classify_with_yolov8(image_array, classifier_model_path, confidence_threshold=0.95):
    """
    Classify bark image using existing YOLOv8 classifier
    
    Args:
        image_array: Numpy array of cropped bark image
        classifier_model_path: Path to trained classifier
        confidence_threshold: Minimum confidence (default: 0.95)
    
    Returns:
        Classification result dict
    """
    # Load classifier
    classifier = YOLO(classifier_model_path)
    
    # Save to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        from PIL import Image
        pil_image = Image.fromarray(image_array)
        pil_image.save(tmp.name, 'JPEG')
        tmp_path = tmp.name
    
    try:
        # Classify
        results = classifier(tmp_path, verbose=False)
        result = results[0]
        
        if hasattr(result, 'probs'):
            probs = result.probs
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            class_names = result.names
            
            predicted_class = class_names[top1_idx]
            confidence = top1_conf
            
            # Apply threshold
            if confidence < confidence_threshold:
                predicted_class = "Other"
            
            return {
                'predicted_class': predicted_class,
                'confidence': confidence
            }
    finally:
        import os
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    return None

def segment_and_classify(image_path, seg_model_path, classifier_model_path):
    """
    Complete pipeline: Segment with YOLOv8-seg, classify with YOLOv8-cls
    
    Args:
        image_path: Path to forest image
        seg_model_path: Path to trained segmentation model
        classifier_model_path: Path to trained classifier model
    
    Returns:
        List of (mask, classification) tuples
    """
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Step 1: Segment
    print("Segmenting bark objects...")
    masks = segment_with_yolov8(image_path, seg_model_path)
    print(f"Found {len(masks)} bark objects")
    
    # Step 2: Classify each
    print("Classifying each object...")
    results = []
    for i, mask in enumerate(masks):
        print(f"  Classifying object {i+1}/{len(masks)}...")
        
        # Crop to mask
        cropped, bbox = crop_mask_region(image_rgb, mask)
        if cropped is None:
            continue
        
        # Classify
        classification = classify_with_yolov8(
            cropped, 
            classifier_model_path
        )
        
        if classification:
            results.append((mask, classification))
    
    return results

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Segment and classify bark using custom models'
    )
    parser.add_argument('--image', required=True, help='Path to forest image')
    parser.add_argument(
        '--seg_model', 
        default='../segmentation_results/bark_segmentation/weights/best.pt',
        help='Path to segmentation model'
    )
    parser.add_argument(
        '--classifier_model',
        default='../data/models/yolov8_results/bark_classifier/weights/best.pt',
        help='Path to classifier model'
    )
    
    args = parser.parse_args()
    
    results = segment_and_classify(
        args.image,
        args.seg_model,
        args.classifier_model
    )
    
    print(f"\nResults: {len(results)} objects classified")
    for i, (mask, classification) in enumerate(results):
        print(f"  Object {i+1}: {classification['predicted_class']} "
              f"(confidence: {classification['confidence']:.2%})")
