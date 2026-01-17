# predict_yolov8_bark.py
# Predict bark images using trained YOLOv8 classification model
# Classifies as Picea, Pinus, or Other (if confidence is below threshold)

import argparse
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import json
from tqdm import tqdm

def predict_image(model, image_path, confidence_threshold=0.95):
    """
    Predict class for a single image
    
    Args:
        model: Trained YOLOv8 classification model
        image_path: Path to image file
        confidence_threshold: Minimum confidence to classify as Picea or Pinus (default: 0.95)
                             If confidence < threshold, classify as "Other"
    
    Returns:
        predicted_class: 'Picea', 'Pinus', or 'Other'
        confidence: Confidence score (0-1)
        all_probs: Dictionary with probabilities for all classes
    """
    # Run prediction
    results = model(str(image_path), verbose=False)
    
    # Get predictions (YOLOv8-cls returns results in a list)
    result = results[0]
    
    # Get class names and probabilities
    # YOLOv8-cls structure: result.probs contains class probabilities
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
            # For "Other", confidence represents uncertainty (how far below threshold)
            other_confidence = confidence  # Store original confidence for reference
        else:
            predicted_class = predicted_class_name
            other_confidence = 0.0
        
        # Add "Other" to probabilities for display
        # "Other" probability is 1 - max_confidence when below threshold, else 0
        picea_conf = all_probs.get("Picea", 0.0)
        pinus_conf = all_probs.get("Pinus", 0.0)
        max_conf = max(picea_conf, pinus_conf)
        
        if max_conf < confidence_threshold:
            all_probs["Other"] = 1.0 - max_conf
        else:
            all_probs["Other"] = 0.0
        
        return predicted_class, confidence, all_probs
    else:
        return None, None, None

def predict_single_image(model_path, image_path, confidence_threshold=0.95):
    """Predict class for a single image"""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    print(f"\nClassifying image: {image_path}")
    predicted_class, confidence, all_probs = predict_image(model, image_path, confidence_threshold)
    
    if predicted_class is None:
        print("Error: Could not predict image")
        return None
    
    print(f"\nPredicted class: {predicted_class}")
    print(f"Confidence: {confidence*100:.2f}%")
    if predicted_class == "Other":
        print(f"  (Classified as 'Other' because confidence < {confidence_threshold*100:.0f}%)")
    
    print(f"\nAll probabilities:")
    for class_name, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        marker = "✓" if class_name == predicted_class else " "
        print(f"  {marker} {class_name}: {prob*100:.2f}%")
    
    return predicted_class, confidence, all_probs

def predict_directory(model_path, image_dir, confidence_threshold=0.95, output_file=None):
    """Predict classes for all images in a directory"""
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)
    
    # Find all images
    image_dir = Path(image_dir)
    image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG', '.png', '.PNG'}
    image_files = []
    for ext in image_extensions:
        image_files.extend(image_dir.rglob(f'*{ext}'))
    
    image_files = sorted(image_files)
    print(f"\nFound {len(image_files)} images to classify")
    print(f"Confidence threshold: {confidence_threshold*100:.0f}%")
    print("-" * 70)
    
    # Predict each image
    results = []
    class_counts = {"Picea": 0, "Pinus": 0, "Other": 0}
    
    for img_path in tqdm(image_files, desc="Classifying"):
        predicted_class, confidence, all_probs = predict_image(model, img_path, confidence_threshold)
        
        if predicted_class is None:
            continue
        
        results.append({
            'image': str(img_path),
            'predicted_class': predicted_class,
            'confidence': float(confidence),
            'probabilities': all_probs
        })
        
        class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
    
    # Print summary
    print("\n" + "=" * 70)
    print("CLASSIFICATION SUMMARY")
    print("=" * 70)
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(results) * 100) if len(results) > 0 else 0
        print(f"{class_name}: {count} images ({percentage:.1f}%)")
    print("=" * 70)
    
    # Save results if output file specified
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump({
                'confidence_threshold': confidence_threshold,
                'summary': class_counts,
                'results': results
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Classify bark images using trained YOLOv8 model')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to trained model checkpoint (.pt file)')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image to classify')
    parser.add_argument('--dir', type=str, default=None,
                        help='Directory containing images to classify')
    parser.add_argument('--confidence_threshold', type=float, default=0.95,
                        help='Confidence threshold for Picea/Pinus classification (default: 0.95). '
                             'Images with confidence below this will be classified as "Other"')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for batch predictions (optional)')
    
    args = parser.parse_args()
    
    if args.image and args.dir:
        print("Error: Please specify either --image or --dir, not both")
        return
    
    if not args.image and not args.dir:
        print("Error: Please specify either --image or --dir")
        return
    
    if args.confidence_threshold < 0 or args.confidence_threshold > 1:
        print("Error: confidence_threshold must be between 0 and 1")
        return
    
    if args.image:
        predict_single_image(args.model, args.image, args.confidence_threshold)
    else:
        predict_directory(args.model, args.dir, args.confidence_threshold, args.output)

if __name__ == '__main__':
    main()

