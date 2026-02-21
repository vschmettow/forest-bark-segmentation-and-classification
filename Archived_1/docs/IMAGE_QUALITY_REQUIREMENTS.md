# Image Quality Requirements for Forest Bark Analyzer

## Input Image Requirements

For best results when analyzing forest images, the input images should meet these requirements:

### Resolution
- **Minimum**: 2-3 megapixels (e.g., 1500x2000px)
- **Recommended**: 5-10 megapixels (e.g., 2500x4000px)
- **Why**: Higher resolution allows SAM to segment objects more accurately and preserves bark texture details for classification

### Quality Factors

1. **Sharpness**
   - Images should be in focus with clear bark texture
   - Blurry images will result in poor segmentation and classification

2. **Lighting**
   - Even, natural lighting is best
   - Avoid extreme shadows or overexposure
   - Good contrast helps SAM identify object boundaries

3. **Distance to Trees**
   - Close enough that bark texture is visible
   - Too far away = bark features are too small for classification
   - Too close = only a small part of tree visible, may miss context

4. **Angle**
   - Perpendicular to tree trunk is ideal (bark texture is clearest)
   - Side angles may distort texture patterns

5. **Background**
   - Clear separation between trees and background helps segmentation
   - Cluttered backgrounds may cause SAM to include non-bark objects

## How Quality Affects Results

### Poor Quality Images May Cause:
- **Segmentation Issues**: SAM may not find objects correctly
- **Classification Errors**: YOLOv8 classifier needs clear bark texture to distinguish Picea vs Pinus
- **False Positives**: Background objects may be incorrectly segmented as "bark"

### Good Quality Images Will:
- Allow SAM to accurately segment individual bark regions
- Provide clear bark texture for the classifier
- Result in more accurate Picea/Pinus classifications

## Current Processing

The analyzer:
1. Uses SAM to segment all objects in the image
2. Crops each segment **tightly** to show only bark (minimal padding)
3. Applies the mask so only the bark region is visible (white background)
4. Classifies each cropped bark region

**Note**: The cropping function has been optimized to:
- Use tight bounding boxes from mask pixels (not loose SAM bboxes)
- Apply minimal padding (3 pixels) to avoid including background
- Keep only the mask region visible (white background)

This ensures the classifier sees primarily bark texture, improving accuracy.

