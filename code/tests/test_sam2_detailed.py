# test_sam2_detailed.py
# Detailed test to explore SAM2 API once installed

import sys
import inspect

print("=" * 70)
print("SAM2 Detailed API Exploration")
print("=" * 70)

try:
    import sam2
    print("✓ SAM2 module imported successfully")
except ImportError as e:
    print(f"✗ SAM2 not installed: {e}")
    print("\nPlease install SAM2 first:")
    print("  git clone https://github.com/facebookresearch/segment-anything-2.git")
    print("  cd segment-anything-2")
    print("  pip install -e .")
    sys.exit(1)

# Explore module structure
print("\n" + "=" * 70)
print("1. SAM2 Module Contents")
print("=" * 70)
print("\nTop-level attributes:")
for attr in sorted(dir(sam2)):
    if not attr.startswith('_'):
        obj = getattr(sam2, attr)
        obj_type = type(obj).__name__
        print(f"  {attr}: {obj_type}")

# Try to find build functions
print("\n" + "=" * 70)
print("2. Looking for Build Functions")
print("=" * 70)
try:
    from sam2 import build_sam2
    print("✓ Found: build_sam2 (direct import)")
    print(f"  Signature: {inspect.signature(build_sam2)}")
except ImportError:
    try:
        from sam2.build_sam import build_sam2
        print("✓ Found: build_sam2 (from sam2.build_sam)")
        print(f"  Signature: {inspect.signature(build_sam2)}")
    except ImportError:
        print("✗ build_sam2 not found")
        # Check if build_sam module exists
        try:
            from sam2 import build_sam
            print(f"✓ Found build_sam module: {dir(build_sam)}")
        except:
            print("✗ build_sam module not found")

# Try to find predictor classes
print("\n" + "=" * 70)
print("3. Looking for Predictor Classes")
print("=" * 70)

predictor_classes = []
for attr in dir(sam2):
    if 'predictor' in attr.lower() or 'predict' in attr.lower():
        obj = getattr(sam2, attr)
        if inspect.isclass(obj):
            predictor_classes.append(attr)
            print(f"✓ Found class: {attr}")

if not predictor_classes:
    # Try submodules
    try:
        from sam2 import sam2_image_predictor
        print("✓ Found: sam2_image_predictor module")
        for attr in dir(sam2_image_predictor):
            if inspect.isclass(getattr(sam2_image_predictor, attr)):
                print(f"  - Class: {attr}")
    except ImportError:
        pass

# Check for image predictor specifically
print("\n" + "=" * 70)
print("4. Image Predictor")
print("=" * 70)
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("✓ Found: SAM2ImagePredictor")
    print(f"  Methods: {[m for m in dir(SAM2ImagePredictor) if not m.startswith('_')][:10]}")
except ImportError:
    try:
        from sam2 import SAM2ImagePredictor
        print("✓ Found: SAM2ImagePredictor (direct)")
    except ImportError:
        print("✗ SAM2ImagePredictor not found")
        print("  Checking what predictor classes exist...")

# Try to find text prompt methods
print("\n" + "=" * 70)
print("5. Text Prompt Support")
print("=" * 70)
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    methods = [m for m in dir(SAM2ImagePredictor) if 'text' in m.lower() or 'prompt' in m.lower()]
    if methods:
        print(f"✓ Found text-related methods: {methods}")
    else:
        print("  No obvious text prompt methods found")
        print("  Available methods:")
        all_methods = [m for m in dir(SAM2ImagePredictor) if not m.startswith('_')]
        print(f"  {all_methods[:15]}...")
except Exception as e:
    print(f"  Could not check: {e}")

print("\n" + "=" * 70)
print("Test Complete")
print("=" * 70)
print("\nPlease share the output above so we can update forest_bark_analyzer.py")
print("with the correct SAM2 API calls.")

