# test_sam2.py
# Test script to verify SAM2 installation and API

import sys
import os

print("=" * 70)
print("SAM2 Installation and API Test")
print("=" * 70)

# Test 1: Check if SAM2 is installed
print("\n1. Checking if SAM2 is installed...")
try:
    import sam2
    print("✓ SAM2 module found")
    print(f"  Location: {sam2.__file__}")
except ImportError as e:
    print(f"✗ SAM2 not installed: {e}")
    print("\nTo install SAM2:")
    print("  git clone https://github.com/facebookresearch/segment-anything-2.git")
    print("  cd segment-anything-2")
    print("  pip install -e .")
    sys.exit(1)

# Test 2: Check available functions/classes
print("\n2. Checking SAM2 API structure...")
try:
    from sam2.build_sam import build_sam2
    print("✓ build_sam2 function found")
except ImportError as e:
    print(f"✗ build_sam2 not found: {e}")
    print("  Available in sam2.build_sam module?")

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("✓ SAM2ImagePredictor class found")
except ImportError as e:
    print(f"✗ SAM2ImagePredictor not found: {e}")
    print("  Checking alternative imports...")
    try:
        # Try different possible import paths
        from sam2 import SAM2ImagePredictor
        print("✓ SAM2ImagePredictor found (direct import)")
    except:
        print("  Could not find SAM2ImagePredictor")
        print("  Listing available sam2 attributes:")
        print(f"  {dir(sam2)}")

# Test 3: Check for model weights
print("\n3. Checking for SAM2 model weights...")
sam2_checkpoints_dir = "Archive/Preprocessing scripts/"
possible_checkpoints = [
    "sam2_hiera_large.pt",
    "sam2_hiera_l.pt", 
    "sam2_hiera_b.pt",
    "sam2_hiera_base.pt",
    "sam2_hiera_tiny.pt"
]

found_checkpoints = []
for checkpoint in possible_checkpoints:
    path = os.path.join(sam2_checkpoints_dir, checkpoint)
    if os.path.exists(path):
        found_checkpoints.append((checkpoint, path))
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"✓ Found: {checkpoint} ({size_mb:.1f} MB)")

if not found_checkpoints:
    print("✗ No SAM2 checkpoints found")
    print(f"  Searched in: {sam2_checkpoints_dir}")
    print("\nTo download SAM2 checkpoints:")
    print("  Visit: https://github.com/facebookresearch/segment-anything-2")
    print("  Download checkpoints and place in Archive/Preprocessing scripts/")

# Test 4: Try to inspect SAM2 module structure
print("\n4. Inspecting SAM2 module structure...")
try:
    import sam2
    print("Available submodules/attributes in sam2:")
    attrs = [attr for attr in dir(sam2) if not attr.startswith('_')]
    for attr in attrs[:20]:  # Show first 20
        print(f"  - {attr}")
    if len(attrs) > 20:
        print(f"  ... and {len(attrs) - 20} more")
except Exception as e:
    print(f"Error inspecting sam2 module: {e}")

# Test 5: Try to find predictor classes
print("\n5. Searching for predictor classes...")
try:
    import inspect
    import sam2
    
    # Look for predictor classes
    for name in dir(sam2):
        obj = getattr(sam2, name)
        if inspect.isclass(obj) and 'predictor' in name.lower():
            print(f"  Found predictor class: {name}")
except Exception as e:
    print(f"Error searching for classes: {e}")

# Summary
print("\n" + "=" * 70)
print("Summary")
print("=" * 70)

sam2_installed = True
try:
    import sam2
except:
    sam2_installed = False

if sam2_installed:
    print("✓ SAM2 is installed")
else:
    print("✗ SAM2 is NOT installed")

if found_checkpoints:
    print(f"✓ Found {len(found_checkpoints)} SAM2 checkpoint(s)")
else:
    print("✗ No SAM2 checkpoints found")

print("\n" + "=" * 70)

