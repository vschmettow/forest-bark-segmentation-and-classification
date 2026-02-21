# SAM2 Installation and Setup Guide

## Installation Steps

### 1. Clone and Install SAM2

```bash
# Clone the SAM2 repository
git clone https://github.com/facebookresearch/segment-anything-2.git
cd segment-anything-2

# Install SAM2
pip install -e .
```

### 2. Download Model Weights

Download SAM2 checkpoints from the official repository:
- Visit: https://github.com/facebookresearch/segment-anything-2
- Download the model checkpoint you want (e.g., `sam2_hiera_large.pt` or `sam2_hiera_b.pt`)
- Place it in: `Archive/Preprocessing scripts/`

### 3. Verify Installation

Run the test script:
```bash
python3 test_sam2.py
```

This will check:
- ✓ SAM2 module installation
- ✓ API structure and available classes
- ✓ Model checkpoint availability

## Testing SAM2 API

After installation, we'll need to verify the actual API structure. SAM2's API might differ from SAM v1, so we may need to adjust the code in `forest_bark_analyzer.py` based on the actual implementation.

## Common Issues

**Issue: Module not found after installation**
- Make sure you ran `pip install -e .` from inside the segment-anything-2 directory
- Check that the installation completed successfully

**Issue: Model weights not found**
- Download the checkpoint files from the SAM2 repository
- Place them in the correct directory: `Archive/Preprocessing scripts/`

**Issue: Import errors**
- SAM2 might use different import paths than expected
- Check the actual structure with: `python3 test_sam2.py`

## Next Steps

Once SAM2 is installed:
1. Run `python3 test_sam2.py` to verify installation
2. We'll update `forest_bark_analyzer.py` based on the actual SAM2 API
3. Test with a sample image to verify text prompt functionality

