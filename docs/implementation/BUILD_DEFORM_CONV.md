# Building Deformable Convolution Extension

## Overview

This document explains how to build the deformable convolution C++/CUDA extension copied from detectron2.

## Files Copied

The following files were copied from `Wheelchair-Panoptic-Segmentation/detectron2/`:

### Python Files:
- `tcd/layers/deform_conv.py` - Python wrapper for deformable conv
- `tcd/layers/wrappers.py` - Utility wrappers (for empty tensor support)
- `tcd/layers/__init__.py` - Module exports

### C++/CUDA Files:
- `tcd/layers/csrc/vision.cpp` - Main extension entry point
- `tcd/layers/csrc/deformable/deform_conv.h` - Header file
- `tcd/layers/csrc/deformable/deform_conv_cuda.cu` - CUDA implementation
- `tcd/layers/csrc/deformable/deform_conv_cuda_kernel.cu` - CUDA kernels

## Building the Extension

### Step 1: Install Requirements

Make sure you have:
- PyTorch with CUDA support
- CUDA toolkit
- C++ compiler (g++ or clang)

### Step 2: Build the Extension

Run the main setup script:

```bash
cd /hdd/hiep/CODE/Tree_Canopy_Detection
python setup.py build_ext --inplace
```

Or install the package (which will build the extension):

```bash
python setup.py develop
```

This will:
1. Compile the C++/CUDA code
2. Create `tcd/_C.so` (or `tcd/_C.pyd` on Windows) extension module
3. Make it available for import as `from tcd import _C`

### Step 3: Verify Installation

Test that the extension works:

```python
from tcd import _C
from tcd.layers import DeformConv, ModulatedDeformConv
print("Deformable convolution extension loaded successfully!")
```

## Troubleshooting

### Error: "CUDA_HOME not found"
- Make sure CUDA is installed and `CUDA_HOME` environment variable is set
- Or install CUDA toolkit

### Error: "Cannot find _C module"
- Make sure you ran `build_ext --inplace` in the project root
- Check that `tcd/_C.so` (or `.pyd`) exists

### Error: "Namespace mismatch"
- All namespace references have been changed from `detectron2` to `tcd`
- If you see errors, check that all `.cu` and `.h` files use `namespace tcd`

## Usage

Once built, the YOSO neck will automatically use the real deformable convolution:

```python
from tcd.models.neck import YOSONeck

neck = YOSONeck(
    in_channels=[128, 256, 512, 1024],
    agg_dim=128,
    hidden_dim=256,
    norm='BN'
)
```

The `DeformLayer` in the neck will use the real `ModulatedDeformConv` implementation.

## Notes

- The extension is built in-place, so it's part of the tcd package
- No need to install detectron2 - we copied only what we need
- The namespace has been changed from `detectron2` to `tcd` to avoid conflicts

