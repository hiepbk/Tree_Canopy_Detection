# Documentation

This folder contains documentation and analysis files for the Tree Canopy Detection project.

## Structure

- **`analysis/`**: Architecture and design analysis documents
  - `YOSO_HEAD_ANALYSIS.md`: Analysis of YOSO head architecture
  - `YOSO_MATCHING_ANALYSIS.md`: Analysis of YOSO matching strategy (Hungarian Matcher)
  - `YOSO_LOSS_ANALYSIS.md`: Analysis of YOSO loss functions
  - `YOSO_SwinT_Architecture_Analysis.md`: Analysis of YOSO architecture with SwinT backbone
  - `FLOAT16_OVERFLOW_EINSUM_FIX.md`: Root cause and fix for float16 overflow in einsum operations

- **`implementation/`**: Implementation guides and verification documents
  - `DEFORM_CONV_VERIFICATION.md`: Verification of deformable convolution wrapper (Python to C++)
  - `YOSO_NECK_IMPLEMENTATION.md`: Implementation details for YOSO neck
  - `BUILD_DEFORM_CONV.md`: Build instructions for deformable convolution extension

## Quick Links

- [YOSO Head Analysis](analysis/YOSO_HEAD_ANALYSIS.md)
- [YOSO Matching Strategy](analysis/YOSO_MATCHING_ANALYSIS.md)
- [Deformable Convolution Verification](implementation/DEFORM_CONV_VERIFICATION.md)
- [Build Instructions](implementation/BUILD_DEFORM_CONV.md)

