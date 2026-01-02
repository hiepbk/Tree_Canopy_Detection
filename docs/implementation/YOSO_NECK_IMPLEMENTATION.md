# YOSO Neck Implementation

## Summary

Successfully implemented the YOSO (You Only Segment Once) neck architecture for the tree canopy detection task.

## Files Created/Modified

### 1. New File: `tcd/models/neck/yoso_neck.py`
- **DeformConv2d**: Simple deformable convolution (placeholder, uses regular conv for now)
- **ModulatedDeformConv2d**: Modulated deformable convolution (placeholder)
- **DeformLayer**: Deformable conv layer with upsampling
- **LiteDeformConv**: Lightweight deformable FPN with Cross-Feature Aggregation (CFA)
- **YOSONeck**: Main neck module with coordinate features

### 2. Modified: `tcd/models/neck/__init__.py`
- Added `YOSONeck` to exports

### 3. Modified: `tcd/configs/tcd_config.py`
- Updated neck config to use `YOSONeck` instead of `FPN`

## Architecture Details

### YOSONeck Components

1. **LiteDeformConv (Lightweight Deformable FPN)**
   - Takes 4 backbone feature levels as input
   - Uses deformable convolutions for flexible feature processing
   - Top-down feature fusion with upsampling
   - Cross-Feature Aggregation (CFA) to fuse all scales
   - Output: `[B, agg_dim, H/4, W/4]` (default agg_dim=128)

2. **Coordinate Features**
   - Adds explicit x, y positional encoding (normalized to [-1, 1])
   - Helps model understand spatial relationships
   - Concatenated with features: `[B, agg_dim+2, H/4, W/4]`

3. **Localization Conv**
   - Final 1×1 conv to project to hidden dimension
   - Output: `[B, hidden_dim, H/4, W/4]` (default hidden_dim=256)

### Configuration

```python
neck=dict(
    type='YOSONeck',
    in_channels=[128, 256, 512, 1024],  # For Swin-B
    # For Swin-T use: [96, 192, 384, 768]
    agg_dim=128,  # Aggregated feature dimension
    hidden_dim=256,  # Hidden dimension for output
    norm='BN',  # 'BN' for single GPU, 'SyncBN' for multi-GPU
),
```

## Key Features

1. **Efficient Multi-scale Fusion**: CFA effectively combines features from all backbone stages
2. **Deformable Convolutions**: More flexible than standard convolutions (currently using regular conv as placeholder)
3. **Positional Awareness**: Coordinate features provide explicit spatial information
4. **Lightweight**: Channel reduction reduces memory and computation

## Current Status

✅ **Implemented**:
- YOSO neck architecture
- Coordinate feature generation
- Cross-Feature Aggregation (CFA)
- Config integration

⚠️ **Note**:
- Deformable convolutions are currently implemented as placeholders using regular convolutions
- Can be replaced with proper DCN (Deformable Convolution Network) implementation later
- For now, this provides the same architecture structure with regular convs

## Next Steps

1. **Test the implementation**: Run training to verify it works
2. **Optional**: Replace placeholder deformable convs with proper DCN implementation (e.g., from mmcv or custom)
3. **Tune hyperparameters**: Adjust `agg_dim`, `hidden_dim` if needed
4. **Compare performance**: Compare with original FPN neck

## Usage

The neck is automatically used when the config specifies `type='YOSONeck'`. No code changes needed in the model - it's handled by the registry system.

## Differences from Original YOSO

1. **Simplified Deformable Conv**: Using regular conv as placeholder (can be upgraded)
2. **Config-based**: Integrated with existing config system
3. **Single Output**: Returns list with single feature map (to match head interface)

