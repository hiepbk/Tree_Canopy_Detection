# YOSO Architecture Analysis (SwinT Backbone)

## Overview

This document analyzes the **YOSO (You Only Segment Once)** architecture from the Wheelchair-Panoptic-Segmentation project, which uses **SwinT (Swin Transformer Tiny)** as the backbone. This is more relevant to our codebase since we also use SwinT.

---

## 1. Neck Architecture: `YOSONeck` with SwinT

### 1.1 Overall Flow

```
SwinT Backbone Features → LiteDeformConv → Coordinate Features → Final Features
[stage2, stage3, stage4, stage5] → [AGG_DIM channels] → [HIDDEN_DIM channels]
```

### 1.2 LiteDeformConv (Lightweight Deformable FPN)

**Purpose**: Efficiently fuses multi-scale SwinT features using deformable convolutions.

#### Input Features from SwinT
- **stage2**: 96 channels (H/4, W/4)
- **stage3**: 192 channels (H/8, W/8)
- **stage4**: 384 channels (H/16, W/16)
- **stage5**: 768 channels (H/32, W/32)

#### Channel Reduction Strategy

The code handles channel reduction differently for SwinT:

```python
for idx, feat in enumerate(in_features):
    cur_channel = backbone_shape[feat].channels
    if idx == 0:
        out_channels.append(cur_channel//2)  # First layer: ÷2
    else:
        out_channels.append(backbone_shape[in_features[idx-1]].channels)  # Others: use previous layer's channel
```

**Example for SwinT**:
- Input: `[96, 192, 384, 768]` (stage2, stage3, stage4, stage5)
- Output channels: `[128, 48, 96, 192, 384]` = `[AGG_DIM, stage2//2, stage2, stage3, stage4]`
  - `AGG_DIM = 128` (final aggregated dimension)
  - `stage2//2 = 48` (first lateral conv output)
  - `stage2 = 96` (second lateral conv output)
  - `stage3 = 192` (third lateral conv output)
  - `stage4 = 384` (fourth lateral conv output)

#### Architecture Components

1. **Lateral Convolutions (1×1)**
   ```python
   lateral_conv0: 768 → 384  # stage5
   lateral_conv1: 384 → 192  # stage4
   lateral_conv2: 192 → 96   # stage3
   lateral_conv3: 96 → 48    # stage2
   ```

2. **DeformLayer (Deformable Conv + Upsampling)**
   - **Modulated Deformable Convolution**:
     - Learns offsets and modulation masks (27 channels: 3×3×3)
     - More flexible than standard convolution
   - **Transposed Convolution**:
     - Upsamples by 2× (kernel=4, stride=2)
     - Bilinear initialization for stable training
   - **Normalization**: Uses `get_norm(norm)` which supports both `SyncBN` (multi-GPU) and `BN` (single-GPU)

3. **Top-Down Feature Fusion**
   ```
   p5 (stage5, 768ch) → lateral_conv0 → 384ch
                        ↓
                        DeformLayer → upsample 2×
                        ↓
   p4 (stage4, 384ch) → lateral_conv1 → 192ch + upsampled
                        ↓
                        DeformLayer → upsample 2×
                        ↓
   p3 (stage3, 192ch) → lateral_conv2 → 96ch + upsampled
                        ↓
                        DeformLayer → upsample 2×
                        ↓
   p2 (stage2, 96ch) → lateral_conv3 → 48ch + upsampled
   ```

4. **Cross-Feature Aggregation (CFA)**
   ```python
   # Project all features to AGG_DIM (128)
   x5 = F.interpolate(conv_a5(x5), scale_factor=8)  # stage5: ×8 upsampling
   x4 = F.interpolate(conv_a4(x4), scale_factor=4)   # stage4: ×4 upsampling
   x3 = F.interpolate(conv_a3(x3), scale_factor=2)   # stage3: ×2 upsampling
   x2 = conv_a2(x2)                                   # stage2: ×1 (already at target size)
   
   # Element-wise addition
   x = x5 + x4 + x3 + x2 + bias
   ```
   - All features are upsampled to the same spatial size (stage2 resolution: H/4, W/4)
   - All features are projected to `AGG_DIM = 128` channels
   - Final output: `[B, 128, H/4, W/4]`

5. **Output Convolution**
   - 3×3 conv for final refinement
   - Output: `[B, 128, H/4, W/4]`

### 1.3 Coordinate Feature Injection

**Purpose**: Provides explicit positional encoding to help spatial reasoning.

```python
def generate_coord(self, input_feat):
    # Generate normalized coordinates [-1, 1]
    x_range = torch.linspace(-1, 1, W)
    y_range = torch.linspace(-1, 1, H)
    y, x = torch.meshgrid(y_range, x_range)
    coord_feat = torch.cat([x, y], 1)  # [B, 2, H, W]
    return coord_feat
```

- Concatenates coordinate features: `[features, coord_feat]` → `[B, 128+2, H/4, W/4]`
- Final 1×1 conv projects to `HIDDEN_DIM = 256`: `[B, 256, H/4, W/4]`

### 1.4 Key Differences from ResNet Version

1. **Channel Handling**: SwinT has different channel progression (96→192→384→768) vs ResNet (256→512→1024→2048)
2. **Normalization**: Uses `get_norm()` to support both `SyncBN` and `BN` for single/multi-GPU training
3. **Feature Names**: Uses `["stage2", "stage3", "stage4", "stage5"]` instead of `["res2", "res3", "res4", "res5"]`

---

## 2. Head Architecture: `YOSOHead`

### 2.1 Overall Structure

YOSO uses **multi-stage iterative refinement** with **proposal kernels** for mask prediction.

### 2.2 Key Concepts

#### Proposal Kernels
- **Learnable kernels** that represent object instances
- Shape: `[NUM_PROPOSALS, HIDDEN_DIM, K, K]` where `K = CONV_KERNEL_SIZE_2D` (typically 1)
- Initialized from a learnable 1×1 conv: `kernels = Conv2d(HIDDEN_DIM, NUM_PROPOSALS, 1)`
- Used to generate masks: `mask = einsum("qc,chw->qhw", kernel, features)`

#### Multi-Stage Refinement
- **Stage 0**: Initial mask prediction from learnable kernels
- **Stage 1 to N-1**: Iterative refinement using `CrossAttenHead`
- **Stage N**: Final prediction with classification

### 2.3 CrossAttenHead (Cross-Attention Head)

The core component that refines masks and kernels iteratively.

#### Input Processing

1. **Feature Extraction from Masks**
   ```python
   # Hard threshold masks (>0.5)
   hard_masks = (mask_preds.sigmoid() > 0.5).float()
   # Extract features: [B, N, C] = einsum('bnhw,bchw->bnc', hard_masks, features)
   f = torch.einsum('bnhw,bchw->bnc', hard_masks, features)
   ```
   - `f`: Feature vectors extracted from regions where masks are active
   - Shape: `[B, NUM_PROPOSALS, HIDDEN_DIM]`

2. **Kernel Reshaping**
   ```python
   # [B, N, C, K, K] -> [B, N, C*K*K]
   k = proposal_kernels.view(B, num_proposals, -1)
   ```
   - `k`: Flattened proposal kernels
   - Shape: `[B, NUM_PROPOSALS, HIDDEN_DIM*K*K]` (typically `[B, 100, 256]`)

#### Attention Mechanisms

1. **Feature-Kernel Cross-Attention (f_atten)**
   - Uses `DySepConvAtten` (Dynamic Separable Convolution Attention)
   - Updates features `f` based on kernels `k`
   - Process: `f = f + Dropout(f_atten(k, f))`
   - Normalization: `f = f_atten_norm(f)`

2. **Kernel-Feature Cross-Attention (k_atten)**
   - Same `DySepConvAtten` mechanism
   - Updates kernels `k` based on features `f`
   - Process: `k = k_atten_norm(f)` (note: uses `f` for normalization, not `k`)

3. **Self-Attention (s_atten)**
   - Standard `MultiheadAttention` (8 heads)
   - Allows kernels to interact with each other
   - Process: `k = k + Dropout(s_atten(k, k, k))`

#### Dynamic Separable Convolution Attention (DySepConvAtten)

**Key Innovation**: Uses dynamic convolution weights generated from query features.

```python
class DySepConvAtten:
    def forward(self, query, value):
        # Generate dynamic weights from query
        dy_conv_weight = self.weight_linear(query)  # [B, N, kernel_size + num_proposals]
        
        # Split into depth-wise and point-wise weights
        dy_depth_weight = dy_conv_weight[:, :, :kernel_size]      # [B, N, 1, K]
        dy_point_weight = dy_conv_weight[:, :, kernel_size:]       # [B, N, N, 1]
        
        # Depth-wise conv (per-channel filtering)
        depth_padding = (kernel_size - 1) // 2  # Explicit padding for ONNX compatibility
        out = F.relu(F.conv1d(value, dy_depth_weight, groups=N, padding=depth_padding))
        
        # Point-wise conv (cross-proposal interaction)
        point_padding = 0  # kernel size is 1
        out = F.conv1d(out, dy_point_weight, padding=point_padding)
        
        return out
```

**Key Differences from Original YOSO**:
- **Explicit Padding**: Uses `calculate_padding()` instead of `padding='same'` for ONNX compatibility
- **ReLU Activation**: Applied after depth-wise conv
- **Batch Loop**: Processes each batch item separately (for dynamic weight generation)

**Advantages**:
- **Adaptive**: Weights adapt to input, more flexible than fixed attention
- **Efficient**: Separable convolution reduces computation
- **Context-aware**: Point-wise conv enables cross-proposal communication

#### Feed-Forward Network (FFN)

- Standard 2-layer MLP: `Linear(C) → ReLU → Dropout → Linear(C)`
- Hidden dimension: 2048 (4× expansion)
- Residual connection: `out = identity + Dropout(FFN(out))`

#### Output Generation

1. **Object Features**
   ```python
   # Reshape kernels: [B, N, C*K*K] -> [B, N, C, K, K]
   obj_feat = k.reshape(B, N, C, K, K)
   obj_feat = FFN(obj_feat)  # [B, N, K*K, C]
   ```

2. **Classification**
   ```python
   cls_feat = obj_feat.sum(-2)  # [B, N, C] - aggregate over spatial dims
   # Apply FC layers (NUM_CLS_FCS = 3)
   for cls_layer in cls_fcs:  # Linear → LayerNorm → ReLU (×3)
       cls_feat = cls_layer(cls_feat)
   cls_score = Linear(cls_feat)  # [B, N, num_classes+1]
   ```

3. **Mask Prediction**
   ```python
   # Apply FC layers (NUM_MASK_FCS = 3)
   for reg_layer in mask_fcs:  # Linear → LayerNorm → ReLU (×3)
       mask_feat = reg_layer(mask_feat)
   mask_kernels = Linear(mask_feat)  # [B, N, C]
   new_mask_preds = einsum("bqc,bchw->bqhw", mask_kernels, features)
   ```

### 2.4 YOSOHead Forward Pass

```python
for stage in range(num_stages + 1):  # num_stages = 2, so stages 0, 1, 2
    if stage == 0:
        # Initial prediction
        mask_preds = kernels(features)  # [B, N, H, W]
        proposal_kernels = kernels.weight.clone()
        object_kernels = proposal_kernels.expand(B, -1, -1, -1)
        
    elif stage == num_stages:  # Final stage (stage 2)
        # Compute classification
        cls_scores, mask_preds, proposal_kernels = mask_head(
            features, object_kernels, mask_preds, train_flag=True
        )
        
    else:  # Intermediate stages (stage 1)
        # Refine masks and kernels
        cls_scores, mask_preds, proposal_kernels = mask_head(
            features, object_kernels, mask_preds, train_flag=(targets is not None)
        )
        object_kernels = proposal_kernels  # Update for next stage
```

### 2.5 Key Differences from Original YOSO

1. **ONNX Compatibility**: Explicit padding calculation instead of `padding='same'`
2. **More FC Layers**: `NUM_CLS_FCS = 3` and `NUM_MASK_FCS = 3` (vs 1 in original)
3. **Device Specification**: `MultiheadAttention` includes `device=cfg.MODEL.DEVICE` parameter

---

## 3. Configuration Parameters (SwinT)

### Neck Parameters
- `AGG_DIM`: 128 (aggregated feature dimension)
- `HIDDEN_DIM`: 256 (hidden dimension for head)
- `IN_FEATURES`: `["stage2", "stage3", "stage4", "stage5"]` (SwinT feature levels)
- `NORM`: `"SyncBN"` (for multi-GPU) or `"BN"` (for single-GPU)

### Head Parameters
- `NUM_PROPOSALS`: 100 (number of proposal kernels)
- `NUM_STAGES`: 2 (number of refinement stages)
- `CONV_KERNEL_SIZE_2D`: 1 (2D kernel size)
- `CONV_KERNEL_SIZE_1D`: 3 (1D kernel size for dynamic conv)
- `NUM_CLS_FCS`: 3 (number of classification FC layers)
- `NUM_MASK_FCS`: 3 (number of mask prediction FC layers)
- `TEMPERATIRE`: 0.05 (temperature scaling for logits)

### Loss Parameters
- `CLASS_WEIGHT`: 2.0
- `MASK_WEIGHT`: 5.0
- `DICE_WEIGHT`: 5.0
- `NO_OBJECT_WEIGHT`: 0.1
- `TRAIN_NUM_POINTS`: 12544 (points sampled for mask loss)

---

## 4. Advantages for Tree Canopy Detection

### Neck Advantages
1. **Multi-scale Fusion**: CFA effectively combines features from all SwinT stages
2. **Deformable Convolutions**: Better handle irregular tree shapes
3. **Efficiency**: Channel reduction (÷2) reduces memory and computation
4. **Positional Awareness**: Coordinate features help with spatial reasoning

### Head Advantages
1. **Efficiency**: Kernel-based prediction is faster than pixel-level operations
2. **Iterative Refinement**: Multi-stage approach progressively improves predictions
3. **Dynamic Adaptation**: DySepConvAtten adapts to different tree patterns
4. **Unified Representation**: Kernels encode both spatial and semantic information

### Potential Improvements for Our Use Case

1. **Increase NUM_PROPOSALS**: May need to increase from 100 to 200-300 for images with many trees
2. **Adjust NUM_STAGES**: Could try 3 stages for better refinement
3. **Fine-tune NUM_CLS_FCS/NUM_MASK_FCS**: Current 3 layers might be good, but could experiment
4. **Coordinate Features**: Explicit positional encoding may help with spatial reasoning for tree locations

---

## 5. Comparison with Our Current Mask2Former

| Component | Mask2Former (Ours) | YOSO (SwinT) |
|-----------|-------------------|--------------|
| **Neck** | Standard FPN | LiteDeformConv (deformable + CFA) |
| **Head** | Transformer decoder | Dynamic convolution attention |
| **Mask Prediction** | Pixel decoder + queries | Proposal kernels + convolution |
| **Attention** | Multi-head self/cross-attention | Dynamic separable convolution |
| **Positional Encoding** | Learned queries | Coordinate features + kernels |
| **Speed** | Slower | ~2× faster |
| **Memory** | Higher | Lower (kernel-based) |

---

## 6. Code References

- **Neck**: `Wheelchair-Panoptic-Segmentation/projects/PanSeg/model/neck/neck.py`
- **Head**: `Wheelchair-Panoptic-Segmentation/projects/PanSeg/model/head/head.py`
- **Config**: `Wheelchair-Panoptic-Segmentation/projects/PanSeg/configs/cityscapes/panoptic-segmentation/YOSO-SWIN_T.yaml`

---

## Summary

YOSO with SwinT backbone offers:

1. **Efficient Neck**: LiteDeformConv uses deformable convolutions and CFA for efficient multi-scale fusion
2. **Fast Head**: Dynamic convolution attention with proposal kernels enables fast, accurate mask prediction
3. **SwinT Compatibility**: Designed specifically for SwinT feature stages
4. **ONNX Ready**: Explicit padding makes it compatible with ONNX export

This architecture could potentially improve both speed and accuracy for tree canopy detection, especially for handling dense instances and irregular shapes. The kernel-based approach is particularly efficient for dense segmentation tasks.

