# YOSO Head Analysis for Instance Segmentation

## Summary

**Can we use the exact same YOSO head? NO** - but we can adapt it with modifications.

## Key Differences

### 1. **Input/Output Interface**

**YOSO Head:**
```python
def forward(self, features, targets):
    # features: Single tensor [B, C, H, W] from neck
    # targets: detectron2 format (Instances objects) or None
    # Returns: (all_stage_loss, cls_scores, mask_preds)
```

**Current Mask2Former Head:**
```python
def forward(self, feats, img_metas, gt_bboxes, gt_labels, gt_masks):
    # feats: List of tensors from neck
    # gt_bboxes: List[Tensor] - each [num_instances, 4]
    # gt_labels: List[Tensor] - each [num_instances]
    # gt_masks: List[Tensor] - each [num_instances, H, W]
    # Returns: Dict with losses (training) or predictions (inference)
```

### 2. **Architecture Differences**

| Aspect | YOSO Head | Mask2Former Head |
|--------|-----------|------------------|
| **Prediction Method** | Kernel-based (proposal kernels) | Query-based (transformer decoder) |
| **Refinement** | Multi-stage (iterative refinement) | Single-stage |
| **Attention** | Dynamic Separable Convolution Attention | Multi-head Cross-Attention |
| **Feature Processing** | Uses hard masks to extract features | Uses transformer decoder |
| **Initialization** | Conv2d kernel generator | Learnable query embeddings |

### 3. **Loss Computation**

**YOSO Head:**
- Loss computation is **integrated** in the head
- Uses `SetCriterion` from detectron2
- Computes loss at each stage
- Returns `all_stage_loss` dict with per-stage losses

**Mask2Former Head:**
- Loss computation is **separate** method
- Uses custom `HungarianMatcher` and loss functions
- Single-stage loss computation
- Returns loss dict directly

### 4. **Target Format**

**YOSO Head:**
- Expects detectron2 `Instances` objects
- Uses detectron2's `SetCriterion` which expects specific format
- Requires detectron2 structures

**Mask2Former Head:**
- Expects raw tensors (bboxes, labels, masks)
- Uses custom target preparation
- No detectron2 dependencies

## YOSO Head Architecture Details

### Multi-Stage Refinement
```python
for stage in range(self.num_stages + 1):
    if stage == 0:
        # Initial prediction from kernels
        mask_preds = self.kernels(features)
        proposal_kernels = self.kernels.weight.clone()
    else:
        # Refine using CrossAttenHead
        cls_scores, mask_preds, proposal_kernels = mask_head(...)
```

### Kernel-Based Prediction
- Uses **proposal kernels** `[B, N, C, K, K]` instead of query embeddings
- Masks generated via: `torch.einsum("bqc,bchw->bqhw", mask_kernels, features)`
- Kernels are refined across stages

### Dynamic Separable Convolution Attention
- `DySepConvAtten`: Depth-wise + point-wise convolution
- Generates dynamic weights from query features
- More efficient than standard attention

## Can We Use It for Instance Segmentation?

### ✅ **YES, YOSO supports instance segmentation:**
- The head outputs `cls_scores` and `mask_preds` which are exactly what we need
- It's designed for panoptic segmentation but works for instance segmentation
- The `instance_inference` method in segmentator.py shows it's used for instances

### ❌ **But NOT directly because:**
1. **Interface mismatch**: Different input/output format
2. **Detectron2 dependencies**: Uses detectron2 structures and utilities
3. **Target format**: Expects detectron2 `Instances`, not our tensor format
4. **Loss integration**: Loss is built-in, we need separate loss method

## Adaptation Strategy

### Option 1: **Adapt YOSO Head** (Recommended)
1. **Change input interface**:
   - Accept single tensor (already compatible with YOSONeck output)
   - Accept our target format (gt_labels, gt_masks as tensors)

2. **Adapt target preparation**:
   - Convert our tensor format to detectron2 format internally
   - Or modify SetCriterion to accept our format

3. **Separate loss computation**:
   - Extract loss computation to separate method
   - Keep forward() clean for inference

4. **Remove detectron2 dependencies**:
   - Replace `SetCriterion` with our own loss computation
   - Or keep it but adapt input format

### Option 2: **Hybrid Approach**
- Keep Mask2Former head structure
- Integrate YOSO's kernel-based prediction
- Use YOSO's multi-stage refinement
- Keep our loss computation

### Option 3: **Full YOSO Integration**
- Use YOSO head as-is
- Adapt our data pipeline to detectron2 format
- Use detectron2's SetCriterion
- More work but gets full YOSO benefits

## Recommendation

**Use Option 1**: Adapt YOSO Head to our interface

**Reasons:**
1. YOSO's kernel-based approach is more efficient than transformer decoder
2. Multi-stage refinement can improve mask quality
3. Dynamic separable convolution attention is faster
4. We can keep our existing data pipeline
5. Minimal changes to rest of codebase

**Key Changes Needed:**
1. Modify `forward()` signature to match our interface
2. Add target format conversion (our format → detectron2 format)
3. Extract loss computation to separate method
4. Adapt output format to match our expectations
5. Remove or adapt detectron2 dependencies

## Implementation Steps

1. Copy YOSO head code
2. Modify `forward()` to accept: `(features, img_metas, gt_labels, gt_masks)`
3. Add target conversion method
4. Extract loss computation to `loss()` method
5. Adapt output format
6. Test with our data pipeline

