# Why Use CONV_KERNEL_SIZE_2D=3?

## Current Situation

**Original YOSO**: `CONV_KERNEL_SIZE_2D = 1`
**Our Implementation**: `conv_kernel_size_2d = 3`

## What is `conv_kernel_size_2d`?

The `conv_kernel_size_2d` parameter determines the **spatial size of proposal kernels** used for mask generation.

### How Kernels Are Used

Looking at the mask generation (line 395 in original, line 249 in ours):
```python
mask_kernels = self.fc_mask(mask_feat).squeeze(2)  # [B, N, C] if K=1, or [B, N, C, K, K] if K>1
new_mask_preds = torch.einsum("bqc,bchw->bqhw", mask_kernels, features)
```

**With K=1**:
- `mask_kernels`: `[B, N, C]` (point-wise kernels)
- `einsum("bqc,bchw->bqhw")`: Each kernel is a single vector that dot-products with features at each spatial location
- **No spatial context** - each location is independent

**With K=3**:
- `mask_kernels`: `[B, N, C, 3, 3]` (3×3 spatial kernels)
- The einsum would need to be different: `einsum("bqckk,bchw->bqhw", mask_kernels, features)`
- **Has spatial context** - can capture local patterns (like a 3×3 convolution)

## Wait - There's an Issue!

Looking at our code (line 249):
```python
mask_kernels = self.fc_mask(mask_feat).squeeze(2)  # [B, N, K*K, C] -> [B, N, C] if K=1, or [B, N, K*K, C] if K>1
new_mask_preds = torch.einsum("bqc,bchw->bqhw", mask_kernels, features)
```

If `K=3`, then `mask_kernels` after `squeeze(2)` would be `[B, N, 9, C]` (if K*K=9), not `[B, N, C]`!

But the einsum expects `[B, N, C]`. So either:
1. We're using `K=3` incorrectly
2. The einsum should be different for `K=3`
3. We should use `K=1` like the original

## Checking the Original Code

Original YOSO (line 394-395):
```python
mask_kernels = self.fc_mask(mask_feat).squeeze(2)  # [B, N, K*K, C] -> [B, N, C] (since K=1, K*K=1)
new_mask_preds = torch.einsum("bqc,bchw->bqhw", mask_kernels, features)
```

With `K=1`:
- `mask_feat`: `[B, N, K*K, C]` = `[B, N, 1, C]`
- `mask_kernels = squeeze(2)`: `[B, N, C]` ✅
- `einsum("bqc,bchw->bqhw")`: Works! ✅

## The Problem with K=3

If we use `K=3`:
- `mask_feat`: `[B, N, K*K, C]` = `[B, N, 9, C]`
- `mask_kernels = squeeze(2)`: `[B, N, 9, C]` (can't squeeze if dim=9) ❌
- `einsum("bqc,bchw->bqhw")`: Expects `[B, N, C]`, gets `[B, N, 9, C]` ❌

## Critical Issue: `squeeze(2)` Only Works with K=1!

Looking at line 377-394 in the original code:
```python
# [B, N, C * K * K] -> [B, N, C, K * K] -> [B, N, K * K, C]
obj_feat = k.reshape(B, self.num_proposals, self.hidden_dim, -1).permute(0, 1, 3, 2)
# obj_feat is [B, N, K*K, hidden_dim]

mask_feat = obj_feat  # [B, N, K*K, hidden_dim]
mask_kernels = self.fc_mask(mask_feat).squeeze(2)  # [B, N, K*K, hidden_dim] -> squeeze(2)
```

**The Problem:**
- `squeeze(2)` only removes dimension 2 **if it has size 1**
- With `K=1`: `K*K=1`, so `squeeze(2)` works: `[B, N, 1, hidden_dim]` → `[B, N, hidden_dim]` ✅
- With `K=3`: `K*K=9`, so `squeeze(2)` **fails** - dimension 2 has size 9, not 1! ❌

## Conclusion

**We MUST use `CONV_KERNEL_SIZE_2D = 1` like the original!**

**Reasons:**
1. **Original YOSO uses K=1** - it's the tested, working configuration
2. **`squeeze(2)` requires K=1** - The code assumes `K*K=1` to squeeze dimension 2
3. **Our einsum expects K=1** - `einsum("bqc,bchw->bqhw")` only works with `[B, N, C]` kernels
4. **No expansion needed** - All shapes match naturally with K=1
5. **Simpler implementation** - No expansion logic needed

**If we want K=3**, we would need to:
- Change `squeeze(2)` to handle `K*K=9` (e.g., use `reshape` or `mean/sum` over dimension 2)
- Change the einsum to handle 3×3 kernels: `einsum("bqckk,bchw->bqhw", ...)` 
- But this would require different mask generation logic
- The original YOSO doesn't do this - it uses K=1

## Recommendation

**✅ Changed `conv_kernel_size_2d` from 3 to 1** to match the original YOSO implementation.

This will:
- Remove the need for expansion logic
- Make all shapes match naturally
- Match the original YOSO design
- Fix the `squeeze(2)` issue

