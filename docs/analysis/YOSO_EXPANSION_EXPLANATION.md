# YOSO Expansion Explanation - Original vs Our Implementation

## 🔑 Key Discovery: Original YOSO Uses `CONV_KERNEL_SIZE_2D = 1`!

**This is the root cause of the difference!**

Looking at the original YOSO config (`Wheelchair-Panoptic-Segmentation/projects/PanSeg/model/config/config.py`, line 14):
```python
cfg.MODEL.YOSO.CONV_KERNEL_SIZE_2D = 1  # NOT 3!
cfg.MODEL.YOSO.CONV_KERNEL_SIZE_1D = 3
```

**Our config** (`tcd/configs/tcd_config.py`):
```python
conv_kernel_size_2d=3,  # We use 3, not 1!
```

## Original YOSO Flow (CONV_KERNEL_SIZE_2D = 1)

### Stage 0:
1. `proposal_kernels = self.kernels.weight.clone()` → `[N, C, 1, 1]`
2. `object_kernels = proposal_kernels[None].expand(...)` → `[B, N, C, 1, 1]`
3. In `CrossAttenHead.forward()`:
   - `f = torch.einsum(...)` → `[B, N, C]` = `[B, N, hidden_dim]` = `[B, N, 256]`
   - `k = proposal_kernels.view(B, N, -1)` → `[B, N, C*1*1]` = `[B, N, 256]`
   - **Both `f` and `k` are `[B, N, 256]`** ✅

4. `DySepConvAtten`:
   - `weight_linear = Linear(hidden_dim, ...)` = `Linear(256, ...)` ✅
   - `norm = LayerNorm(hidden_dim)` = `LayerNorm(256)` ✅
   - Input: `[B, N, 256]` ✅
   - Output: `[B, N, 256]` ✅

5. `f_atten_norm`:
   - `LayerNorm(hidden_dim * conv_kernel_size_2d**2)` = `LayerNorm(256 * 1**2)` = `LayerNorm(256)` ✅
   - Input: `[B, N, 256]` ✅

### Stage 1+:
1. `proposal_kernels` returned from previous stage (line 398):
   ```python
   return ..., obj_feat.permute(0, 1, 3, 2).reshape(B, N, hidden_dim, conv_kernel_size_2d, conv_kernel_size_2d)
   ```
   - Since `conv_kernel_size_2d = 1`, this is `[B, N, hidden_dim, 1, 1]` ✅

2. In `CrossAttenHead.forward()`:
   - `f = [B, N, 256]` (from einsum)
   - `k = proposal_kernels.view(B, N, -1)` → `[B, N, 256*1*1]` = `[B, N, 256]` ✅
   - **Both `f` and `k` are still `[B, N, 256]`** ✅

**Conclusion**: With `CONV_KERNEL_SIZE_2D = 1`, everything stays at `[B, N, hidden_dim]` - **NO EXPANSION NEEDED!**

## Our Implementation (CONV_KERNEL_SIZE_2D = 3)

### Stage 0:
1. `proposal_kernels = self.kernels.weight.clone()` → `[N, C, 1, 1]`
2. `object_kernels = proposal_kernels[None].expand(...)` → `[B, N, C, 1, 1]`
3. In `CrossAttenHead.forward()`:
   - `f = torch.einsum(...)` → `[B, N, C]` = `[B, N, 256]`
   - `k = proposal_kernels.view(B, N, -1)` → `[B, N, C*1*1]` = `[B, N, 256]` ❌
   - But `f_atten_norm` expects `[B, N, 256*3*3]` = `[B, N, 2304]` ❌

4. `DySepConvAtten`:
   - `weight_linear = Linear(input_dim, ...)` = `Linear(2304, ...)` ✅
   - `norm = LayerNorm(input_dim)` = `LayerNorm(2304)` ✅
   - But input is `[B, N, 256]` ❌

**Solution**: We need to expand:
- `proposal_kernels`: `[B, N, C, 1, 1]` → `[B, N, C, 3, 3]`
- `k`: `[B, N, C*3*3]` = `[B, N, 2304]`
- `f`: `[B, N, C]` → `[B, N, C*3*3]` = `[B, N, 2304]`

## Why We Need Expansion

| Component | Original (K=1) | Our (K=3) | Why Different? |
|-----------|----------------|-----------|---------------|
| `f` (from einsum) | `[B, N, 256]` | `[B, N, 256]` | Same |
| `k` (from kernels) | `[B, N, 256]` | `[B, N, 256]` → needs `[B, N, 2304]` | We use K=3 |
| `f_atten_norm` expects | `[B, N, 256]` | `[B, N, 2304]` | We use K=3 |
| `DySepConvAtten.input_dim` | `256` | `2304` | We use K=3 |

## Summary

**Original YOSO**: Uses `CONV_KERNEL_SIZE_2D = 1`, so:
- `hidden_dim * conv_kernel_size_2d**2 = 256 * 1**2 = 256`
- Everything is `[B, N, hidden_dim]` = `[B, N, 256]`
- `f_atten_norm = LayerNorm(256)` expects `[B, N, 256]` ✅
- `DySepConvAtten.weight_linear = Linear(256, ...)` expects `[B, N, 256]` ✅
- `f = [B, N, 256]` from einsum ✅
- `k = proposal_kernels.view(B, N, -1)` where `proposal_kernels` is `[B, N, C, 1, 1]` → `k = [B, N, 256]` ✅
- **No expansion needed!** All shapes match naturally.

**Our Implementation**: Uses `CONV_KERNEL_SIZE_2D = 3`, so:
- `hidden_dim * conv_kernel_size_2d**2 = 256 * 3**2 = 256 * 9 = 2304`
- `f_atten_norm = LayerNorm(2304)` expects `[B, N, 2304]` ✅
- `DySepConvAtten.weight_linear = Linear(2304, ...)` expects `[B, N, 2304]` ✅
- `f = [B, N, 256]` from einsum ❌ (needs expansion to `[B, N, 2304]`)
- `k = proposal_kernels.view(B, N, -1)` where `proposal_kernels` is `[B, N, C, 1, 1]` → `k = [B, N, 256]` ❌ (needs expansion to `[B, N, 2304]`)
- **We expand both `f` and `k` to `[B, N, 2304]`** ✅
- This is **correct** for `K=3`!

## Why Different Kernel Sizes?

- **Original YOSO (`K=1`)**: Simpler, no expansion needed, but potentially less expressive
- **Our Implementation (`K=3`)**: More expressive (3x3 kernels can capture more spatial context), but requires expansion

The original didn't face this issue because it uses `K=1`, where `hidden_dim * 1**2 = hidden_dim`, so no expansion is needed!

