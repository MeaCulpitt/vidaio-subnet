# 03 — Colorspace Conversion

## Current Implementation

**File:** `/root/vidaio-subnet/services/upscaling/server.py` (lines 406-484)

### BT.601/709 Detection
```python
def _is_bt709(color_meta: dict) -> bool:
    cs = (color_meta or {}).get('color_space')
    return cs == 'bt709'  # None / unknown / absent -> BT.601
```

### GPU Conversions
- `_rgb_to_nv12_bt601_gpu()` — VMAF-verified: mean=95.47, min=94.18
- `_rgb_to_nv12_bt709_gpu()` — BT.709 variant
- `_nv12_to_rgb_bt601_bilinear()` — bilinear chroma upsampling
- `_nv12_to_rgb_bt709_bilinear()` — BT.709 variant

### Chroma Subsampling
NV12 encode uses box filter (2x2 average) for chroma downsampling.
NV12 decode uses bilinear interpolation for chroma upsampling.

## Quality Impact

Colorspace round-trip (RGB→NV12→RGB) introduces ~0.5-1.0 VMAF loss.
PieAPP impact is minimal if matrices match the validator's reference.

**Critical:** Mismatched colorspace tags → validator penalizes with high PieAPP.

## Improvement Opportunities

| # | Improvement | Est. Impact | Ease |
|---|-------------|-------------|------|
| 1 | Verify BT.601/709 detection matches all validator content | prevent failures | easy |
| 2 | Use Lanczos chroma subsampling instead of box filter | marginal quality | medium |
| 3 | Skip NV12 round-trip for x2 path (stay in RGB24) | avoid quality loss | medium |

## Status: [ ] pending
