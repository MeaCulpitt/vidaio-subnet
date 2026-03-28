# 04 — Upscale x2 (SD2HD / HD24K Path)

## Current Implementation

**File:** `/root/vidaio-subnet/services/upscaling/server.py`

### Primary: nvidia-vfx VideoSuperRes (TensorRT)
- **Library:** `nvvfx` (NVIDIA Video Effects SDK)
- **Quality:** `QualityLevel.HIGH`
- **Output:** input × 2 (e.g., 1080p → 4K 3840×2160)
- **Performance:** ~18s for 300 frames (1080p input), ~17 FPS
- **VMAF:** mean 91.55, min 89.21

### Fallback: SPAN x2
- **Model:** `~/.cache/span/2xHFA2kSPAN.safetensors`
- **Performance:** ~60s for 300 frames, ~5 FPS
- **Used when:** nvidia-vfx import fails or crashes

### Pipeline Flow (nvidia-vfx streaming)
1. ffmpeg URL decode → rgb24 rawvideo pipe
2. GPU: rgb24→float32 CHW [0,1]
3. nvidia-vfx VSR inference (TensorRT)
4. RGB→NV12 on GPU (BT.601/709)
5. ffmpeg encode (hevc_nvenc, CQ=20)

## Baseline Quality

| Metric | Value | Status |
|--------|-------|--------|
| VMAF mean | 91.55 | PASS (>50 gate) |
| VMAF min | 89.21 | PASS |
| PieAPP | **UNKNOWN** | CRITICAL GAP |
| S_F estimate | **UNKNOWN** | Need benchmark |

## Key Issue: PieAPP UNKNOWN

nvidia-vfx is a black-box TensorRT model. We don't know its PieAPP score.
If PieAPP > 0.092, the x2 path will NOT reach S_F > 0.32.

## SOTA Research

### Models to Evaluate

| Model | Architecture | Scale | Params | Speed (est.) | Expected PieAPP |
|-------|-------------|-------|--------|-------------|-----------------|
| nvidia-vfx HIGH | TensorRT (current) | x2 | unknown | ~17 FPS | unknown |
| SPAN x2 (fallback) | SPAN | x2 | ~0.5M | ~5 FPS | unknown |
| HAT-S x2 | Hybrid Attention | x2 | ~10M | ~8 FPS | ~0.03 |
| SwinIR x2 | Swin Transformer | x2 | ~12M | ~6 FPS | ~0.03 |
| RealESRGAN x2 | RRDB+GAN | x2 | ~16M | ~10 FPS | >0.5 (GAN) |
| OmniSR x2 | Omni-SR | x2 | ~0.8M | ~15 FPS | ~0.05 |

### Key Constraint
Must process 300 frames of 1080p in <40s to stay within 90s timeout.
nvidia-vfx at 17 FPS = ~18s. Alternative must be >7 FPS at 1080p resolution.

## Ranked Improvements

| # | Improvement | Est. Impact | Ease |
|---|-------------|-------------|------|
| 1 | Measure nvidia-vfx PieAPP (critical!) | baseline data | easy |
| 2 | Compare SPAN x2 PieAPP vs nvidia-vfx | comparative data | easy |
| 3 | Test HAT-S x2 if nvvfx PieAPP is poor | potentially +0.05 S_F | medium |
| 4 | Self-ensemble on nvidia-vfx output | marginal, 2x slower | medium |

## Status: [ ] pending
