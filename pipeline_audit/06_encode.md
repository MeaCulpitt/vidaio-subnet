# 06 — Encoding

## Current Implementation

**File:** `/root/vidaio-subnet/services/upscaling/server.py` (lines 296-308)

### Encoder Settings
```
Codec: hevc_nvenc (NVIDIA GPU-accelerated H.265)
CQ: 20 (constant quality)
Preset: p4 (balanced)
Profile: Main
Pixel Format: yuv420p
SAR: 1:1
Container: MP4 with faststart
Color flags: mirrored from input metadata
```

### Pipe Format
- x4 path: NV12 input (50% less bandwidth than RGB24)
- x2 fallback: RGB24 input
- x2 nvidia-vfx: NV12 input

## Baseline Timing

Encoding is overlapped with inference (streaming pipe), so dedicated encode time
is hard to measure. Estimated ~2-3s for 300 4K frames.

## Quality Impact

Encoding introduces quality loss. CQ=20 is a moderate quality setting.

| CQ Value | Quality | Bitrate (4K) | PieAPP Impact |
|----------|---------|-------------|---------------|
| 15 | very high | ~50-80 Mbps | minimal |
| 18 | high | ~25-40 Mbps | small |
| 20 | moderate (current) | ~15-25 Mbps | measurable |
| 23 | lower | ~8-15 Mbps | significant |

### Validator Requirements (from scoring server)
- Codec: HEVC (for upscaling task)
- Profile: Main or Main 10
- Pixel format: yuv420p
- Container: MP4
- SAR: 1:1
- FPS: must match reference ±0.3
- Colorspace: must match reference tags

## Ranked Improvements

| # | Improvement | Est. PieAPP Gain | Ease | Risk |
|---|-------------|-----------------|------|------|
| 1 | Lower CQ from 20 → 16-18 | 0.005-0.02 | trivial | larger file, upload slower |
| 2 | Use preset p7 (highest quality) | 0.002-0.005 | trivial | slower encode |
| 3 | Use Main 10 profile (10-bit) | ~0.003 | easy | need to verify validator accepts |
| 4 | Tune -rc-lookahead, -b_ref_mode | marginal | medium | untested |

## Status: [ ] pending
