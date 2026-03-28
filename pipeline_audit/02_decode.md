# 02 — Decode

## Current Implementation

**File:** `/root/vidaio-subnet/services/upscaling/server.py`

### ffmpeg CPU Decode (all paths)
```python
decoder = subprocess.Popen(
    ['ffmpeg', '-i', str(input_source),
     '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
)
```

Outputs rgb24 rawvideo frames via pipe. CPU decoding (no hardware acceleration).

### PyNvVideoCodec HW Decode (blocked)
The nvidia-vfx `_upscale_nvvfx()` path uses PyNvVideoCodec for NV12 hardware decode,
but it has color bugs (MAE 8.72 vs reference). The streaming path `_upscale_nvvfx_streaming()`
uses ffmpeg CPU decode instead.

## Baseline Timing

Decode is overlapped with inference in the streaming pipeline. Estimated ~1-3s overhead.

## Improvement Opportunities

| # | Improvement | Est. Impact | Ease |
|---|-------------|-------------|------|
| 1 | NVDEC hardware decode via ffmpeg | ~1s faster | easy |
| 2 | Fix PyNvVideoCodec color bugs | enable full GPU pipeline | hard (NVIDIA bug) |
| 3 | NV12 direct decode (skip rgb24 conversion) | bandwidth saving | medium |

## Status: [ ] pending
