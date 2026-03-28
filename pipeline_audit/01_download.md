# 01 — Download

## Current Implementation

Both x2 and x4 paths now stream directly from URL via ffmpeg (no separate download step).

**File:** `/root/vidaio-subnet/services/upscaling/server.py` (line 820-821)
```python
# Both x4 and x2 now stream from URL directly (no download)
processed_video_path = upscale_video(payload_url, task_type, is_url=True)
```

ffmpeg handles the HTTP download internally as part of the decode pipe.

## Baseline Timing

Download is embedded in the decode step. Estimated 2-8s depending on network.

## Improvement Opportunities

| # | Improvement | Est. Impact | Ease |
|---|-------------|-------------|------|
| 1 | Parallel download + decode (prefetch) | marginal (already streaming) | medium |
| 2 | Multi-connection download | ~1-2s faster | medium |

## Status: [ ] pending
