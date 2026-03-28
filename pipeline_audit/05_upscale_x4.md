# 05 — Upscale x4 (SD24K Path)

## Current Implementation

**File:** `/root/vidaio-subnet/services/upscaling/server.py`

### Model Selection
```python
_X4_MAX_PIXELS = 700_000  # ~1050x667
if input_pixels <= 700_000:
    model = PLKSR_X4_DF2K.pth     # Quality model (L1 loss)
else:
    model = realesr-general-x4v3.pth  # GAN fallback (fast, PieAPP>2.0)
```

### PLKSR x4
- **Model:** `~/.cache/span/PLKSR_X4_DF2K.pth`
- **Architecture:** PLKSR (Partial Large Kernel Super-Resolution)
- **Training:** DF2K dataset, L1 loss
- **Scale:** 4x
- **Performance:** ~21ms/frame at 480x270, batch=4, torch.compile, FP16
- **PieAPP:** ~0.07 (BONUS tier at 10s)
- **S_F estimate:** ~0.34 (just above 0.32 target)

### realesr-x4v3 (fallback)
- **Model:** `~/.cache/span/realesr-general-x4v3.pth`
- **Architecture:** GAN-based RealESRGAN
- **PieAPP:** >2.0 (PENALTY — S_F≈0)
- **Only used** for inputs >700k pixels (>1050x667)

### Pipeline Flow
1. ffprobe metadata extraction
2. Optional pre-downscale to 540p (if height > 540)
3. ffmpeg decode → rgb24 rawvideo pipe
4. GPU inference (PLKSR batch=4, FP16, torch.compile)
5. RGB→NV12 on GPU (BT.601/709 matching input)
6. ffmpeg encode (hevc_nvenc, CQ=20)

## Baseline Timing

| Step | Time |
|------|------|
| Probe | ~0.1s |
| Pre-downscale | ~1.5s (if needed) |
| PLKSR inference | ~6s (300 frames @540p) |
| Total x4 path | ~12-20s |

## Known Issues

1. **Frame variance:** PLKSR produces varying quality across frames — some frames have high PieAPP
2. **realesr-x4v3 quality is terrible** — PieAPP>2.0 makes it useless for scoring
3. **Input resolution threshold** at 700k pixels — inputs just above get GAN path

## Comprehensive Model Benchmark (2026-03-27)

**Test conditions:** 480x270 input → 1920x1080 output, 60-frame benchmark, PieAPP on 16 samples

### Full Results (sorted by FPS)

| Model | Params | PieAPP | S_F | FPS | 300f | VRAM | Bonus? |
|-------|--------|--------|-----|-----|------|------|--------|
| PLKSR (prod) | 7.40M | 0.410 | 0.132 | 19.2 | 16s | 694MB | no |
| **DAT-light** | **0.57M** | **0.165** | **0.258** | **11.3** | **27s** | **1500MB** | **no (best non-bonus)** |
| DAT (full) TRT dynamo | 14.80M | ~0.064 | ~0.37 | 3.0 | 100s | 1820MB | YES (too slow) |
| DAT-S (bf16) | 11.21M | ??? | ??? | 2.6 | 113s | 1906MB | FP16 broken |
| DAT-2 (bf16) | 11.21M | ??? | ??? | 2.4 | 124s | 2902MB | FP16 broken |
| DAT (full) FP16 | 14.80M | 0.064 | ~0.37 | 2.0 | 148s | 2878MB | YES (too slow) |
| HAT-S | 9.62M | 0.135 | 0.282 | 1.8 | 168s | 3153MB | no (too slow) |
| DAT-S (fp32) | 11.21M | ??? | ??? | 1.1 | 269s | 3799MB | untested |
| HAT-L (bf16) | ~40M | 0.170 | 0.254 | 0.85 | 352s | 3374MB | no (too slow) |
| SwinIR | 12M | 0.326 | 0.164 | 0.5 | 610s | 1835MB | no |
| DRCT | ~14M | 0.449 | 0.120 | 0.1 | 2322s | 4094MB | no |

### TRT/Optimization Attempts

| Approach | Result | Speedup |
|----------|--------|---------|
| torch_tensorrt dynamo (DAT full) | 3.0 FPS | 1.5x |
| torch.compile default (DAT) | FAILED (C++ codegen bug in torch 2.11) | - |
| torch.compile reduce-overhead | FAILED (CUDAGraphs + self.mean issue) | - |
| ONNX export (DAT full, GPU) | OOM (20GB VRAM during tracing) | - |
| ONNX export (DAT full, CPU) | OOM-killed (>62GB RAM) | - |
| ONNX export (DAT-light, GPU) | OOM (same attention tracing issue) | - |
| Tiled inference (DAT full) | 0.08-0.80x SLOWER than full-frame | negative |
| DAT-light batch=2 | 11.1 f/s (no improvement over b=1) | 0.97x |
| DAT-light batch=4 | 10.8 f/s (slower) | 0.94x |

### Key Findings

1. **No model achieves both PieAPP<0.092 AND FPS≥15.** This is a fundamental quality-speed tradeoff wall for 4x SR on RTX 4090 at 512x320.

2. **DAT-light is the best practical upgrade** over PLKSR:
   - PieAPP: 0.41 → 0.165 (2.5x improvement)
   - S_F: 0.132 → 0.258 (nearly 2x improvement)
   - FPS: 19.2 → 11.3 (40% slower but within 90s timeout at 27s)
   - Does NOT hit bonus tier (S_F>0.32)

3. **TRT gives minimal speedup** for transformer SR models (~1.5x). The attention mechanism is compute-bound, not launch-overhead-bound.

4. **ONNX export is blocked** for all DAT variants — JIT tracing allocates ~20GB+ intermediates.

5. **DAT-S and DAT-2 produce garbage in FP16.** Only FP32/BF16 work.

## Recommended Action

### Option A: Deploy DAT-light (pragmatic improvement)
- Replace PLKSR with DAT-light x4
- S_F improves from 0.132 → 0.258 (still below 0.32 bonus)
- 11.3 FPS → 27s inference for 300 frames (within budget)
- Risk: slower processing may hit timeout on edge cases

### Option B: Keep PLKSR (status quo)
- Faster (19.2 FPS) but worse quality
- Focus optimization effort elsewhere (x2 path, encoding, etc.)

### Option C: Knowledge distillation (hard, high potential)
- Train a PLKSR-architecture model to mimic DAT output
- Could potentially get PieAPP<0.15 at 19+ FPS
- Requires training infrastructure and dataset preparation

### Option D: Wait for better models
- New efficient SR architectures emerge regularly
- Monitor for models that crack the quality-speed barrier

## Status: [x] investigated — DAT TRT path exhausted, DAT-light identified as best alternative
