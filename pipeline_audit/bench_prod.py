#!/usr/bin/env python3
"""Benchmark at ACTUAL production resolution: 960x540→3840x2160 (speed only)."""
import sys, os, time, gc
sys.path.insert(0, '/root/vidaio-subnet')
sys.stdout.reconfigure(line_buffering=True)
import torch, torch.nn.functional as F, numpy as np
from spandrel import ModelLoader

S = os.path.expanduser('~/.cache/span')

# Production: 960x540 input, 4x output = 3840x2160
# Also test 480x270 for comparison
CONFIGS = [
    (480, 270, "480x270 (benchmark)"),
    (960, 540, "960x540 (production)"),
]

MODELS = [
    (f'{S}/PLKSR_X4_DF2K.pth', 'PLKSR', 'fp16', 4),  # pad to mult of 4
    (f'{S}/DAT_light_x4.pth', 'DAT-light', 'fp16', 64),  # pad to mult of 64
]

def bench_eager(model_path, name, dtype_str, pad_mult, w, h, n=8):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    model = ModelLoader(device="cuda:0").load_from_file(model_path).model.eval()
    if dtype_str == 'fp16': model = model.half(); dt = torch.float16
    elif dtype_str == 'bf16': model = model.bfloat16(); dt = torch.bfloat16
    else: dt = torch.float32
    model = model.cuda()

    ph = (pad_mult - h % pad_mult) % pad_mult
    pw = (pad_mult - w % pad_mult) % pad_mult
    dummy = torch.randn(1, 3, h+ph, w+pw, device='cuda', dtype=dt)

    with torch.no_grad():
        for _ in range(2): model(dummy); torch.cuda.synchronize()

    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.time()
            model(dummy)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

    fps = 1.0 / np.mean(times)
    vram = torch.cuda.max_memory_allocated() / 1e6
    del model; torch.cuda.empty_cache(); gc.collect()
    return fps, vram

def bench_compiled(model_path, name, dtype_str, pad_mult, w, h, n=10):
    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    model = ModelLoader(device="cuda:0").load_from_file(model_path).model.eval()
    if dtype_str == 'fp16': model = model.half(); dt = torch.float16
    elif dtype_str == 'bf16': model = model.bfloat16(); dt = torch.bfloat16
    else: dt = torch.float32
    model = model.cuda()

    ph = (pad_mult - h % pad_mult) % pad_mult
    pw = (pad_mult - w % pad_mult) % pad_mult
    dummy = torch.randn(1, 3, h+ph, w+pw, device='cuda', dtype=dt)

    compiled = torch.compile(model, mode="max-autotune")
    print(f"    Compiling (first run triggers autotune)...", flush=True)
    with torch.no_grad():
        for i in range(3):
            compiled(dummy); torch.cuda.synchronize()
            print(f"    warmup {i+1}/3 done", flush=True)

    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.time()
            compiled(dummy)
            torch.cuda.synchronize()
            times.append(time.time() - t0)

    fps = 1.0 / np.mean(times)
    vram = torch.cuda.max_memory_allocated() / 1e6
    del model, compiled; torch.cuda.empty_cache(); gc.collect()
    return fps, vram

print("="*70)
print("Production Resolution Benchmark")
print("="*70)

results = []
for w, h, res_label in CONFIGS:
    print(f"\n{'─'*70}")
    print(f"  Resolution: {res_label}")
    print(f"{'─'*70}")

    for model_path, name, dtype_str, pad_mult in MODELS:
        if not os.path.exists(model_path):
            print(f"  SKIP {name}"); continue

        # Eager
        print(f"\n  {name} {dtype_str} eager @ {w}x{h}...", flush=True)
        try:
            fps_e, vram_e = bench_eager(model_path, name, dtype_str, pad_mult, w, h)
            t300 = 300/fps_e
            fit = "YES" if t300 < 82 else "NO"
            print(f"    {fps_e:.2f} FPS, VRAM={vram_e:.0f}MB, 300f={t300:.0f}s [{fit}]")
            results.append((res_label, f"{name} eager", fps_e, vram_e, t300, fit))
        except Exception as e:
            print(f"    FAILED: {e}")

        # Compiled
        print(f"\n  {name} {dtype_str} compiled @ {w}x{h}...", flush=True)
        try:
            fps_c, vram_c = bench_compiled(model_path, name, dtype_str, pad_mult, w, h)
            t300 = 300/fps_c
            fit = "YES" if t300 < 82 else "NO"
            print(f"    {fps_c:.2f} FPS, VRAM={vram_c:.0f}MB, 300f={t300:.0f}s [{fit}]")
            results.append((res_label, f"{name} compiled", fps_c, vram_c, t300, fit))
        except Exception as e:
            print(f"    FAILED: {e}")

# Summary
print(f"\n{'='*70}")
print(f"SUMMARY (90s timeout, 8s overhead = 82s inference budget)")
print(f"{'='*70}")
for res, name, fps, vram, t300, fit in results:
    print(f"  {res:<25} {name:<22} {fps:>6.1f} FPS  300f={t300:>5.0f}s  VRAM={vram:>5.0f}MB  [{fit}]")
