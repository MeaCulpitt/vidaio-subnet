#!/usr/bin/env python3
"""Push DAT-light x4 toward 15 FPS with torch.compile + batch, and re-test DAT-S in fp32."""
import sys, os, json, time, math, gc
import numpy as np
sys.path.insert(0, '/root/vidaio-subnet')
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F

MODEL_PATH = os.path.expanduser('~/.cache/span/DAT_light_x4.pth')
DATS_PATH = os.path.expanduser('~/.cache/span/DAT_S_x4.pth')
H, W = 270, 480
PAD_H = (64 - H % 64) % 64
PAD_W = (64 - W % 64) % 64
H_PAD, W_PAD = H + PAD_H, W + PAD_W

from spandrel import ModelLoader

def bench(model_fn, dummy, warmup=10, n=40, label=""):
    with torch.no_grad():
        for _ in range(warmup):
            _ = model_fn(dummy)
            torch.cuda.synchronize()
    times = []
    with torch.no_grad():
        for _ in range(n):
            t0 = time.time()
            _ = model_fn(dummy)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
    fps = 1.0 / np.mean(times)
    vram = torch.cuda.max_memory_allocated() / 1e6
    print(f"  {label}: {fps:.2f} FPS ({np.mean(times)*1000:.1f} ms/frame), VRAM={vram:.0f}MB")
    return fps, vram

results = {}

# === DAT-light FP16 eager (baseline) ===
print(f"\n=== DAT-light FP16 eager ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = ModelLoader(device="cuda:0").load_from_file(MODEL_PATH).model.eval().half().cuda()
dummy = torch.randn(1, 3, H_PAD, W_PAD, device='cuda', dtype=torch.float16)
fps1, v1 = bench(model, dummy, label="FP16 eager")
results['dat_light_fp16_eager'] = (fps1, v1)
del model; torch.cuda.empty_cache(); gc.collect()

# === DAT-light FP16 + torch.compile (default) ===
print(f"\n=== DAT-light FP16 + torch.compile (default) ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = ModelLoader(device="cuda:0").load_from_file(MODEL_PATH).model.eval().half().cuda()
try:
    compiled = torch.compile(model, mode="default")
    fps2, v2 = bench(compiled, dummy, warmup=5, n=40, label="compile-default")
    results['dat_light_compile_default'] = (fps2, v2)
except Exception as e:
    print(f"  FAILED: {e}")
del model; torch.cuda.empty_cache(); gc.collect()

# === DAT-light FP16 + torch.compile (max-autotune-no-cudagraphs) ===
print(f"\n=== DAT-light FP16 + torch.compile (max-autotune-no-cudagraphs) ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = ModelLoader(device="cuda:0").load_from_file(MODEL_PATH).model.eval().half().cuda()
try:
    compiled = torch.compile(model, mode="max-autotune-no-cudagraphs")
    fps3, v3 = bench(compiled, dummy, warmup=5, n=40, label="max-autotune")
    results['dat_light_compile_mat'] = (fps3, v3)
except Exception as e:
    print(f"  FAILED: {e}")
del model; torch.cuda.empty_cache(); gc.collect()

# === DAT-light FP16 batch=2 ===
print(f"\n=== DAT-light FP16 batch=2 ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = ModelLoader(device="cuda:0").load_from_file(MODEL_PATH).model.eval().half().cuda()
dummy_b2 = torch.randn(2, 3, H_PAD, W_PAD, device='cuda', dtype=torch.float16)
try:
    fps4, v4 = bench(model, dummy_b2, label="FP16 batch=2")
    # Report per-frame FPS
    fps4_per = fps4 * 2
    print(f"  → {fps4_per:.2f} frames/sec (batch throughput)")
    results['dat_light_fp16_b2'] = (fps4_per, v4)
except Exception as e:
    print(f"  FAILED: {e}")
del model; torch.cuda.empty_cache(); gc.collect()

# === DAT-light FP16 batch=4 ===
print(f"\n=== DAT-light FP16 batch=4 ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = ModelLoader(device="cuda:0").load_from_file(MODEL_PATH).model.eval().half().cuda()
dummy_b4 = torch.randn(4, 3, H_PAD, W_PAD, device='cuda', dtype=torch.float16)
try:
    fps5, v5 = bench(model, dummy_b4, label="FP16 batch=4")
    fps5_per = fps5 * 4
    print(f"  → {fps5_per:.2f} frames/sec (batch throughput)")
    results['dat_light_fp16_b4'] = (fps5_per, v5)
except Exception as e:
    print(f"  FAILED: {e}")
del model; torch.cuda.empty_cache(); gc.collect()

# === DAT-S FP32 (fix the FP16 garbage) ===
print(f"\n=== DAT-S x4 FP32 (fixing FP16 garbage) ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = ModelLoader(device="cuda:0").load_from_file(DATS_PATH).model.eval().float().cuda()
dummy_f32 = torch.randn(1, 3, H_PAD, W_PAD, device='cuda', dtype=torch.float32)
try:
    fps6, v6 = bench(model, dummy_f32, warmup=3, n=10, label="DAT-S FP32")
    results['dat_s_fp32'] = (fps6, v6)
except Exception as e:
    print(f"  FAILED: {e}")
del model; torch.cuda.empty_cache(); gc.collect()

# === DAT-S BF16 ===
print(f"\n=== DAT-S x4 BF16 ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = ModelLoader(device="cuda:0").load_from_file(DATS_PATH).model.eval().bfloat16().cuda()
dummy_bf16 = torch.randn(1, 3, H_PAD, W_PAD, device='cuda', dtype=torch.bfloat16)
try:
    fps7, v7 = bench(model, dummy_bf16, warmup=3, n=10, label="DAT-S BF16")
    results['dat_s_bf16'] = (fps7, v7)
except Exception as e:
    print(f"  FAILED: {e}")
del model; torch.cuda.empty_cache(); gc.collect()

# === Summary ===
print(f"\n{'='*70}")
print(f"SUMMARY — all at {W_PAD}x{H_PAD}")
print(f"{'='*70}")
for name, (fps, vram) in sorted(results.items(), key=lambda x: -x[1][0]):
    ext300 = 300 / fps if fps > 0 else 99999
    tag = "TARGET" if fps >= 15 else ("CLOSE" if fps >= 10 else "SLOW")
    print(f"  {name:<35} {fps:>6.1f} FPS  300f≈{ext300:>5.0f}s  VRAM={vram:>5.0f}MB  [{tag}]")
print(f"\n  Need: ≥15 FPS for viable deployment")
