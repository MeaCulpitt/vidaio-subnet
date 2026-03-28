#!/usr/bin/env python3
"""Compile DAT x4 with torch_tensorrt and torch.compile — benchmark all paths."""
import sys, os, time, gc
sys.path.insert(0, '/root/vidaio-subnet')
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
import numpy as np

MODEL_PATH = os.path.expanduser('~/.cache/span/DAT_x4.safetensors')
H, W = 270, 480
PAD_H = (64 - H % 64) % 64
PAD_W = (64 - W % 64) % 64
H_PAD, W_PAD = H + PAD_H, W + PAD_W

print(f"torch: {torch.__version__}")
print(f"Input: {W}x{H}, Padded: {W_PAD}x{H_PAD}")

from spandrel import ModelLoader

def load_model(dtype=torch.float16):
    descriptor = ModelLoader(device="cuda:0").load_from_file(MODEL_PATH)
    model = descriptor.model.eval()
    if dtype == torch.float16:
        model = model.half()
    elif dtype == torch.bfloat16:
        model = model.bfloat16()
    return model.cuda()

def bench(model_fn, dummy, warmup=10, n=40, label=""):
    """Benchmark a callable. Returns FPS."""
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

# --- 1. PyTorch FP16 eager ---
print("\n=== 1. PyTorch FP16 eager ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = load_model(torch.float16)
dummy = torch.randn(1, 3, H_PAD, W_PAD, device='cuda', dtype=torch.float16)
fps_pt, vram_pt = bench(model, dummy, label="FP16 eager")
results['pytorch_fp16'] = (fps_pt, vram_pt)
del model; torch.cuda.empty_cache(); gc.collect()

# --- 2. torch.compile default mode ---
print("\n=== 2. torch.compile (default) ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = load_model(torch.float16)
try:
    compiled = torch.compile(model, mode="default")
    fps_co, vram_co = bench(compiled, dummy, warmup=5, label="compile-default")
    results['compile_default'] = (fps_co, vram_co)
except Exception as e:
    print(f"  FAILED: {e}")
del model; torch.cuda.empty_cache(); gc.collect()

# --- 3. torch.compile max-autotune (no CUDA graphs) ---
print("\n=== 3. torch.compile (max-autotune-no-cudagraphs) ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = load_model(torch.float16)
try:
    compiled = torch.compile(model, mode="max-autotune-no-cudagraphs")
    fps_ma, vram_ma = bench(compiled, dummy, warmup=5, label="max-autotune-no-cg")
    results['compile_max_autotune'] = (fps_ma, vram_ma)
except Exception as e:
    print(f"  FAILED: {e}")
del model; torch.cuda.empty_cache(); gc.collect()

# --- 4. torch_tensorrt via dynamo ---
print("\n=== 4. torch_tensorrt (dynamo) ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = load_model(torch.float16)
try:
    import torch_tensorrt
    trt_model = torch_tensorrt.compile(
        model,
        ir="dynamo",
        inputs=[torch_tensorrt.Input(
            shape=[1, 3, H_PAD, W_PAD],
            dtype=torch.float16,
        )],
        use_explicit_typing=False,
        enabled_precisions={torch.float16},
        workspace_size=8 << 30,
        min_block_size=1,
    )
    fps_trt, vram_trt = bench(trt_model, dummy, label="torch_tensorrt dynamo")
    results['torch_tensorrt'] = (fps_trt, vram_trt)

    # Numerical comparison
    with torch.no_grad():
        model2 = load_model(torch.float16)
        out_pt = model2(dummy)
        out_trt = trt_model(dummy)
        diff = (out_trt.float() - out_pt.float()).abs()
        print(f"  Max diff vs PT: {diff.max().item():.6f}, Mean: {diff.mean().item():.6f}")
        del model2
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
del model; torch.cuda.empty_cache(); gc.collect()

# --- 5. torch_tensorrt via torchscript ---
print("\n=== 5. torch_tensorrt (torchscript) ===")
torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
model = load_model(torch.float16)
try:
    import torch_tensorrt
    trt_ts = torch_tensorrt.compile(
        model,
        ir="torchscript",
        inputs=[torch_tensorrt.Input(
            shape=[1, 3, H_PAD, W_PAD],
            dtype=torch.float16,
        )],
        enabled_precisions={torch.float16},
        workspace_size=8 << 30,
    )
    fps_ts, vram_ts = bench(trt_ts, dummy, label="torch_tensorrt torchscript")
    results['torch_tensorrt_ts'] = (fps_ts, vram_ts)
except Exception as e:
    print(f"  FAILED: {e}")
    import traceback; traceback.print_exc()
del model; torch.cuda.empty_cache(); gc.collect()

# --- Summary ---
print(f"\n{'='*70}")
print(f"SUMMARY — DAT x4 at {W_PAD}x{H_PAD} (padded {W}x{H} input)")
print(f"{'='*70}")
for name, (fps, vram) in sorted(results.items(), key=lambda x: -x[1][0]):
    ext300 = 300 / fps if fps > 0 else 99999
    feasible = "OK" if ext300 < 20 else ("MAYBE" if ext300 < 60 else "TOO SLOW")
    print(f"  {name:<30} {fps:>6.2f} FPS  300f≈{ext300:>5.0f}s  VRAM={vram:>5.0f}MB  [{feasible}]")
print(f"\n  Target: ≥15 FPS (300f in <20s)")
print(f"  Current PLKSR: 19.2 FPS, PieAPP=0.41")
print(f"  DAT quality: PieAPP≈0.064 (BONUS tier)")
