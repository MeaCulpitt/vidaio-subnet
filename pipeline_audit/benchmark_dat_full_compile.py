#!/usr/bin/env python3
"""Benchmark DAT-full x4 with torch.compile on torch 2.7 (RTX 4090).

Goal: determine if DAT-full x4 (PieAPP=0.049, S_F=0.366 BONUS tier)
can fit 300 frames within 90s timeout using torch.compile speedups.
"""

import sys
sys.path.insert(0, "/root/vidaio-subnet")

import gc
import time
import torch
import spandrel

MODEL_PATH = "/root/.cache/span/DAT_x4.safetensors"
# 480x270 padded to multiples of 64
H_IN, W_IN = 270, 480
PAD_H = (64 - H_IN % 64) % 64  # 50
PAD_W = (64 - W_IN % 64) % 64  # 32
H_PAD, W_PAD = H_IN + PAD_H, W_IN + PAD_W  # 320, 512

WARMUP = 3
BENCH_ITERS = 15
NUM_FRAMES = 300
OVERHEAD_S = 8.0
TIMEOUT_S = 90.0


def load_model(dtype):
    """Load DAT-full x4 fresh from disk."""
    loader = spandrel.ModelLoader(device="cuda")
    m = loader.load_from_file(MODEL_PATH)
    model = m.model
    model.eval()
    if dtype == torch.bfloat16:
        model.bfloat16()
    elif dtype == torch.float16:
        model.half()
    return model


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def benchmark_config(name, dtype, compile_mode=None):
    print(f"\n{'='*70}")
    print(f"Config: {name}")
    print(f"  dtype={dtype}, compile_mode={compile_mode}")
    print(f"  Input: {W_PAD}x{H_PAD} (padded from {W_IN}x{H_IN})")
    print(f"{'='*70}")

    cleanup()

    try:
        model = load_model(dtype)

        if compile_mode is not None:
            print(f"  Compiling with mode={compile_mode}...")
            t0 = time.perf_counter()
            model = torch.compile(model, mode=compile_mode)
            compile_time = time.perf_counter() - t0
            print(f"  torch.compile() call took {compile_time:.1f}s")

        inp = torch.randn(1, 3, H_PAD, W_PAD, device="cuda", dtype=dtype)

        # Warmup
        print(f"  Warming up ({WARMUP} iters)...")
        for i in range(WARMUP):
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(inp)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            print(f"    warmup {i+1}: {elapsed:.3f}s  ({1.0/elapsed:.2f} FPS)")
            del out

        # Benchmark
        print(f"  Benchmarking ({BENCH_ITERS} iters)...")
        times = []
        for i in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(inp)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            times.append(elapsed)
            del out

        vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        avg = sum(times) / len(times)
        fps = 1.0 / avg
        min_t = min(times)
        max_t = max(times)
        fps_best = 1.0 / min_t

        inference_time = NUM_FRAMES / fps
        total_time = inference_time + OVERHEAD_S
        fits = total_time <= TIMEOUT_S

        print(f"\n  Results:")
        print(f"    Avg time/frame:  {avg*1000:.1f} ms")
        print(f"    Min time/frame:  {min_t*1000:.1f} ms")
        print(f"    Max time/frame:  {max_t*1000:.1f} ms")
        print(f"    Avg FPS:         {fps:.2f}")
        print(f"    Best FPS:        {fps_best:.2f}")
        print(f"    Peak VRAM:       {vram_mb:.0f} MB")
        print(f"    ---")
        print(f"    {NUM_FRAMES} frames @ {fps:.2f} FPS = {inference_time:.1f}s inference")
        print(f"    + {OVERHEAD_S:.0f}s overhead = {total_time:.1f}s total")
        print(f"    Timeout: {TIMEOUT_S:.0f}s -> {'FITS' if fits else 'EXCEEDS'}")
        if not fits:
            needed_fps = NUM_FRAMES / (TIMEOUT_S - OVERHEAD_S)
            print(f"    Need >= {needed_fps:.2f} FPS to fit in {TIMEOUT_S:.0f}s")

        del model
        cleanup()

        return {
            "name": name,
            "fps": fps,
            "fps_best": fps_best,
            "avg_ms": avg * 1000,
            "vram_mb": vram_mb,
            "total_time": total_time,
            "fits": fits,
        }

    except Exception as e:
        print(f"\n  FAILED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        cleanup()
        return {"name": name, "fps": 0, "error": str(e)}


def main():
    print(f"torch {torch.__version__}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Input: {W_IN}x{H_IN} -> padded {W_PAD}x{H_PAD}")
    print(f"Output: {W_PAD*4}x{H_PAD*4} = {W_PAD*4}x{H_PAD*4}")
    print(f"Warmup: {WARMUP}, Bench: {BENCH_ITERS}, Frames: {NUM_FRAMES}")

    configs = [
        ("bf16 eager (baseline)",         torch.bfloat16, None),
        ("fp16 eager",                    torch.float16,  None),
        ("fp16 compile default",          torch.float16,  "default"),
        ("fp16 compile max-autotune",     torch.float16,  "max-autotune"),
        ("fp16 compile max-autotune-no-cudagraphs", torch.float16, "max-autotune-no-cudagraphs"),
    ]

    results = []
    for name, dtype, mode in configs:
        r = benchmark_config(name, dtype, mode)
        results.append(r)

    # Summary
    print(f"\n\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<45} {'FPS':>6} {'Best':>6} {'VRAM':>7} {'Total':>7} {'Fit?':>5}")
    print("-" * 80)

    baseline_fps = None
    for r in results:
        if "error" in r:
            print(f"{r['name']:<45} FAILED: {r['error'][:30]}")
            continue
        if baseline_fps is None:
            baseline_fps = r["fps"]
        speedup = r["fps"] / baseline_fps if baseline_fps else 0
        print(
            f"{r['name']:<45} {r['fps']:>5.2f}x {r['fps_best']:>5.2f}x"
            f" {r['vram_mb']:>6.0f}M {r['total_time']:>6.1f}s"
            f" {'YES' if r['fits'] else 'NO':>5}"
            f"  ({speedup:.2f}x)"
        )

    # Final verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    feasible = [r for r in results if r.get("fits")]
    if feasible:
        best = max(feasible, key=lambda r: r["fps"])
        print(f"BEST FEASIBLE: {best['name']}")
        print(f"  {best['fps']:.2f} FPS, {best['total_time']:.1f}s total (within {TIMEOUT_S:.0f}s)")
        print(f"  DAT-full x4 + torch.compile CAN fit in timeout!")
    else:
        best = max(results, key=lambda r: r.get("fps", 0))
        print(f"NO config fits in {TIMEOUT_S:.0f}s timeout.")
        print(f"  Best: {best['name']} at {best.get('fps',0):.2f} FPS, {best.get('total_time',999):.1f}s total")
        needed = NUM_FRAMES / (TIMEOUT_S - OVERHEAD_S)
        print(f"  Need {needed:.2f} FPS minimum.")


if __name__ == "__main__":
    main()
