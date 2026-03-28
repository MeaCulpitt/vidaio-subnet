#!/usr/bin/env python3
"""Benchmark DAT-light x4 with torch.compile on torch 2.7.

Tests FP16 eager vs 4 torch.compile modes on 480x270 -> padded 512x320 input.
Target: >= 15 FPS (baseline eager is ~11.3 FPS).
"""

import sys
sys.path.insert(0, "/root/vidaio-subnet")

import gc
import time
import torch
import traceback
from spandrel import ModelLoader

MODEL_PATH = "/root/.cache/span/DAT_light_x4.pth"
WARMUP = 5
BENCHMARK = 40
# 480x270 padded to 512x320
PAD_H, PAD_W = 50, 32
INPUT_H, INPUT_W = 270 + PAD_H, 480 + PAD_W  # 320, 512


def load_model():
    """Load DAT-light x4 via spandrel, FP16 CUDA."""
    m = ModelLoader().load_from_file(MODEL_PATH)
    model = m.model.half().cuda().eval()
    return model


def get_vram_mb():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_vram():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    gc.collect()


def run_benchmark(label, model, inp, use_cudagraph_mark=False):
    """Run warmup + benchmark, return (fps, vram_mb) or (error_str,)."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    reset_vram()

    # Warmup
    try:
        for i in range(WARMUP):
            if use_cudagraph_mark:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model(inp)
            torch.cuda.synchronize()
        print(f"  Warmup {WARMUP} iters OK")
    except Exception as e:
        msg = f"  FAILED during warmup: {e}"
        print(msg)
        traceback.print_exc()
        return (msg,)

    # Benchmark
    reset_vram()
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for i in range(BENCHMARK):
            if use_cudagraph_mark:
                torch.compiler.cudagraph_mark_step_begin()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
                _ = model(inp)
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        fps = BENCHMARK / elapsed
        vram = get_vram_mb()
        print(f"  {BENCHMARK} iters in {elapsed:.2f}s  =>  {fps:.2f} FPS")
        print(f"  Peak VRAM: {vram:.0f} MB")
        return (fps, vram)

    except Exception as e:
        msg = f"  FAILED during benchmark: {e}"
        print(msg)
        traceback.print_exc()
        return (msg,)


def main():
    print(f"torch {torch.__version__}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Input shape: 1x3x{INPUT_H}x{INPUT_W} (FP16)")
    print(f"Warmup: {WARMUP}, Benchmark: {BENCHMARK} iters")

    inp = torch.randn(1, 3, INPUT_H, INPUT_W, dtype=torch.float16, device="cuda")

    configs = [
        ("1. FP16 eager (baseline)", None, False),
        ("2. compile mode='default' FP16", "default", False),
        ("3. compile mode='reduce-overhead' FP16", "reduce-overhead", False),
        ("4. compile mode='max-autotune' FP16", "max-autotune", False),
        ("5. compile mode='max-autotune-no-cudagraphs' FP16", "max-autotune-no-cudagraphs", False),
    ]

    results = {}

    for label, mode, cudagraph_mark in configs:
        # Reload model fresh for each config to avoid state leakage
        model = load_model()

        if mode is not None:
            try:
                model = torch.compile(model, mode=mode)
                print(f"\n  torch.compile(mode='{mode}') applied")
            except Exception as e:
                results[label] = (f"compile failed: {e}",)
                print(f"\n  torch.compile(mode='{mode}') FAILED: {e}")
                del model
                gc.collect()
                torch.cuda.empty_cache()
                continue

        res = run_benchmark(label, model, inp, use_cudagraph_mark=cudagraph_mark)

        # If reduce-overhead failed, retry with cudagraph_mark_step_begin
        if mode == "reduce-overhead" and len(res) == 1:
            print("\n  >> Retrying reduce-overhead with cudagraph_mark_step_begin()...")
            # Need fresh model + compile
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = load_model()
            model = torch.compile(model, mode="reduce-overhead")
            res = run_benchmark(
                "3b. reduce-overhead + cudagraph_mark_step_begin",
                model, inp, use_cudagraph_mark=True
            )
            results["3b. reduce-overhead + cudagraph_mark"] = res

        results[label] = res

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n\n{'='*60}")
    print("  SUMMARY — DAT-light x4 torch.compile benchmark (torch 2.7)")
    print(f"{'='*60}")
    print(f"  {'Config':<50} {'FPS':>7}  {'VRAM':>8}  {'Status'}")
    print(f"  {'-'*50} {'-'*7}  {'-'*8}  {'-'*10}")

    for label, res in results.items():
        if len(res) == 2:
            fps, vram = res
            status = "PASS" if fps >= 15 else "BELOW 15"
            print(f"  {label:<50} {fps:>7.2f}  {vram:>7.0f}MB  {status}")
        else:
            print(f"  {label:<50} {'N/A':>7}  {'N/A':>8}  FAILED")

    print()


if __name__ == "__main__":
    main()
