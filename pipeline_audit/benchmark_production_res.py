#!/usr/bin/env python3
"""
Benchmark SR models at ACTUAL production resolution: 960x540 -> 3840x2160 (4x).

Previous benchmarks used 480x270->1080p which is 1/4 of production pixels.
The validator sends 960x540 for SD24K tasks and expects 3840x2160 output.

Ground truth is 1080p so PieAPP can't be used for 4K. This script measures
SPEED (FPS) and VRAM only.

Run 1 results: PLKSR compiled batch=4 = 0.17 FPS (5.8s/frame). Totally unviable.
DAT-light compile took >20min and didn't finish. Skipping compiled DAT variants.
"""

import sys
import time
import gc
import subprocess
import pathlib
import numpy as np

import torch
torch.backends.cudnn.benchmark = True

sys.path.insert(0, "/root/vidaio-subnet")
from spandrel import ModelLoader

# -- Config --
INPUT_VIDEO  = "/root/pipeline_audit/payload_540p.mp4"
INPUT_W, INPUT_H = 960, 540
SCALE = 4
WARMUP_FRAMES = 3
BENCH_FRAMES  = 10
TIMEOUT_S     = 90
OVERHEAD_S    = 8   # decode + encode + upload
TARGET_FRAMES = 300  # 10s @ 30fps
DEVICE = "cuda:0"

# -- Helpers --

def decode_frames(path: str, n: int, w: int, h: int) -> list[np.ndarray]:
    cmd = [
        "ffmpeg", "-i", path,
        "-vf", f"scale={w}:{h}",
        "-pix_fmt", "rgb24", "-f", "rawvideo",
        "-frames:v", str(n),
        "-loglevel", "error", "pipe:1"
    ]
    proc = subprocess.run(cmd, capture_output=True)
    raw = proc.stdout
    frame_bytes = w * h * 3
    frames = []
    for i in range(n):
        arr = np.frombuffer(raw[i*frame_bytes:(i+1)*frame_bytes], dtype=np.uint8).reshape(h, w, 3)
        frames.append(arr)
    return frames


def frames_to_tensor(frames: list[np.ndarray], device: str, dtype: torch.dtype) -> torch.Tensor:
    t = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div_(255.0)
    return t.to(device=device, dtype=dtype)


def pad_to_multiple(tensor: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int, int]:
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h > 0 or pad_w > 0:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, pad_h, pad_w


def get_vram_mb() -> float:
    return torch.cuda.max_memory_allocated(DEVICE) / 1024**2


def reset_vram():
    torch.cuda.reset_peak_memory_stats(DEVICE)
    torch.cuda.empty_cache()
    gc.collect()


def load_model(path: str, dtype: torch.dtype):
    desc = ModelLoader(device=DEVICE).load_from_file(path)
    model = desc.model.eval()
    if dtype == torch.bfloat16:
        model = model.bfloat16()
    else:
        model = model.half()
    return model


def benchmark_model(
    name: str,
    model_path: str,
    dtype: torch.dtype,
    batch_size: int,
    pad_multiple: int,
    compile_model: bool,
    compile_mode: str = "max-autotune",
):
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  dtype={dtype}, batch={batch_size}, pad={pad_multiple}, compile={compile_model}")
    print(f"{'='*70}")

    reset_vram()
    print("Loading model...", end=" ", flush=True)
    model = load_model(model_path, dtype)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{nparams:.2f}M params")

    if compile_model:
        print(f"Compiling with torch.compile(mode='{compile_mode}', dynamic=True)...", flush=True)
        model = torch.compile(model, mode=compile_mode, dynamic=True)

    # Decode frames
    total_needed = WARMUP_FRAMES + BENCH_FRAMES
    raw_frames = decode_frames(INPUT_VIDEO, total_needed, INPUT_W, INPUT_H)
    print(f"Decoded {len(raw_frames)} frames @ {INPUT_W}x{INPUT_H}")

    all_tensors = frames_to_tensor(raw_frames, DEVICE, dtype)
    all_tensors, pad_h, pad_w = pad_to_multiple(all_tensors, pad_multiple)
    _, _, ph, pw = all_tensors.shape
    print(f"Padded input: {pw}x{ph} (pad_h={pad_h}, pad_w={pad_w})")
    print(f"Expected output: {pw*SCALE}x{ph*SCALE}")

    # Warmup
    print(f"Warming up ({WARMUP_FRAMES} frames)...", flush=True)
    reset_vram()
    try:
        for i in range(0, WARMUP_FRAMES, batch_size):
            end_i = min(i + batch_size, WARMUP_FRAMES)
            batch = all_tensors[i:end_i]
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                out = model(batch)
            del out
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  FAILED during warmup: {e}")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return None

    print(f"Warmup done. VRAM after warmup: {get_vram_mb():.0f} MB")

    # Benchmark
    print(f"Benchmarking ({BENCH_FRAMES} frames, batch_size={batch_size})...", flush=True)
    reset_vram()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    frames_done = 0
    try:
        for i in range(0, BENCH_FRAMES, batch_size):
            end_i = min(i + batch_size, BENCH_FRAMES)
            batch = all_tensors[WARMUP_FRAMES + i : WARMUP_FRAMES + end_i]
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=False):
                out = model(batch)
            frames_done += end_i - i
            del out
        torch.cuda.synchronize()
    except Exception as e:
        print(f"  FAILED during benchmark: {e}")
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return None

    elapsed = time.perf_counter() - t0
    vram_peak = get_vram_mb()
    fps = frames_done / elapsed
    ms_per_frame = (elapsed / frames_done) * 1000
    time_for_target = TARGET_FRAMES / fps
    total_time = time_for_target + OVERHEAD_S
    fits = total_time <= TIMEOUT_S

    print(f"\n  Results:")
    print(f"    FPS:           {fps:.2f}")
    print(f"    ms/frame:      {ms_per_frame:.1f}")
    print(f"    Peak VRAM:     {vram_peak:.0f} MB")
    print(f"    Time for {TARGET_FRAMES}f:  {time_for_target:.1f}s")
    print(f"    + {OVERHEAD_S}s overhead: {total_time:.1f}s")
    print(f"    Fits in {TIMEOUT_S}s?  {'YES' if fits else 'NO'}")

    result = {
        "name": name,
        "fps": fps,
        "ms_per_frame": ms_per_frame,
        "vram_mb": vram_peak,
        "time_300f": time_for_target,
        "total_time": total_time,
        "fits_90s": fits,
    }

    del model, all_tensors
    torch.cuda.empty_cache()
    gc.collect()
    return result


def main():
    print("=" * 70)
    print("  PRODUCTION RESOLUTION BENCHMARK: 960x540 -> 3840x2160 (4x)")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Warmup: {WARMUP_FRAMES} frames, Benchmark: {BENCH_FRAMES} frames")
    print(f"Target: {TARGET_FRAMES} frames in <{TIMEOUT_S}s (incl {OVERHEAD_S}s overhead)")

    results = []
    home = pathlib.Path.home()

    # ── Prior result from Run 1 (compiled PLKSR took too long to re-run) ──
    plksr_compiled = {
        "name": "PLKSR x4 FP16 compiled batch=4 (PROD)",
        "fps": 0.17,
        "ms_per_frame": 5807.5,
        "vram_mb": 2340,
        "time_300f": 1742.2,
        "total_time": 1750.2,
        "fits_90s": False,
    }
    results.append(plksr_compiled)
    print(f"\n  [cached] PLKSR compiled: 0.17 FPS, 5807ms/f — from Run 1")

    # 1. PLKSR FP16 eager batch=4 (to see compile overhead vs benefit)
    r = benchmark_model(
        name="PLKSR x4 FP16 eager batch=4",
        model_path=str(home / ".cache/span/PLKSR_X4_DF2K.pth"),
        dtype=torch.float16,
        batch_size=4,
        pad_multiple=4,
        compile_model=False,
    )
    if r: results.append(r)

    # 2. PLKSR FP16 eager batch=1 (less VRAM)
    r = benchmark_model(
        name="PLKSR x4 FP16 eager batch=1",
        model_path=str(home / ".cache/span/PLKSR_X4_DF2K.pth"),
        dtype=torch.float16,
        batch_size=1,
        pad_multiple=4,
        compile_model=False,
    )
    if r: results.append(r)

    # 3. DAT-light FP16 eager batch=1
    r = benchmark_model(
        name="DAT-light x4 FP16 eager batch=1",
        model_path=str(home / ".cache/span/DAT_light_x4.pth"),
        dtype=torch.float16,
        batch_size=1,
        pad_multiple=64,
        compile_model=False,
    )
    if r: results.append(r)

    # 4. DAT-full — try FP16 first, fall back to BF16
    for attempt_dtype, label in [(torch.float16, "FP16"), (torch.bfloat16, "BF16")]:
        r = benchmark_model(
            name=f"DAT-full x4 {label} eager batch=1",
            model_path=str(home / ".cache/span/DAT_x4.safetensors"),
            dtype=attempt_dtype,
            batch_size=1,
            pad_multiple=64,
            compile_model=False,
        )
        if r:
            results.append(r)
            break
        else:
            print(f"  DAT-full {label} failed, trying next dtype...")

    # Summary
    print(f"\n\n{'='*70}")
    print(f"  SUMMARY — 960x540 -> 3840x2160 (4x) on RTX 4090")
    print(f"{'='*70}")
    print(f"{'Model':<42} {'FPS':>6} {'ms/f':>8} {'VRAM':>7} {'300f':>7} {'Total':>7} {'OK?':>4}")
    print("-" * 82)
    for r in results:
        print(
            f"{r['name']:<42} {r['fps']:>6.2f} {r['ms_per_frame']:>8.1f} "
            f"{r['vram_mb']:>6.0f}M {r['time_300f']:>6.1f}s {r['total_time']:>6.1f}s "
            f"{'YES' if r['fits_90s'] else 'NO':>4}"
        )
    print("-" * 82)
    print(f"Target: {TARGET_FRAMES} frames + {OVERHEAD_S}s overhead < {TIMEOUT_S}s timeout")
    print(f"Required FPS: >= {TARGET_FRAMES / (TIMEOUT_S - OVERHEAD_S):.1f}")


if __name__ == "__main__":
    main()
