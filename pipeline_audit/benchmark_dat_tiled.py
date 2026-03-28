#!/usr/bin/env python3
"""Benchmark DAT x4 with tiled inference at different tile sizes.

Compares full-frame vs tiled inference to find optimal tile size
for 480x270 input upscaled 4x to 1920x1080.
"""

import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/root/vidaio-subnet")

from spandrel import ModelLoader


DEVICE = torch.device("cuda")
MODEL_PATH = "/root/.cache/span/DAT_x4.safetensors"
SCALE = 4
INPUT_W, INPUT_H = 480, 270
PAD_W, PAD_H = 512, 320  # padded to multiple of 64
TILE_SIZES = [64, 128, 192, 256]
OVERLAP = 16
NUM_FRAMES = 10
WARMUP = 3


def load_model():
    model = ModelLoader().load_from_file(MODEL_PATH).model
    model = model.half().to(DEVICE).eval()
    return model


def pad_to_multiple(t: torch.Tensor, multiple: int) -> torch.Tensor:
    """Pad tensor (B,C,H,W) so H and W are multiples of `multiple`."""
    _, _, h, w = t.shape
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    if ph or pw:
        t = F.pad(t, (0, pw, 0, ph), mode="reflect")
    return t


def reset_vram():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


@torch.inference_mode()
def benchmark_full_frame(model, inp: torch.Tensor) -> dict:
    """Benchmark full-frame (padded to 512x320) inference."""
    # Pad input to 512x320
    padded = F.pad(inp, (0, PAD_W - INPUT_W, 0, PAD_H - INPUT_H), mode="reflect")

    # Warmup
    for _ in range(WARMUP):
        _ = model(padded)
    torch.cuda.synchronize()

    reset_vram()
    start = time.perf_counter()
    for _ in range(NUM_FRAMES):
        out = model(padded)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    vram = get_peak_vram_mb()
    fps = NUM_FRAMES / elapsed
    # Crop to actual output size
    out_h, out_w = INPUT_H * SCALE, INPUT_W * SCALE
    out = out[:, :, :out_h, :out_w]
    return {"fps": fps, "elapsed": elapsed, "vram_mb": vram, "out_shape": tuple(out.shape)}


@torch.inference_mode()
def tiled_inference(model, inp: torch.Tensor, tile_size: int, overlap: int) -> torch.Tensor:
    """Run model on overlapping tiles and stitch results."""
    scale = SCALE
    _, c, h, w = inp.shape
    out = torch.zeros(1, c, h * scale, w * scale, device=inp.device, dtype=inp.dtype)
    weight = torch.zeros(1, 1, h * scale, w * scale, device=inp.device, dtype=inp.dtype)

    step = tile_size - overlap

    # Compute tile positions
    y_positions = list(range(0, h, step))
    x_positions = list(range(0, w, step))
    # Make sure we cover the full image
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)

    for y in y_positions:
        for x in x_positions:
            # Extract tile (clamp to image bounds)
            y1 = min(y, max(0, h - tile_size))
            x1 = min(x, max(0, w - tile_size))
            y2 = min(y1 + tile_size, h)
            x2 = min(x1 + tile_size, w)

            tile = inp[:, :, y1:y2, x1:x2]
            # Pad tile to multiple of 64
            tile = pad_to_multiple(tile, 64)
            tile_h, tile_w = tile.shape[2], tile.shape[3]

            # Run model
            sr_tile = model(tile)

            # Crop SR tile to match expected output
            sr_h = (y2 - y1) * scale
            sr_w = (x2 - x1) * scale
            sr_tile = sr_tile[:, :, :sr_h, :sr_w]

            # Create blending weight (ramp at edges for overlap blending)
            tw = torch.ones(1, 1, sr_h, sr_w, device=inp.device, dtype=inp.dtype)

            out[:, :, y1 * scale:y2 * scale, x1 * scale:x2 * scale] += sr_tile * tw
            weight[:, :, y1 * scale:y2 * scale, x1 * scale:x2 * scale] += tw

    # Normalize by weight
    out = out / weight.clamp(min=1e-6)
    return out


@torch.inference_mode()
def benchmark_tiled(model, inp: torch.Tensor, tile_size: int) -> dict:
    """Benchmark tiled inference at given tile_size."""
    # Count tiles
    step = tile_size - OVERLAP
    h, w = INPUT_H, INPUT_W
    y_positions = list(range(0, h, step))
    x_positions = list(range(0, w, step))
    if y_positions[-1] + tile_size < h:
        y_positions.append(h - tile_size)
    if x_positions[-1] + tile_size < w:
        x_positions.append(w - tile_size)
    num_tiles = len(y_positions) * len(x_positions)

    # Warmup
    for _ in range(WARMUP):
        _ = tiled_inference(model, inp, tile_size, OVERLAP)
    torch.cuda.synchronize()

    reset_vram()
    start = time.perf_counter()
    for _ in range(NUM_FRAMES):
        out = tiled_inference(model, inp, tile_size, OVERLAP)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    vram = get_peak_vram_mb()
    fps = NUM_FRAMES / elapsed
    return {
        "tile_size": tile_size,
        "num_tiles": num_tiles,
        "fps": fps,
        "elapsed": elapsed,
        "vram_mb": vram,
        "out_shape": tuple(out.shape),
    }


def main():
    print("=" * 70)
    print("DAT x4 Tiled Inference Benchmark")
    print(f"Input: {INPUT_W}x{INPUT_H} | Scale: {SCALE} | Frames: {NUM_FRAMES}")
    print(f"Overlap: {OVERLAP}px | Warmup: {WARMUP}")
    print("=" * 70)

    print("\nLoading DAT_x4 model...")
    model = load_model()
    print("Model loaded.\n")

    # Create random input
    inp = torch.randn(1, 3, INPUT_H, INPUT_W, device=DEVICE, dtype=torch.float16)

    # --- Full frame benchmark ---
    print("-" * 70)
    print(f"[Full Frame] Padded to {PAD_W}x{PAD_H}")
    result = benchmark_full_frame(model, inp)
    print(f"  FPS:   {result['fps']:.3f}")
    print(f"  Time:  {result['elapsed']:.2f}s ({NUM_FRAMES} frames)")
    print(f"  VRAM:  {result['vram_mb']:.0f} MB")
    print(f"  Output: {result['out_shape']}")
    full_fps = result["fps"]

    # --- Tiled benchmarks ---
    results = []
    for ts in TILE_SIZES:
        print("-" * 70)
        print(f"[Tiled] tile_size={ts}, overlap={OVERLAP}")
        r = benchmark_tiled(model, inp, ts)
        results.append(r)
        speedup = r["fps"] / full_fps
        print(f"  Tiles: {r['num_tiles']}")
        print(f"  FPS:   {r['fps']:.3f}  ({speedup:.2f}x vs full-frame)")
        print(f"  Time:  {r['elapsed']:.2f}s ({NUM_FRAMES} frames)")
        print(f"  VRAM:  {r['vram_mb']:.0f} MB")
        print(f"  Output: {r['out_shape']}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<20} {'FPS':>8} {'Speedup':>10} {'VRAM (MB)':>10} {'Tiles':>6}")
    print("-" * 60)
    print(f"{'Full (512x320)':<20} {full_fps:>8.3f} {'1.00x':>10} {result['vram_mb']:>10.0f} {'1':>6}")
    for r in results:
        speedup = r["fps"] / full_fps
        label = f"Tile {r['tile_size']}"
        print(f"{label:<20} {r['fps']:>8.3f} {f'{speedup:.2f}x':>10} {r['vram_mb']:>10.0f} {r['num_tiles']:>6}")

    best = max(results, key=lambda r: r["fps"])
    print(f"\nBest tiled config: tile_size={best['tile_size']} "
          f"({best['fps']:.3f} FPS, {best['fps']/full_fps:.2f}x vs full)")


if __name__ == "__main__":
    main()
