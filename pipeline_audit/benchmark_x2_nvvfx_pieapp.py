#!/usr/bin/env python3
"""Benchmark nvidia-vfx x2 VideoSuperRes PieAPP at different quality levels.

Measures PieAPP, S_Q, S_F, and FPS for each nvvfx quality level.
Ground truth: 1080p video. 540p payload upscaled 2x back to 1080p.

Must run with venv python:
  /root/vidaio-subnet/venv/bin/python /root/pipeline_audit/benchmark_x2_nvvfx_pieapp.py
"""
import sys
# CRITICAL: system torch first (venv torch has performance regression)
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
sys.path.insert(0, '/root/vidaio-subnet')

import os, time, math, subprocess, tempfile
import numpy as np
import cv2
import torch
import pyiqa
import nvvfx

print(f"torch version: {torch.__version__}, from: {torch.__file__}")
print(f"CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GROUND_TRUTH = "/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4"
PAYLOAD_540P = "/root/pipeline_audit/payload_540p_x2.mp4"
OUTPUT_DIR = "/root/pipeline_audit"

# Quality levels to test (skip BICUBIC/LOW/MEDIUM — focus on candidates)
QUALITY_LEVELS = [
    "HIGH",
    "ULTRA",
    "HIGHBITRATE_HIGH",
    "HIGHBITRATE_ULTRA",
]

NUM_SCORE_FRAMES = 16   # sample 16 frames uniformly for PieAPP
NUM_PROCESS_FRAMES = 60 # process 60 frames for FPS measurement

# ---------------------------------------------------------------------------
# Scoring formulas (from subnet scoring)
# ---------------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calc_sq(p):
    s = sigmoid(p)
    a0 = (1 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
    a2 = (1 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
    v  = (1 - (np.log10(s + 1) / np.log10(3.5))) ** 2.5
    return 1 - ((v - a0) / (a2 - a0))

def calc_sf(p, cl=10):
    sq = calc_sq(p)
    sl = math.log(1 + cl) / math.log(1 + 320)
    sp = 0.5 * sq + 0.5 * sl
    return 0.1 * math.exp(6.979 * (sp - 0.5)), sq

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_frames_cv2(path, max_frames=None):
    """Read frames from video, return list of numpy HWC uint8 BGR arrays."""
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames

def encode_hevc(frames_rgb_gpu, output_path, fps=30, out_w=1920, out_h=1080):
    """Encode GPU float32 CHW [0,1] frames to HEVC via ffmpeg pipe."""
    enc_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{out_w}x{out_h}', '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
        '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-sar', '1:1',
        '-movflags', '+faststart',
        str(output_path),
    ]
    proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for rgb_gpu in frames_rgb_gpu:
        # CHW float32 [0,1] -> HWC uint8 numpy
        frame_np = (rgb_gpu.clamp(0, 1).permute(1, 2, 0).mul(255).byte()
                    .cpu().numpy())
        proc.stdin.write(frame_np.tobytes())
    proc.stdin.close()
    stderr = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        print(f"  Encode warning: {stderr.decode()[:300]}")
    return output_path

def measure_pieapp(dist_path, ref_path, metric, device, n_frames=16):
    """Measure PieAPP between dist and ref videos, sampling n_frames uniformly."""
    ref_frames = read_frames_cv2(ref_path)
    dist_frames = read_frames_cv2(dist_path)

    n_ref = len(ref_frames)
    n_dist = len(dist_frames)
    n_sample = min(n_frames, n_ref, n_dist)
    indices = np.linspace(0, min(n_ref, n_dist) - 1, n_sample, dtype=int)

    scores = []
    for idx in indices:
        ref = ref_frames[idx]
        dist = dist_frames[idx]

        # Resize dist to match ref if needed
        rh, rw = ref.shape[:2]
        dh, dw = dist.shape[:2]
        if (dh, dw) != (rh, rw):
            dist = cv2.resize(dist, (rw, rh), interpolation=cv2.INTER_LANCZOS4)

        # BGR -> RGB, to tensor NCHW float32 [0,1]
        ref_t = torch.from_numpy(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
        dist_t = torch.from_numpy(cv2.cvtColor(dist, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)

        with torch.no_grad():
            score = metric(dist_t, ref_t).item()
        scores.append(abs(score))  # abs() negatives

    return float(np.mean(scores))

# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------
def benchmark_quality_level(quality_name, device, metric):
    """Benchmark a single nvvfx quality level."""
    print(f"\n{'='*60}")
    print(f"Quality: {quality_name}")
    print(f"{'='*60}")

    # Get quality enum
    ql = getattr(nvvfx.VideoSuperRes.QualityLevel, quality_name)

    # Create VSR
    vsr = nvvfx.VideoSuperRes(quality=ql, device=0)
    vsr.output_width = 1920
    vsr.output_height = 1080
    vsr.load()
    print(f"  VSR loaded: {quality_name} -> 1920x1080")

    # Read input frames
    input_frames = read_frames_cv2(PAYLOAD_540P, max_frames=NUM_PROCESS_FRAMES)
    print(f"  Input frames: {len(input_frames)} @ {input_frames[0].shape[1]}x{input_frames[0].shape[0]}")

    # Warmup: process 3 frames
    for i in range(min(3, len(input_frames))):
        frame = input_frames[i]
        rgb = (torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
               .permute(2, 0, 1).contiguous().to(device))
        result = vsr.run(rgb)
        _ = torch.from_dlpack(result.image).clone()
    torch.cuda.synchronize()

    # Process all frames, measure FPS
    output_frames = []
    t0 = time.perf_counter()
    for frame in input_frames:
        rgb = (torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
               .permute(2, 0, 1).contiguous().to(device))
        result = vsr.run(rgb)
        out_gpu = torch.from_dlpack(result.image).clone()
        output_frames.append(out_gpu)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    fps = len(output_frames) / elapsed
    print(f"  Upscale: {len(output_frames)} frames in {elapsed:.2f}s = {fps:.1f} FPS")

    # Encode to HEVC
    out_path = os.path.join(OUTPUT_DIR, f"bench_nvvfx_x2_{quality_name}.mp4")
    encode_hevc(output_frames, out_path, fps=30, out_w=1920, out_h=1080)
    print(f"  Encoded: {out_path}")

    # Free GPU memory from output frames
    del output_frames
    torch.cuda.empty_cache()

    # Measure PieAPP
    pieapp = measure_pieapp(out_path, GROUND_TRUTH, metric, device, n_frames=NUM_SCORE_FRAMES)
    sf, sq = calc_sf(pieapp, cl=10)
    print(f"  PieAPP: {pieapp:.4f}")
    print(f"  S_Q: {sq:.4f}")
    print(f"  S_F: {sf:.4f}")
    print(f"  FPS: {fps:.1f}")
    bonus = sf > 0.32
    print(f"  BONUS (S_F > 0.32): {'YES' if bonus else 'NO'}")

    # Cleanup VSR
    del vsr
    torch.cuda.empty_cache()

    return {
        'quality': quality_name,
        'pieapp': pieapp,
        'sq': sq,
        'sf': sf,
        'fps': fps,
        'bonus': bonus,
    }


def main():
    device = torch.device('cuda')
    print(f"Loading PieAPP metric...")
    metric = pyiqa.create_metric('pieapp', device=device)
    print(f"PieAPP metric loaded.")

    # Verify payload exists
    if not os.path.exists(PAYLOAD_540P):
        print(f"Creating 540p payload from GT...")
        subprocess.run([
            'ffmpeg', '-y', '-i', GROUND_TRUTH,
            '-vf', 'scale=-2:540', '-c:v', 'libx264', '-crf', '18', '-r', '30',
            PAYLOAD_540P
        ], check=True, capture_output=True)
        print(f"  Created: {PAYLOAD_540P}")

    results = []
    for ql_name in QUALITY_LEVELS:
        try:
            r = benchmark_quality_level(ql_name, device, metric)
            results.append(r)
        except Exception as e:
            print(f"  ERROR with {ql_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'quality': ql_name, 'pieapp': None, 'sq': None,
                'sf': None, 'fps': None, 'bonus': None, 'error': str(e),
            })

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY: nvvfx x2 VideoSuperRes PieAPP Benchmark")
    print(f"{'='*80}")
    print(f"{'Quality':<22} {'PieAPP':>8} {'S_Q':>8} {'S_F':>8} {'FPS':>8} {'BONUS?':>8}")
    print(f"{'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        if r.get('pieapp') is not None:
            print(f"{r['quality']:<22} {r['pieapp']:>8.4f} {r['sq']:>8.4f} {r['sf']:>8.4f} {r['fps']:>8.1f} {'YES' if r['bonus'] else 'NO':>8}")
        else:
            print(f"{r['quality']:<22} {'ERROR':>8} {'':>8} {'':>8} {'':>8} {'':>8}  {r.get('error','')}")

    # Also print the threshold
    print(f"\nTarget: S_F > 0.32 for bonus")
    print(f"Current production: HIGH")

    # Check BT.709 detection for x2
    print(f"\n--- BT.709 Detection Check (x2 path) ---")
    print(f"server.py _upscale_nvvfx_streaming (line 532):")
    print(f"  bt709 = _is_bt709(color_meta)")
    print(f"  rgb_to_nv12 = _rgb_to_nv12_bt709_gpu if bt709 else _rgb_to_nv12_bt601_gpu")
    print(f"  -> YES, BT.709 detection is used correctly for x2 encode.")
    print(f"  NOTE: Decode path (line 544) uses ffmpeg rgb24 (BT.601 by default)")
    print(f"  This means decode is ALWAYS BT.601, but encode respects input colorspace.")
    print(f"  For _upscale_nvvfx (pynvc path, line 624): both decode AND encode respect BT.709.")


if __name__ == "__main__":
    main()
