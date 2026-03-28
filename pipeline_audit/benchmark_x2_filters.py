#!/usr/bin/env python3
"""Benchmark pre/post-processing filters around nvvfx x2 HIGHBITRATE_ULTRA.

Goal: reduce PieAPP from 0.113 toward 0.092 by applying ffmpeg filters
before and/or after nvvfx upscaling.

Run: /root/vidaio-subnet/venv/bin/python /root/pipeline_audit/benchmark_x2_filters.py
"""
import sys
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
sys.path.insert(0, '/root/vidaio-subnet')

import os, time, math, subprocess, tempfile, json
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
NUM_SCORE_FRAMES = 16
NUM_PROCESS_FRAMES = 60

# ---------------------------------------------------------------------------
# Scoring
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

def decode_with_filter(input_path, vf_filter, max_frames=60):
    """Decode video applying an ffmpeg filter, return list of BGR numpy arrays."""
    cmd = ['ffmpeg', '-i', input_path]
    if vf_filter:
        cmd += ['-vf', vf_filter]
    cmd += ['-f', 'rawvideo', '-pix_fmt', 'bgr24', '-v', 'error', 'pipe:1']

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read input resolution
    probe = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0', input_path],
        capture_output=True, text=True
    )
    w, h = map(int, probe.stdout.strip().split(','))

    frame_size = w * h * 3
    frames = []
    while len(frames) < max_frames:
        data = proc.stdout.read(frame_size)
        if len(data) < frame_size:
            break
        frame = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        frames.append(frame.copy())
    proc.stdout.close()
    proc.stderr.close()
    proc.terminate()
    proc.wait()
    return frames

def bgr_to_rgb_tensor(frame_bgr, device):
    """BGR uint8 HWC -> float32 CHW [0,1] contiguous on GPU."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().to(device)

def encode_hevc(frames_rgb_gpu, output_path, fps=30, out_w=1920, out_h=1080, post_filter=None):
    """Encode GPU float32 CHW [0,1] frames to HEVC via ffmpeg pipe."""
    enc_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', f'{out_w}x{out_h}', '-r', str(fps),
        '-i', 'pipe:0',
    ]
    if post_filter:
        enc_cmd += ['-vf', post_filter]
    enc_cmd += [
        '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
        '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-sar', '1:1', '-movflags', '+faststart',
        str(output_path),
    ]
    proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for rgb_gpu in frames_rgb_gpu:
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
        rh, rw = ref.shape[:2]
        dh, dw = dist.shape[:2]
        if (dh, dw) != (rh, rw):
            dist = cv2.resize(dist, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
        ref_t = torch.from_numpy(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
        dist_t = torch.from_numpy(cv2.cvtColor(dist, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).float().div(255).to(device)
        with torch.no_grad():
            score = metric(dist_t, ref_t).item()
        scores.append(abs(score))
    return float(np.mean(scores))

# ---------------------------------------------------------------------------
# Filter configs
# ---------------------------------------------------------------------------
FILTER_CONFIGS = [
    # (name, pre_filter, post_filter)
    ("0_baseline",                    None,                              None),
    # A) Pre-processing
    ("A1_pre_unsharp_mild",           "unsharp=3:3:0.5:3:3:0.5",        None),
    ("A2_pre_unsharp_strong",         "unsharp=5:5:1.0:5:5:1.0",        None),
    ("A3_pre_hqdn3d",                 "hqdn3d=2:2:3:3",                 None),
    ("A4_pre_nlmeans",                "nlmeans=s=3:p=7:r=5",            None),
    ("A5_pre_bilateral",              "bilateral=sigmaS=1:sigmaR=0.1",  None),
    # B) Post-processing
    ("B6_post_sharpen_mild",          None,  "unsharp=3:3:0.3:3:3:0.3"),
    ("B7_post_sharpen_moderate",      None,  "unsharp=5:5:0.5:5:5:0.5"),
    ("B8_post_cas",                   None,  "cas=0.3"),
    ("B9_post_blur_antialias",        None,  "gblur=sigma=0.3"),
]

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_config(name, pre_filter, post_filter, vsr, device, metric):
    print(f"\n{'='*60}")
    print(f"Config: {name}")
    print(f"  Pre-filter:  {pre_filter or 'None'}")
    print(f"  Post-filter: {post_filter or 'None'}")
    print(f"{'='*60}")

    # Decode frames (with optional pre-filter)
    if pre_filter:
        input_frames = decode_with_filter(PAYLOAD_540P, pre_filter, max_frames=NUM_PROCESS_FRAMES)
    else:
        input_frames = read_frames_cv2(PAYLOAD_540P, max_frames=NUM_PROCESS_FRAMES)
    print(f"  Input: {len(input_frames)} frames @ {input_frames[0].shape[1]}x{input_frames[0].shape[0]}")

    # Warmup
    for i in range(min(3, len(input_frames))):
        rgb = bgr_to_rgb_tensor(input_frames[i], device)
        result = vsr.run(rgb)
        _ = torch.from_dlpack(result.image).clone()
    torch.cuda.synchronize()

    # Upscale all frames
    output_frames = []
    t0 = time.perf_counter()
    for frame in input_frames:
        rgb = bgr_to_rgb_tensor(frame, device)
        result = vsr.run(rgb)
        out_gpu = torch.from_dlpack(result.image).clone()
        output_frames.append(out_gpu)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    fps = len(output_frames) / elapsed
    print(f"  Upscale: {len(output_frames)} frames in {elapsed:.2f}s = {fps:.1f} FPS")

    # Encode
    out_path = os.path.join(OUTPUT_DIR, f"bench_filter_{name}.mp4")
    encode_hevc(output_frames, out_path, fps=30, out_w=1920, out_h=1080, post_filter=post_filter)
    print(f"  Encoded: {out_path}")

    del output_frames
    torch.cuda.empty_cache()

    # Measure PieAPP
    pieapp = measure_pieapp(out_path, GROUND_TRUTH, metric, device, n_frames=NUM_SCORE_FRAMES)
    sf, sq = calc_sf(pieapp, cl=10)
    print(f"  PieAPP: {pieapp:.4f}")
    print(f"  S_Q: {sq:.4f}")
    print(f"  S_F: {sf:.4f}")

    return {
        'name': name,
        'pre_filter': pre_filter,
        'post_filter': post_filter,
        'pieapp': pieapp,
        'sq': sq,
        'sf': sf,
        'fps': fps,
    }


def main():
    device = torch.device('cuda')
    print("Loading PieAPP metric...")
    metric = pyiqa.create_metric('pieapp', device=device)
    print("PieAPP metric loaded.")

    # Verify payload
    if not os.path.exists(PAYLOAD_540P):
        print("Creating 540p payload from GT...")
        subprocess.run([
            'ffmpeg', '-y', '-i', GROUND_TRUTH,
            '-vf', 'scale=-2:540', '-c:v', 'libx264', '-crf', '18', '-r', '30',
            PAYLOAD_540P
        ], check=True, capture_output=True)
        print(f"  Created: {PAYLOAD_540P}")

    # Create VSR once and reuse
    vsr = nvvfx.VideoSuperRes(
        quality=nvvfx.VideoSuperRes.QualityLevel.HIGHBITRATE_ULTRA, device=0
    )
    vsr.output_width = 1920
    vsr.output_height = 1080
    vsr.load()
    print("VSR loaded: HIGHBITRATE_ULTRA -> 1920x1080")

    BASELINE_PIEAPP = 0.113
    TARGET_PIEAPP = 0.092

    results = []
    best_pre = None
    best_post = None
    best_pre_pieapp = 999
    best_post_pieapp = 999

    for name, pre_f, post_f in FILTER_CONFIGS:
        try:
            r = run_config(name, pre_f, post_f, vsr, device, metric)
            results.append(r)

            delta = r['pieapp'] - BASELINE_PIEAPP
            print(f"  Delta vs baseline: {delta:+.4f}")

            # Track best pre and post
            if pre_f and not post_f and r['pieapp'] < best_pre_pieapp:
                best_pre_pieapp = r['pieapp']
                best_pre = (name, pre_f)
            if post_f and not pre_f and r['pieapp'] < best_post_pieapp:
                best_post_pieapp = r['pieapp']
                best_post = (name, post_f)

            if r['pieapp'] <= TARGET_PIEAPP:
                print(f"\n*** TARGET REACHED: PieAPP {r['pieapp']:.4f} <= {TARGET_PIEAPP} ***")
                print(f"*** Config: {name} ***")
                # Print summary and stop
                _print_summary(results, BASELINE_PIEAPP)
                return

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': name, 'pre_filter': pre_f, 'post_filter': post_f,
                'pieapp': None, 'sq': None, 'sf': None, 'fps': None, 'error': str(e),
            })

    # C) Combined: best pre + best post
    if best_pre and best_post:
        combo_name = "C10_combined_best"
        print(f"\n--- Combined: best_pre={best_pre[0]}, best_post={best_post[0]} ---")
        try:
            r = run_config(combo_name, best_pre[1], best_post[1], vsr, device, metric)
            results.append(r)
            delta = r['pieapp'] - BASELINE_PIEAPP
            print(f"  Delta vs baseline: {delta:+.4f}")
            if r['pieapp'] <= TARGET_PIEAPP:
                print(f"\n*** TARGET REACHED: PieAPP {r['pieapp']:.4f} <= {TARGET_PIEAPP} ***")
        except Exception as e:
            print(f"  ERROR in combined: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'name': combo_name, 'pre_filter': best_pre[1], 'post_filter': best_post[1],
                'pieapp': None, 'error': str(e),
            })

    del vsr
    torch.cuda.empty_cache()

    _print_summary(results, BASELINE_PIEAPP)


def _print_summary(results, baseline):
    print(f"\n{'='*90}")
    print(f"SUMMARY: nvvfx x2 HIGHBITRATE_ULTRA Filter Benchmark")
    print(f"{'='*90}")
    print(f"{'Config':<30} {'PieAPP':>8} {'S_F':>8} {'Delta':>8} {'FPS':>8}")
    print(f"{'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for r in results:
        if r.get('pieapp') is not None:
            delta = r['pieapp'] - baseline
            print(f"{r['name']:<30} {r['pieapp']:>8.4f} {r['sf']:>8.4f} {delta:>+8.4f} {r['fps']:>8.1f}")
        else:
            print(f"{r['name']:<30} {'ERROR':>8} {'':>8} {'':>8} {'':>8}  {r.get('error','')[:40]}")

    # Find best
    valid = [r for r in results if r.get('pieapp') is not None]
    if valid:
        best = min(valid, key=lambda r: r['pieapp'])
        print(f"\nBest: {best['name']} -> PieAPP={best['pieapp']:.4f}, S_F={best['sf']:.4f}")
        print(f"Baseline: PieAPP={baseline:.4f}")
        print(f"Target: PieAPP<=0.092")


if __name__ == "__main__":
    main()
