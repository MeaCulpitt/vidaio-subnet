#!/usr/bin/env python3
"""Benchmark nvvfx 4x super resolution (270p -> 1080p).

Tests multiple quality levels for speed and PieAPP quality.
"""
import sys
sys.path.insert(0, '/root/vidaio-subnet')

import nvvfx
import numpy as np
import subprocess
import json
import time
import math
import os
import tempfile

import torch
import cv2

# ── Config ──────────────────────────────────────────────────────────────
INPUT_VIDEO  = '/root/pipeline_audit/payload_270p.mp4'
REF_VIDEO    = '/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
NFRAMES      = 60
IN_W, IN_H   = 480, 270
OUT_W, OUT_H  = 1920, 1080   # 4x
PIEAPP_SAMPLES = 16
GPU_ID = 0

# ── Quality levels ──────────────────────────────────────────────────────
QUALITIES = [
    ('HIGH',  nvvfx.VideoSuperRes.QualityLevel.HIGH),
    ('ULTRA', nvvfx.VideoSuperRes.QualityLevel.ULTRA),
]
for qname in ('HIGHBITRATE_HIGH', 'HIGHBITRATE_ULTRA'):
    try:
        QUALITIES.append((qname, getattr(nvvfx.VideoSuperRes.QualityLevel, qname)))
    except AttributeError:
        pass

# ── Scoring helpers ─────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def calc_sq(p):
    s = sigmoid(p)
    a0 = (1.0 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
    a2 = (1.0 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
    v  = (1.0 - (np.log10(s + 1) / np.log10(3.5))) ** 2.5
    return 1.0 - ((v - a0) / (a2 - a0))

def calc_sf(p, cl=10):
    sq = calc_sq(p)
    sl = math.log(1 + cl) / math.log(1 + 320)
    sp = 0.5 * sq + 0.5 * sl
    sf = 0.1 * math.exp(6.979 * (sp - 0.5))
    return sf, sq

# ── Decode input frames ────────────────────────────────────────────────
def decode_frames(path, n, w, h):
    """Decode n frames from video as list of numpy arrays (H, W, 3) uint8 RGB."""
    cmd = [
        'ffmpeg', '-i', path,
        '-vframes', str(n),
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw = proc.stdout
    frame_size = w * h * 3
    frames = []
    for i in range(n):
        start = i * frame_size
        end = start + frame_size
        if end > len(raw):
            break
        arr = np.frombuffer(raw[start:end], dtype=np.uint8).reshape(h, w, 3).copy()
        frames.append(arr)
    return frames

# ── NV12 conversion (BT.601) ───────────────────────────────────────────
def rgb_to_nv12_bt601(rgb_chw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Convert float32 CHW [0,1] RGB tensor to NV12 bytes on GPU."""
    rgb = rgb_chw.clamp(0, 1)
    R, G, B = rgb[0], rgb[1], rgb[2]
    Y  = ( 0.299  * R + 0.587  * G + 0.114  * B) * 255.0
    Cb = (-0.1687 * R - 0.3313 * G + 0.5    * B) * 255.0 + 128.0
    Cr = ( 0.5    * R - 0.4187 * G - 0.0813 * B) * 255.0 + 128.0
    Y_u8  = Y.clamp(0, 255).to(torch.uint8)
    Cb_s  = Cb.view(h // 2, 2, w // 2, 2).mean(dim=(1, 3)).clamp(0, 255).to(torch.uint8)
    Cr_s  = Cr.view(h // 2, 2, w // 2, 2).mean(dim=(1, 3)).clamp(0, 255).to(torch.uint8)
    uv = torch.zeros((h // 2, w), dtype=torch.uint8, device=rgb.device)
    uv[:, 0::2] = Cb_s
    uv[:, 1::2] = Cr_s
    return torch.cat([Y_u8, uv], dim=0)

# ── PieAPP measurement ─────────────────────────────────────────────────
def measure_pieapp(test_video, ref_video, n_samples=PIEAPP_SAMPLES):
    """Sample n_samples frames uniformly from both videos, compute mean PieAPP."""
    import pyiqa

    device = torch.device('cuda')
    metric = pyiqa.create_metric('pieapp', device=device)

    # Get frame counts
    def get_frame_count(path):
        cmd = ['ffprobe', '-v', 'error', '-count_frames',
               '-select_streams', 'v:0',
               '-show_entries', 'stream=nb_read_frames',
               '-of', 'csv=p=0', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            return int(r.stdout.strip())
        except:
            # fallback: use nb_frames
            cmd2 = ['ffprobe', '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'stream=nb_frames',
                    '-of', 'csv=p=0', path]
            r2 = subprocess.run(cmd2, capture_output=True, text=True)
            return int(r2.stdout.strip())

    # We'll sample specific frame indices
    # Decode specific frames via seeking
    def decode_frame_at(path, frame_idx, fps):
        ts = frame_idx / fps
        cmd = [
            'ffmpeg', '-ss', f'{ts:.4f}', '-i', path,
            '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return r.stdout

    # Get fps of ref
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', ref_video]
    r = subprocess.run(cmd, capture_output=True, text=True)
    fps_str = r.stdout.strip()
    num, den = fps_str.split('/')
    ref_fps = float(num) / float(den)

    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
           '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', test_video]
    r = subprocess.run(cmd, capture_output=True, text=True)
    fps_str = r.stdout.strip()
    num, den = fps_str.split('/')
    test_fps = float(num) / float(den)

    # Get resolutions
    def get_res(path):
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', path]
        r = subprocess.run(cmd, capture_output=True, text=True)
        parts = r.stdout.strip().split(',')
        return int(parts[0]), int(parts[1])

    test_w, test_h = get_res(test_video)
    ref_w, ref_h = get_res(ref_video)
    print(f"  Test: {test_w}x{test_h}, Ref: {ref_w}x{ref_h}")

    # Use minimum frame count (test video is only 60 frames)
    test_nframes = NFRAMES  # we encoded exactly this many
    indices = np.linspace(0, test_nframes - 1, n_samples, dtype=int)

    scores = []
    for idx in indices:
        # Decode test frame
        test_raw = decode_frame_at(test_video, idx, test_fps)
        if len(test_raw) < test_w * test_h * 3:
            continue
        test_frame = np.frombuffer(test_raw[:test_w*test_h*3],
                                    dtype=np.uint8).reshape(test_h, test_w, 3)

        # Decode corresponding ref frame
        ref_raw = decode_frame_at(ref_video, idx, ref_fps)
        if len(ref_raw) < ref_w * ref_h * 3:
            continue
        ref_frame = np.frombuffer(ref_raw[:ref_w*ref_h*3],
                                   dtype=np.uint8).reshape(ref_h, ref_w, 3)

        # Resize ref to match test dimensions if needed
        if (ref_w, ref_h) != (test_w, test_h):
            ref_frame = cv2.resize(ref_frame, (test_w, test_h),
                                    interpolation=cv2.INTER_LANCZOS4)

        # Convert to torch tensors (B, C, H, W) float32 [0, 1]
        t_test = torch.from_numpy(test_frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        t_ref  = torch.from_numpy(ref_frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            score = metric(t_test, t_ref).item()
        scores.append(abs(score))

    return float(np.mean(scores)), scores

# ── Main benchmark ──────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("NVVFX 4x Super Resolution Benchmark")
    print(f"Input:  {INPUT_VIDEO} ({IN_W}x{IN_H})")
    print(f"Output: {OUT_W}x{OUT_H} (4x)")
    print(f"Ref:    {REF_VIDEO}")
    print("=" * 70)

    # Decode input frames
    print(f"\nDecoding {NFRAMES} input frames...")
    frames = decode_frames(INPUT_VIDEO, NFRAMES, IN_W, IN_H)
    print(f"  Decoded {len(frames)} frames")

    device = torch.device(f'cuda:{GPU_ID}')
    results = {}

    for qname, qlevel in QUALITIES:
        print(f"\n{'─' * 60}")
        print(f"Testing quality: {qname}")
        print(f"{'─' * 60}")

        try:
            # 1. Create VSR instance
            vsr = nvvfx.VideoSuperRes(quality=qlevel, device=GPU_ID)
            vsr.output_width = OUT_W
            vsr.output_height = OUT_H
            print(f"  Loading model (output {OUT_W}x{OUT_H})...")
            vsr.load()
            print(f"  Model loaded OK")

            # 2. Warmup (3 frames)
            for i in range(min(3, len(frames))):
                rgb = (
                    torch.from_numpy(frames[i].astype(np.float32) / 255.0)
                    .permute(2, 0, 1).contiguous().to(device)
                )
                _ = vsr.run(rgb)
            torch.cuda.synchronize()
            print(f"  Warmup done")

            # 3. Benchmark inference
            t0 = time.perf_counter()
            outputs = []
            for i in range(len(frames)):
                rgb = (
                    torch.from_numpy(frames[i].astype(np.float32) / 255.0)
                    .permute(2, 0, 1).contiguous().to(device)
                )
                result = vsr.run(rgb)
                out_gpu = torch.from_dlpack(result.image).clone()
                outputs.append(out_gpu)

            torch.cuda.synchronize()
            t1 = time.perf_counter()

            elapsed = t1 - t0
            fps = len(frames) / elapsed
            time_300 = 300.0 / fps
            print(f"  Inference: {len(frames)} frames in {elapsed:.2f}s = {fps:.1f} FPS")
            print(f"  300 frames extrapolated: {time_300:.1f}s")

            # 4. Encode to HEVC
            out_path = f'/root/pipeline_audit/bench_nvvfx_x4_{qname}.mp4'
            print(f"  Encoding to {out_path}...")

            enc_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-pix_fmt', 'nv12',
                '-s', f'{OUT_W}x{OUT_H}', '-r', '30',
                '-i', 'pipe:0',
                '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
                '-profile:v', 'main', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart',
                out_path,
            ]
            encoder = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

            for out_gpu in outputs:
                nv12 = rgb_to_nv12_bt601(out_gpu, OUT_H, OUT_W)
                encoder.stdin.write(nv12.contiguous().cpu().numpy().tobytes())

            encoder.stdin.close()
            enc_stderr = encoder.stderr.read()
            encoder.wait()
            if encoder.returncode != 0:
                print(f"  Encoder error: {enc_stderr.decode()[:300]}")
                continue

            fsize = os.path.getsize(out_path)
            print(f"  Encoded OK ({fsize / 1024:.0f} KB)")

            # 5. PieAPP
            print(f"  Measuring PieAPP ({PIEAPP_SAMPLES} samples)...")
            pieapp_mean, pieapp_scores = measure_pieapp(out_path, REF_VIDEO)
            sf, sq = calc_sf(pieapp_mean, cl=elapsed)
            bonus = sf > 0.32

            print(f"\n  *** RESULTS for {qname} ***")
            print(f"  4x works:       YES")
            print(f"  FPS (infer):    {fps:.1f}")
            print(f"  300f time:      {time_300:.1f}s")
            print(f"  PieAPP mean:    {pieapp_mean:.4f}")
            print(f"  S_Q:            {sq:.4f}")
            print(f"  S_F:            {sf:.4f}")
            print(f"  BONUS (>0.32):  {'YES' if bonus else 'NO'}")

            results[qname] = {
                'works': True,
                'fps': round(fps, 1),
                'time_300f': round(time_300, 1),
                'pieapp_mean': round(pieapp_mean, 4),
                'pieapp_scores': [round(s, 4) for s in pieapp_scores],
                'sq': round(sq, 4),
                'sf': round(sf, 4),
                'bonus': bonus,
            }

            # Free GPU memory
            del vsr, outputs
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()
            results[qname] = {'works': False, 'error': str(e)}

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    for qname, r in results.items():
        if r.get('works'):
            print(f"  {qname:20s}  FPS={r['fps']:5.1f}  PieAPP={r['pieapp_mean']:.4f}  "
                  f"S_F={r['sf']:.4f}  BONUS={'YES' if r['bonus'] else 'NO'}")
        else:
            print(f"  {qname:20s}  FAILED: {r.get('error', '?')}")

    # Check the critical threshold
    for qname, r in results.items():
        if r.get('works') and r.get('pieapp_mean', 99) < 0.092 and r.get('fps', 0) > 15:
            print(f"\n  *** CRITICAL: {qname} hits PieAPP<0.092 at >{r['fps']} FPS!")
            print(f"  *** The audit could be solved with a one-line code change!")

    # Save JSON
    out_json = '/root/pipeline_audit/nvvfx_x4_benchmark.json'
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

if __name__ == '__main__':
    main()
