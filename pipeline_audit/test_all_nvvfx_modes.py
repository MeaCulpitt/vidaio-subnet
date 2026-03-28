#!/usr/bin/env python3
"""Test ALL nvidia-vfx quality modes for x2 super resolution.

Phase 1: Encode all modes (nvvfx GPU pipeline)
Phase 2: Measure PieAPP for each output (separate GPU context, crop-based to avoid OOM)
"""

import sys
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
sys.path.insert(0, '/root/vidaio-subnet')

import nvvfx
import torch
import numpy as np
import subprocess
import time
import math
import gc
from pathlib import Path

PAYLOAD = '/root/pipeline_audit/payload_540p_x2.mp4'
REFERENCE = '/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
OUT_DIR = Path('/root/pipeline_audit/outputs')
OUT_DIR.mkdir(exist_ok=True)

NUM_FRAMES = 60
IN_W, IN_H = 960, 540
OUT_W, OUT_H = 1920, 1080

# Scoring functions
def sigmoid(x): return 1/(1+np.exp(-x))
def calc_sq(p):
    s = sigmoid(p)
    a0 = (1-(np.log10(sigmoid(0)+1)/np.log10(3.5)))**2.5
    a2 = (1-(np.log10(sigmoid(2.0)+1)/np.log10(3.5)))**2.5
    v = (1-(np.log10(s+1)/np.log10(3.5)))**2.5
    return 1-((v-a0)/(a2-a0))

def calc_sf(p, cl=10):
    sq = calc_sq(p)
    sl = math.log(1+cl)/math.log(1+320)
    sp = 0.5*sq + 0.5*sl
    sf = 0.1*math.exp(6.979*(sp-0.5))
    return sf, sq

# BT.601 RGB->NV12 on GPU
def rgb_to_nv12_bt601(rgb_chw, h, w):
    R, G, B = rgb_chw[0], rgb_chw[1], rgb_chw[2]
    Y  = (65.481 * R + 128.553 * G + 24.966 * B + 16).clamp_(0, 255)
    Cb = (-37.797 * R - 74.203 * G + 112.0 * B + 128).clamp_(0, 255)
    Cr = (112.0 * R - 93.786 * G - 18.214 * B + 128).clamp_(0, 255)
    Y_plane = Y.to(torch.uint8)
    Cb_sub = Cb.reshape(h//2, 2, w//2, 2).mean(dim=(1,3)).to(torch.uint8)
    Cr_sub = Cr.reshape(h//2, 2, w//2, 2).mean(dim=(1,3)).to(torch.uint8)
    uv = torch.stack([Cb_sub, Cr_sub], dim=2).reshape(h//2, w)
    nv12 = torch.cat([Y_plane, uv], dim=0)
    return nv12


def process_quality_mode(quality_val, mode_name):
    """Process frames with a given quality mode."""
    out_path = OUT_DIR / f'nvvfx_{mode_name}.mp4'

    try:
        vsr = nvvfx.VideoSuperRes(quality=quality_val, device=0)
        vsr.output_width = OUT_W
        vsr.output_height = OUT_H
        vsr.load()
    except Exception as e:
        return None, str(e)

    device = torch.device('cuda:0')
    frame_size = IN_W * IN_H * 3

    decoder = subprocess.Popen(
        ['ffmpeg', '-i', PAYLOAD, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    enc_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'nv12',
        '-s', f'{OUT_W}x{OUT_H}', '-r', '30',
        '-i', 'pipe:0',
        '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
        '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        str(out_path),
    ]
    encoder = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    t0 = time.time()
    count = 0
    try:
        for _ in range(NUM_FRAMES):
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(IN_H, IN_W, 3).copy()
            rgb = (
                torch.from_numpy(frame_np.astype(np.float32) / 255.0)
                .permute(2, 0, 1).contiguous().to(device)
            )
            result = vsr.run(rgb)
            out_gpu = torch.from_dlpack(result.image).clone()
            out_bytes = (
                rgb_to_nv12_bt601(out_gpu.clamp_(0, 1), OUT_H, OUT_W)
                .contiguous().cpu().numpy().tobytes()
            )
            encoder.stdin.write(out_bytes)
            count += 1
    except Exception as e:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        encoder.wait()
        del vsr
        torch.cuda.empty_cache()
        gc.collect()
        return None, f"run() failed: {e}"

    elapsed = time.time() - t0
    decoder.stdout.close()
    decoder.wait()
    encoder.stdin.close()
    encoder.wait()

    del vsr
    torch.cuda.empty_cache()
    gc.collect()

    fps = count / elapsed if elapsed > 0 else 0
    return str(out_path), f"{count}f {fps:.1f}FPS {elapsed:.1f}s"


# ---- PHASE 1: Encode all modes ----
ql = nvvfx.VideoSuperRes.QualityLevel

named_modes = {
    'BICUBIC': ql.BICUBIC,            # 0
    'LOW': ql.LOW,                      # 1
    'MEDIUM': ql.MEDIUM,                # 2
    'HIGH': ql.HIGH,                    # 3
    'ULTRA': ql.ULTRA,                  # 4
    'DENOISE_LOW': ql.DENOISE_LOW,      # 8
    'DENOISE_MEDIUM': ql.DENOISE_MEDIUM,# 9
    'DENOISE_HIGH': ql.DENOISE_HIGH,    # 10
    'DENOISE_ULTRA': ql.DENOISE_ULTRA,  # 11
    'DEBLUR_LOW': ql.DEBLUR_LOW,        # 12
    'DEBLUR_MEDIUM': ql.DEBLUR_MEDIUM,  # 13
    'DEBLUR_HIGH': ql.DEBLUR_HIGH,      # 14
    'DEBLUR_ULTRA': ql.DEBLUR_ULTRA,    # 15
    'HIGHBITRATE_LOW': ql.HIGHBITRATE_LOW,        # 16
    'HIGHBITRATE_MEDIUM': ql.HIGHBITRATE_MEDIUM,  # 17
    'HIGHBITRATE_HIGH': ql.HIGHBITRATE_HIGH,      # 18
    'HIGHBITRATE_ULTRA': ql.HIGHBITRATE_ULTRA,    # 19
}

hidden_ints = [5, 6, 7, 20, 21, 22, 23, 24]

print("=" * 90)
print("NVIDIA VFX x2 Super Resolution — ALL Quality Modes")
print(f"Input: {IN_W}x{IN_H} -> Output: {OUT_W}x{OUT_H}, {NUM_FRAMES} frames")
print("=" * 90)

encode_results = {}  # mode_name -> (val, out_path, info)

for name, qval in named_modes.items():
    print(f"\nEncoding {name} (val={int(qval)})...", end=' ', flush=True)
    out_path, info = process_quality_mode(qval, name)
    if out_path:
        print(info)
        encode_results[name] = (int(qval), out_path, info)
    else:
        print(f"FAILED: {info}")
        encode_results[name] = (int(qval), None, info)

for ival in hidden_ints:
    name = f"HIDDEN_{ival}"
    print(f"\nEncoding hidden int {ival}...", end=' ', flush=True)
    try:
        qval = ql(ival)
        out_path, info = process_quality_mode(qval, name)
        if out_path:
            print(info)
            encode_results[name] = (ival, out_path, info)
        else:
            print(f"FAILED: {info}")
            encode_results[name] = (ival, None, info)
    except (ValueError, Exception) as e:
        print(f"REJECTED: {e}")
        encode_results[name] = (ival, None, f"REJECTED: {e}")

# Free ALL GPU memory before PieAPP
torch.cuda.empty_cache()
gc.collect()

print("\n\n" + "=" * 90)
print("PHASE 2: PieAPP Measurement (crop-based to avoid OOM)")
print("=" * 90)

# ---- PHASE 2: PieAPP per-mode ----
import cv2
import pyiqa

CROP_SIZE = 256  # PieAPP on 256x256 crops to stay within GPU memory
SAMPLE_FRAMES = 20  # Sample 20 frames from the 60

device = torch.device('cuda')
pieapp_metric = pyiqa.create_metric('pieapp', device=device)

def measure_pieapp_crops(processed_path, reference_path):
    """Measure PieAPP using center crops to avoid OOM."""
    ref_cap = cv2.VideoCapture(reference_path)
    proc_cap = cv2.VideoCapture(processed_path)
    if not ref_cap.isOpened() or not proc_cap.isOpened():
        return None

    # Center crop coordinates
    cy, cx = OUT_H // 2, OUT_W // 2
    y1, y2 = cy - CROP_SIZE, cy + CROP_SIZE
    x1, x2 = cx - CROP_SIZE, cx + CROP_SIZE

    scores = []
    frame_idx = 0
    interval = NUM_FRAMES // SAMPLE_FRAMES  # sample every Nth frame

    while frame_idx < NUM_FRAMES:
        ref_ret, ref_frame = ref_cap.read()
        proc_ret, proc_frame = proc_cap.read()
        if not ref_ret or not proc_ret:
            break

        if frame_idx % interval == 0:
            ref_crop = cv2.cvtColor(ref_frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            proc_crop = cv2.cvtColor(proc_frame[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)

            ref_t = torch.from_numpy(ref_crop).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
            proc_t = torch.from_numpy(proc_crop).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0

            with torch.no_grad():
                score = pieapp_metric(proc_t, ref_t)
                val = abs(score.item())
                scores.append(val)

            del ref_t, proc_t
            torch.cuda.empty_cache()

        frame_idx += 1

    ref_cap.release()
    proc_cap.release()
    return np.mean(scores) if scores else None


final_results = []

for name, (val, out_path, info) in encode_results.items():
    if out_path is None:
        final_results.append((name, val, None, None, None, info))
        continue

    print(f"  PieAPP for {name}...", end=' ', flush=True)
    pieapp = measure_pieapp_crops(out_path, REFERENCE)
    if pieapp is not None:
        sf, sq = calc_sf(pieapp)
        flag = " *** PieAPP < 0.092 ***" if pieapp < 0.092 else ""
        print(f"PieAPP={pieapp:.4f} S_Q={sq:.4f} S_F={sf:.4f}{flag}")
        final_results.append((name, val, pieapp, sq, sf, info))
    else:
        print("FAILED")
        final_results.append((name, val, None, None, None, info))


# Summary
print("\n\n" + "=" * 100)
print(f"{'Mode':<22} {'Val':>3}  {'PieAPP':>8}  {'S_Q':>8}  {'S_F':>8}  {'Flag':>15}  {'Info'}")
print("-" * 100)
for name, val, pieapp, sq, sf, info in sorted(final_results, key=lambda x: (x[2] if x[2] is not None else 999)):
    if pieapp is not None:
        flag = "<<< BEST >>>" if pieapp < 0.092 else ""
        print(f"{name:<22} {val:>3}  {pieapp:>8.4f}  {sq:>8.4f}  {sf:>8.4f}  {flag:>15}  {info}")
    else:
        print(f"{name:<22} {val:>3}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}  {'FAILED':>15}  {info}")

print("=" * 100)
print("Current production mode: HIGHBITRATE_ULTRA (19)")
print("=" * 100)
