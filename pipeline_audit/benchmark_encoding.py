#!/usr/bin/env python3
"""Benchmark: measure PieAPP quality loss from HEVC encoding at different CQ values.

Steps:
1. Load DAT-light x4 via spandrel, upscale 60 frames of payload_270p.mp4 to 1080p
2. Save lossless (ffv1) and hevc_nvenc at CQ=15,16,18,20,23
3. Measure PieAPP of each against ground truth
4. Also test preset p7 at the best CQ
"""

import sys
import os
import time
import math
import subprocess
import tempfile
import numpy as np
import cv2
import torch

sys.path.insert(0, '/root/vidaio-subnet')

from spandrel import ModelLoader

# ── paths ──
MODEL_PATH = os.path.expanduser("~/.cache/span/DAT_light_x4.pth")
INPUT_VIDEO = "/root/pipeline_audit/payload_270p.mp4"
REF_VIDEO = "/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4"
OUT_DIR = "/root/pipeline_audit/encoding_bench"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_FRAMES = 60
PAD_MULT = 64

# ── scoring functions (exact copy) ──
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
    return 0.1*math.exp(6.979*(sp-0.5)), sq


# ── Step 1: Load model ──
print("Loading DAT-light x4 model...")
device = torch.device('cuda')
model = ModelLoader(device="cuda:0").load_from_file(MODEL_PATH).model.eval().half().cuda()
print("Model loaded.")

# ── Step 2: Decode input frames ──
print(f"Decoding {NUM_FRAMES} frames from {INPUT_VIDEO}...")
cap = cv2.VideoCapture(INPUT_VIDEO)
frames_bgr = []
for i in range(NUM_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    frames_bgr.append(frame)
cap.release()
print(f"Decoded {len(frames_bgr)} frames, shape={frames_bgr[0].shape}")

# ── Step 3: Upscale all frames ──
print("Upscaling frames with DAT-light x4 (FP16)...")
upscaled_bgr = []
t0 = time.time()
for i, frame in enumerate(frames_bgr):
    # BGR -> RGB -> tensor
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).half().to(device)

    # Pad to multiple of PAD_MULT
    _, _, h, w = t.shape
    pad_h = (PAD_MULT - h % PAD_MULT) % PAD_MULT
    pad_w = (PAD_MULT - w % PAD_MULT) % PAD_MULT
    if pad_h > 0 or pad_w > 0:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        out = model(t)

    # Remove padding (scale factor 4)
    out_h = h * 4
    out_w = w * 4
    out = out[:, :, :out_h, :out_w]

    # Tensor -> numpy BGR
    out_np = out.squeeze(0).clamp(0, 1).float().mul(255).byte().permute(1, 2, 0).cpu().numpy()
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    upscaled_bgr.append(out_bgr)

    if (i + 1) % 10 == 0:
        print(f"  Upscaled {i+1}/{len(frames_bgr)} frames")

upscale_time = time.time() - t0
print(f"Upscaling done in {upscale_time:.1f}s, output shape={upscaled_bgr[0].shape}")

# Free model memory
del model
torch.cuda.empty_cache()

# ── Step 4: Get FPS from input ──
cap_tmp = cv2.VideoCapture(INPUT_VIDEO)
fps = cap_tmp.get(cv2.CAP_PROP_FPS)
cap_tmp.release()
if fps <= 0:
    fps = 30.0
print(f"Using FPS={fps}")

out_h, out_w = upscaled_bgr[0].shape[:2]

# ── Step 5: Write lossless (FFV1) ──
lossless_path = os.path.join(OUT_DIR, "lossless_ffv1.mkv")
print(f"Writing lossless FFV1 to {lossless_path}...")
t0 = time.time()
# Write frames via ffmpeg pipe
proc = subprocess.Popen([
    'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
    '-s', f'{out_w}x{out_h}', '-r', str(fps),
    '-i', 'pipe:0',
    '-c:v', 'ffv1', '-level', '3', '-slicecrc', '1',
    lossless_path
], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for frame in upscaled_bgr:
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
proc.wait()
lossless_time = time.time() - t0
lossless_size = os.path.getsize(lossless_path)
print(f"  Lossless: {lossless_size/1024/1024:.1f} MB in {lossless_time:.1f}s")

# ── Step 6: Encode with hevc_nvenc at different CQ values ──
cq_values = [15, 16, 18, 20, 23]
encoded_files = {}

for cq in cq_values:
    out_path = os.path.join(OUT_DIR, f"hevc_cq{cq}.mp4")
    print(f"Encoding hevc_nvenc CQ={cq} ...")
    t0 = time.time()
    proc = subprocess.Popen([
        'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{out_w}x{out_h}', '-r', str(fps),
        '-i', 'pipe:0',
        '-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq',
        '-rc', 'vbr', '-cq', str(cq), '-b:v', '0',
        '-pix_fmt', 'yuv420p',
        out_path
    ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for frame in upscaled_bgr:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()
    enc_time = time.time() - t0
    fsize = os.path.getsize(out_path)
    encoded_files[cq] = {'path': out_path, 'time': enc_time, 'size': fsize}
    print(f"  CQ={cq}: {fsize/1024/1024:.1f} MB in {enc_time:.1f}s")

# ── Step 7: Measure PieAPP ──
print("\nLoading PieAPP metric...")
import pyiqa
metric = pyiqa.create_metric('pieapp', device=device)

SAMPLE_COUNT = 16

def measure_pieapp(ref_path, dist_path, sample_count=SAMPLE_COUNT):
    """Measure PieAPP between ref and dist videos, sampling frames uniformly."""
    ref_cap = cv2.VideoCapture(str(ref_path))
    dist_cap = cv2.VideoCapture(str(dist_path))

    ref_count = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dist_count = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(ref_count, dist_count)

    if total < sample_count:
        sample_count = total

    if total > sample_count:
        indices = np.linspace(0, total - 1, sample_count, dtype=int)
    else:
        indices = list(range(total))

    scores = []
    for idx in indices:
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        dist_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret1, ref_frame = ref_cap.read()
        ret2, dist_frame = dist_cap.read()
        if not ret1 or not ret2:
            continue

        # Resize dist to match ref if needed
        if ref_frame.shape[:2] != dist_frame.shape[:2]:
            dist_frame = cv2.resize(dist_frame, (ref_frame.shape[1], ref_frame.shape[0]),
                                     interpolation=cv2.INTER_LANCZOS4)

        ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        dist_rgb = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2RGB)

        ref_t = torch.from_numpy(ref_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        dist_t = torch.from_numpy(dist_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            score = metric(dist_t, ref_t).item()
            if score < 0:
                score = abs(score)
        scores.append(score)

    ref_cap.release()
    dist_cap.release()
    return float(np.mean(scores)) if scores else 5.0


# Measure lossless
print("Measuring PieAPP: lossless vs reference...")
pieapp_lossless = measure_pieapp(REF_VIDEO, lossless_path)
sf_lossless, sq_lossless = calc_sf(pieapp_lossless)
print(f"  Lossless: PieAPP={pieapp_lossless:.4f}  S_Q={sq_lossless:.4f}  S_F={sf_lossless:.4f}")

# Measure each CQ
results = {}
for cq in cq_values:
    info = encoded_files[cq]
    print(f"Measuring PieAPP: CQ={cq} vs reference...")
    pieapp = measure_pieapp(REF_VIDEO, info['path'])
    sf, sq = calc_sf(pieapp)
    results[cq] = {
        'pieapp': pieapp, 'sq': sq, 'sf': sf,
        'size_mb': info['size'] / 1024 / 1024,
        'enc_time': info['time']
    }
    print(f"  CQ={cq}: PieAPP={pieapp:.4f}  S_Q={sq:.4f}  S_F={sf:.4f}  size={info['size']/1024/1024:.1f}MB  time={info['time']:.1f}s")

# ── Step 8: Find best CQ and test p7 preset ──
best_cq = max(results, key=lambda c: results[c]['sf'])
print(f"\nBest CQ by S_F: {best_cq}")

# Encode with p7 at best CQ
p7_path = os.path.join(OUT_DIR, f"hevc_cq{best_cq}_p7.mp4")
print(f"Encoding hevc_nvenc CQ={best_cq} preset=p7 ...")
t0 = time.time()
proc = subprocess.Popen([
    'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
    '-s', f'{out_w}x{out_h}', '-r', str(fps),
    '-i', 'pipe:0',
    '-c:v', 'hevc_nvenc', '-preset', 'p7', '-tune', 'hq',
    '-rc', 'vbr', '-cq', str(best_cq), '-b:v', '0',
    '-pix_fmt', 'yuv420p',
    p7_path
], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for frame in upscaled_bgr:
    proc.stdin.write(frame.tobytes())
proc.stdin.close()
proc.wait()
p7_time = time.time() - t0
p7_size = os.path.getsize(p7_path)

print(f"Measuring PieAPP: CQ={best_cq} p7 vs reference...")
pieapp_p7 = measure_pieapp(REF_VIDEO, p7_path)
sf_p7, sq_p7 = calc_sf(pieapp_p7)

# ── Final results table ──
print("\n" + "="*90)
print(f"ENCODING BENCHMARK RESULTS  (DAT-light x4, {len(frames_bgr)} frames, {out_w}x{out_h})")
print("="*90)
print(f"{'Variant':<25} {'PieAPP':>8} {'S_Q':>8} {'S_F':>8} {'Size(MB)':>10} {'Enc(s)':>8}")
print("-"*90)
print(f"{'Lossless (FFV1)':<25} {pieapp_lossless:>8.4f} {sq_lossless:>8.4f} {sf_lossless:>8.4f} {lossless_size/1024/1024:>10.1f} {lossless_time:>8.1f}")
print("-"*90)
for cq in cq_values:
    r = results[cq]
    label = f"HEVC CQ={cq} (p4)"
    print(f"{label:<25} {r['pieapp']:>8.4f} {r['sq']:>8.4f} {r['sf']:>8.4f} {r['size_mb']:>10.1f} {r['enc_time']:>8.1f}")
print("-"*90)
label_p7 = f"HEVC CQ={best_cq} (p7)"
print(f"{label_p7:<25} {pieapp_p7:>8.4f} {sq_p7:>8.4f} {sf_p7:>8.4f} {p7_size/1024/1024:>10.1f} {p7_time:>8.1f}")
print("="*90)

# Show delta from lossless
print(f"\n{'Encoding quality loss (PieAPP delta from lossless):'}")
print(f"  Lossless baseline PieAPP: {pieapp_lossless:.4f}")
for cq in cq_values:
    delta = results[cq]['pieapp'] - pieapp_lossless
    print(f"  CQ={cq} (p4): +{delta:.4f}")
delta_p7 = pieapp_p7 - pieapp_lossless
print(f"  CQ={best_cq} (p7): +{delta_p7:.4f}")

print("\nDone.")
