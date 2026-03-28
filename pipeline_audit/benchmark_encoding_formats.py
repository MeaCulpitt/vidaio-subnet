#!/usr/bin/env python3
"""Benchmark: find HEVC encoding settings that preserve DAT-light x4 quality.

Problem: DAT-light x4 achieves PieAPP~0.098 lossless, but HEVC CQ=15-23
degrades to 0.21-0.23. We test CQ=1,5,8,10,12,15,20 plus 10-bit and p7 variants.

Validator constraints: HEVC, Main/Main10 profile, yuv420p, MP4, SAR 1:1.
"""

import sys
import os
import time
import math
import subprocess
import tempfile
import json
import numpy as np
import cv2
import torch

sys.path.insert(0, '/root/vidaio-subnet')

from spandrel import ModelLoader

# ── paths ──
MODEL_PATH = os.path.expanduser("~/.cache/span/DAT_light_x4.pth")
INPUT_VIDEO = "/root/pipeline_audit/payload_270p.mp4"
REF_VIDEO = "/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4"
OUT_DIR = "/root/pipeline_audit/encoding_bench_v2"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_FRAMES = 60
PAD_MULT = 64
SAMPLE_COUNT = 16  # match validator sampling

# ── scoring functions ──
def sigmoid(x): return 1 / (1 + np.exp(-x))

def calc_sq(p):
    s = sigmoid(p)
    a0 = (1 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
    a2 = (1 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
    v = (1 - (np.log10(s + 1) / np.log10(3.5))) ** 2.5
    return 1 - ((v - a0) / (a2 - a0))

def calc_sf(p, cl=10):
    sq = calc_sq(p)
    sl = math.log(1 + cl) / math.log(1 + 320)
    sp = 0.5 * sq + 0.5 * sl
    return 0.1 * math.exp(6.979 * (sp - 0.5)), sq

BONUS_THRESHOLD = 0.092  # PieAPP below this -> bonus score

# ── Step 1: Load model ──
print("=" * 80)
print("ENCODING FORMAT BENCHMARK — DAT-light x4")
print("=" * 80)
print("\n[1/7] Loading DAT-light x4 model...")
device = torch.device('cuda')
model = ModelLoader(device="cuda:0").load_from_file(MODEL_PATH).model.eval().half().cuda()
print("  Model loaded on GPU (FP16).")

# ── Step 2: Decode input frames ──
print(f"\n[2/7] Decoding {NUM_FRAMES} frames from {INPUT_VIDEO}...")
cap = cv2.VideoCapture(INPUT_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    fps = 30.0
frames_bgr = []
for i in range(NUM_FRAMES):
    ret, frame = cap.read()
    if not ret:
        break
    frames_bgr.append(frame)
cap.release()
print(f"  Decoded {len(frames_bgr)} frames, shape={frames_bgr[0].shape}, fps={fps}")

# ── Step 3: Upscale all frames ──
print(f"\n[3/7] Upscaling {len(frames_bgr)} frames with DAT-light x4 (FP16)...")
upscaled_bgr = []
t0 = time.time()
for i, frame in enumerate(frames_bgr):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).half().to(device)

    _, _, h, w = t.shape
    pad_h = (PAD_MULT - h % PAD_MULT) % PAD_MULT
    pad_w = (PAD_MULT - w % PAD_MULT) % PAD_MULT
    if pad_h > 0 or pad_w > 0:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='reflect')

    with torch.no_grad():
        out = model(t)

    out_h = h * 4
    out_w = w * 4
    out = out[:, :, :out_h, :out_w]

    out_np = out.squeeze(0).clamp(0, 1).float().mul(255).byte().permute(1, 2, 0).cpu().numpy()
    out_bgr = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
    upscaled_bgr.append(out_bgr)

    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f"  Upscaled {i+1}/{len(frames_bgr)} frames ({elapsed:.1f}s)")

upscale_time = time.time() - t0
out_h, out_w = upscaled_bgr[0].shape[:2]
print(f"  Done in {upscale_time:.1f}s, output={out_w}x{out_h}")

# Free model memory
del model
torch.cuda.empty_cache()

# ── Step 4: Write raw frames to a temp file for reuse ──
print(f"\n[4/7] Writing raw frames to temp file...")
raw_path = os.path.join(OUT_DIR, "raw_frames.rgb")
t0 = time.time()
with open(raw_path, 'wb') as f:
    for frame in upscaled_bgr:
        f.write(frame.tobytes())
raw_size = os.path.getsize(raw_path)
print(f"  Raw file: {raw_size/1024/1024:.1f} MB ({time.time()-t0:.1f}s)")

# ── Step 5: Define encoding variants ──
# All must output HEVC in MP4 with yuv420p and SAR 1:1
VARIANTS = [
    {
        'name': 'HEVC CQ=1 (near-lossless)',
        'filename': 'hevc_cq1.mp4',
        'extra_args': ['-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq',
                       '-rc', 'vbr', '-cq', '1', '-b:v', '0',
                       '-profile:v', 'main', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart'],
    },
    {
        'name': 'HEVC CQ=5',
        'filename': 'hevc_cq5.mp4',
        'extra_args': ['-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq',
                       '-rc', 'vbr', '-cq', '5', '-b:v', '0',
                       '-profile:v', 'main', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart'],
    },
    {
        'name': 'HEVC CQ=8',
        'filename': 'hevc_cq8.mp4',
        'extra_args': ['-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq',
                       '-rc', 'vbr', '-cq', '8', '-b:v', '0',
                       '-profile:v', 'main', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart'],
    },
    {
        'name': 'HEVC CQ=10',
        'filename': 'hevc_cq10.mp4',
        'extra_args': ['-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq',
                       '-rc', 'vbr', '-cq', '10', '-b:v', '0',
                       '-profile:v', 'main', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart'],
    },
    {
        'name': 'HEVC CQ=12',
        'filename': 'hevc_cq12.mp4',
        'extra_args': ['-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq',
                       '-rc', 'vbr', '-cq', '12', '-b:v', '0',
                       '-profile:v', 'main', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart'],
    },
    {
        'name': 'HEVC CQ=15',
        'filename': 'hevc_cq15.mp4',
        'extra_args': ['-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq',
                       '-rc', 'vbr', '-cq', '15', '-b:v', '0',
                       '-profile:v', 'main', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart'],
    },
    {
        'name': 'HEVC CQ=20 (production)',
        'filename': 'hevc_cq20.mp4',
        'extra_args': ['-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq',
                       '-rc', 'vbr', '-cq', '20', '-b:v', '0',
                       '-profile:v', 'main', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart'],
    },
    {
        'name': 'HEVC CQ=1 main10 10bit',
        'filename': 'hevc_cq1_main10.mp4',
        'extra_args': ['-c:v', 'hevc_nvenc', '-preset', 'p4', '-tune', 'hq',
                       '-rc', 'vbr', '-cq', '1', '-b:v', '0',
                       '-profile:v', 'main10', '-pix_fmt', 'yuv420p10le',
                       '-movflags', '+faststart'],
    },
    {
        'name': 'HEVC CQ=10 p7 lookahead32',
        'filename': 'hevc_cq10_p7.mp4',
        'extra_args': ['-c:v', 'hevc_nvenc', '-preset', 'p7', '-tune', 'hq',
                       '-rc', 'vbr', '-cq', '10', '-b:v', '0',
                       '-rc-lookahead', '32',
                       '-profile:v', 'main', '-pix_fmt', 'yuv420p',
                       '-movflags', '+faststart'],
    },
]

# ── Step 6: Encode all variants from raw frames ──
print(f"\n[5/7] Encoding {len(VARIANTS)} variants...")
encode_results = []

for v in VARIANTS:
    out_path = os.path.join(OUT_DIR, v['filename'])
    print(f"  Encoding: {v['name']} -> {v['filename']}")

    cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24',
        '-s', f'{out_w}x{out_h}', '-r', str(fps),
        '-i', raw_path,
        '-vf', 'setsar=1:1',
    ] + v['extra_args'] + [out_path]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    enc_time = time.time() - t0

    if result.returncode != 0:
        print(f"    FAILED: {result.stderr[-300:]}")
        encode_results.append({
            'name': v['name'],
            'filename': v['filename'],
            'path': out_path,
            'enc_time': enc_time,
            'size_mb': 0,
            'failed': True,
            'error': result.stderr[-200:],
        })
        continue

    fsize = os.path.getsize(out_path)
    print(f"    OK: {fsize/1024/1024:.1f} MB in {enc_time:.1f}s")

    # Verify with ffprobe
    probe = subprocess.run([
        'ffprobe', '-v', 'quiet', '-show_entries',
        'stream=codec_name,profile,pix_fmt,width,height,nb_frames',
        '-of', 'json', out_path
    ], capture_output=True, text=True)
    probe_data = json.loads(probe.stdout) if probe.returncode == 0 else {}
    stream = probe_data.get('streams', [{}])[0] if probe_data.get('streams') else {}

    encode_results.append({
        'name': v['name'],
        'filename': v['filename'],
        'path': out_path,
        'enc_time': enc_time,
        'size_mb': fsize / 1024 / 1024,
        'failed': False,
        'codec': stream.get('codec_name', '?'),
        'profile': stream.get('profile', '?'),
        'pix_fmt': stream.get('pix_fmt', '?'),
        'nb_frames': stream.get('nb_frames', '?'),
    })

print(f"\n  Encoded {sum(1 for r in encode_results if not r.get('failed'))} / {len(VARIANTS)} variants successfully.")

# ── Step 7: Measure PieAPP for each variant ──
print(f"\n[6/7] Loading PieAPP metric and measuring quality...")
import pyiqa
metric = pyiqa.create_metric('pieapp', device=device)

# Decode reference frames
print("  Loading reference frames...")
ref_cap = cv2.VideoCapture(REF_VIDEO)
ref_total = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
ref_frames = []
for _ in range(ref_total):
    ret, frame = ref_cap.read()
    if not ret:
        break
    ref_frames.append(frame)
ref_cap.release()
print(f"  Reference: {len(ref_frames)} frames, shape={ref_frames[0].shape}")

# Choose sample indices (match validator: uniform sampling)
sample_indices = np.linspace(0, min(len(ref_frames), NUM_FRAMES) - 1, SAMPLE_COUNT, dtype=int)
print(f"  Sampling {SAMPLE_COUNT} frames at indices: {list(sample_indices)}")


def measure_pieapp_from_file(dist_path):
    """Measure PieAPP between reference and distorted video."""
    dist_cap = cv2.VideoCapture(dist_path)
    dist_frames_list = []
    for _ in range(NUM_FRAMES):
        ret, frame = dist_cap.read()
        if not ret:
            break
        dist_frames_list.append(frame)
    dist_cap.release()

    if len(dist_frames_list) == 0:
        return 5.0

    scores = []
    for idx in sample_indices:
        if idx >= len(ref_frames) or idx >= len(dist_frames_list):
            continue
        ref_frame = ref_frames[idx]
        dist_frame = dist_frames_list[idx]

        # Resize dist to ref size if needed
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

    return float(np.mean(scores)) if scores else 5.0


# Also write a lossless reference: encode raw -> FFV1 for comparison
lossless_path = os.path.join(OUT_DIR, "lossless_ffv1.mkv")
cmd_ll = [
    'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
    '-s', f'{out_w}x{out_h}', '-r', str(fps),
    '-i', raw_path,
    '-c:v', 'ffv1', '-level', '3', '-slicecrc', '1',
    lossless_path
]
subprocess.run(cmd_ll, capture_output=True)
lossless_size = os.path.getsize(lossless_path)

print("  Measuring lossless PieAPP...")
pieapp_ll = measure_pieapp_from_file(lossless_path)
sf_ll, sq_ll = calc_sf(pieapp_ll)
print(f"    Lossless: PieAPP={pieapp_ll:.4f}, S_Q={sq_ll:.4f}, S_F={sf_ll:.4f}")

# Measure each variant
for r in encode_results:
    if r.get('failed'):
        r['pieapp'] = None
        r['sq'] = None
        r['sf'] = None
        continue

    print(f"  Measuring PieAPP for {r['name']}...")
    pieapp = measure_pieapp_from_file(r['path'])
    sf, sq = calc_sf(pieapp)
    r['pieapp'] = pieapp
    r['sq'] = sq
    r['sf'] = sf
    bonus = "BONUS!" if pieapp < BONUS_THRESHOLD else ""
    print(f"    PieAPP={pieapp:.4f}, S_Q={sq:.4f}, S_F={sf:.4f} {bonus}")


# ── Step 7: Print final results ──
print(f"\n\n{'=' * 110}")
print(f"ENCODING FORMAT BENCHMARK RESULTS — DAT-light x4, {len(frames_bgr)} frames, {out_w}x{out_h}")
print(f"Ground truth: {REF_VIDEO}")
print(f"Bonus threshold: PieAPP < {BONUS_THRESHOLD}")
print(f"{'=' * 110}")
header = f"{'Variant':<30} {'Codec':>6} {'Profile':>10} {'PixFmt':>12} {'PieAPP':>8} {'S_Q':>7} {'S_F':>7} {'Size MB':>9} {'Enc(s)':>7} {'Bonus':>6}"
print(header)
print("-" * 110)

# Lossless row
print(f"{'Lossless (FFV1)':<30} {'ffv1':>6} {'—':>10} {'bgr24':>12} {pieapp_ll:>8.4f} {sq_ll:>7.4f} {sf_ll:>7.4f} {lossless_size/1024/1024:>9.1f} {'—':>7} {'YES' if pieapp_ll < BONUS_THRESHOLD else 'no':>6}")
print("-" * 110)

for r in encode_results:
    if r.get('failed'):
        print(f"{r['name']:<30} {'FAILED':>6} {'':>10} {'':>12} {'—':>8} {'—':>7} {'—':>7} {'—':>9} {r['enc_time']:>7.1f} {'':>6}")
        continue

    bonus_str = "YES" if r['pieapp'] < BONUS_THRESHOLD else "no"
    print(f"{r['name']:<30} {r.get('codec','?'):>6} {r.get('profile','?'):>10} {r.get('pix_fmt','?'):>12} {r['pieapp']:>8.4f} {r['sq']:>7.4f} {r['sf']:>7.4f} {r['size_mb']:>9.1f} {r['enc_time']:>7.1f} {bonus_str:>6}")

print("=" * 110)

# Delta analysis
print(f"\nPieAPP delta from lossless ({pieapp_ll:.4f}):")
for r in encode_results:
    if r.get('failed') or r['pieapp'] is None:
        continue
    delta = r['pieapp'] - pieapp_ll
    pct = (delta / pieapp_ll * 100) if pieapp_ll > 0 else 0
    size_ratio = r['size_mb'] / (lossless_size / 1024 / 1024) * 100
    print(f"  {r['name']:<30} delta=+{delta:.4f} ({pct:+.1f}%)  size={size_ratio:.1f}% of lossless")

# Find the threshold CQ
print(f"\n{'=' * 80}")
print("KEY FINDING: CQ threshold for bonus (PieAPP < 0.092)")
print(f"{'=' * 80}")
valid = [(r['name'], r['pieapp'], r['sf'], r['size_mb'])
         for r in encode_results if not r.get('failed') and r['pieapp'] is not None]
valid.sort(key=lambda x: x[1])

for name, pa, sf, sz in valid:
    marker = " <<< BONUS" if pa < BONUS_THRESHOLD else ""
    print(f"  {name:<30} PieAPP={pa:.4f}  S_F={sf:.4f}  size={sz:.1f}MB{marker}")

# Best recommendation
best_bonus = [(r['name'], r['pieapp'], r['sf'], r['size_mb'])
              for r in encode_results
              if not r.get('failed') and r['pieapp'] is not None and r['pieapp'] < BONUS_THRESHOLD]
if best_bonus:
    best_bonus.sort(key=lambda x: x[3])  # smallest file that still gets bonus
    name, pa, sf, sz = best_bonus[0]
    print(f"\n>>> RECOMMENDATION: {name}")
    print(f"    PieAPP={pa:.4f}, S_F={sf:.4f}, file size={sz:.1f} MB")
    print(f"    This is the smallest file that achieves bonus score.")
else:
    # Find closest to threshold
    closest = min(valid, key=lambda x: x[1])
    name, pa, sf, sz = closest
    print(f"\n>>> NO variant achieved bonus threshold PieAPP < {BONUS_THRESHOLD}")
    print(f"    Closest: {name} with PieAPP={pa:.4f}")
    print(f"    The encoding loss is too large even at CQ=1.")

# Cleanup raw file
os.remove(raw_path)
print(f"\nCleaned up raw frames file.")
print("Done.")
