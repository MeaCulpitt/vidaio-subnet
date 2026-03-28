#!/usr/bin/env python3
"""Benchmark encoding format variations for the x2 nvvfx HIGHBITRATE_ULTRA path.

Tests whether encoding overhead (yuv420p chroma subsampling, CQ compression)
adds measurable PieAPP loss vs the raw nvvfx output.

Validator REQUIRES yuv420p (server.py:1375) — yuv444p variants skipped.

Variants tested:
  a) yuv420p CQ=20           — current production baseline
  b) yuv420p CQ=1            — near-lossless
  e) yuv420p CQ=20 lanczos   — better chroma downsampling filter
  f) yuv420p CQ=20 bt709     — explicit colorspace tags
  g) FFV1 RGB lossless        — theoretical max (no encode loss)
  h) Raw nvvfx (no encode)    — theoretical ceiling

Run:
  /root/vidaio-subnet/venv/bin/python /root/pipeline_audit/benchmark_x2_encoding.py
"""
import sys
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
sys.path.insert(0, '/root/vidaio-subnet')

import os, time, math, subprocess, tempfile
import numpy as np
import cv2
import torch
import pyiqa
import nvvfx

print(f"torch: {torch.__version__} from {torch.__file__}")
print(f"CUDA: {torch.cuda.get_device_name(0)}")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GROUND_TRUTH = "/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4"
PAYLOAD_540P = "/root/pipeline_audit/payload_540p_x2.mp4"
OUTPUT_DIR   = "/root/pipeline_audit/encoding_bench_x2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_SCORE_FRAMES = 16   # PieAPP frames sampled uniformly
CLIP_SEC = 10           # for S_F length score

# ---------------------------------------------------------------------------
# Scoring (exact validator formulas)
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

def measure_pieapp(dist_frames_bgr, ref_frames_bgr, metric, device, n_frames=16):
    """Measure PieAPP between two lists of BGR frames, sampling uniformly."""
    n = min(n_frames, len(dist_frames_bgr), len(ref_frames_bgr))
    indices = np.linspace(0, min(len(dist_frames_bgr), len(ref_frames_bgr)) - 1, n, dtype=int)
    scores = []
    for idx in indices:
        ref = ref_frames_bgr[idx]
        dist = dist_frames_bgr[idx]
        # Resize if needed
        rh, rw = ref.shape[:2]
        dh, dw = dist.shape[:2]
        if (dh, dw) != (rh, rw):
            dist = cv2.resize(dist, (rw, rh), interpolation=cv2.INTER_LANCZOS4)
        ref_t = (torch.from_numpy(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
                 .permute(2, 0, 1).unsqueeze(0).float().div(255).to(device))
        dist_t = (torch.from_numpy(cv2.cvtColor(dist, cv2.COLOR_BGR2RGB))
                  .permute(2, 0, 1).unsqueeze(0).float().div(255).to(device))
        with torch.no_grad():
            score = metric(dist_t, ref_t).item()
        scores.append(abs(score))
    return float(np.mean(scores)), scores

def file_size_mb(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0

# ---------------------------------------------------------------------------
# Encoding variants — each takes list of RGB uint8 HWC numpy arrays
# ---------------------------------------------------------------------------
def encode_variant_hevc(frames_rgb, out_path, cq=20, extra_input_args=None,
                        extra_output_args=None, pix_fmt="yuv420p"):
    """Encode RGB uint8 HWC frames to HEVC mp4."""
    h, w = frames_rgb[0].shape[:2]
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{w}x{h}', '-r', '30', '-i', 'pipe:0']
    if extra_input_args:
        cmd.extend(extra_input_args)
    cmd.extend([
        '-c:v', 'hevc_nvenc', '-cq', str(cq), '-preset', 'p4',
        '-profile:v', 'main', '-pix_fmt', pix_fmt,
        '-sar', '1:1', '-movflags', '+faststart',
    ])
    if extra_output_args:
        cmd.extend(extra_output_args)
    cmd.append(str(out_path))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames_rgb:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()[-500:]
        print(f"    ENCODE ERROR: {stderr}")
    return out_path

def encode_ffv1_lossless(frames_rgb, out_path):
    """Encode to FFV1 RGB lossless (mkv)."""
    h, w = frames_rgb[0].shape[:2]
    cmd = ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{w}x{h}', '-r', '30', '-i', 'pipe:0',
           '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'gbrp',
           str(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames_rgb:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()[-500:]
        print(f"    FFV1 ENCODE ERROR: {stderr}")
    return out_path

def decode_to_bgr_frames(video_path, max_frames=None):
    """Decode any video file to BGR frames via cv2."""
    return read_frames_cv2(video_path, max_frames)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device('cuda')
    print(f"\nLoading PieAPP metric...")
    metric = pyiqa.create_metric('pieapp', device=device)
    print(f"PieAPP loaded.\n")

    # Ensure 540p payload exists
    if not os.path.exists(PAYLOAD_540P):
        print(f"Creating 540p payload from GT...")
        subprocess.run([
            'ffmpeg', '-y', '-i', GROUND_TRUTH,
            '-vf', 'scale=-2:540', '-c:v', 'libx264', '-crf', '18', '-r', '30',
            PAYLOAD_540P
        ], check=True, capture_output=True)

    # Read ground truth frames
    print(f"Reading ground truth: {GROUND_TRUTH}")
    gt_frames = read_frames_cv2(GROUND_TRUTH)
    print(f"  GT: {gt_frames[0].shape[1]}x{gt_frames[0].shape[0]}, {len(gt_frames)} frames")

    # Read 540p input
    print(f"Reading payload: {PAYLOAD_540P}")
    input_frames = read_frames_cv2(PAYLOAD_540P)
    print(f"  Input: {input_frames[0].shape[1]}x{input_frames[0].shape[0]}, {len(input_frames)} frames")

    # -----------------------------------------------------------------------
    # Step 1: Run nvvfx HIGHBITRATE_ULTRA, capture lossless RGB intermediate
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Running nvvfx HIGHBITRATE_ULTRA upscale (540p -> 1080p)...")
    print(f"{'='*70}")

    ql = nvvfx.VideoSuperRes.QualityLevel.HIGHBITRATE_ULTRA
    vsr = nvvfx.VideoSuperRes(quality=ql, device=0)
    vsr.output_width = 1920
    vsr.output_height = 1080
    vsr.load()
    print(f"  VSR loaded: HIGHBITRATE_ULTRA -> 1920x1080")

    # Warmup
    for i in range(min(3, len(input_frames))):
        rgb_in = (torch.from_numpy(
            cv2.cvtColor(input_frames[i], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            .permute(2, 0, 1).contiguous().to(device))
        _ = vsr.run(rgb_in)
    torch.cuda.synchronize()

    # Process all frames -> collect as RGB uint8 numpy (lossless intermediate)
    nvvfx_rgb_frames = []   # RGB uint8 HWC numpy
    nvvfx_bgr_frames = []   # BGR uint8 HWC numpy (for direct PieAPP)
    t0 = time.perf_counter()
    for frame in input_frames:
        rgb_in = (torch.from_numpy(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            .permute(2, 0, 1).contiguous().to(device))
        result = vsr.run(rgb_in)
        out_gpu = torch.from_dlpack(result.image).clone()
        # CHW float32 [0,1] -> HWC uint8
        out_np = out_gpu.clamp(0, 1).permute(1, 2, 0).mul(255).byte().cpu().numpy()
        nvvfx_rgb_frames.append(out_np.copy())
        nvvfx_bgr_frames.append(cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    fps = len(nvvfx_rgb_frames) / elapsed
    print(f"  Upscaled {len(nvvfx_rgb_frames)} frames in {elapsed:.2f}s = {fps:.1f} FPS")
    print(f"  Output shape: {nvvfx_rgb_frames[0].shape}")

    # Cleanup VSR
    del vsr
    torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Step 2: Save lossless FFV1 intermediate
    # -----------------------------------------------------------------------
    ffv1_path = os.path.join(OUTPUT_DIR, "nvvfx_lossless_ffv1.mkv")
    print(f"\nSaving FFV1 lossless intermediate: {ffv1_path}")
    encode_ffv1_lossless(nvvfx_rgb_frames, ffv1_path)
    print(f"  FFV1 size: {file_size_mb(ffv1_path):.1f} MB")

    # -----------------------------------------------------------------------
    # Step 3: Measure PieAPP of raw nvvfx output (no encoding at all)
    # This is the theoretical ceiling.
    # -----------------------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Measuring raw nvvfx PieAPP (no encoding) — theoretical ceiling")
    print(f"{'='*70}")
    raw_pieapp, raw_scores = measure_pieapp(nvvfx_bgr_frames, gt_frames, metric, device, NUM_SCORE_FRAMES)
    raw_sf, raw_sq = calc_sf(raw_pieapp, cl=CLIP_SEC)
    print(f"  Raw nvvfx PieAPP: {raw_pieapp:.6f}")
    print(f"  Raw nvvfx S_Q:    {raw_sq:.6f}")
    print(f"  Raw nvvfx S_F:    {raw_sf:.6f}")
    if raw_pieapp <= 0.092:
        print(f"  *** RAW nvvfx already hits PieAPP <= 0.092! ***")
    else:
        print(f"  Raw nvvfx PieAPP = {raw_pieapp:.6f} > 0.092 — encoding can't close the gap alone")

    # -----------------------------------------------------------------------
    # Step 4: Test encoding variants
    # -----------------------------------------------------------------------
    variants = []

    # (a) yuv420p CQ=20 — current production baseline
    variants.append({
        'name': 'a) yuv420p CQ=20 (production)',
        'encode_fn': lambda frames, path: encode_variant_hevc(frames, path, cq=20),
        'ext': 'mp4',
    })

    # (b) yuv420p CQ=1 — near-lossless
    variants.append({
        'name': 'b) yuv420p CQ=1 (near-lossless)',
        'encode_fn': lambda frames, path: encode_variant_hevc(frames, path, cq=1),
        'ext': 'mp4',
    })

    # (c) yuv444p — SKIPPED, validator rejects non-yuv420p
    # (d) yuv444p CQ=1 — SKIPPED

    # (e) yuv420p CQ=20 with lanczos chroma downsampling
    variants.append({
        'name': 'e) yuv420p CQ=20 lanczos sws',
        'encode_fn': lambda frames, path: encode_variant_hevc(
            frames, path, cq=20,
            extra_output_args=['-sws_flags', 'lanczos']),
        'ext': 'mp4',
    })

    # (f) yuv420p CQ=20 with explicit bt709 colorspace
    variants.append({
        'name': 'f) yuv420p CQ=20 bt709',
        'encode_fn': lambda frames, path: encode_variant_hevc(
            frames, path, cq=20,
            extra_output_args=[
                '-vf', 'colorspace=all=bt709:iall=bt709',
                '-color_primaries', 'bt709',
                '-color_trc', 'bt709',
                '-colorspace', 'bt709',
            ]),
        'ext': 'mp4',
    })

    # (g) FFV1 RGB lossless — theoretical max (no lossy encode)
    variants.append({
        'name': 'g) FFV1 RGB lossless',
        'encode_fn': lambda frames, path: encode_ffv1_lossless(frames, path),
        'ext': 'mkv',
    })

    # Run each variant
    results = []

    # First: add raw nvvfx result
    results.append({
        'name': 'h) Raw nvvfx (no encode)',
        'pieapp': raw_pieapp,
        'sq': raw_sq,
        'sf': raw_sf,
        'file_size_mb': 0.0,
        'hit_target': raw_pieapp <= 0.092,
    })

    for v in variants:
        name = v['name']
        ext = v['ext']
        safe_name = name.split(')')[0].strip() + ')'
        out_path = os.path.join(OUTPUT_DIR, f"enc_{safe_name.replace(' ','_').replace('(','').replace(')','')}.{ext}")

        print(f"\n{'─'*70}")
        print(f"  {name}")
        print(f"{'─'*70}")

        try:
            v['encode_fn'](nvvfx_rgb_frames, out_path)
            fsize = file_size_mb(out_path)
            print(f"  File: {out_path} ({fsize:.2f} MB)")

            # Decode back and measure PieAPP
            decoded_frames = decode_to_bgr_frames(out_path)
            print(f"  Decoded: {len(decoded_frames)} frames")

            pieapp, scores = measure_pieapp(decoded_frames, gt_frames, metric, device, NUM_SCORE_FRAMES)
            sf, sq = calc_sf(pieapp, cl=CLIP_SEC)
            hit = pieapp <= 0.092

            print(f"  PieAPP: {pieapp:.6f}")
            print(f"  S_Q:    {sq:.6f}")
            print(f"  S_F:    {sf:.6f}")
            if hit:
                print(f"  *** HIT TARGET: PieAPP <= 0.092 ***")

            results.append({
                'name': name,
                'pieapp': pieapp,
                'sq': sq,
                'sf': sf,
                'file_size_mb': fsize,
                'hit_target': hit,
            })

            del decoded_frames
            torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({
                'name': name,
                'pieapp': None,
                'sq': None,
                'sf': None,
                'file_size_mb': 0.0,
                'hit_target': False,
                'error': str(e),
            })

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*90}")
    print(f"ENCODING BENCHMARK SUMMARY — nvvfx HIGHBITRATE_ULTRA x2")
    print(f"{'='*90}")
    print(f"Validator constraint: yuv420p ONLY (yuv444p rejected)")
    print(f"Ground truth: {GROUND_TRUTH}")
    print(f"{'='*90}")
    print(f"{'Variant':<38} {'PieAPP':>8} {'S_Q':>8} {'S_F':>8} {'Size MB':>8} {'<=0.092?':>8}")
    print(f"{'-'*38} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for r in results:
        if r.get('pieapp') is not None:
            flag = ' <<<' if r['hit_target'] else ''
            print(f"{r['name']:<38} {r['pieapp']:>8.4f} {r['sq']:>8.4f} {r['sf']:>8.4f} "
                  f"{r['file_size_mb']:>8.2f} {'YES' if r['hit_target'] else 'NO':>8}{flag}")
        else:
            print(f"{r['name']:<38} {'ERROR':>8} {'':>8} {'':>8} {'':>8} {'':>8}  {r.get('error','')}")

    # Encoding overhead analysis
    print(f"\n{'─'*90}")
    print(f"ENCODING OVERHEAD ANALYSIS")
    print(f"{'─'*90}")
    raw = results[0]  # raw nvvfx
    if raw['pieapp'] is not None:
        for r in results[1:]:
            if r.get('pieapp') is not None:
                delta = r['pieapp'] - raw['pieapp']
                print(f"  {r['name']:<38} delta_PieAPP = {delta:+.6f}")

    # Key finding
    print(f"\n{'─'*90}")
    print(f"KEY FINDING")
    print(f"{'─'*90}")
    if raw['pieapp'] is not None:
        if raw['pieapp'] > 0.092:
            print(f"  Raw nvvfx PieAPP = {raw['pieapp']:.6f} > 0.092")
            print(f"  -> Even with PERFECT lossless encoding, PieAPP target 0.092 is UNREACHABLE")
            print(f"     for this model. Encoding optimization alone cannot close the gap.")
        else:
            print(f"  Raw nvvfx PieAPP = {raw['pieapp']:.6f} <= 0.092")
            print(f"  -> Target is reachable! Check which encoding variants preserve it.")
            for r in results[1:]:
                if r.get('hit_target'):
                    print(f"     {r['name']} also hits target!")

    hits = [r for r in results if r.get('hit_target')]
    if hits:
        print(f"\n  VARIANTS HITTING PieAPP <= 0.092:")
        for r in hits:
            print(f"    {r['name']}: PieAPP={r['pieapp']:.6f}, S_F={r['sf']:.4f}")
    else:
        print(f"\n  NO variant hits PieAPP <= 0.092.")


if __name__ == "__main__":
    main()
