#!/usr/bin/env python3
"""
benchmark_chroma.py — Investigate chroma subsampling quality loss in HEVC yuv420p pipeline.

DAT-light x4 achieves PieAPP=0.098 lossless (RGB), but yuv420p encoding pushes
it to 0.231. This script tests whether better chroma handling can close the gap.
"""

import sys, os, math, time, subprocess, json
import numpy as np

sys.path.insert(0, "/root/vidaio-subnet")

import torch
import torch.nn.functional as F
import cv2
import pyiqa
import spandrel

# ── Config ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda")
PAYLOAD = "/root/pipeline_audit/payload_270p.mp4"
GT_VIDEO = "/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4"
MODEL_PATH = os.path.expanduser("~/.cache/span/DAT_light_x4.pth")
OUT_DIR = "/root/pipeline_audit/chroma_bench"
LOSSLESS = os.path.join(OUT_DIR, "lossless.mkv")
NUM_UPSCALE_FRAMES = 60
NUM_METRIC_FRAMES = 16
CQ = 20

os.makedirs(OUT_DIR, exist_ok=True)


# ── Scoring helpers ─────────────────────────────────────────────────────────
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


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


# ── Step 1: Upscale with DAT-light x4 and save lossless ────────────────────
def upscale_and_save_lossless():
    if os.path.exists(LOSSLESS):
        print(f"[skip] Lossless already exists: {LOSSLESS}")
        return

    print("[1/3] Loading DAT-light x4 model...")
    model = spandrel.ModelLoader(device=DEVICE).load_from_file(MODEL_PATH).eval()
    if hasattr(model, "model"):
        net = model.model
    else:
        net = model

    print("[2/3] Upscaling frames...")
    cap = cv2.VideoCapture(PAYLOAD)
    frames_rgb = []
    count = 0
    while count < NUM_UPSCALE_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break
        # BGR -> RGB, float32, NCHW
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames_rgb.append(rgb)
        count += 1
    cap.release()
    print(f"  Read {len(frames_rgb)} frames ({frames_rgb[0].shape})")

    # Upscale in batches
    upscaled = []
    batch_size = 4
    with torch.no_grad():
        for i in range(0, len(frames_rgb), batch_size):
            batch_np = np.stack(frames_rgb[i : i + batch_size])  # (B, H, W, 3)
            batch_t = (
                torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(DEVICE)
                / 255.0
            )
            out = model(batch_t)  # spandrel model is callable
            out = out.clamp(0, 1).mul(255).byte().cpu().permute(0, 2, 3, 1).numpy()
            for j in range(out.shape[0]):
                upscaled.append(out[j])  # (H, W, 3) RGB uint8
            print(f"  Upscaled batch {i // batch_size + 1}/{(len(frames_rgb) + batch_size - 1) // batch_size}")

    h, w = upscaled[0].shape[:2]
    print(f"  Output resolution: {w}x{h}")

    # Pipe raw RGB24 to ffmpeg for lossless FFV1 (gbrp = planar RGB, truly lossless)
    print("[3/3] Saving lossless FFV1...")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo",
        "-pix_fmt", "rgb24",
        "-s", f"{w}x{h}",
        "-r", "30",
        "-i", "pipe:0",
        "-c:v", "ffv1",
        "-pix_fmt", "gbrp",
        LOSSLESS,
    ]
    raw_data = b"".join(frame.tobytes() for frame in upscaled)
    result = subprocess.run(cmd, input=raw_data, capture_output=True)
    if result.returncode != 0:
        print(f"  ffmpeg error: {result.stderr.decode()[-500:]}")
        raise RuntimeError("ffmpeg lossless encode failed")
    fsize = os.path.getsize(LOSSLESS) / 1024 / 1024
    print(f"  Saved {LOSSLESS} ({fsize:.1f} MB, {len(upscaled)} frames)")

    # Free GPU memory
    del model, net
    torch.cuda.empty_cache()


# ── Step 2: Encode variants ────────────────────────────────────────────────
ENCODE_VARIANTS = {
    "a_default": [
        "ffmpeg", "-y", "-i", LOSSLESS,
        "-c:v", "hevc_nvenc", "-cq", str(CQ), "-pix_fmt", "yuv420p", "-preset", "p4",
        os.path.join(OUT_DIR, "a_default.mp4"),
    ],
    "b_colorspace": [
        "ffmpeg", "-y", "-i", LOSSLESS,
        "-vf", "colorspace=all=bt709:iall=bt709",
        "-c:v", "hevc_nvenc", "-cq", str(CQ), "-pix_fmt", "yuv420p", "-preset", "p4",
        os.path.join(OUT_DIR, "b_colorspace.mp4"),
    ],
    "c_lanczos": [
        "ffmpeg", "-y", "-i", LOSSLESS,
        "-sws_flags", "lanczos",
        "-c:v", "hevc_nvenc", "-cq", str(CQ), "-pix_fmt", "yuv420p", "-preset", "p4",
        os.path.join(OUT_DIR, "c_lanczos.mp4"),
    ],
    "d_spline": [
        "ffmpeg", "-y", "-i", LOSSLESS,
        "-sws_flags", "spline",
        "-c:v", "hevc_nvenc", "-cq", str(CQ), "-pix_fmt", "yuv420p", "-preset", "p4",
        os.path.join(OUT_DIR, "d_spline.mp4"),
    ],
    "e_bicubic": [
        "ffmpeg", "-y", "-i", LOSSLESS,
        "-sws_flags", "bicubic",
        "-c:v", "hevc_nvenc", "-cq", str(CQ), "-pix_fmt", "yuv420p", "-preset", "p4",
        os.path.join(OUT_DIR, "e_bicubic.mp4"),
    ],
    "f_area": [
        "ffmpeg", "-y", "-i", LOSSLESS,
        "-sws_flags", "area",
        "-c:v", "hevc_nvenc", "-cq", str(CQ), "-pix_fmt", "yuv420p", "-preset", "p4",
        os.path.join(OUT_DIR, "f_area.mp4"),
    ],
    "g_preblur": [
        "ffmpeg", "-y", "-i", LOSSLESS,
        "-vf", "format=yuv444p,gblur=sigma=0.5:planes=6",
        "-c:v", "hevc_nvenc", "-cq", str(CQ), "-pix_fmt", "yuv420p", "-preset", "p4",
        os.path.join(OUT_DIR, "g_preblur.mp4"),
    ],
    "h_neighbor": [
        "ffmpeg", "-y", "-i", LOSSLESS,
        "-sws_flags", "neighbor",
        "-c:v", "hevc_nvenc", "-cq", str(CQ), "-pix_fmt", "yuv420p", "-preset", "p4",
        os.path.join(OUT_DIR, "h_neighbor.mp4"),
    ],
    "i_10bit": [
        "ffmpeg", "-y", "-i", LOSSLESS,
        "-c:v", "hevc_nvenc", "-cq", str(CQ), "-pix_fmt", "p010le", "-profile:v", "main10",
        "-preset", "p4",
        os.path.join(OUT_DIR, "i_10bit.mp4"),
    ],
}


def encode_variants():
    print("\n=== Encoding variants ===")
    for name, cmd in ENCODE_VARIANTS.items():
        outpath = cmd[-1]
        if os.path.exists(outpath):
            print(f"  [skip] {name} already exists")
            continue
        print(f"  Encoding {name}...")
        t0 = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        if result.returncode != 0:
            print(f"    FAILED ({result.returncode}): {result.stderr[-300:]}")
            # Try fallback without problematic options
            continue
        fsize = os.path.getsize(outpath) / 1024
        print(f"    OK ({dt:.1f}s, {fsize:.0f} KB)")


# ── Step 3: Measure PieAPP ─────────────────────────────────────────────────
def read_frames(video_path, n_frames):
    """Read n_frames evenly spaced from video, return list of RGB uint8 numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = 300  # fallback
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def measure_pieapp(ref_path, dist_path, metric, n_frames=NUM_METRIC_FRAMES):
    """Compute mean PieAPP between ref and dist videos."""
    ref_frames = read_frames(ref_path, n_frames)
    dist_frames = read_frames(dist_path, n_frames)

    if len(ref_frames) == 0 or len(dist_frames) == 0:
        return float("nan")

    scores = []
    for rf, df in zip(ref_frames, dist_frames):
        # Resize dist to ref size if needed
        rh, rw = rf.shape[:2]
        dh, dw = df.shape[:2]
        if (rh, rw) != (dh, dw):
            df = cv2.resize(df, (rw, rh), interpolation=cv2.INTER_LANCZOS4)

        # To tensor: (1, 3, H, W) float [0, 1]
        rt = torch.from_numpy(rf).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        dt = torch.from_numpy(df).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0

        with torch.no_grad():
            score = metric(dt, rt).item()
        scores.append(abs(score))

    return float(np.mean(scores))


def evaluate_all():
    print("\n=== Measuring PieAPP ===")
    print(f"  Ground truth: {GT_VIDEO}")

    metric = pyiqa.create_metric("pieapp", device=DEVICE)

    results = []

    # First measure lossless
    if os.path.exists(LOSSLESS):
        print(f"  Measuring lossless...")
        p = measure_pieapp(GT_VIDEO, LOSSLESS, metric)
        sf, sq = calc_sf(p)
        fsize = os.path.getsize(LOSSLESS) / 1024
        results.append(("lossless (FFV1 RGB)", p, sf, sq, fsize))
        print(f"    PieAPP={p:.4f}  S_F={sf:.4f}  S_Q={sq:.4f}")

    # Measure each variant
    for name in sorted(ENCODE_VARIANTS.keys()):
        outpath = ENCODE_VARIANTS[name][-1]
        if not os.path.exists(outpath):
            print(f"  [skip] {name} — file not found")
            continue
        print(f"  Measuring {name}...")
        p = measure_pieapp(GT_VIDEO, outpath, metric)
        sf, sq = calc_sf(p)
        fsize = os.path.getsize(outpath) / 1024
        results.append((name, p, sf, sq, fsize))
        print(f"    PieAPP={p:.4f}  S_F={sf:.4f}  S_Q={sq:.4f}  size={fsize:.0f}KB")

    return results


def print_table(results):
    print("\n" + "=" * 85)
    print(f"{'Method':<30} {'PieAPP':>8} {'S_Q':>8} {'S_F':>8} {'Size(KB)':>10}")
    print("-" * 85)
    for name, p, sf, sq, fsize in results:
        marker = " ***" if p < 0.15 else ""
        print(f"{name:<30} {p:>8.4f} {sq:>8.4f} {sf:>8.4f} {fsize:>10.0f}{marker}")
    print("=" * 85)

    # Summary
    lossless_p = None
    best_encoded = None
    for name, p, sf, sq, fsize in results:
        if "lossless" in name:
            lossless_p = p
        elif best_encoded is None or p < best_encoded[1]:
            best_encoded = (name, p, sf, sq, fsize)

    if lossless_p is not None and best_encoded is not None:
        gap = best_encoded[1] - lossless_p
        print(f"\nLossless PieAPP:       {lossless_p:.4f}")
        print(f"Best encoded:          {best_encoded[0]} = {best_encoded[1]:.4f}")
        print(f"Chroma gap:            +{gap:.4f}")
        print(f"Gap vs 0.231 baseline: {0.231 - best_encoded[1]:+.4f} improvement")
        if best_encoded[1] < 0.15:
            print(">>> SIGNIFICANT: Best method achieves PieAPP < 0.15! <<<")


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t_start = time.time()

    upscale_and_save_lossless()
    encode_variants()
    results = evaluate_all()
    print_table(results)

    print(f"\nTotal time: {time.time() - t_start:.1f}s")
