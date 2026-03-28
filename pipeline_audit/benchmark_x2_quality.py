#!/usr/bin/env python3
"""Benchmark x2 upscaling quality (PieAPP) for SPAN/RPLKSR 2x models.

Since nvidia-vfx (nvvfx) is not available on this machine, we test
spandrel-based x2 models as the fallback path used by the server.

Pipeline:
  1. Downscale 1080p ground truth → 540p (payload)
  2. Upscale 540p → 1080p with each x2 model (60 frames, FP16)
  3. Encode output with hevc_nvenc CQ=20
  4. Measure PieAPP vs ground truth
  5. Compute S_Q, S_F
"""
import sys, os, time, math, subprocess, tempfile
sys.path.insert(0, "/root/vidaio-subnet")

import numpy as np
import torch
import cv2
import pyiqa
from pathlib import Path
from spandrel import ModelLoader

# ── Paths ──────────────────────────────────────────────────────────────
GROUND_TRUTH = "/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4"
PAYLOAD_540P = "/root/pipeline_audit/payload_540p.mp4"
OUT_DIR = Path("/root/pipeline_audit")

MODELS = {
    "2xHFA2kSPAN":        Path.home() / ".cache/span/2xHFA2kSPAN.safetensors",
    "2xHFA2k_LUDVAE_SPAN": Path.home() / ".cache/span/2xHFA2k_LUDVAE_SPAN.safetensors",
    "2xNomosUni_span":    Path.home() / ".cache/span/2xNomosUni_span.safetensors",
    "2x_RPLKSR":          Path.home() / ".cache/span/2x_RPLKSR.pth",
}

NUM_UPSCALE_FRAMES = 60
NUM_PIEAPP_FRAMES = 16
DEVICE = torch.device("cuda")

# ── Scoring ────────────────────────────────────────────────────────────
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

# ── Helpers ────────────────────────────────────────────────────────────
def ensure_payload():
    """Create 540p payload by downscaling ground truth if not present."""
    if os.path.exists(PAYLOAD_540P):
        sz = os.path.getsize(PAYLOAD_540P)
        if sz > 10000:
            print(f"[payload] Using existing {PAYLOAD_540P} ({sz/1e6:.1f} MB)")
            return
    print("[payload] Creating 540p payload from ground truth...")
    subprocess.run([
        "ffmpeg", "-y", "-i", GROUND_TRUTH,
        "-vf", "scale=-2:540",
        "-c:v", "libx264", "-crf", "18",
        PAYLOAD_540P,
    ], check=True, capture_output=True)
    print(f"[payload] Created {PAYLOAD_540P}")

def read_frames(path, n_frames):
    """Read first n_frames from video as list of BGR numpy arrays."""
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < n_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    return frames

def read_frames_uniform(path, n_frames):
    """Read n_frames uniformly sampled from video."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames

def upscale_frames(model, frames, batch_size=1):
    """Upscale BGR frames with a spandrel model (FP16). Returns list of BGR numpy arrays."""
    results = []
    for frame in frames:
        # BGR -> RGB, HWC -> CHW, normalize
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).half().to(DEVICE) / 255.0
        with torch.no_grad():
            out = model(tensor)
        out = out.squeeze(0).clamp(0, 1).cpu().float().numpy()
        out = (out * 255).astype(np.uint8)
        out_bgr = cv2.cvtColor(np.transpose(out, (1, 2, 0)), cv2.COLOR_RGB2BGR)
        results.append(out_bgr)
    return results

def encode_hevc(frames, output_path, fps=30.0):
    """Encode frames to HEVC with hevc_nvenc CQ=20."""
    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "hevc_nvenc", "-rc", "constqp", "-qp", "20",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()

def measure_pieapp(ref_path, dist_path, n_frames=16):
    """Measure PieAPP between ref and dist videos, sampling n_frames uniformly."""
    metric = pyiqa.create_metric("pieapp", device=DEVICE)
    ref_frames = read_frames_uniform(ref_path, n_frames)
    dist_frames = read_frames_uniform(dist_path, n_frames)

    scores = []
    for rf, df in zip(ref_frames, dist_frames):
        # Resize dist to match ref if needed
        rh, rw = rf.shape[:2]
        dh, dw = df.shape[:2]
        if (rh, rw) != (dh, dw):
            df = cv2.resize(df, (rw, rh), interpolation=cv2.INTER_LANCZOS4)

        # BGR -> RGB, to tensor
        r_rgb = cv2.cvtColor(rf, cv2.COLOR_BGR2RGB)
        d_rgb = cv2.cvtColor(df, cv2.COLOR_BGR2RGB)
        r_t = torch.from_numpy(r_rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
        d_t = torch.from_numpy(d_rgb).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0

        with torch.no_grad():
            s = metric(d_t, r_t).item()
        scores.append(abs(s))

    return float(np.mean(scores))

# ── Main ───────────────────────────────────────────────────────────────
def main():
    ensure_payload()

    print(f"\nGround truth: {GROUND_TRUTH}")
    print(f"Payload 540p: {PAYLOAD_540P}")
    print(f"Upscale frames: {NUM_UPSCALE_FRAMES}, PieAPP frames: {NUM_PIEAPP_FRAMES}")
    print("=" * 80)

    results = []

    for name, model_path in MODELS.items():
        if not model_path.exists() or model_path.stat().st_size < 1000:
            print(f"\n[SKIP] {name}: file missing or empty ({model_path})")
            continue

        print(f"\n{'─' * 60}")
        print(f"[{name}] Loading model from {model_path}...")

        try:
            descriptor = ModelLoader(device="cuda").load_from_file(str(model_path))
            model = descriptor.model.eval().half()
            nparams = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"[{name}] Loaded: {nparams:.2f}M params")
        except Exception as e:
            print(f"[{name}] FAILED to load: {e}")
            continue

        # Read 540p frames
        payload_frames = read_frames(PAYLOAD_540P, NUM_UPSCALE_FRAMES)
        print(f"[{name}] Read {len(payload_frames)} payload frames ({payload_frames[0].shape[1]}x{payload_frames[0].shape[0]})")

        # Warm up
        warmup_t = torch.from_numpy(cv2.cvtColor(payload_frames[0], cv2.COLOR_BGR2RGB)).permute(2, 0, 1).unsqueeze(0).half().to(DEVICE) / 255.0
        with torch.no_grad():
            _ = model(warmup_t)
        torch.cuda.synchronize()

        # Upscale with timing
        t0 = time.perf_counter()
        upscaled = upscale_frames(model, payload_frames)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        elapsed = t1 - t0
        fps = len(payload_frames) / elapsed
        print(f"[{name}] Upscaled {len(upscaled)} frames in {elapsed:.2f}s ({fps:.1f} FPS)")
        print(f"[{name}] Output size: {upscaled[0].shape[1]}x{upscaled[0].shape[0]}")

        # Encode
        out_path = OUT_DIR / f"bench_x2_{name}.mp4"
        print(f"[{name}] Encoding to {out_path} (hevc_nvenc CQ=20)...")
        encode_hevc(upscaled, out_path, fps=30.0)
        print(f"[{name}] Encoded: {out_path.stat().st_size / 1e6:.2f} MB")

        # PieAPP
        print(f"[{name}] Measuring PieAPP ({NUM_PIEAPP_FRAMES} frames)...")
        pieapp = measure_pieapp(GROUND_TRUTH, str(out_path), NUM_PIEAPP_FRAMES)
        sf, sq = calc_sf(pieapp, cl=10)

        results.append({
            "model": name,
            "params_M": nparams,
            "pieapp": pieapp,
            "S_Q": sq,
            "S_F": sf,
            "fps": fps,
            "elapsed_s": elapsed,
        })

        print(f"[{name}] PieAPP={pieapp:.4f}  S_Q={sq:.4f}  S_F={sf:.4f}  FPS={fps:.1f}")

        # Cleanup GPU
        del model, descriptor
        torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 80)
    print(f"{'Model':<28} {'PieAPP':>8} {'S_Q':>8} {'S_F':>8} {'FPS':>8} {'Params':>8}")
    print("-" * 80)
    for r in results:
        marker = " *" if r["S_F"] > 0.32 else ""
        print(f"{r['model']:<28} {r['pieapp']:>8.4f} {r['S_Q']:>8.4f} {r['S_F']:>8.4f} {r['fps']:>8.1f} {r['params_M']:>7.2f}M{marker}")
    print("-" * 80)
    print("* = S_F > 0.32 threshold met")

if __name__ == "__main__":
    main()
