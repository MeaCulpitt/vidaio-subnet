#!/usr/bin/env python3
"""Benchmark x2 upscaling path — SPAN x2 models (nvvfx not available).

Measures PieAPP, S_Q, S_F, and FPS for each 2x model in ~/.cache/span/.
Ground truth: 1080p video. We downscale to 540p, upscale 2x back to 1080p,
then measure PieAPP against the original.
"""
import sys, os, time, math, subprocess, tempfile
import numpy as np
import cv2
import torch
import pyiqa

sys.path.insert(0, "/root/vidaio-subnet")
from spandrel import ModelLoader

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
GROUND_TRUTH = "/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4"
PAYLOAD_540P = "/root/pipeline_audit/payload_540p.mp4"
SPAN_DIR = os.path.expanduser("~/.cache/span")

X2_MODELS = [
    "2xHFA2kSPAN.safetensors",
    "2xHFA2k_LUDVAE_SPAN.safetensors",
    "2xHFA2k_SPAN.safetensors",
    "2xNomosUni_span.safetensors",
    "2x_PLKSR.pth",
    "2x_RPLKSR.pth",
    "2x_SPAN_106k.pth",
]

FRAME_INTERVAL = 10  # score every 10th frame to keep runtime reasonable
PIEAPP_SIZE = 512     # resize both ref/dist to 512px height for PieAPP (avoids OOM)

# ---------------------------------------------------------------------------
# Scoring formulas
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
# Step 1: Create 540p payload
# ---------------------------------------------------------------------------
def create_540p_payload():
    if os.path.exists(PAYLOAD_540P):
        print(f"[info] 540p payload already exists: {PAYLOAD_540P}")
        return
    print("[info] Creating 540p payload from ground truth...")
    cmd = [
        "ffmpeg", "-y", "-i", GROUND_TRUTH,
        "-vf", "scale=-2:540",
        "-c:v", "libx264", "-crf", "18",
        PAYLOAD_540P,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"[info] Created {PAYLOAD_540P}")

# ---------------------------------------------------------------------------
# Step 2: Upscale with a model and measure PieAPP + FPS
# ---------------------------------------------------------------------------
def benchmark_model(model_name: str, device: torch.device, pieapp_metric):
    model_path = os.path.join(SPAN_DIR, model_name)
    if not os.path.exists(model_path):
        print(f"[SKIP] {model_name} not found")
        return None

    print(f"\n{'='*60}")
    print(f"  Model: {model_name}")
    print(f"{'='*60}")

    # Load model
    descriptor = ModelLoader(device=str(device)).load_from_file(model_path)
    model = descriptor.model.eval().half()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Params: {nparams:.2f}M")

    # Open 540p payload
    cap_in = cv2.VideoCapture(PAYLOAD_540P)
    if not cap_in.isOpened():
        print(f"[ERROR] Cannot open {PAYLOAD_540P}")
        return None

    in_w = int(cap_in.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap_in.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Input: {in_w}x{in_h}, {total_frames} frames")

    # Open ground truth
    cap_gt = cv2.VideoCapture(GROUND_TRUTH)
    gt_w = int(cap_gt.get(cv2.CAP_PROP_FRAME_WIDTH))
    gt_h = int(cap_gt.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Ground truth: {gt_w}x{gt_h}")

    # Process frames
    pieapp_scores = []
    frame_times = []
    frame_idx = 0

    while True:
        ret_in, frame_in = cap_in.read()
        ret_gt, frame_gt = cap_gt.read()
        if not ret_in or not ret_gt:
            break

        # Upscale every frame for FPS measurement, but only score every Nth
        # Convert BGR->RGB, HWC->CHW, normalize
        rgb = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device).half()

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(tensor)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

        if frame_idx % FRAME_INTERVAL == 0:
            # Move output to CPU to free GPU memory for PieAPP
            out_rgb = out.float().clamp(0, 1).squeeze(0).cpu()  # (3, H, W)

            # Ground truth tensor
            gt_rgb = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gt_tensor = torch.from_numpy(gt_rgb).permute(2, 0, 1)  # (3, H, W)

            # Resize both to match & fit in memory for PieAPP
            _, gh, gw = gt_tensor.shape
            # First ensure output matches GT size
            _, oh, ow = out_rgb.shape
            if (oh, ow) != (gh, gw):
                out_rgb = torch.nn.functional.interpolate(
                    out_rgb.unsqueeze(0), size=(gh, gw), mode="bicubic", align_corners=False
                ).squeeze(0).clamp(0, 1)

            # Downscale both to PIEAPP_SIZE to avoid OOM
            scale_h = min(1.0, PIEAPP_SIZE / gh)
            new_h = int(gh * scale_h)
            new_w = int(gw * scale_h)
            # Make divisible by 2
            new_h = new_h - (new_h % 2)
            new_w = new_w - (new_w % 2)

            out_small = torch.nn.functional.interpolate(
                out_rgb.unsqueeze(0), size=(new_h, new_w), mode="bicubic", align_corners=False
            ).clamp(0, 1)
            gt_small = torch.nn.functional.interpolate(
                gt_tensor.unsqueeze(0), size=(new_h, new_w), mode="bicubic", align_corners=False
            ).clamp(0, 1)

            with torch.no_grad():
                score = pieapp_metric(
                    out_small.to(device),
                    gt_small.to(device),
                )
            sv = abs(score.item())
            pieapp_scores.append(sv)
            del out_small, gt_small
            torch.cuda.empty_cache()

        frame_idx += 1

    cap_in.release()
    cap_gt.release()

    # Cleanup model
    del model, descriptor
    torch.cuda.empty_cache()

    if not pieapp_scores:
        print("  [ERROR] No scores collected")
        return None

    avg_pieapp = np.mean(pieapp_scores)
    capped_pieapp = min(avg_pieapp, 2.0)
    avg_fps = 1.0 / np.mean(frame_times) if frame_times else 0
    sf, sq = calc_sf(capped_pieapp, cl=10)

    print(f"  Frames processed: {frame_idx} (scored {len(pieapp_scores)})")
    print(f"  PieAPP (raw avg): {avg_pieapp:.4f}")
    print(f"  PieAPP (capped):  {capped_pieapp:.4f}")
    print(f"  S_Q:              {sq:.4f}")
    print(f"  S_F:              {sf:.4f}  {'> 0.32 OK' if sf > 0.32 else '< 0.32 FAIL'}")
    print(f"  FPS:              {avg_fps:.1f}")

    return {
        "model": model_name,
        "params_M": nparams,
        "pieapp_raw": avg_pieapp,
        "pieapp_capped": capped_pieapp,
        "S_Q": sq,
        "S_F": sf,
        "fps": avg_fps,
        "frames_scored": len(pieapp_scores),
    }

# ---------------------------------------------------------------------------
# Bicubic baseline
# ---------------------------------------------------------------------------
def benchmark_bicubic(device, pieapp_metric):
    """Bicubic upscale baseline — no model, just resize."""
    cap_in = cv2.VideoCapture(PAYLOAD_540P)
    cap_gt = cv2.VideoCapture(GROUND_TRUTH)
    if not cap_in.isOpened() or not cap_gt.isOpened():
        print("[ERROR] Cannot open video files")
        return None

    gt_w = int(cap_gt.get(cv2.CAP_PROP_FRAME_WIDTH))
    gt_h = int(cap_gt.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap_in.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Input: 960x540 -> {gt_w}x{gt_h} (bicubic)")

    pieapp_scores = []
    frame_idx = 0
    while True:
        ret_in, frame_in = cap_in.read()
        ret_gt, frame_gt = cap_gt.read()
        if not ret_in or not ret_gt:
            break
        if frame_idx % FRAME_INTERVAL == 0:
            # Bicubic upscale
            up = cv2.resize(frame_in, (gt_w, gt_h), interpolation=cv2.INTER_CUBIC)
            out_rgb = cv2.cvtColor(up, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            gt_rgb = cv2.cvtColor(frame_gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            out_t = torch.from_numpy(out_rgb).permute(2, 0, 1).unsqueeze(0)
            gt_t = torch.from_numpy(gt_rgb).permute(2, 0, 1).unsqueeze(0)

            # Downscale for PieAPP
            scale_h = min(1.0, PIEAPP_SIZE / gt_h)
            new_h = int(gt_h * scale_h) - (int(gt_h * scale_h) % 2)
            new_w = int(gt_w * scale_h) - (int(gt_w * scale_h) % 2)

            out_small = torch.nn.functional.interpolate(out_t, size=(new_h, new_w), mode="bicubic", align_corners=False).clamp(0, 1)
            gt_small = torch.nn.functional.interpolate(gt_t, size=(new_h, new_w), mode="bicubic", align_corners=False).clamp(0, 1)

            with torch.no_grad():
                score = pieapp_metric(out_small.to(device), gt_small.to(device))
            pieapp_scores.append(abs(score.item()))
            del out_small, gt_small
            torch.cuda.empty_cache()
        frame_idx += 1

    cap_in.release()
    cap_gt.release()

    avg_pieapp = np.mean(pieapp_scores)
    capped = min(avg_pieapp, 2.0)
    sf, sq = calc_sf(capped, cl=10)
    print(f"  PieAPP (raw avg): {avg_pieapp:.4f}")
    print(f"  PieAPP (capped):  {capped:.4f}")
    print(f"  S_Q:              {sq:.4f}")
    print(f"  S_F:              {sf:.4f}  {'> 0.32 OK' if sf > 0.32 else '< 0.32 FAIL'}")
    return {
        "model": "BICUBIC (baseline)",
        "params_M": 0,
        "pieapp_raw": avg_pieapp,
        "pieapp_capped": capped,
        "S_Q": sq,
        "S_F": sf,
        "fps": 999.0,
        "frames_scored": len(pieapp_scores),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  x2 Upscaling Benchmark — SPAN models (nvvfx unavailable)")
    print("=" * 60)

    device = torch.device("cuda:0")
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Torch: {torch.__version__}")

    create_540p_payload()

    # Init PieAPP metric once
    print("\n[info] Loading PieAPP metric...")
    pieapp_metric = pyiqa.create_metric("pieapp", device=device)

    results = []
    for model_name in X2_MODELS:
        try:
            r = benchmark_model(model_name, device, pieapp_metric)
        except Exception as e:
            print(f"  [ERROR] {model_name}: {e}")
            torch.cuda.empty_cache()
            r = None
        if r:
            results.append(r)

    # Bicubic baseline for sanity check
    print(f"\n{'='*60}")
    print(f"  Baseline: Bicubic x2 (no model)")
    print(f"{'='*60}")
    r_bicubic = benchmark_bicubic(device, pieapp_metric)
    if r_bicubic:
        results.append(r_bicubic)

    # Summary table
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY — x2 Upscaling (540p -> 1080p)")
    print(f"{'='*80}")
    print(f"{'Model':<35} {'PieAPP':>7} {'S_Q':>7} {'S_F':>7} {'FPS':>7} {'Pass?':>6}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: x["S_F"], reverse=True):
        ok = "YES" if r["S_F"] > 0.32 else "NO"
        print(f"{r['model']:<35} {r['pieapp_capped']:>7.4f} {r['S_Q']:>7.4f} {r['S_F']:>7.4f} {r['fps']:>7.1f} {ok:>6}")
    print("-" * 80)

if __name__ == "__main__":
    main()
