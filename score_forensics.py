#!/usr/bin/env python3
"""
SN85 Score Forensics — measure PieAPP, S_Q, S_pre, S_f exactly as the validator does.

Flow (matches validator):
  1. Start with high-res ground truth video
  2. Downscale to create miner input (simulates what validator sends)
  3. Upscale with our model
  4. Encode through HEVC (simulates what validator receives)
  5. Compare HEVC-decoded output vs ground truth using PieAPP (same resolution)
"""
import sys, os, math, time, subprocess, tempfile
import numpy as np
import cv2
import torch
import pyiqa
from pathlib import Path

os.chdir("/root/vidaio-subnet")

# ── Validator scoring formulas (exact from services/scoring/server.py) ──

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def calculate_quality_score(pieapp_score):
    sn = sigmoid(pieapp_score)
    at_zero = (1 - (math.log10(sigmoid(0) + 1) / math.log10(3.5))) ** 2.5
    at_two  = (1 - (math.log10(sigmoid(2.0) + 1) / math.log10(3.5))) ** 2.5
    value   = (1 - (math.log10(sn + 1) / math.log10(3.5))) ** 2.5
    return 1 - ((value - at_zero) / (at_two - at_zero))

def calculate_length_score(sec):
    return math.log(1 + sec) / math.log(1 + 320)

def calculate_final_score(s_pre):
    return 0.1 * math.exp(6.979 * (s_pre - 0.5))

# ── PieAPP (mirrors validator exactly) ──

_pieapp_cache = {}
def get_pieapp(device):
    key = str(device)
    if key not in _pieapp_cache:
        _pieapp_cache[key] = pyiqa.create_metric('pieapp', device=device)
    return _pieapp_cache[key]

def measure_pieapp(ref_frames, dist_frames, device):
    """ref_frames/dist_frames: lists of BGR numpy arrays at SAME resolution."""
    # Use CPU for large frames to avoid OOM (PieAPP is lightweight)
    h, w = ref_frames[0].shape[:2]
    pieapp_device = torch.device('cpu') if (h * w) > 2_500_000 else device
    metric = get_pieapp(pieapp_device)
    scores = []
    for i, (rf, df) in enumerate(zip(ref_frames, dist_frames)):
        ref_rgb = cv2.cvtColor(rf, cv2.COLOR_BGR2RGB)
        dist_rgb = cv2.cvtColor(df, cv2.COLOR_BGR2RGB)
        ref_t = torch.from_numpy(ref_rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
        dist_t = torch.from_numpy(dist_rgb).permute(2, 0, 1).float().div_(255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            s = metric(dist_t, ref_t).item()
            if s < 0:
                s = abs(s)
        scores.append(s)
        print(f"    Frame {i}: PieAPP = {s:.6f}")
    avg = np.mean(scores) if scores else 5.0
    return min(avg, 2.0), scores

# ── Load SR models (same as server.py) ──

_model_cache = {}
def load_model(scale, gpu_id=0):
    if scale in _model_cache:
        return _model_cache[scale]
    from spandrel import ModelLoader
    paths = {
        2: Path.home() / ".cache" / "span" / "2xHFA2kSPAN.safetensors",
        4: Path.home() / ".cache" / "span" / "PLKSR_X4_DF2K.pth",
        "4fast": Path.home() / ".cache" / "span" / "realesr-general-x4v3.pth",
    }
    p = paths[scale]
    print(f"  Loading {p.stem} x{scale}...")
    from spandrel import ModelLoader
    desc = ModelLoader(device=f"cuda:{gpu_id}").load_from_file(str(p))
    model = desc.model.eval().half()
    nparams = sum(pp.numel() for pp in model.parameters()) / 1e6
    print(f"  {nparams:.2f}M params, fp16")
    _model_cache[scale] = model
    return model

def upscale_frames_sr(model, frames_bgr, device, batch_size=4):
    """Upscale BGR frames using SR model. Returns list of BGR numpy arrays."""
    results = []
    for i in range(0, len(frames_bgr), batch_size):
        batch_bgr = frames_bgr[i:i+batch_size]
        tensors = []
        for f in batch_bgr:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
            tensors.append(t)
        batch = torch.stack(tensors).to(device).half()
        with torch.no_grad():
            out = model(batch)
        out = out.float().clamp_(0, 1).mul_(255).round_().to(torch.uint8).cpu().numpy()
        for j in range(out.shape[0]):
            rgb_out = out[j].transpose(1, 2, 0)
            bgr_out = cv2.cvtColor(rgb_out, cv2.COLOR_RGB2BGR)
            results.append(bgr_out)
    return results

def upscale_frames_nvvfx(frames_bgr, device):
    """Upscale using nvidia-vfx VSR (x2)."""
    try:
        import pynvvfx as nvvfx
    except ImportError:
        print("  nvidia-vfx not available")
        return None
    h, w = frames_bgr[0].shape[:2]
    out_w, out_h = w * 2, h * 2
    try:
        vsr = nvvfx.VideoSuperRes(quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=0)
        vsr.output_width = out_w
        vsr.output_height = out_h
        vsr.load()
    except Exception as e:
        print(f"  nvidia-vfx load failed: {e}")
        return None

    results = []
    for f in frames_bgr:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t_in = torch.from_numpy(rgb).to(device).permute(2, 0, 1).unsqueeze(0).contiguous().to(torch.uint8)
        try:
            t_out = vsr.process(t_in)
            out_np = t_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
            bgr_out = cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)
            results.append(bgr_out)
        except Exception as e:
            print(f"  nvidia-vfx error: {e}")
            return None
    return results

# ── Encode/decode through HEVC (simulates what validator receives) ──

def encode_decode_hevc(frames_bgr, fps=30, cq=20):
    h, w = frames_bgr[0].shape[:2]
    tmpdir = tempfile.mkdtemp()
    mp4_path = os.path.join(tmpdir, "encoded.mp4")

    enc_cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "pipe:0",
        "-c:v", "hevc_nvenc", "-cq", str(cq), "-preset", "p4",
        "-profile:v", "main", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", "-an", mp4_path
    ]
    proc = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    for f in frames_bgr:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        stderr = proc.stderr.read().decode()[-300:]
        print(f"  HEVC encode failed: {stderr}")
        return frames_bgr, 0

    fsize = os.path.getsize(mp4_path)

    decoded = []
    cap = cv2.VideoCapture(mp4_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        decoded.append(frame)
    cap.release()
    os.unlink(mp4_path)
    os.rmdir(tmpdir)

    bpp = (fsize * 8) / (len(decoded) * w * h) if decoded else 0
    print(f"  HEVC round-trip: {len(frames_bgr)}→{len(decoded)} frames, {w}x{h}, cq={cq}, "
          f"size={fsize/1024:.0f}KB, bpp={bpp:.4f}")
    return decoded, fsize

# ── Read frames from video file ──

def read_frames(path, max_frames=300):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def downscale_frames(frames, target_w, target_h):
    return [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA) for f in frames]

# ── Run forensics for one path ──

def run_path(label, ground_truth_frames, scale, model_key, cq=20):
    print(f"\n{'='*70}")
    gt_h, gt_w = ground_truth_frames[0].shape[:2]
    input_w, input_h = gt_w // scale, gt_h // scale
    print(f"  {label}")
    print(f"  Ground truth: {gt_w}x{gt_h} → Downscale x{scale} → {input_w}x{input_h} → Upscale → Compare")
    print(f"{'='*70}")

    device = torch.device("cuda:0")
    num_frames = len(ground_truth_frames)
    duration_sec = num_frames / 30.0

    # 1. Downscale ground truth to create miner input
    print(f"\n[1] Downscaling {gt_w}x{gt_h} → {input_w}x{input_h} ({num_frames} frames)...")
    input_frames = downscale_frames(ground_truth_frames, input_w, input_h)

    # 2. Upscale with our model
    print(f"\n[2] Upscaling with {label}...")
    t0 = time.time()

    if model_key == "nvvfx":
        upscaled = upscale_frames_nvvfx(input_frames, device)
        if upscaled is None:
            print("  Falling back to SPAN x2")
            model = load_model(2)
            upscaled = upscale_frames_sr(model, input_frames, device)
            label += " (SPAN fallback)"
    else:
        model = load_model(model_key)
        upscaled = upscale_frames_sr(model, input_frames, device)

    upscale_time = time.time() - t0
    out_h, out_w = upscaled[0].shape[:2]
    ms_per_frame = upscale_time / len(upscaled) * 1000
    print(f"  Output: {out_w}x{out_h}, {len(upscaled)} frames, {upscale_time:.1f}s ({ms_per_frame:.1f}ms/f)")

    # Verify resolution match
    if out_w != gt_w or out_h != gt_h:
        print(f"  WARNING: Output {out_w}x{out_h} != ground truth {gt_w}x{gt_h}!")
        # Resize ground truth to match (shouldn't happen if scale factor is correct)
        ground_truth_resized = downscale_frames(ground_truth_frames, out_w, out_h)
    else:
        ground_truth_resized = ground_truth_frames

    # 3. HEVC encode/decode round-trip
    print(f"\n[3] HEVC encode/decode (cq={cq})...")
    decoded, fsize = encode_decode_hevc(upscaled, fps=30, cq=cq)

    # 4. PieAPP: compare decoded output vs ground truth
    # Free GPU memory from upscaling before PieAPP
    if model_key != "nvvfx":
        del upscaled
    torch.cuda.empty_cache()

    print(f"\n[4] PieAPP: comparing {len(decoded)} decoded frames vs ground truth...")
    num_trials = 5
    all_avgs = []
    all_per_frame = []
    for trial in range(num_trials):
        sample_size = min(4, len(ground_truth_resized), len(decoded))
        max_start = min(len(ground_truth_resized), len(decoded)) - sample_size
        start = 0 if max_start <= 0 else np.random.randint(0, max_start)

        ref_sample = ground_truth_resized[start:start+sample_size]
        dist_sample = decoded[start:start+sample_size]

        print(f"  Trial {trial+1} (frames {start}-{start+sample_size-1}):")
        avg, scores = measure_pieapp(ref_sample, dist_sample, device)
        all_avgs.append(avg)
        all_per_frame.extend(scores)

    pieapp_mean = np.mean(all_avgs)
    pieapp_std = np.std(all_avgs)

    # 5. Calculate validator scores
    s_q = calculate_quality_score(pieapp_mean)
    s_l = calculate_length_score(duration_sec)
    s_pre = 0.5 * s_q + 0.5 * s_l
    s_f = calculate_final_score(s_pre)

    if s_f > 0.32:
        tier = "BONUS (>0.32)"
    elif s_f < 0.07:
        tier = "PENALTY (<0.07)"
    else:
        tier = "NEUTRAL (0.07-0.32)"

    print(f"\n{'─'*60}")
    print(f"  SCORE REPORT: {label}")
    print(f"{'─'*60}")
    print(f"  PieAPP raw      = {pieapp_mean:.6f} ± {pieapp_std:.6f}")
    print(f"  PieAPP range    = {np.min(all_per_frame):.6f} - {np.max(all_per_frame):.6f}")
    print(f"  S_Q  (quality)  = {s_q:.6f}")
    print(f"  S_L  (length)   = {s_l:.6f}  ({duration_sec:.0f}s video)")
    print(f"  S_pre           = {s_pre:.6f}")
    print(f"  S_f  (final)    = {s_f:.6f}")
    print(f"  Tier            = {tier}")
    print(f"  Speed           = {ms_per_frame:.1f}ms/frame")
    print(f"  HEVC CQ         = {cq}")
    print(f"{'─'*60}")

    # Score sensitivity
    print(f"\n  Score sensitivity (S_L={s_l:.3f}):")
    for tsq in [0.70, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98]:
        tp = 0.5 * tsq + 0.5 * s_l
        tsf = calculate_final_score(tp)
        marker = " ◀ CURRENT" if abs(tsq - s_q) < 0.02 else ""
        tl = "BONUS" if tsf > 0.32 else ("PENALTY" if tsf < 0.07 else "neutral")
        print(f"    S_Q={tsq:.2f} → S_f={tsf:.4f} [{tl}]{marker}")

    # PieAPP targets
    print(f"\n  PieAPP targets:")
    for target in [0.90, 0.92, 0.95]:
        lo, hi = 0.0, 2.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if calculate_quality_score(mid) > target:
                lo = mid
            else:
                hi = mid
        gap = "ACHIEVED" if pieapp_mean <= hi else f"need {pieapp_mean - hi:+.4f} improvement"
        print(f"    PieAPP < {hi:.6f} for S_Q >= {target:.2f}  ({gap})")

    return {
        "label": label, "pieapp": pieapp_mean, "pieapp_std": pieapp_std,
        "s_q": s_q, "s_l": s_l, "s_pre": s_pre, "s_f": s_f, "tier": tier,
        "ms_per_frame": ms_per_frame, "cq": cq,
    }


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    # Load ground truth source video
    src_path = "/root/test_540p_300f.mp4"
    print(f"Loading ground truth from {src_path}...")
    gt_frames_540p = read_frames(src_path, max_frames=300)
    gt_h, gt_w = gt_frames_540p[0].shape[:2]
    print(f"  Loaded: {gt_w}x{gt_h}, {len(gt_frames_540p)} frames")

    # For x4 path: ground truth is the 540p video, miner input is 540/4 ≈ 135p
    # But that's unrealistically small. Real validators use 2160p sources.
    # We need higher-res ground truth. Let's create one by upscaling 540p to 1080p with bicubic
    # (this simulates having a 1080p source that gets downscaled to 270p for SD24K)
    print("\nCreating 1080p ground truth (bicubic upscale from 540p)...")
    gt_frames_1080p = [cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_CUBIC) for f in gt_frames_540p]

    # Create 2160p ground truth for x2 tests
    print("Creating 2160p ground truth (bicubic upscale from 540p)...")
    gt_frames_2160p = [cv2.resize(f, (3840, 2160), interpolation=cv2.INTER_CUBIC) for f in gt_frames_540p]

    results = {}

    # ── x4 path: PLKSR ──
    # Ground truth 1080p → downscale to 270p → PLKSR x4 → 1080p → compare vs GT 1080p
    results["x4_plksr"] = run_path(
        "x4 PLKSR (SD24K)", gt_frames_1080p, scale=4, model_key=4, cq=20
    )

    # ── x4 path: realesr (fast fallback) ──
    results["x4_fast"] = run_path(
        "x4 RealESR (fast)", gt_frames_1080p, scale=4, model_key="4fast", cq=20
    )

    # ── x2 path: SPAN (nvidia-vfx not available in this env) ──
    # Ground truth 2160p → downscale to 1080p → SPAN x2 → 2160p → compare vs GT 2160p
    results["x2_span"] = run_path(
        "x2 SPAN (SD2HD)", gt_frames_2160p, scale=2, model_key=2, cq=20
    )

    # ── CQ sensitivity on x4 PLKSR ──
    print(f"\n{'='*70}")
    print(f"  CQ SENSITIVITY TEST (x4 PLKSR)")
    print(f"{'='*70}")
    cq_results = {}
    for cq in [14, 16, 18, 22]:
        r = run_path(f"x4 PLKSR cq={cq}", gt_frames_1080p, scale=4, model_key=4, cq=cq)
        cq_results[cq] = r
    cq_results[20] = results["x4_plksr"]  # reuse existing

    # ── Final summary ──
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Model':<28s}  {'PieAPP':>8s}  {'S_Q':>6s}  {'S_f':>8s}  {'Tier':<20s}  {'ms/f':>6s}")
    print(f"  {'─'*28}  {'─'*8}  {'─'*6}  {'─'*8}  {'─'*20}  {'─'*6}")
    for k, v in results.items():
        if v:
            print(f"  {v['label']:<28s}  {v['pieapp']:8.4f}  {v['s_q']:6.4f}  {v['s_f']:8.4f}  {v['tier']:<20s}  {v['ms_per_frame']:5.0f}ms")
    print()
    print(f"  CQ sensitivity (x4 PLKSR):")
    for cq, v in sorted(cq_results.items()):
        if v:
            print(f"    cq={cq:2d}: PieAPP={v['pieapp']:.4f}  S_Q={v['s_q']:.4f}  S_f={v['s_f']:.4f}  [{v['tier']}]")
    print(f"{'='*70}")
