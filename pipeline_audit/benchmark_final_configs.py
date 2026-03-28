#!/usr/bin/env python3
"""
Benchmark: end-to-end viable 4x upscale configurations at production resolution.

GT is 1080p (1920x1080, 300 frames).  Input = 480x270 for true 4x.
Configs:
  A  PLKSR eager 4x  (480x270 -> 1920x1080)
  B  DAT-light compiled 4x (480x270 -> 1920x1080)  [two-stage skipped: GT is 1080p]
  C  DAT-light eager  4x  (480x270 -> 1920x1080)
"""

import sys, os, time, math, json, subprocess, tempfile, shutil
sys.path.insert(0, "/root/vidaio-subnet")

import numpy as np
import torch
import cv2
import pyiqa
from spandrel import ModelLoader

# ── constants ──────────────────────────────────────────────────────────────
GT_PATH   = "/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4"
INPUT_270 = "/root/pipeline_audit/payload_270p.mp4"
OUT_DIR   = "/root/pipeline_audit"
TIMEOUT   = 90
OVERHEAD  = 8          # seconds for decode + encode + upload overhead
N_FRAMES  = 60         # frames to actually process (measure)
TOTAL_FRAMES = 300     # full video length
CQ        = 20

PLKSR_PATH    = os.path.expanduser("~/.cache/span/PLKSR_X4_DF2K.pth")
DAT_LIGHT_PATH = os.path.expanduser("~/.cache/span/DAT_light_x4.pth")

DEVICE = torch.device("cuda")

# ── scoring formulas ───────────────────────────────────────────────────────
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
    sf = 0.1 * math.exp(6.979 * (sp - 0.5))
    return sf, sq

# ── helpers ────────────────────────────────────────────────────────────────
def ensure_input():
    """Create 480x270 input from GT if not present."""
    if not os.path.exists(INPUT_270):
        print("[prep] Creating 480x270 payload from GT …")
        subprocess.run([
            "ffmpeg", "-y", "-i", GT_PATH,
            "-vf", "scale=480:270",
            "-c:v", "libx264", "-crf", "18",
            INPUT_270
        ], check=True, capture_output=True)
    # verify
    cap = cv2.VideoCapture(INPUT_270)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"[prep] Input: {w}x{h}, {n} frames")
    return w, h, n


def read_frames(path, n):
    """Read first n frames as uint8 numpy (H,W,3 BGR)."""
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(n):
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()
    return frames


def frames_to_tensor_batch(frames):
    """List[np HWC BGR uint8] -> (N,3,H,W) float16 RGB on GPU."""
    tensors = []
    for f in frames:
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).permute(2, 0, 1).half() / 255.0
        tensors.append(t)
    return torch.stack(tensors).to(DEVICE)


def pad_to_multiple(tensor, mult=16):
    """Pad spatial dims to nearest multiple of mult. Returns (padded, (ph, pw))."""
    _, _, h, w = tensor.shape
    ph = (mult - h % mult) % mult
    pw = (mult - w % mult) % mult
    if ph or pw:
        tensor = torch.nn.functional.pad(tensor, (0, pw, 0, ph), mode="reflect")
    return tensor, (ph, pw)


def unpad(tensor, pad_hw):
    ph, pw = pad_hw
    if ph or pw:
        h = tensor.shape[2] - ph * 4  # scale factor 4
        w = tensor.shape[3] - pw * 4
        tensor = tensor[:, :, :h, :w]
    return tensor


def encode_frames(frames_np, out_path, fps=30):
    """Encode list of uint8 BGR frames to hevc_nvenc CQ=20 yuv420p."""
    h, w = frames_np[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "hevc_nvenc", "-cq", str(CQ),
        "-pix_fmt", "yuv420p",
        out_path
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for f in frames_np:
        proc.stdin.write(f.tobytes())
    proc.stdin.close()
    proc.wait()


def compute_pieapp(gt_frames, upscaled_frames, sample=10):
    """Compute PieAPP on `sample` evenly-spaced frames. Returns mean score."""
    metric = pyiqa.create_metric("pieapp", device=DEVICE)
    n = min(len(gt_frames), len(upscaled_frames))
    indices = np.linspace(0, n - 1, min(sample, n), dtype=int)
    scores = []
    for i in indices:
        gt_rgb = cv2.cvtColor(gt_frames[i], cv2.COLOR_BGR2RGB)
        up_rgb = cv2.cvtColor(upscaled_frames[i], cv2.COLOR_BGR2RGB)

        # Resize upscaled to GT size if mismatch
        gh, gw = gt_rgb.shape[:2]
        uh, uw = up_rgb.shape[:2]
        if (uh, uw) != (gh, gw):
            up_rgb = cv2.resize(up_rgb, (gw, gh), interpolation=cv2.INTER_LANCZOS4)

        gt_t = torch.from_numpy(gt_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        up_t = torch.from_numpy(up_rgb).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            s = metric(up_t.to(DEVICE), gt_t.to(DEVICE)).item()
            scores.append(abs(s))
    del metric
    torch.cuda.empty_cache()
    return float(np.mean(scores))


def load_model(path, compile_model=False):
    """Load a spandrel model, FP16 on GPU."""
    model = ModelLoader().load_from_file(path).model
    model = model.half().to(DEVICE).eval()
    if compile_model:
        model = torch.compile(model, mode="max-autotune")
    return model


# ── benchmark runner ───────────────────────────────────────────────────────
def run_config(name, model_path, compile_model, input_frames, gt_frames, n_frames):
    """Run a single configuration benchmark. Returns dict of results."""
    print(f"\n{'='*60}")
    print(f"  Config {name}")
    print(f"{'='*60}")

    # Load model
    print(f"[{name}] Loading model …")
    model = load_model(model_path, compile_model=compile_model)

    # Prepare input tensor
    inp_tensor = frames_to_tensor_batch(input_frames[:n_frames])
    inp_tensor, pad_hw = pad_to_multiple(inp_tensor, mult=16)
    print(f"[{name}] Input tensor: {inp_tensor.shape}")

    # Warmup (2 frames)
    with torch.no_grad():
        for _ in range(2):
            _ = model(inp_tensor[:1])
    torch.cuda.synchronize()

    # Timed inference
    print(f"[{name}] Running inference on {n_frames} frames …")
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    upscaled_list = []
    with torch.no_grad():
        for i in range(n_frames):
            if compile_model:
                torch.compiler.cudagraph_mark_step_begin()
            out = model(inp_tensor[i:i+1])
            upscaled_list.append(out.clone())
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    infer_time = t1 - t0
    fps = n_frames / infer_time
    time_300 = TOTAL_FRAMES / fps
    total_time = time_300 + OVERHEAD

    # Unpad and convert to numpy
    upscaled_tensor = torch.cat(upscaled_list, dim=0)
    upscaled_tensor = unpad(upscaled_tensor, pad_hw)
    print(f"[{name}] Output tensor: {upscaled_tensor.shape}")

    upscaled_np = []
    for i in range(upscaled_tensor.shape[0]):
        frame = upscaled_tensor[i].clamp(0, 1).float().cpu().permute(1, 2, 0).numpy()
        frame = (frame * 255).astype(np.uint8)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        upscaled_np.append(frame)

    # Encode to check it works
    out_path = os.path.join(OUT_DIR, f"bench_{name}.mp4")
    print(f"[{name}] Encoding to {out_path} …")
    encode_frames(upscaled_np, out_path)

    # PieAPP
    print(f"[{name}] Computing PieAPP (10 sample frames) …")
    pieapp_mean = compute_pieapp(gt_frames[:n_frames], upscaled_np[:n_frames], sample=10)
    sf, sq = calc_sf(pieapp_mean, cl=10)

    # Cleanup
    del model, inp_tensor, upscaled_tensor, upscaled_list
    torch.cuda.empty_cache()

    results = {
        "config": name,
        "n_frames_tested": n_frames,
        "fps_inference": round(fps, 2),
        "time_300_frames_s": round(time_300, 2),
        "total_time_s": round(total_time, 2),
        "fits_90s": total_time <= TIMEOUT,
        "pieapp_mean": round(pieapp_mean, 4),
        "S_Q": round(sq, 4),
        "S_F": round(sf, 4),
        "BONUS": sf > 0.32,
        "output_resolution": f"{upscaled_np[0].shape[1]}x{upscaled_np[0].shape[0]}",
    }

    print(f"\n[{name}] Results:")
    for k, v in results.items():
        print(f"  {k:25s}: {v}")

    return results


# ── main ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FINAL CONFIGS BENCHMARK — Production Resolution")
    print("=" * 60)

    # Ensure input exists
    iw, ih, in_frames = ensure_input()
    print(f"[info] GT: 1920x1080, Input: {iw}x{ih}")
    print(f"[info] Testing {N_FRAMES} frames, extrapolating to {TOTAL_FRAMES}")

    # Read frames
    print("[prep] Reading input frames …")
    input_frames = read_frames(INPUT_270, N_FRAMES)
    print(f"[prep] Read {len(input_frames)} input frames")

    print("[prep] Reading GT frames …")
    gt_frames = read_frames(GT_PATH, N_FRAMES)
    print(f"[prep] Read {len(gt_frames)} GT frames")

    all_results = []

    # ── Config A: PLKSR eager 4x ───────────────────────────────────────
    try:
        res = run_config(
            name="A_PLKSR_eager_4x",
            model_path=PLKSR_PATH,
            compile_model=False,
            input_frames=input_frames,
            gt_frames=gt_frames,
            n_frames=N_FRAMES,
        )
        all_results.append(res)
    except Exception as e:
        print(f"[A] FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── Config B: DAT-light compiled 4x ────────────────────────────────
    try:
        res = run_config(
            name="B_DATlight_compiled_4x",
            model_path=DAT_LIGHT_PATH,
            compile_model=True,
            input_frames=input_frames,
            gt_frames=gt_frames,
            n_frames=N_FRAMES,
        )
        all_results.append(res)
    except Exception as e:
        print(f"[B] FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── Config C: DAT-light eager 4x ──────────────────────────────────
    try:
        res = run_config(
            name="C_DATlight_eager_4x",
            model_path=DAT_LIGHT_PATH,
            compile_model=False,
            input_frames=input_frames,
            gt_frames=gt_frames,
            n_frames=30,           # slower model, 30 frames only
        )
        all_results.append(res)
    except Exception as e:
        print(f"[C] FAILED: {e}")
        import traceback; traceback.print_exc()

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  SUMMARY")
    print("=" * 80)
    header = f"{'Config':<28s} {'FPS':>6s} {'300f(s)':>8s} {'Total':>7s} {'90s?':>5s} {'PieAPP':>7s} {'S_Q':>6s} {'S_F':>6s} {'BONUS':>6s}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        line = (
            f"{r['config']:<28s} "
            f"{r['fps_inference']:>6.1f} "
            f"{r['time_300_frames_s']:>8.1f} "
            f"{r['total_time_s']:>7.1f} "
            f"{'YES' if r['fits_90s'] else 'NO':>5s} "
            f"{r['pieapp_mean']:>7.4f} "
            f"{r['S_Q']:>6.4f} "
            f"{r['S_F']:>6.4f} "
            f"{'YES' if r['BONUS'] else 'NO':>6s}"
        )
        print(line)
    print()

    # Save JSON
    out_json = os.path.join(OUT_DIR, "benchmark_final_configs.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[done] Results saved to {out_json}")


if __name__ == "__main__":
    main()
