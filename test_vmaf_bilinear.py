#!/usr/bin/env python3
"""VMAF comparison: production nvvfx-pipe vs NV12 GPU-direct with bilinear chroma fix.

Reference: high-quality bicubic lanczos 4K (cq=10, preset=p7)
Both pipelines evaluated vs same reference using vmaf_4k_v0.6.1.json.
"""
import sys, os, time, json, subprocess
sys.path.insert(0, '/root/vidaio-subnet')
os.chdir('/root/vidaio-subnet')

import torch
import nvvfx
import PyNvVideoCodec as nvc
from pathlib import Path

TEST_INPUT   = Path("/tmp/test_1080p_10s.mp4")
OUT_DIR      = Path("/tmp/vmaf_bilinear")
VMAF_MODEL   = "/tmp/vmaf/model/vmaf_4k_v0.6.1.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# BT.709 color conversion — GPU
# --------------------------------------------------------------------------
def nv12_to_rgb_chw_bilinear(nv12_tensor, H, W):
    """NV12 → float32 RGB CHW [0,1] with BILINEAR chroma upsampling."""
    Y = nv12_tensor[:H].float() - 16.0                           # (H, W)
    uv_plane = nv12_tensor[H:].view(H//2, W//2, 2).float()       # (H/2, W/2, 2)

    # Bilinear 2x upsampling: (1,2,H/2,W/2) → (1,2,H,W)
    uv_nchw = uv_plane.permute(2, 0, 1).unsqueeze(0)
    uv_up   = torch.nn.functional.interpolate(
        uv_nchw, scale_factor=2, mode='bilinear', align_corners=False
    )
    U = uv_up[0, 0] - 128.0   # (H, W)
    V = uv_up[0, 1] - 128.0   # (H, W)

    R = (1.164*Y + 1.793*V          ).clamp_(0, 255)
    G = (1.164*Y - 0.213*U - 0.533*V).clamp_(0, 255)
    B = (1.164*Y + 2.112*U          ).clamp_(0, 255)
    return torch.stack([R, G, B], dim=0).div_(255.0)   # (3,H,W) float32

def rgb_chw_to_nv12(rgb, H, W):
    """float32 RGB CHW [0,1] → NV12 BT.709 limited-range uint8."""
    p = rgb.mul(255.0).clamp_(0, 255)
    R, G, B = p[0], p[1], p[2]
    Y  = (16.0  + 0.2568*R + 0.5041*G + 0.0979*B).clamp_(16, 235)
    Cb = (128.0 - 0.1482*R - 0.2910*G + 0.4392*B).clamp_(16, 240)
    Cr = (128.0 + 0.4392*R - 0.3678*G - 0.0714*B).clamp_(16, 240)
    Y_u8  = Y.to(torch.uint8)
    Cb_s  = Cb.view(H//2, 2, W//2, 2).mean(dim=(1,3))
    Cr_s  = Cr.view(H//2, 2, W//2, 2).mean(dim=(1,3))
    uv    = torch.stack([Cb_s, Cr_s], dim=2).to(torch.uint8).view(H//2, W)
    return torch.cat([Y_u8, uv], dim=0)

# --------------------------------------------------------------------------
# Pipeline A: current production (nvvfx + ffmpeg rgb24 pipes)
# --------------------------------------------------------------------------
def run_production(vsr, inp, out):
    r = subprocess.run(
        ['ffprobe','-v','quiet','-select_streams','v:0',
         '-show_entries','stream=width,height,r_frame_rate','-of','csv=p=0', str(inp)],
        stdout=subprocess.PIPE, text=True
    )
    w, h, fps_frac = r.stdout.strip().split(',')
    W, H = int(w), int(h)
    n, d  = map(int, fps_frac.split('/'))
    fps   = n / d
    OW, OH = W*2, H*2
    fsz   = W * H * 3

    dec = subprocess.Popen(
        ['ffmpeg','-i',str(inp),'-f','rawvideo','-pix_fmt','rgb24','-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc = subprocess.Popen(
        ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}',
         '-r',str(fps),'-i','pipe:0',
         '-c:v','hevc_nvenc','-cq','20','-preset','p4','-profile:v','main',
         '-pix_fmt','yuv420p','-sar','1:1',
         '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',
         '-movflags','+faststart', str(out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    n_frames = 0
    t0 = time.time()
    while True:
        raw = dec.stdout.read(fsz)
        if len(raw) < fsz: break
        t = (torch.frombuffer(bytearray(raw), dtype=torch.uint8)
             .reshape(H,W,3).cuda().float().div_(255.0).permute(2,0,1).contiguous())
        res = vsr.run(t)
        out_rgb = torch.from_dlpack(res.image).clone()
        enc.stdin.write(out_rgb.clamp_(0,1).mul_(255).round_().to(torch.uint8)
                        .permute(1,2,0).contiguous().cpu().numpy().tobytes())
        n_frames += 1

    enc.stdin.close(); enc.stderr.read(); enc.wait(); dec.wait()
    elapsed = time.time() - t0
    print(f"  Production: {n_frames} frames, {elapsed:.2f}s, {n_frames/elapsed:.1f} fps")
    return n_frames, elapsed

# --------------------------------------------------------------------------
# Pipeline B: NV12 GPU-direct with bilinear chroma
# --------------------------------------------------------------------------
def run_nv12_bilinear(vsr, inp, out):
    dec = nvc.CreateSimpleDecoder(
        str(inp), gpuid=0, useDeviceMemory=True,
        outputColorType=nvc.OutputColorType.NATIVE, decoderCacheSize=1)
    meta = dec.get_stream_metadata()
    W, H = meta.width, meta.height
    FPS   = meta.average_fps
    OW, OH = W*2, H*2

    enc = nvc.CreateEncoder(OW, OH, 'NV12', False,
                            codec='hevc', preset='P4',
                            rc='constqp', constqp=20, profile='main')

    # warmup
    warmup = dec.get_batch_frames(3)
    for wf in warmup:
        nv12 = torch.from_dlpack(wf)
        rgb  = nv12_to_rgb_chw_bilinear(nv12, H, W)
        res  = vsr.run(rgb)
        out_rgb = torch.from_dlpack(res.image).clone()
        enc.Encode(rgb_chw_to_nv12(out_rgb, OH, OW))
    torch.cuda.synchronize()

    chunks = []
    total  = len(warmup)
    t0 = time.time()
    while True:
        batch = dec.get_batch_frames(8)
        if not batch: break
        for frame in batch:
            nv12 = torch.from_dlpack(frame)
            rgb  = nv12_to_rgb_chw_bilinear(nv12, H, W)
            res  = vsr.run(rgb)
            out_rgb = torch.from_dlpack(res.image).clone()
            bits = enc.Encode(rgb_chw_to_nv12(out_rgb, OH, OW))
            if bits: chunks.append(bits)
            total += 1
    eos = enc.EndEncode()
    if eos: chunks.append(eos)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    timed   = total - len(warmup)

    hevc_path = out.with_suffix('.hevc')
    hevc_path.write_bytes(b''.join(chunks))
    subprocess.run(
        ['ffmpeg','-y','-r',str(FPS),'-i',str(hevc_path),
         '-c','copy','-movflags','+faststart', str(out)],
        capture_output=True, check=True)
    hevc_path.unlink()

    print(f"  NV12 bilinear: {total} frames ({len(warmup)} warmup + {timed} timed), "
          f"{elapsed:.2f}s, {timed/elapsed:.1f} fps")
    return total, elapsed, timed

# --------------------------------------------------------------------------
# VMAF: decode mp4 → raw yuv, run vmaf CLI
# --------------------------------------------------------------------------
def run_vmaf(dist_mp4, ref_yuv, label):
    dist_yuv = OUT_DIR / f"{label}.yuv"
    vmaf_json = OUT_DIR / f"{label}_vmaf.json"

    # probe output dimensions
    r = subprocess.run(
        ['ffprobe','-v','quiet','-select_streams','v:0',
         '-show_entries','stream=width,height','-of','csv=p=0', str(dist_mp4)],
        stdout=subprocess.PIPE, text=True)
    W, H = map(int, r.stdout.strip().split(','))

    print(f"  Decoding {label} to YUV ({W}x{H})...")
    subprocess.run(
        ['ffmpeg','-y','-i',str(dist_mp4),'-c:v','rawvideo','-pix_fmt','yuv420p',str(dist_yuv)],
        capture_output=True, check=True)

    print(f"  Running VMAF for {label}...")
    result = subprocess.run(
        ['vmaf','-r',str(ref_yuv),'-d',str(dist_yuv),
         '-w',str(W),'-h',str(H),'-p','420','-b','8',
         '-m',f'path={VMAF_MODEL}',
         '--json','-o',str(vmaf_json)],
        capture_output=True, text=True)

    dist_yuv.unlink(missing_ok=True)   # free space immediately

    if result.returncode != 0:
        print(f"  VMAF stderr: {result.stderr[:500]}")
        return None, None

    with open(vmaf_json) as f:
        data = json.load(f)
    scores = [fr['metrics']['vmaf'] for fr in data['frames']]
    return sum(scores)/len(scores), min(scores)


# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
print("="*65)
print("VMAF COMPARISON: nvvfx-pipe vs NV12 GPU-direct (bilinear chroma)")
print(f"Input: {TEST_INPUT}")
print("="*65)

# [1] Load VSR
print("\n[1/5] Loading nvidia-vfx VSR (1080p→4K)...")
vsr = nvvfx.VideoSuperRes(quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=0)
W_IN, H_IN = 1920, 1080
vsr.output_width  = W_IN * 2
vsr.output_height = H_IN * 2
vsr.load()
print("  VSR ready.")

# [2] Create bicubic 4K reference (HIGH quality, reuse if exists)
ref_mp4 = OUT_DIR / "reference_bicubic_4k.mp4"
ref_yuv = OUT_DIR / "reference_bicubic_4k.yuv"
if not ref_mp4.exists():
    print("\n[2/5] Creating bicubic lanczos 4K reference (cq=10, preset=p7)...")
    subprocess.run([
        'ffmpeg','-y','-i',str(TEST_INPUT),
        '-vf','scale=3840:2160:flags=lanczos',
        '-c:v','hevc_nvenc','-cq','10','-preset','p7',
        '-pix_fmt','yuv420p','-movflags','+faststart', str(ref_mp4)
    ], capture_output=True, check=True)
    print(f"  Reference: {ref_mp4}")
else:
    print(f"\n[2/5] Reusing existing reference: {ref_mp4}")

if not ref_yuv.exists():
    print("  Decoding reference to YUV...")
    subprocess.run(
        ['ffmpeg','-y','-i',str(ref_mp4),
         '-c:v','rawvideo','-pix_fmt','yuv420p', str(ref_yuv)],
        capture_output=True, check=True)
    print(f"  Reference YUV: {ref_yuv.stat().st_size/1e9:.2f} GB")

# [3] Production pipeline
prod_mp4 = OUT_DIR / "production_nvvfx.mp4"
print("\n[3/5] Running production pipeline (nvvfx + ffmpeg rgb24 pipes)...")
prod_frames, prod_elapsed = run_production(vsr, TEST_INPUT, prod_mp4)

# [4] NV12 GPU-direct with bilinear chroma
nv12_mp4 = OUT_DIR / "nv12_bilinear.mp4"
print("\n[4/5] Running NV12 GPU-direct pipeline (bilinear chroma)...")
nv12_total, nv12_elapsed, nv12_timed = run_nv12_bilinear(vsr, TEST_INPUT, nv12_mp4)
vsr.close()

# [5] VMAF
print("\n[5/5] Running VMAF (both vs bicubic 4K reference)...")
prod_mean, prod_min = run_vmaf(prod_mp4, ref_yuv, "production")
nv12_mean, nv12_min = run_vmaf(nv12_mp4, ref_yuv, "nv12_bilinear")

# Cleanup reference yuv (large)
ref_yuv.unlink(missing_ok=True)

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print(f"\n{'='*65}")
print(f"RESULTS — {TEST_INPUT.name}")
print(f"{'='*65}")
print(f"{'Pipeline':<38} {'Time':>7} {'fps':>6}  {'VMAF mean':>10} {'VMAF min':>10}")
print(f"{'-'*73}")

prod_fps = prod_frames / prod_elapsed
nv12_fps = nv12_timed  / nv12_elapsed

def fmt(m, mn):
    if m is None: return "   N/A        N/A"
    return f"{m:>10.2f} {mn:>10.2f}"

print(f"{'Production (nvvfx+pipe)':<38} {prod_elapsed:>6.1f}s {prod_fps:>6.1f}  {fmt(prod_mean, prod_min)}")
print(f"{'NV12 GPU-direct (bilinear)':<38} {nv12_elapsed:>6.1f}s {nv12_fps:>6.1f}  {fmt(nv12_mean, nv12_min)}")
print()
print(f"Speedup:          {nv12_fps/prod_fps:.1f}x")
print(f"Projected 300f:   production={300/prod_fps:.1f}s   nv12={300/nv12_fps:.1f}s")
print()
if nv12_mean is not None:
    gate_mean = "✓ PASS" if nv12_mean >= 89 else "✗ FAIL"
    gate_min  = "✓ PASS" if nv12_min  >= 89 else "✗ FAIL"
    print(f"VMAF gate (≥89)   mean={nv12_mean:.2f} {gate_mean}   min={nv12_min:.2f} {gate_min}")
    if nv12_mean >= 89:
        print("\n✅ NV12 GPU-direct PASSES quality gate — ready for production integration.")
    else:
        print("\n❌ NV12 GPU-direct FAILS quality gate.")
