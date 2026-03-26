#!/usr/bin/env python3
"""VMAF comparison after color matrix fix:
  Decode NV12 → RGB using BT.601 (matches ffmpeg default for unknown-colorspace H.264)
  Encode RGB → NV12 using BT.709 (matches production ffmpeg encoder -colorspace bt709)

Root cause: previous test used BT.709 for decode, causing 25px mean G-channel error.
"""
import sys, os, time, json, subprocess
sys.path.insert(0, '/root/vidaio-subnet')
os.chdir('/root/vidaio-subnet')

import torch
import nvvfx
import PyNvVideoCodec as nvc
from pathlib import Path

TEST_INPUT   = Path("/tmp/test_1080p_10s.mp4")
OUT_DIR      = Path("/tmp/vmaf_fixed")
VMAF_MODEL   = "/tmp/vmaf/model/vmaf_4k_v0.6.1.json"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# BT.601 limited-range decode  (ffmpeg default for unknown colorspace H.264)
# --------------------------------------------------------------------------
def nv12_to_rgb_bt601_bilinear(nv12, H, W):
    """NV12 → float32 RGB CHW [0,1]  — BT.601, bilinear chroma."""
    Y = nv12[:H].float() - 16.0
    uv = nv12[H:].view(H//2, W//2, 2).float()
    uv_up = torch.nn.functional.interpolate(
        uv.permute(2,0,1).unsqueeze(0), scale_factor=2,
        mode='bilinear', align_corners=False)   # (1,2,H,W)
    U = uv_up[0,0] - 128.0
    V = uv_up[0,1] - 128.0
    R = (1.164*Y + 1.596*V          ).clamp_(0, 255)
    G = (1.164*Y - 0.392*U - 0.813*V).clamp_(0, 255)
    B = (1.164*Y + 2.017*U          ).clamp_(0, 255)
    return torch.stack([R,G,B],0).div_(255.0)

# --------------------------------------------------------------------------
# BT.709 limited-range encode  (matches production ffmpeg -colorspace bt709)
# --------------------------------------------------------------------------
def rgb_to_nv12_bt709(rgb, H, W):
    """float32 RGB CHW [0,1] → NV12 uint8 — BT.709 limited range."""
    p = rgb.mul(255.0).clamp_(0, 255)
    R, G, B = p[0], p[1], p[2]
    Y  = (16.0  + 0.1826*R + 0.6142*G + 0.0620*B).clamp_(16, 235)
    Cb = (128.0 - 0.1006*R - 0.3386*G + 0.4392*B).clamp_(16, 240)
    Cr = (128.0 + 0.4392*R - 0.3990*G - 0.0402*B).clamp_(16, 240)
    Y_u8 = Y.to(torch.uint8)
    Cb_s = Cb.view(H//2, 2, W//2, 2).mean(dim=(1,3))
    Cr_s = Cr.view(H//2, 2, W//2, 2).mean(dim=(1,3))
    uv   = torch.stack([Cb_s, Cr_s], dim=2).to(torch.uint8).view(H//2, W)
    return torch.cat([Y_u8, uv], dim=0)

# --------------------------------------------------------------------------
# Production pipeline (reference, unchanged)
# --------------------------------------------------------------------------
def run_production(vsr, inp, out):
    r = subprocess.run(
        ['ffprobe','-v','quiet','-select_streams','v:0',
         '-show_entries','stream=width,height,r_frame_rate','-of','csv=p=0',str(inp)],
        stdout=subprocess.PIPE, text=True)
    w,h,fps_frac = r.stdout.strip().split(',')
    W,H = int(w), int(h)
    n,d = map(int, fps_frac.split('/'))
    fps = n/d
    OW, OH = W*2, H*2
    fsz = W*H*3

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

    n_frames, t0 = 0, time.time()
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
    elapsed = time.time()-t0
    print(f"  Production: {n_frames} frames, {elapsed:.2f}s, {n_frames/elapsed:.1f} fps")
    return n_frames, elapsed

# --------------------------------------------------------------------------
# NV12 GPU-direct pipeline  (BT.601 decode → nvvfx → BT.709 encode)
# --------------------------------------------------------------------------
def run_nv12_fixed(vsr, inp, out):
    dec = nvc.CreateSimpleDecoder(str(inp), gpuid=0, useDeviceMemory=True,
          outputColorType=nvc.OutputColorType.NATIVE, decoderCacheSize=1)
    meta = dec.get_stream_metadata()
    W, H = meta.width, meta.height
    FPS  = meta.average_fps
    OW, OH = W*2, H*2

    enc = nvc.CreateEncoder(OW, OH, 'NV12', False,
                            codec='hevc', preset='P4',
                            rc='constqp', constqp=20, profile='main')

    # warmup (3 frames, not timed)
    warmup = dec.get_batch_frames(3)
    for wf in warmup:
        nv12 = torch.from_dlpack(wf)
        rgb  = nv12_to_rgb_bt601_bilinear(nv12, H, W)
        res  = vsr.run(rgb)
        enc.Encode(rgb_to_nv12_bt709(torch.from_dlpack(res.image).clone(), OH, OW))
    torch.cuda.synchronize()

    chunks = []
    total  = len(warmup)
    t0 = time.time()
    while True:
        batch = dec.get_batch_frames(8)
        if not batch: break
        for frame in batch:
            nv12 = torch.from_dlpack(frame)
            rgb  = nv12_to_rgb_bt601_bilinear(nv12, H, W)
            res  = vsr.run(rgb)
            bits = enc.Encode(rgb_to_nv12_bt709(torch.from_dlpack(res.image).clone(), OH, OW))
            if bits: chunks.append(bits)
            total += 1
    eos = enc.EndEncode()
    if eos: chunks.append(eos)
    torch.cuda.synchronize()
    elapsed = time.time()-t0
    timed   = total - len(warmup)

    hevc_path = out.with_suffix('.hevc')
    hevc_path.write_bytes(b''.join(chunks))
    # Mux with BT.709 color tags to match production
    subprocess.run(
        ['ffmpeg','-y','-r',str(FPS),'-i',str(hevc_path),
         '-c','copy',
         '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',
         '-movflags','+faststart', str(out)],
        capture_output=True, check=True)
    hevc_path.unlink()
    print(f"  NV12 fixed: {total} frames ({len(warmup)} warmup+{timed} timed), "
          f"{elapsed:.2f}s, {timed/elapsed:.1f} fps")
    return total, elapsed, timed

# --------------------------------------------------------------------------
# VMAF via vmaf CLI
# --------------------------------------------------------------------------
def run_vmaf(dist_mp4, ref_mp4, label):
    dist_yuv  = OUT_DIR / f"{label}.yuv"
    ref_yuv   = OUT_DIR / "reference.yuv"
    vmaf_json = OUT_DIR / f"{label}_vmaf.json"

    r = subprocess.run(
        ['ffprobe','-v','quiet','-select_streams','v:0',
         '-show_entries','stream=width,height','-of','csv=p=0',str(dist_mp4)],
        stdout=subprocess.PIPE, text=True)
    W, H = map(int, r.stdout.strip().split(','))

    print(f"  Decoding {label} ({W}x{H}) to YUV...")
    subprocess.run(
        ['ffmpeg','-y','-i',str(dist_mp4),
         '-c:v','rawvideo','-pix_fmt','yuv420p',str(dist_yuv)],
        capture_output=True, check=True)

    if not ref_yuv.exists():
        print(f"  Decoding reference to YUV...")
        subprocess.run(
            ['ffmpeg','-y','-i',str(ref_mp4),
             '-c:v','rawvideo','-pix_fmt','yuv420p',str(ref_yuv)],
            capture_output=True, check=True)

    print(f"  Running VMAF for {label}...")
    result = subprocess.run(
        ['vmaf','-r',str(ref_yuv),'-d',str(dist_yuv),
         '-w',str(W),'-h',str(H),'-p','420','-b','8',
         '-m',f'path={VMAF_MODEL}',
         '--json','-o',str(vmaf_json)],
        capture_output=True, text=True)
    dist_yuv.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"  VMAF error: {result.stderr[:300]}")
        return None, None

    with open(vmaf_json) as f:
        scores = [fr['metrics']['vmaf'] for fr in json.load(f)['frames']]
    return sum(scores)/len(scores), min(scores)

# --------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------
print("="*65)
print("VMAF TEST: NV12 GPU-direct (BT.601 decode / BT.709 encode fix)")
print(f"Input: {TEST_INPUT}")
print("="*65)

print("\n[1/5] Loading VSR (1080p→4K)...")
vsr = nvvfx.VideoSuperRes(quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=0)
vsr.output_width  = 3840
vsr.output_height = 2160
vsr.load()
print("  VSR ready.")

# Reference
ref_mp4 = OUT_DIR / "reference_bicubic_4k.mp4"
if not ref_mp4.exists():
    print("\n[2/5] Creating bicubic lanczos 4K reference (cq=10, p7)...")
    subprocess.run([
        'ffmpeg','-y','-i',str(TEST_INPUT),
        '-vf','scale=3840:2160:flags=lanczos',
        '-c:v','hevc_nvenc','-cq','10','-preset','p7',
        '-pix_fmt','yuv420p',
        '-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709',
        '-movflags','+faststart', str(ref_mp4)
    ], capture_output=True, check=True)
    print(f"  Reference: {ref_mp4}")
else:
    print(f"\n[2/5] Reusing reference: {ref_mp4}")

# Production
prod_mp4 = OUT_DIR / "production_nvvfx.mp4"
print("\n[3/5] Production pipeline (nvvfx + ffmpeg rgb24 pipe)...")
prod_frames, prod_elapsed = run_production(vsr, TEST_INPUT, prod_mp4)

# NV12 fixed
nv12_mp4 = OUT_DIR / "nv12_bt601dec_bt709enc.mp4"
print("\n[4/5] NV12 GPU-direct (BT.601 decode, BT.709 encode, bilinear)...")
nv12_total, nv12_elapsed, nv12_timed = run_nv12_fixed(vsr, TEST_INPUT, nv12_mp4)
vsr.close()

# VMAF
print("\n[5/5] VMAF vs bicubic 4K reference...")
prod_mean, prod_min = run_vmaf(prod_mp4, ref_mp4, "production")
nv12_mean, nv12_min = run_vmaf(nv12_mp4, ref_mp4, "nv12_fixed")
# Cleanup ref yuv
(OUT_DIR / "reference.yuv").unlink(missing_ok=True)

# Bonus: PSNR between the two outputs
r2 = subprocess.run(
    ['ffmpeg','-i',str(nv12_mp4),'-i',str(prod_mp4),
     '-filter_complex','psnr','-f','null','-'],
    capture_output=True, text=True)
import re
pm = re.search(r'average:([\d.]+)', r2.stderr)
psnr_between = float(pm.group(1)) if pm else None

print(f"\n{'='*65}")
print(f"RESULTS — {TEST_INPUT.name}")
print(f"{'='*65}")
print(f"{'Pipeline':<38} {'Time':>7} {'fps':>6}  {'VMAF mean':>10} {'VMAF min':>10}")
print(f"{'-'*73}")

prod_fps = prod_frames / prod_elapsed
nv12_fps = nv12_timed  / nv12_elapsed

def fmt(m, mn):
    return f"{m:>10.2f} {mn:>10.2f}" if m else "       N/A        N/A"

print(f"{'Production (nvvfx+pipe)':<38} {prod_elapsed:>6.1f}s {prod_fps:>6.1f}  {fmt(prod_mean, prod_min)}")
print(f"{'NV12 GPU-direct (BT.601/709 fix)':<38} {nv12_elapsed:>6.1f}s {nv12_fps:>6.1f}  {fmt(nv12_mean, nv12_min)}")
print()
print(f"Speedup:          {nv12_fps/prod_fps:.1f}x")
print(f"Projected 300f:   production={300/prod_fps:.1f}s   nv12={300/nv12_fps:.1f}s")
if psnr_between: print(f"PSNR NV12 vs prod: {psnr_between:.2f} dB")
print()
if nv12_mean:
    gate_mean = "✓ PASS" if nv12_mean >= 89 else "✗ FAIL"
    gate_min  = "✓ PASS" if nv12_min  >= 89 else "✗ FAIL"
    print(f"VMAF gate (≥89)   mean={nv12_mean:.2f} {gate_mean}   min={nv12_min:.2f} {gate_min}")
    overall = nv12_mean >= 89
    print()
    if overall:
        print("✅  NV12 GPU-direct PASSES — ready for production integration.")
    else:
        print("❌  NV12 GPU-direct FAILS quality gate.")
