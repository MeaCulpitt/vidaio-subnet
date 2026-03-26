#!/usr/bin/env python3
"""Test C: pynvc decode (BT.601) → VSR → RGB→NV12 → ffmpeg encode (pipe NV12)
Bypasses our rgb_to_nv12_bt709 + nvc entirely for encoding.
"""
import sys, os, time, json, subprocess, re
sys.path.insert(0, '/root/vidaio-subnet')
os.chdir('/root/vidaio-subnet')

import torch
import nvvfx
import PyNvVideoCodec as nvc
from pathlib import Path

TEST = Path("/tmp/test_1080p_10s.mp4")
OUT  = Path("/tmp/isolate")
VMAF = "/tmp/vmaf/model/vmaf_4k_v0.6.1.json"

def probe(p):
    r = subprocess.run(['ffprobe','-v','quiet','-select_streams','v:0',
         '-show_entries','stream=width,height,r_frame_rate','-of','csv=p=0',str(p)],
        stdout=subprocess.PIPE, text=True)
    w,h,fps = r.stdout.strip().split(','); n,d = map(int, fps.split('/'))
    return int(w), int(h), n/d

def psnr_between(a, b):
    r = subprocess.run(['ffmpeg','-i',str(a),'-i',str(b),
         '-filter_complex','psnr','-f','null','-'], capture_output=True, text=True)
    m = re.search(r'average:([\d.]+)', r.stderr)
    return float(m.group(1)) if m else None

def run_vmaf_score(dist, ref_mp4, label):
    dist_yuv = OUT / f"{label}.yuv"
    ref_yuv  = OUT / "ref.yuv"
    vjson    = OUT / f"{label}_vmaf.json"
    W, H = map(int, subprocess.run(['ffprobe','-v','quiet','-select_streams','v:0',
         '-show_entries','stream=width,height','-of','csv=p=0',str(dist)],
        stdout=subprocess.PIPE, text=True).stdout.strip().split(','))
    subprocess.run(['ffmpeg','-y','-i',str(dist),'-c:v','rawvideo','-pix_fmt','yuv420p',str(dist_yuv)],capture_output=True,check=True)
    if not ref_yuv.exists():
        subprocess.run(['ffmpeg','-y','-i',str(ref_mp4),'-c:v','rawvideo','-pix_fmt','yuv420p',str(ref_yuv)],capture_output=True,check=True)
    r = subprocess.run(['vmaf','-r',str(ref_yuv),'-d',str(dist_yuv),
         '-w',str(W),'-h',str(H),'-p','420','-b','8',
         '-m',f'path={VMAF}','--json','-o',str(vjson)], capture_output=True, text=True)
    dist_yuv.unlink(missing_ok=True)
    if r.returncode != 0: return None, None
    scores = [fr['metrics']['vmaf'] for fr in json.load(open(vjson))['frames']]
    return sum(scores)/len(scores), min(scores)

W, H, FPS = probe(TEST); OW, OH = W*2, H*2

print("Loading VSR...")
vsr = nvvfx.VideoSuperRes(quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=0)
vsr.output_width = OW; vsr.output_height = OH; vsr.load()
print("  VSR ready.")

# ── Create bicubic reference if needed ──────────────────────────────
ref_mp4 = OUT / "reference_bicubic_4k.mp4"
if not ref_mp4.exists():
    print("Creating bicubic reference...")
    subprocess.run([
        'ffmpeg','-y','-i',str(TEST),'-vf','scale=3840:2160:flags=lanczos',
        '-c:v','hevc_nvenc','-cq','10','-preset','p7','-pix_fmt','yuv420p',
        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
        '-movflags','+faststart', str(ref_mp4)
    ], capture_output=True, check=True)

def nv12_to_rgb_bt601(nv12, H, W):
    Y = nv12[:H].float() - 16.0
    uv = nv12[H:].view(H//2, W//2, 2).float()
    uv_up = torch.nn.functional.interpolate(
        uv.permute(2,0,1).unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False)
    U = uv_up[0,0] - 128.0; V = uv_up[0,1] - 128.0
    R = (1.164*Y + 1.596*V).clamp_(0, 255)
    G = (1.164*Y - 0.392*U - 0.813*V).clamp_(0, 255)
    B = (1.164*Y + 2.017*U).clamp_(0, 255)
    return torch.stack([R,G,B],0).div_(255.0)

# ── TEST C: pynvc decode → VSR → RGB → ffmpeg encode ────────────────
testc_out = OUT / "testc_nv12dec_ffenc_fullclip.mp4"
print(f"\nTEST C: pynvc NV12 decode (BT.601) → VSR → ffmpeg encode (full 300 frames)...")

dec_nvc = nvc.CreateSimpleDecoder(str(TEST), gpuid=0, useDeviceMemory=True,
          outputColorType=nvc.OutputColorType.NATIVE, decoderCacheSize=1)
meta = dec_nvc.get_stream_metadata()

enc = subprocess.Popen(
    ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}',
     '-r',str(FPS),'-i','pipe:0',
     '-c:v','hevc_nvenc','-cq','20','-preset','p4','-profile:v','main',
     '-pix_fmt','yuv420p',
     '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
     '-movflags','+faststart', str(testc_out)],
    stdin=subprocess.PIPE, stderr=subprocess.PIPE)

t0 = time.time()
# warmup (3 frames, included in output)
warmup = dec_nvc.get_batch_frames(3)
for wf in warmup:
    nv12 = torch.from_dlpack(wf)
    rgb = nv12_to_rgb_bt601(nv12, H, W)
    res = vsr.run(rgb)
    out_rgb = torch.from_dlpack(res.image).clone()
    enc.stdin.write(out_rgb.clamp_(0,1).mul_(255).round_().to(torch.uint8).permute(1,2,0).contiguous().cpu().numpy().tobytes())

total = len(warmup)
while True:
    batch = dec_nvc.get_batch_frames(8)
    if not batch: break
    for frame in batch:
        nv12 = torch.from_dlpack(frame)
        rgb = nv12_to_rgb_bt601(nv12, H, W)
        res = vsr.run(rgb)
        out_rgb = torch.from_dlpack(res.image).clone()
        enc.stdin.write(out_rgb.clamp_(0,1).mul_(255).round_().to(torch.uint8).permute(1,2,0).contiguous().cpu().numpy().tobytes())
        total += 1

enc.stdin.close(); enc.stderr.read(); enc.wait()
elapsed = time.time() - t0
print(f"  Done: {total} frames in {elapsed:.1f}s, {total/elapsed:.1f} fps")
vsr.close()

# Production for comparison
prod_out = OUT / "ref_prod_fullclip.mp4"
if not prod_out.exists():
    print("\nRunning production for PSNR comparison (300 frames)...")
    vsr2 = nvvfx.VideoSuperRes(quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=0)
    vsr2.output_width = OW; vsr2.output_height = OH; vsr2.load()
    fsz = W*H*3
    dec = subprocess.Popen(['ffmpeg','-i',str(TEST),'-f','rawvideo','-pix_fmt','rgb24','-'],
         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    enc2 = subprocess.Popen(
        ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}',
         '-r',str(FPS),'-i','pipe:0',
         '-c:v','hevc_nvenc','-cq','20','-preset','p4','-profile:v','main',
         '-pix_fmt','yuv420p','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
         '-movflags','+faststart', str(prod_out)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    n=0
    while True:
        raw = dec.stdout.read(fsz)
        if len(raw) < fsz: break
        t = torch.frombuffer(bytearray(raw),dtype=torch.uint8).reshape(H,W,3).cuda().float().div_(255.0).permute(2,0,1).contiguous()
        res = vsr2.run(t)
        out_rgb = torch.from_dlpack(res.image).clone()
        enc2.stdin.write(out_rgb.clamp_(0,1).mul_(255).round_().to(torch.uint8).permute(1,2,0).contiguous().cpu().numpy().tobytes())
        n += 1
    enc2.stdin.close(); enc2.stderr.read(); enc2.wait(); dec.wait()
    vsr2.close()
    print(f"  Production done: {n} frames.")

# Compare
print("\nPSNR: test_C vs production...")
p_c = psnr_between(testc_out, prod_out)
print(f"  PSNR test_C vs prod: {p_c:.2f} dB")

print("\nVMAF: test_C vs bicubic ref...")
vmaf_mean, vmaf_min = run_vmaf_score(testc_out, ref_mp4, "testc")
(OUT / "ref.yuv").unlink(missing_ok=True)

print(f"\n{'='*55}")
print(f"TEST C RESULTS (pynvc BT.601 decode + ffmpeg encode)")
print(f"{'='*55}")
print(f"  Time: {elapsed:.1f}s for {total} frames = {total/elapsed:.1f} fps")
print(f"  Projected 300f: {300*elapsed/total:.1f}s")
print(f"  PSNR vs production: {p_c:.2f} dB")
if vmaf_mean:
    print(f"  VMAF mean: {vmaf_mean:.2f}  min: {vmaf_min:.2f}")
    gate = "✓ PASS" if vmaf_mean >= 89 else "✗ FAIL"
    print(f"  Gate (≥89): {gate}")
