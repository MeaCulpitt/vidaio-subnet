#!/usr/bin/env python3
"""Isolate where the 23 dB PSNR gap comes from:
Test A: ffmpeg rgb24 decode → VSR → nvc encode (is nvc encode wrong?)
Test B: pynvc NV12 decode → BT.601 → VSR → ffmpeg encode (is decode wrong?)
Test C: pynvc NV12 decode → BT.601 → VSR → RGB compare vs production (pre-encode diff?)
"""
import sys, os, time, subprocess
sys.path.insert(0, '/root/vidaio-subnet')
os.chdir('/root/vidaio-subnet')

import torch
import nvvfx
import PyNvVideoCodec as nvc
from pathlib import Path
import re

TEST = Path("/tmp/test_1080p_10s.mp4")

def probe(path):
    r = subprocess.run(
        ['ffprobe','-v','quiet','-select_streams','v:0',
         '-show_entries','stream=width,height,r_frame_rate','-of','csv=p=0',str(path)],
        stdout=subprocess.PIPE, text=True)
    w,h,fps = r.stdout.strip().split(',')
    n,d = map(int, fps.split('/'))
    return int(w), int(h), n/d

def psnr_between(a, b):
    r = subprocess.run(
        ['ffmpeg','-i',str(a),'-i',str(b),'-filter_complex','psnr','-f','null','-'],
        capture_output=True, text=True)
    m = re.search(r'average:([\d.]+)', r.stderr)
    return float(m.group(1)) if m else None

print("="*60)
print("ISOLATION TEST — finding 23 dB PSNR gap root cause")
print("="*60)

W, H, FPS = probe(TEST)
OW, OH = W*2, H*2
fsz = W*H*3

print(f"\nLoading VSR ({W}x{H} → {OW}x{OH})...")
vsr = nvvfx.VideoSuperRes(quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=0)
vsr.output_width  = OW
vsr.output_height = OH
vsr.load()
print("  VSR ready.")

OUT = Path("/tmp/isolate")
OUT.mkdir(exist_ok=True)
N = 30  # test with 30 frames for speed

# ─────────────────────────────────────────────────────────────────────
# REFERENCE: ffmpeg decode → VSR → ffmpeg encode  (= production)
# ─────────────────────────────────────────────────────────────────────
print(f"\n[REF] ffmpeg rgb24 decode → VSR → ffmpeg encode ({N} frames)...")
prod_out = OUT / "ref_prod.mp4"
dec = subprocess.Popen(
    ['ffmpeg','-i',str(TEST),'-vframes',str(N),'-f','rawvideo','-pix_fmt','rgb24','-'],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
enc = subprocess.Popen(
    ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}',
     '-r',str(FPS),'-i','pipe:0',
     '-c:v','hevc_nvenc','-cq','20','-preset','p4','-profile:v','main',
     '-pix_fmt','yuv420p','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
     '-movflags','+faststart', str(prod_out)],
    stdin=subprocess.PIPE, stderr=subprocess.PIPE)
n = 0
while n < N:
    raw = dec.stdout.read(fsz)
    if len(raw) < fsz: break
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(H,W,3).cuda().float().div_(255.0).permute(2,0,1).contiguous()
    res = vsr.run(t)
    out_rgb = torch.from_dlpack(res.image).clone()
    enc.stdin.write(out_rgb.clamp_(0,1).mul_(255).round_().to(torch.uint8).permute(1,2,0).contiguous().cpu().numpy().tobytes())
    n += 1
enc.stdin.close(); enc.stderr.read(); enc.wait(); dec.wait()
print(f"  Done. {n} frames.")

# ─────────────────────────────────────────────────────────────────────
# TEST A: ffmpeg rgb24 decode → VSR → nvc encode
# (replaces production ffmpeg encoder with nvc, same decode/VSR)
# ─────────────────────────────────────────────────────────────────────
print(f"\n[A] ffmpeg rgb24 decode → VSR → nvc encode (NV12 BT.709)...")

def rgb_to_nv12_bt709(rgb, H, W):
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

testa_out = OUT / "testa_nvenc_nv12.mp4"
enc_nvc = nvc.CreateEncoder(OW, OH, 'NV12', False, codec='hevc', preset='P4', rc='constqp', constqp=20, profile='main')

dec = subprocess.Popen(
    ['ffmpeg','-i',str(TEST),'-vframes',str(N),'-f','rawvideo','-pix_fmt','rgb24','-'],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
chunks = []
n = 0
while n < N:
    raw = dec.stdout.read(fsz)
    if len(raw) < fsz: break
    t = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(H,W,3).cuda().float().div_(255.0).permute(2,0,1).contiguous()
    res = vsr.run(t)
    out_rgb = torch.from_dlpack(res.image).clone()
    bits = enc_nvc.Encode(rgb_to_nv12_bt709(out_rgb, OH, OW))
    if bits: chunks.append(bits)
    n += 1
dec.wait()
eos = enc_nvc.EndEncode()
if eos: chunks.append(eos)
hevc = OUT / "testa.hevc"
hevc.write_bytes(b''.join(chunks))
subprocess.run(['ffmpeg','-y','-r',str(FPS),'-i',str(hevc),'-c','copy',
                '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
                '-movflags','+faststart',str(testa_out)], capture_output=True)
hevc.unlink()
p_a = psnr_between(testa_out, prod_out)
print(f"  PSNR test_A vs ref: {p_a:.2f} dB  {'← if ~23 dB: nvc encoder is the problem' if p_a and p_a < 30 else ''}")

# ─────────────────────────────────────────────────────────────────────
# TEST B: pynvc decode → BT.601 → VSR → ffmpeg encode
# (replaces production pynvc decode, same ffmpeg encoder)
# ─────────────────────────────────────────────────────────────────────
print(f"\n[B] pynvc NV12 decode (BT.601) → VSR → ffmpeg encode ({N} frames)...")

def nv12_to_rgb_bt601(nv12, H, W):
    Y = nv12[:H].float() - 16.0
    uv = nv12[H:].view(H//2, W//2, 2).float()
    uv_up = torch.nn.functional.interpolate(
        uv.permute(2,0,1).unsqueeze(0), scale_factor=2, mode='bilinear', align_corners=False)
    U = uv_up[0,0] - 128.0
    V = uv_up[0,1] - 128.0
    R = (1.164*Y + 1.596*V).clamp_(0, 255)
    G = (1.164*Y - 0.392*U - 0.813*V).clamp_(0, 255)
    B = (1.164*Y + 2.017*U).clamp_(0, 255)
    return torch.stack([R,G,B],0).div_(255.0)

testb_out = OUT / "testb_pynvc_dec_ffenc.mp4"
dec_nvc = nvc.CreateSimpleDecoder(str(TEST), gpuid=0, useDeviceMemory=True,
          outputColorType=nvc.OutputColorType.NATIVE, decoderCacheSize=1)
meta = dec_nvc.get_stream_metadata()

enc = subprocess.Popen(
    ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{OW}x{OH}',
     '-r',str(FPS),'-i','pipe:0',
     '-c:v','hevc_nvenc','-cq','20','-preset','p4','-profile:v','main',
     '-pix_fmt','yuv420p','-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709',
     '-vframes',str(N+3),
     '-movflags','+faststart', str(testb_out)],
    stdin=subprocess.PIPE, stderr=subprocess.PIPE)

# warmup
warmup = dec_nvc.get_batch_frames(3)
for wf in warmup:
    nv12 = torch.from_dlpack(wf)
    rgb = nv12_to_rgb_bt601(nv12, H, W)
    res = vsr.run(rgb)
    out_rgb = torch.from_dlpack(res.image).clone()
    enc.stdin.write(out_rgb.clamp_(0,1).mul_(255).round_().to(torch.uint8).permute(1,2,0).contiguous().cpu().numpy().tobytes())

n = 0
while n < N:
    batch = dec_nvc.get_batch_frames(8)
    if not batch: break
    for frame in batch:
        if n >= N: break
        nv12 = torch.from_dlpack(frame)
        rgb = nv12_to_rgb_bt601(nv12, H, W)
        res = vsr.run(rgb)
        out_rgb = torch.from_dlpack(res.image).clone()
        enc.stdin.write(out_rgb.clamp_(0,1).mul_(255).round_().to(torch.uint8).permute(1,2,0).contiguous().cpu().numpy().tobytes())
        n += 1
enc.stdin.close(); enc.stderr.read(); enc.wait()
p_b = psnr_between(testb_out, prod_out)
print(f"  PSNR test_B vs ref: {p_b:.2f} dB  {'← if ~23 dB: pynvc decode is the problem' if p_b and p_b < 30 else ''}")

vsr.close()
print(f"\nSUMMARY:")
print(f"  [A] ffmpeg-dec + nvc-enc:     PSNR={p_a:.2f} dB vs production")
print(f"  [B] pynvc-dec + ffmpeg-enc:   PSNR={p_b:.2f} dB vs production")
if p_a and p_b:
    if p_a < 30 and p_b > 35:
        print("  → ROOT CAUSE: nvc encoder (rgb_to_nv12_bt709 conversion)")
    elif p_b < 30 and p_a > 35:
        print("  → ROOT CAUSE: pynvc decoder (nv12_to_rgb_bt601 conversion)")
    elif p_a < 30 and p_b < 30:
        print("  → ROOT CAUSE: BOTH encoder AND decoder have issues")
    else:
        print("  → Both OK — issue may be elsewhere")
