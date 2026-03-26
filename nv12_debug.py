#!/usr/bin/env python3
"""Debug NV12 pipeline: find where the PSNR gap comes from."""
import sys, os
sys.path.insert(0, '/root/vidaio-subnet')
os.chdir('/root/vidaio-subnet')

import torch, subprocess, numpy as np
import PyNvVideoCodec as nvc
from pathlib import Path

TEST = "/tmp/test_1080p_10s.mp4"

# ── Bilinear NV12→RGB (our conversion) ──────────────────────────────
def nv12_to_rgb_bilinear(nv12, H, W):
    Y = nv12[:H].float() - 16.0
    uv = nv12[H:].view(H//2, W//2, 2).float()
    uv_up = torch.nn.functional.interpolate(
        uv.permute(2,0,1).unsqueeze(0), scale_factor=2,
        mode='bilinear', align_corners=False)
    U = uv_up[0,0] - 128.0
    V = uv_up[0,1] - 128.0
    R = (1.164*Y + 1.793*V          ).clamp_(0, 255)
    G = (1.164*Y - 0.213*U - 0.533*V).clamp_(0, 255)
    B = (1.164*Y + 2.112*U          ).clamp_(0, 255)
    return torch.stack([R,G,B],0).div_(255.0)   # (3,H,W) [0,1]

# ── Get frame 0 via pynvc ────────────────────────────────────────────
dec = nvc.CreateSimpleDecoder(TEST, gpuid=0, useDeviceMemory=True,
      outputColorType=nvc.OutputColorType.NATIVE, decoderCacheSize=1)
meta = dec.get_stream_metadata()
W, H = meta.width, meta.height
print(f"pynvc: {W}x{H}, {meta.num_frames} frames, {meta.average_fps:.2f} fps")

batch = dec.get_batch_frames(1)
nv12_gpu = torch.from_dlpack(batch[0])
print(f"NV12 tensor shape: {nv12_gpu.shape}, dtype: {nv12_gpu.dtype}, device: {nv12_gpu.device}")

# Stats on raw NV12
Y_plane = nv12_gpu[:H]
UV_plane = nv12_gpu[H:]
print(f"Y  plane stats: min={Y_plane.min().item()}, max={Y_plane.max().item()}, mean={Y_plane.float().mean().item():.1f}")
print(f"UV plane stats: min={UV_plane.min().item()}, max={UV_plane.max().item()}, mean={UV_plane.float().mean().item():.1f}")

# Our RGB conversion
rgb_ours = nv12_to_rgb_bilinear(nv12_gpu, H, W)
print(f"\nOur RGB: min={rgb_ours.min().item():.4f}, max={rgb_ours.max().item():.4f}, mean={rgb_ours.float().mean().item():.4f}")

# ── Get same frame via ffmpeg rgb24 ──────────────────────────────────
proc = subprocess.Popen(
    ['ffmpeg', '-i', TEST, '-vframes', '1', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
raw = proc.stdout.read(W * H * 3)
proc.wait()

rgb_ffmpeg = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(H, W, 3).cuda().float().div_(255.0).permute(2,0,1)
print(f"ffmpeg RGB: min={rgb_ffmpeg.min().item():.4f}, max={rgb_ffmpeg.max().item():.4f}, mean={rgb_ffmpeg.float().mean().item():.4f}")

# ── Compare ────────────────────────────────────────────────────────
diff = (rgb_ours - rgb_ffmpeg).abs()
print(f"\nAbs diff: mean={diff.mean().item()*255:.2f} px, max={diff.max().item()*255:.2f} px")
mse = ((rgb_ours - rgb_ffmpeg)**2).mean().item()
psnr = -10*np.log10(mse) + 20*np.log10(1.0) if mse > 0 else float('inf')
print(f"PSNR (ours vs ffmpeg RGB): {psnr:.2f} dB")

# Per-channel errors
for i, ch in enumerate(['R','G','B']):
    d = (rgb_ours[i] - rgb_ffmpeg[i]).abs()
    print(f"  {ch}: mean={d.mean().item()*255:.2f} px, max={d.max().item()*255:.2f} px")
