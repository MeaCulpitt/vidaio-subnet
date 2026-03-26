#!/usr/bin/env python3
"""Find root cause of G channel error."""
import sys, os
sys.path.insert(0, '/root/vidaio-subnet')
os.chdir('/root/vidaio-subnet')

import torch, subprocess, numpy as np
import PyNvVideoCodec as nvc
from pathlib import Path

TEST = "/tmp/test_1080p_10s.mp4"

def yuv_to_rgb_limited_bt709(Y, U_c, V_c):
    """BT.709 limited range. Y, U_c, V_c are already centered (U-128, V-128)."""
    R = (1.164*Y + 1.793*V_c).clamp(0, 255)
    G = (1.164*Y - 0.213*U_c - 0.533*V_c).clamp(0, 255)
    B = (1.164*Y + 2.112*U_c).clamp(0, 255)
    return R, G, B

dec = nvc.CreateSimpleDecoder(TEST, gpuid=0, useDeviceMemory=True,
      outputColorType=nvc.OutputColorType.NATIVE, decoderCacheSize=1)
meta = dec.get_stream_metadata()
W, H = meta.width, meta.height

batch = dec.get_batch_frames(1)
nv12 = torch.from_dlpack(batch[0])  # (H*1.5, W) uint8 cuda

Y_plane  = nv12[:H].cpu().numpy()   # (H, W)
UV_plane = nv12[H:].cpu().numpy()   # (H//2, W)

# Print a few raw NV12 values
print("=== RAW NV12 VALUES (first 8 pixels of row 0) ===")
print(f"Y  row0[0:8]: {Y_plane[0, 0:8]}")
print(f"UV row0[0:16]: {UV_plane[0, 0:16]}")  # alternating U,V,U,V...
print(f"U[0,0:8] = {UV_plane[0,0:16:2]}")     # even indices = U
print(f"V[0,0:8] = {UV_plane[0,1:16:2]}")     # odd indices  = V

# Compute our RGB for pixel (0, 0)
y0 = float(Y_plane[0, 0])
u0 = float(UV_plane[0, 0])  # U at chroma position (0,0)
v0 = float(UV_plane[0, 1])  # V at chroma position (0,0)
print(f"\nPixel (0,0): Y={y0}, U={u0}, V={v0}")
Y_c = y0 - 16.0
U_c = u0 - 128.0
V_c = v0 - 128.0
R_ours = 1.164*Y_c + 1.793*V_c
G_ours = 1.164*Y_c - 0.213*U_c - 0.533*V_c
B_ours = 1.164*Y_c + 2.112*U_c
R_ours = max(0, min(255, R_ours))
G_ours = max(0, min(255, G_ours))
B_ours = max(0, min(255, B_ours))
print(f"Our RGB: R={R_ours:.1f}, G={G_ours:.1f}, B={B_ours:.1f}")

# Get ffmpeg RGB for pixel (0, 0)
proc = subprocess.Popen(
    ['ffmpeg','-i',TEST,'-vframes','1','-f','rawvideo','-pix_fmt','rgb24','-'],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
raw = proc.stdout.read(W*H*3)
proc.wait()
rgb_ffmpeg = np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3)
r_ff, g_ff, b_ff = rgb_ffmpeg[0,0,0], rgb_ffmpeg[0,0,1], rgb_ffmpeg[0,0,2]
print(f"ffmpeg RGB: R={r_ff}, G={g_ff}, B={b_ff}")

# Also try nearest-neighbor (repeat_interleave)
Y_t = nv12[:H].float() - 16.0
uv_p = nv12[H:].view(H//2, W//2, 2).float()
U_nn = uv_p[:,:,0].repeat_interleave(2,0).repeat_interleave(2,1) - 128.0
V_nn = uv_p[:,:,1].repeat_interleave(2,0).repeat_interleave(2,1) - 128.0
R_nn = (1.164*Y_t + 1.793*V_nn).clamp_(0,255)
G_nn = (1.164*Y_t - 0.213*U_nn - 0.533*V_nn).clamp_(0,255)
B_nn = (1.164*Y_t + 2.112*U_nn).clamp_(0,255)
rgb_nn = torch.stack([R_nn,G_nn,B_nn],0).div_(255.0)

rgb_ff_t = torch.from_numpy(rgb_ffmpeg.astype(np.float32)/255.0).permute(2,0,1)

diff_nn = (rgb_nn.cpu() - rgb_ff_t).abs()
print(f"\n=== NEAREST-NEIGHBOR vs ffmpeg ===")
for i, ch in enumerate(['R','G','B']):
    print(f"  {ch}: mean={diff_nn[i].mean().item()*255:.2f}px, max={diff_nn[i].max().item()*255:.2f}px")

# Bilinear
uv_nchw = uv_p.permute(2,0,1).unsqueeze(0)
uv_up = torch.nn.functional.interpolate(uv_nchw, scale_factor=2, mode='bilinear', align_corners=False)
U_bl = uv_up[0,0] - 128.0
V_bl = uv_up[0,1] - 128.0
R_bl = (1.164*Y_t + 1.793*V_bl).clamp_(0,255)
G_bl = (1.164*Y_t - 0.213*U_bl - 0.533*V_bl).clamp_(0,255)
B_bl = (1.164*Y_t + 2.112*U_bl).clamp_(0,255)
rgb_bl = torch.stack([R_bl,G_bl,B_bl],0).div_(255.0)

diff_bl = (rgb_bl.cpu() - rgb_ff_t).abs()
print(f"\n=== BILINEAR vs ffmpeg ===")
for i, ch in enumerate(['R','G','B']):
    print(f"  {ch}: mean={diff_bl[i].mean().item()*255:.2f}px, max={diff_bl[i].max().item()*255:.2f}px")

# Try swapping U and V
R_sw = (1.164*Y_t + 1.793*U_nn).clamp_(0,255)
G_sw = (1.164*Y_t - 0.213*V_nn - 0.533*U_nn).clamp_(0,255)
B_sw = (1.164*Y_t + 2.112*V_nn).clamp_(0,255)
rgb_sw = torch.stack([R_sw,G_sw,B_sw],0).div_(255.0)
diff_sw = (rgb_sw.cpu() - rgb_ff_t).abs()
print(f"\n=== SWAPPED U↔V vs ffmpeg ===")
for i, ch in enumerate(['R','G','B']):
    print(f"  {ch}: mean={diff_sw[i].mean().item()*255:.2f}px, max={diff_sw[i].max().item()*255:.2f}px")

# Try ffmpeg BGR interpretation
rgb_bgr_t = torch.from_numpy(rgb_ffmpeg[:,:,::-1].copy().astype(np.float32)/255.0).permute(2,0,1)
diff_bgr = (rgb_nn.cpu() - rgb_bgr_t).abs()
print(f"\n=== NN vs ffmpeg (treating ffmpeg as BGR) ===")
for i, ch in enumerate(['R','G','B']):
    print(f"  {ch}: mean={diff_bgr[i].mean().item()*255:.2f}px, max={diff_bgr[i].max().item()*255:.2f}px")

