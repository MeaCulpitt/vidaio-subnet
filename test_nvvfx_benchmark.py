"""Benchmark: SPAN x2 vs nvidia-vfx VideoSuperRes on 1080p → 2160p upscaling."""
import time
import subprocess
import numpy as np
import torch
import nvvfx
from pathlib import Path

TEST_VIDEO = Path("tmp/7b03d158-3585-499d-8f10-997c73eebeaf.mp4")
W, H = 1920, 1080
FRAME_SIZE = W * H * 3

def decode_frames(video_path, max_frames=100):
    """Decode raw RGB frames from video."""
    dec = subprocess.Popen(
        ['ffmpeg', '-i', str(video_path), '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-vframes', str(max_frames), '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    frames = []
    while True:
        raw = dec.stdout.read(FRAME_SIZE)
        if len(raw) < FRAME_SIZE:
            break
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3).copy())
    dec.wait()
    return frames

print(f"Decoding 100 frames from {TEST_VIDEO} ({W}x{H})...")
frames = decode_frames(TEST_VIDEO, max_frames=100)
print(f"Decoded {len(frames)} frames\n")

# --- nvidia-vfx VideoSuperRes benchmark ---
print("=" * 60)
print("NVIDIA VFX VideoSuperRes (RTX VSR)")
print("=" * 60)

vsr = nvvfx.VideoSuperRes(quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=0)
# Set output to 4K (2x)
vsr.output_width = W * 2
vsr.output_height = H * 2
vsr.load()
print(f"Output resolution: {vsr.output_width}x{vsr.output_height}")
print(f"Quality: {vsr.quality}")

# Warmup (3 frames)
for i in range(min(3, len(frames))):
    t = torch.from_numpy(frames[i].astype(np.float32) / 255.0).permute(2, 0, 1).contiguous().cuda()
    result = vsr.run(t)
    out = torch.from_dlpack(result.image).clone()

torch.cuda.synchronize()
print(f"Warmup done. Output shape: {out.shape}")

# Timed run
start = time.time()
for i, frame in enumerate(frames):
    t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous().cuda()
    result = vsr.run(t)
    out = torch.from_dlpack(result.image).clone()
torch.cuda.synchronize()
nvvfx_elapsed = time.time() - start
nvvfx_fps = len(frames) / nvvfx_elapsed
print(f"nvidia-vfx: {len(frames)} frames in {nvvfx_elapsed:.2f}s = {nvvfx_fps:.1f} fps")
print(f"Output shape: {out.shape}, dtype: {out.dtype}, range: [{out.min():.3f}, {out.max():.3f}]")

vsr.close()

# --- SPAN x2 benchmark ---
print()
print("=" * 60)
print("SPAN x2 (current pipeline)")
print("=" * 60)

from spandrel import ModelLoader
model_path = Path.home() / ".cache" / "span" / "2xHFA2kSPAN.safetensors"
descriptor = ModelLoader(device="cuda:0").load_from_file(str(model_path))
model = descriptor.model.eval().half()

# Warmup
for i in range(min(3, len(frames))):
    t = torch.from_numpy(frames[i].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda().half()
    with torch.no_grad():
        out = model(t)

torch.cuda.synchronize()
print(f"Warmup done. Output shape: {out.shape}")

# Timed run (batch_size=4 to match production)
BATCH = 4
start = time.time()
for i in range(0, len(frames), BATCH):
    batch_frames = frames[i:i+BATCH]
    tensors = [torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1) for f in batch_frames]
    batch = torch.stack(tensors).cuda().half()
    with torch.no_grad():
        out = model(batch)
torch.cuda.synchronize()
span_elapsed = time.time() - start
span_fps = len(frames) / span_elapsed
print(f"SPAN x2:    {len(frames)} frames in {span_elapsed:.2f}s = {span_fps:.1f} fps")
print(f"Output shape: {out.shape}, dtype: {out.dtype}")

# Summary
print()
print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"nvidia-vfx:  {nvvfx_fps:.1f} fps  ({nvvfx_elapsed:.2f}s for {len(frames)} frames)")
print(f"SPAN x2:     {span_fps:.1f} fps  ({span_elapsed:.2f}s for {len(frames)} frames)")
print(f"Speedup:     {nvvfx_fps/span_fps:.2f}x" if span_fps > 0 else "N/A")
