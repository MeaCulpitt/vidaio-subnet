"""Quality comparison: SPAN x2 vs nvidia-vfx vs bicubic reference.

Produces:
  tmp/quality_test/bicubic_4k.mp4    — ffmpeg bicubic upscale (reference)
  tmp/quality_test/span_x2_4k.mp4    — SPAN x2 upscale
  tmp/quality_test/nvvfx_4k_4k.mp4   — nvidia-vfx upscale
  tmp/quality_test/frame_span.png     — sample frame from SPAN
  tmp/quality_test/frame_nvvfx.png    — sample frame from nvidia-vfx
  tmp/quality_test/frame_bicubic.png  — sample frame from bicubic (reference)

Then runs VMAF: span vs bicubic, nvvfx vs bicubic.
"""
import os
import time
import subprocess
import numpy as np
import torch
import nvvfx
from pathlib import Path
from spandrel import ModelLoader

TEST_VIDEO = Path("tmp/7b03d158-3585-499d-8f10-997c73eebeaf.mp4")
OUT_DIR = Path("tmp/quality_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 1920, 1080
OUT_W, OUT_H = 3840, 2160
FRAME_SIZE = W * H * 3
SAMPLE_FRAME_IDX = 50  # frame to save as PNG

# -------------------------------------------------------------------------
# Step 1: Bicubic 4K reference
# -------------------------------------------------------------------------
print("=" * 60)
print("Step 1: Generating bicubic 4K reference")
print("=" * 60)
bicubic_path = OUT_DIR / "bicubic_4k.mp4"
subprocess.run([
    'ffmpeg', '-y', '-i', str(TEST_VIDEO),
    '-vf', f'scale={OUT_W}:{OUT_H}:flags=bicubic',
    '-c:v', 'hevc_nvenc', '-cq', '10', '-preset', 'p7',
    '-pix_fmt', 'yuv420p', str(bicubic_path)
], check=True, capture_output=True)
print(f"  -> {bicubic_path}")

# -------------------------------------------------------------------------
# Decode all frames once
# -------------------------------------------------------------------------
print("\nDecoding source frames...")
dec = subprocess.Popen(
    ['ffmpeg', '-i', str(TEST_VIDEO), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
)
frames = []
while True:
    raw = dec.stdout.read(FRAME_SIZE)
    if len(raw) < FRAME_SIZE:
        break
    frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(H, W, 3).copy())
dec.wait()
print(f"  Decoded {len(frames)} frames @ {W}x{H}")

# Get frame rate
fr_out = subprocess.run(
    ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
     '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', str(TEST_VIDEO)],
    capture_output=True, text=True
)
num, den = map(int, fr_out.stdout.strip().split('/'))
FPS = num / den
print(f"  Frame rate: {FPS} fps")

# -------------------------------------------------------------------------
# Step 2: SPAN x2 upscale
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 2: SPAN x2 upscale")
print("=" * 60)

span_path = OUT_DIR / "span_x2_4k.mp4"
model_path = Path.home() / ".cache" / "span" / "2xHFA2kSPAN.safetensors"
descriptor = ModelLoader(device="cuda:0").load_from_file(str(model_path))
span_model = descriptor.model.eval().half()

# Warmup
t = torch.from_numpy(frames[0].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda().half()
with torch.no_grad():
    span_model(t)
torch.cuda.synchronize()

encoder = subprocess.Popen(
    ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
     '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', 'pipe:0',
     '-c:v', 'hevc_nvenc', '-cq', '10', '-preset', 'p7',
     '-pix_fmt', 'yuv420p', str(span_path)],
    stdin=subprocess.PIPE, stderr=subprocess.PIPE
)

span_sample_frame = None
BATCH = 4
start = time.time()
for i in range(0, len(frames), BATCH):
    batch_frames = frames[i:i + BATCH]
    tensors = [torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1) for f in batch_frames]
    batch = torch.stack(tensors).cuda().half()
    with torch.no_grad():
        out = span_model(batch)
    out_np = out.float().clamp_(0, 1).mul_(255.0).round_().to(torch.uint8).permute(0, 2, 3, 1).contiguous().cpu().numpy()
    for j in range(len(batch_frames)):
        frame_idx = i + j
        encoder.stdin.write(out_np[j].tobytes())
        if frame_idx == SAMPLE_FRAME_IDX:
            span_sample_frame = out_np[j]

encoder.stdin.close()
encoder.wait()
torch.cuda.synchronize()
span_elapsed = time.time() - start
print(f"  SPAN x2: {len(frames)} frames in {span_elapsed:.2f}s = {len(frames)/span_elapsed:.1f} fps")
print(f"  -> {span_path}")

# Save SPAN sample frame
if span_sample_frame is not None:
    from PIL import Image
    Image.fromarray(span_sample_frame).save(OUT_DIR / "frame_span.png")
    print(f"  -> saved frame {SAMPLE_FRAME_IDX} as frame_span.png")

# -------------------------------------------------------------------------
# Step 3: nvidia-vfx upscale
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 3: nvidia-vfx VideoSuperRes upscale")
print("=" * 60)

nvvfx_path = OUT_DIR / "nvvfx_4k.mp4"
# Free SPAN model to reclaim GPU memory
del span_model
torch.cuda.empty_cache()

vsr = nvvfx.VideoSuperRes(quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=0)
vsr.output_width = OUT_W
vsr.output_height = OUT_H
vsr.load()

# Warmup
t = torch.from_numpy(frames[0].astype(np.float32) / 255.0).permute(2, 0, 1).contiguous().cuda()
result = vsr.run(t)
torch.from_dlpack(result.image).clone()
torch.cuda.synchronize()

encoder2 = subprocess.Popen(
    ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
     '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', 'pipe:0',
     '-c:v', 'hevc_nvenc', '-cq', '10', '-preset', 'p7',
     '-pix_fmt', 'yuv420p', str(nvvfx_path)],
    stdin=subprocess.PIPE, stderr=subprocess.PIPE
)

nvvfx_sample_frame = None
start = time.time()
for i, frame in enumerate(frames):
    t = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous().cuda()
    result = vsr.run(t)
    out = torch.from_dlpack(result.image).clone()
    out_np = out.float().clamp_(0, 1).mul_(255.0).round_().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy()
    raw_bytes = out_np.tobytes()
    assert len(raw_bytes) == OUT_W * OUT_H * 3, f"Bad frame size: {len(raw_bytes)} vs {OUT_W*OUT_H*3}"
    encoder2.stdin.write(raw_bytes)
    if i == SAMPLE_FRAME_IDX:
        nvvfx_sample_frame = out_np

encoder2.stdin.close()
encoder2.wait()
torch.cuda.synchronize()
vsr.close()
nvvfx_elapsed = time.time() - start
print(f"  nvidia-vfx: {len(frames)} frames in {nvvfx_elapsed:.2f}s = {len(frames)/nvvfx_elapsed:.1f} fps")
print(f"  -> {nvvfx_path}")

if nvvfx_sample_frame is not None:
    from PIL import Image
    Image.fromarray(nvvfx_sample_frame).save(OUT_DIR / "frame_nvvfx.png")
    print(f"  -> saved frame {SAMPLE_FRAME_IDX} as frame_nvvfx.png")

# Save bicubic sample frame too
print("\nExtracting bicubic reference frame...")
subprocess.run([
    'ffmpeg', '-y', '-i', str(bicubic_path),
    '-vf', f'select=eq(n\\,{SAMPLE_FRAME_IDX})',
    '-frames:v', '1', str(OUT_DIR / "frame_bicubic.png")
], capture_output=True)
print(f"  -> frame_bicubic.png")

# -------------------------------------------------------------------------
# Step 4: VMAF scoring
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Step 4: VMAF scoring (vs bicubic reference)")
print("=" * 60)

def run_vmaf(distorted, reference, label):
    """Run VMAF and return the mean score."""
    cmd = [
        'ffmpeg', '-i', str(distorted), '-i', str(reference),
        '-lavfi', f'libvmaf=model=version=vmaf_v0.6.1:log_fmt=json:log_path=/tmp/vmaf_{label}.json',
        '-f', 'null', '-'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    # Parse VMAF score from json
    import json
    with open(f'/tmp/vmaf_{label}.json') as f:
        data = json.load(f)
    mean_vmaf = data['pooled_metrics']['vmaf']['mean']
    min_vmaf = data['pooled_metrics']['vmaf']['min']
    return mean_vmaf, min_vmaf

print("\nRunning VMAF: SPAN x2 vs bicubic reference...")
span_vmaf_mean, span_vmaf_min = run_vmaf(span_path, bicubic_path, "span")
print(f"  SPAN x2 VMAF:    mean={span_vmaf_mean:.2f}  min={span_vmaf_min:.2f}")

print("\nRunning VMAF: nvidia-vfx vs bicubic reference...")
nvvfx_vmaf_mean, nvvfx_vmaf_min = run_vmaf(nvvfx_path, bicubic_path, "nvvfx")
print(f"  nvidia-vfx VMAF: mean={nvvfx_vmaf_mean:.2f}  min={nvvfx_vmaf_min:.2f}")

# -------------------------------------------------------------------------
# Summary
# -------------------------------------------------------------------------
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"{'Method':<20} {'FPS':>8} {'Time':>8} {'VMAF mean':>10} {'VMAF min':>10}")
print("-" * 60)
print(f"{'SPAN x2':<20} {len(frames)/span_elapsed:>7.1f}  {span_elapsed:>7.2f}s {span_vmaf_mean:>10.2f} {span_vmaf_min:>10.2f}")
print(f"{'nvidia-vfx':<20} {len(frames)/nvvfx_elapsed:>7.1f}  {nvvfx_elapsed:>7.2f}s {nvvfx_vmaf_mean:>10.2f} {nvvfx_vmaf_min:>10.2f}")
print(f"\nSpeedup: {(len(frames)/nvvfx_elapsed)/(len(frames)/span_elapsed):.2f}x")
print(f"\nValidator threshold: VMAF >= 89")
print(f"nvidia-vfx passes:  {'YES' if nvvfx_vmaf_mean >= 89 else 'NO'} (mean={nvvfx_vmaf_mean:.2f})")
print(f"\nSample frames saved to {OUT_DIR}/")
print(f"  frame_span.png, frame_nvvfx.png, frame_bicubic.png")
