"""GPU-Direct Pipeline v2: PyNvVideoCodec decode → nvidia-vfx → PyNvVideoCodec encode
Zero CPU↔GPU copies. Fixed color handling + proper bitstream muxing."""
import time
import struct
import torch
import nvvfx
import PyNvVideoCodec as nvc
from pathlib import Path
import subprocess

TEST_VIDEO = "tmp/7b03d158-3585-499d-8f10-997c73eebeaf.mp4"
OUTPUT_HEVC = "tmp/quality_test/gpu_direct_v2.hevc"
OUTPUT_MP4 = "tmp/quality_test/gpu_direct_v2.mp4"

# =====================================================================
# Setup
# =====================================================================
print("Setting up GPU-direct decoder...")
decoder = nvc.CreateSimpleDecoder(
    TEST_VIDEO,
    gpuid=0,
    useDeviceMemory=True,
    outputColorType=nvc.OutputColorType.RGBP,
    decoderCacheSize=1
)
meta = decoder.get_stream_metadata()
W, H = meta.width, meta.height
FPS = meta.average_fps
print(f"  Input: {W}x{H}, {FPS} fps, {meta.num_frames} frames")

OUT_W, OUT_H = W * 2, H * 2

print("Setting up nvidia-vfx...")
vsr = nvvfx.VideoSuperRes(quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=0)
vsr.output_width = OUT_W
vsr.output_height = OUT_H
vsr.load()
print(f"  Output: {OUT_W}x{OUT_H}")

print("Setting up GPU-direct encoder...")
encoder = nvc.CreateEncoder(
    OUT_W, OUT_H, 'ARGB', False,
    codec='hevc', preset='P4', rc='constqp', constqp=20, profile='main'
)

# Pre-allocate alpha channel
alpha = torch.full((OUT_H, OUT_W, 1), 255, dtype=torch.uint8, device='cuda')

# =====================================================================
# Warmup
# =====================================================================
print("\nWarming up (3 frames)...")
warmup_frames = decoder.get_batch_frames(3)
for wf in warmup_frames:
    frame_gpu = torch.from_dlpack(wf)
    input_t = frame_gpu.to(torch.float32).div_(255.0)
    result = vsr.run(input_t)
    out_gpu = torch.from_dlpack(result.image).clone()
    rgb_hwc = out_gpu.clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(1, 2, 0).contiguous()
    argb = torch.cat([alpha, rgb_hwc], dim=2).contiguous()
    encoder.Encode(argb)
torch.cuda.synchronize()
print("Warmup done.")

# =====================================================================
# Timed run
# =====================================================================
print(f"\nProcessing remaining frames...")
total_frames = len(warmup_frames)
bitstream_chunks = []
BATCH = 8

start = time.time()
while True:
    batch = decoder.get_batch_frames(BATCH)
    if not batch:
        break

    for frame in batch:
        frame_gpu = torch.from_dlpack(frame)  # (3, H, W) uint8 CUDA — zero copy
        input_t = frame_gpu.to(torch.float32).div_(255.0)  # GPU-only conversion
        result = vsr.run(input_t)  # GPU inference
        out_gpu = torch.from_dlpack(result.image).clone()  # GPU output
        rgb_hwc = out_gpu.clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(1, 2, 0).contiguous()
        argb = torch.cat([alpha, rgb_hwc], dim=2).contiguous()
        bits = encoder.Encode(argb)  # GPU-direct encode
        if bits:
            bitstream_chunks.append(bits)
        total_frames += 1

# Flush encoder
eos = encoder.EndEncode()
if eos:
    bitstream_chunks.append(eos)

torch.cuda.synchronize()
elapsed = time.time() - start
timed_frames = total_frames - len(warmup_frames)

# Write raw HEVC
hevc_data = b''.join(bitstream_chunks)
Path(OUTPUT_HEVC).write_bytes(hevc_data)

# Mux to MP4
subprocess.run([
    'ffmpeg', '-y', '-r', str(FPS),
    '-i', OUTPUT_HEVC,
    '-c', 'copy', '-movflags', '+faststart',
    OUTPUT_MP4
], capture_output=True)

mp4_size = Path(OUTPUT_MP4).stat().st_size / 1024 / 1024
fps = timed_frames / elapsed

print(f"\n{'='*60}")
print(f"GPU-DIRECT PIPELINE v2 RESULTS")
print(f"{'='*60}")
print(f"Total frames:    {total_frames} ({len(warmup_frames)} warmup + {timed_frames} timed)")
print(f"Timed portion:   {timed_frames} frames in {elapsed:.3f}s = {fps:.1f} fps")
print(f"Per frame:       {elapsed/timed_frames*1000:.1f} ms")
print(f"Output:          {mp4_size:.1f} MB MP4")
print(f"Projected 300f:  {300/fps:.1f}s")

print(f"\n{'='*60}")
print(f"COMPARISON")
print(f"{'='*60}")
print(f"{'Method':<30} {'FPS':>8} {'300 frames':>12} {'Speedup':>10}")
print(f"{'-'*60}")
print(f"{'SPAN x2 (production)':<30} {'~5':>8} {'~60s':>12} {'1.0x':>10}")
print(f"{'nvvfx + ffmpeg pipes':<30} {'~11':>8} {'~27s':>12} {'2.2x':>10}")
print(f"{'nvvfx + GPU-direct':<30} {f'{fps:.1f}':>8} {f'{300/fps:.1f}s':>12} {f'{60/(300/fps):.1f}x':>10}")

vsr.close()
