"""GPU-Direct Pipeline: PyNvVideoCodec decode → nvidia-vfx → PyNvVideoCodec encode
Zero CPU↔GPU copies for the video data path."""
import time
import torch
import nvvfx
import PyNvVideoCodec as nvc
from pathlib import Path

TEST_VIDEO = "tmp/7b03d158-3585-499d-8f10-997c73eebeaf.mp4"
OUTPUT_GPU = "tmp/quality_test/nvvfx_gpu_direct.hevc"
OUTPUT_MP4 = "tmp/quality_test/nvvfx_gpu_direct.mp4"
W, H = 1920, 1080
OUT_W, OUT_H = 3840, 2160

# =====================================================================
# Setup decoder (GPU-direct, RGBP output → planar RGB on CUDA)
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
print(f"  Input: {meta.width}x{meta.height}, {meta.average_fps} fps, {meta.num_frames} frames")

# =====================================================================
# Setup nvidia-vfx (GPU-resident)
# =====================================================================
print("Setting up nvidia-vfx VideoSuperRes...")
vsr = nvvfx.VideoSuperRes(quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=0)
vsr.output_width = OUT_W
vsr.output_height = OUT_H
vsr.load()
print(f"  Output: {OUT_W}x{OUT_H}")

# =====================================================================
# Setup encoder (GPU-direct, ARGB input from CUDA)
# =====================================================================
print("Setting up GPU-direct encoder...")
encoder = nvc.CreateEncoder(
    OUT_W, OUT_H,
    'ARGB',
    False,  # GPU memory input
    codec='hevc',
    preset='P4',
    rc='constqp',
    constqp=20,
    profile='main'
)

# Pre-allocate alpha channel on GPU
alpha_row = torch.full((OUT_H, OUT_W, 1), 255, dtype=torch.uint8, device='cuda')

# =====================================================================
# Warmup: decode + infer + encode 3 frames
# =====================================================================
print("\nWarming up (3 frames)...")
warmup_frames = decoder.get_batch_frames(3)
for wf in warmup_frames:
    # Decode: DecodedFrame → torch CUDA tensor via DLPack (zero-copy)
    frame_gpu = torch.from_dlpack(wf)  # (3, H, W) uint8 on CUDA

    # Convert uint8 RGBP → float32 CHW [0,1] for nvidia-vfx
    input_tensor = frame_gpu.to(torch.float32).div_(255.0)

    # Inference (GPU-resident)
    result = vsr.run(input_tensor)
    out_gpu = torch.from_dlpack(result.image).clone()  # (3, OUT_H, OUT_W) float32

    # Convert float32 CHW → uint8 HWC ARGB for encoder
    rgb_hwc = out_gpu.clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(1, 2, 0).contiguous()
    argb = torch.cat([alpha_row, rgb_hwc], dim=2).contiguous()

    # Encode (GPU-direct)
    _ = encoder.Encode(argb)

torch.cuda.synchronize()
print("Warmup done.")

# =====================================================================
# Timed run: decode all remaining frames
# =====================================================================
print(f"\nProcessing remaining frames...")
total_frames = len(warmup_frames)
encoded_bytes = bytearray()
BATCH = 8

start = time.time()
while True:
    batch = decoder.get_batch_frames(BATCH)
    if not batch:
        break

    for frame in batch:
        # DECODE: zero-copy DLPack → CUDA tensor
        frame_gpu = torch.from_dlpack(frame)  # (3, H, W) uint8 CUDA

        # PREPROCESS: uint8 → float32 on GPU (no CPU touch)
        input_tensor = frame_gpu.to(torch.float32).div_(255.0)

        # INFERENCE: nvidia-vfx (all GPU)
        result = vsr.run(input_tensor)
        out_gpu = torch.from_dlpack(result.image).clone()  # (3, OH, OW) float32

        # POSTPROCESS: float32 → uint8 ARGB on GPU (no CPU touch)
        rgb_hwc = out_gpu.clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(1, 2, 0).contiguous()
        argb = torch.cat([alpha_row, rgb_hwc], dim=2).contiguous()

        # ENCODE: GPU-direct nvenc
        bits = encoder.Encode(argb)
        if bits:
            encoded_bytes.extend(bits)

        total_frames += 1

# Flush encoder
eos_bits = encoder.EndEncode()
if eos_bits:
    encoded_bytes.extend(eos_bits)

torch.cuda.synchronize()
elapsed = time.time() - start
timed_frames = total_frames - len(warmup_frames)

# Write raw HEVC bitstream
Path(OUTPUT_GPU).write_bytes(encoded_bytes)

print(f"\n{'='*60}")
print(f"GPU-DIRECT PIPELINE RESULTS")
print(f"{'='*60}")
print(f"Total frames:    {total_frames} ({len(warmup_frames)} warmup + {timed_frames} timed)")
print(f"Timed portion:   {timed_frames} frames in {elapsed:.3f}s = {timed_frames/elapsed:.1f} fps")
print(f"Per frame:       {elapsed/timed_frames*1000:.1f} ms")
print(f"Output:          {len(encoded_bytes)/1024/1024:.1f} MB raw HEVC")
print(f"Projected 300f:  {300/(timed_frames/elapsed):.1f}s")

# Mux to MP4 with ffmpeg (just container, no re-encode)
import subprocess
subprocess.run([
    'ffmpeg', '-y', '-r', str(meta.average_fps),
    '-i', OUTPUT_GPU,
    '-c', 'copy',
    '-movflags', '+faststart',
    OUTPUT_MP4
], capture_output=True)
mp4_size = Path(OUTPUT_MP4).stat().st_size / 1024 / 1024
print(f"Muxed MP4:       {mp4_size:.1f} MB")

# =====================================================================
# Comparison
# =====================================================================
print(f"\n{'='*60}")
print(f"COMPARISON")
print(f"{'='*60}")
fps_gpu = timed_frames / elapsed
print(f"{'Method':<30} {'FPS':>8} {'300 frames':>12} {'Speedup':>10}")
print(f"{'-'*60}")
print(f"{'SPAN x2 (production)':<30} {'~5':>8} {'~60s':>12} {'1.0x':>10}")
print(f"{'nvvfx + ffmpeg pipes':<30} {'~11':>8} {'~27s':>12} {'2.2x':>10}")
print(f"{'nvvfx + GPU-direct':<30} {f'{fps_gpu:.1f}':>8} {f'{300/fps_gpu:.1f}s':>12} {f'{60/(300/fps_gpu):.1f}x':>10}")

vsr.close()
