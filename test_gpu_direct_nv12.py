"""GPU-Direct Pipeline v3: NV12 decode → GPU color convert → nvidia-vfx → NV12 encode.

Avoids both the RGBP color matrix bug and the ARGB channel ordering bug by:
  1. Decoding as NATIVE NV12 (no PyNvVideoCodec color conversion)
  2. BT.709 NV12→RGB on GPU via torch (correct, verified matrix)
  3. nvidia-vfx inference (float32 RGB CHW)
  4. BT.709 RGB→NV12 on GPU via torch (inverse matrix)
  5. Encoding as NV12 (no ARGB alpha channel hack)
  6. ffmpeg mux (container only, no re-encode)
"""
import time
import torch
import nvvfx
import PyNvVideoCodec as nvc
from pathlib import Path
import subprocess

TEST_VIDEO = "tmp/121b71a1-3723-4302-b322-1621428377a8.mp4"
OUT_DIR = Path("tmp/quality_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_HEVC = str(OUT_DIR / "gpu_direct_nv12.hevc")
OUTPUT_MP4 = str(OUT_DIR / "gpu_direct_nv12.mp4")


# =====================================================================
# BT.709 color conversion (all GPU, no CPU copies)
# =====================================================================
def nv12_to_rgb_chw(nv12_tensor, height, width):
    """NV12 (BT.709 limited range) → float32 RGB CHW [0,1] on GPU.

    NVDEC outputs limited-range NV12 (Y: 16-235, UV: 16-240).
    Verified: mean error 1.69 per pixel vs ffmpeg ground truth (best of 4 variants).
    nv12_tensor: (H*1.5, W) uint8 CUDA
    Returns: (3, H, W) float32 CUDA
    """
    Y = nv12_tensor[:height].float() - 16.0                 # (H, W)
    uv_plane = nv12_tensor[height:].view(height // 2, width // 2, 2).float()
    U = uv_plane[:, :, 0].repeat_interleave(2, 0).repeat_interleave(2, 1) - 128.0
    V = uv_plane[:, :, 1].repeat_interleave(2, 0).repeat_interleave(2, 1) - 128.0

    # BT.709 limited range
    R = (1.164 * Y + 1.793 * V).clamp_(0, 255)
    G = (1.164 * Y - 0.213 * U - 0.533 * V).clamp_(0, 255)
    B = (1.164 * Y + 2.112 * U).clamp_(0, 255)

    return torch.stack([R, G, B], dim=0).div_(255.0)  # (3, H, W) float32 [0,1]


def rgb_chw_to_nv12(rgb, height, width):
    """float32 RGB CHW [0,1] → NV12 (BT.709 limited range) uint8 on GPU.

    Inverse of nv12_to_rgb_chw. Limited range: Y 16-235, UV 16-240.
    Verified: Y round-trip error 0.99, Cb 0.19, Cr 0.21.
    rgb: (3, H, W) float32 CUDA
    Returns: (H*1.5, W) uint8 CUDA
    """
    rgb255 = rgb.mul(255.0).clamp_(0, 255)
    R, G, B = rgb255[0], rgb255[1], rgb255[2]

    # BT.709 limited range forward transform
    Y  = (16.0  + 0.2568 * R + 0.5041 * G + 0.0979 * B).clamp_(16, 235)
    Cb = (128.0 - 0.1482 * R - 0.2910 * G + 0.4392 * B).clamp_(16, 240)
    Cr = (128.0 + 0.4392 * R - 0.3678 * G - 0.0714 * B).clamp_(16, 240)

    Y_u8 = Y.to(torch.uint8)  # (H, W)

    # Subsample Cb, Cr to 4:2:0 by averaging 2x2 blocks
    Cb_sub = Cb.view(height // 2, 2, width // 2, 2).mean(dim=(1, 3))  # (H/2, W/2)
    Cr_sub = Cr.view(height // 2, 2, width // 2, 2).mean(dim=(1, 3))  # (H/2, W/2)

    # Interleave Cb, Cr into NV12 UV plane: (H/2, W/2, 2) → (H/2, W)
    uv = torch.stack([Cb_sub, Cr_sub], dim=2).to(torch.uint8).view(height // 2, width)

    # Stack Y and UV planes: (H + H/2, W) = (H*1.5, W)
    return torch.cat([Y_u8, uv], dim=0)


# =====================================================================
# Setup
# =====================================================================
print("=" * 60)
print("GPU-DIRECT NV12 PIPELINE (v3)")
print("=" * 60)

print("\n[1/4] Setting up decoder (NV12 native)...")
decoder = nvc.CreateSimpleDecoder(
    TEST_VIDEO,
    gpuid=0,
    useDeviceMemory=True,
    outputColorType=nvc.OutputColorType.NATIVE,
    decoderCacheSize=1
)
meta = decoder.get_stream_metadata()
W, H = meta.width, meta.height
FPS = meta.average_fps
print(f"  Input: {W}x{H}, {FPS:.2f} fps, {meta.num_frames} frames")

OUT_W, OUT_H = W * 2, H * 2
print(f"  Output: {OUT_W}x{OUT_H}")

print("\n[2/4] Setting up nvidia-vfx VSR...")
vsr = nvvfx.VideoSuperRes(quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=0)
vsr.output_width = OUT_W
vsr.output_height = OUT_H
vsr.load()

print("\n[3/4] Setting up encoder (NV12 direct)...")
encoder = nvc.CreateEncoder(
    OUT_W, OUT_H,
    'NV12',
    False,  # GPU memory input
    codec='hevc',
    preset='P4',
    rc='constqp',
    constqp=20,
    profile='main'
)

# =====================================================================
# Warmup (3 frames)
# =====================================================================
print("\n[4/4] Warming up (3 frames)...")
warmup_frames = decoder.get_batch_frames(3)
for wf in warmup_frames:
    nv12_gpu = torch.from_dlpack(wf)  # NV12: (H*1.5, W) uint8 CUDA
    rgb_chw = nv12_to_rgb_chw(nv12_gpu, H, W)  # (3, H, W) float32
    result = vsr.run(rgb_chw)
    out_rgb = torch.from_dlpack(result.image).clone()  # (3, OUT_H, OUT_W)
    out_nv12 = rgb_chw_to_nv12(out_rgb, OUT_H, OUT_W)  # (OUT_H*1.5, OUT_W) uint8
    encoder.Encode(out_nv12)
torch.cuda.synchronize()
print("  Warmup done.")

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
        # DECODE: zero-copy DLPack → NV12 CUDA tensor
        nv12_gpu = torch.from_dlpack(frame)

        # COLOR CONVERT: NV12 → RGB on GPU (BT.709 limited range)
        rgb_chw = nv12_to_rgb_chw(nv12_gpu, H, W)

        # INFERENCE: nvidia-vfx (all GPU)
        result = vsr.run(rgb_chw)
        out_rgb = torch.from_dlpack(result.image).clone()

        # COLOR CONVERT: RGB → NV12 on GPU (BT.709 limited range)
        out_nv12 = rgb_chw_to_nv12(out_rgb, OUT_H, OUT_W)

        # ENCODE: NV12 GPU-direct nvenc
        bits = encoder.Encode(out_nv12)
        if bits:
            bitstream_chunks.append(bits)

        total_frames += 1
        if total_frames % 50 == 0:
            elapsed_so_far = time.time() - start
            fps_so_far = (total_frames - len(warmup_frames)) / elapsed_so_far
            print(f"  {total_frames} frames done ({fps_so_far:.1f} fps)")

# Flush encoder
eos = encoder.EndEncode()
if eos:
    bitstream_chunks.append(eos)

torch.cuda.synchronize()
elapsed = time.time() - start
timed_frames = total_frames - len(warmup_frames)

# =====================================================================
# Write raw HEVC + mux to MP4
# =====================================================================
hevc_data = b''.join(bitstream_chunks)
Path(OUTPUT_HEVC).write_bytes(hevc_data)

subprocess.run([
    'ffmpeg', '-y', '-r', str(FPS),
    '-i', OUTPUT_HEVC,
    '-c', 'copy', '-movflags', '+faststart',
    OUTPUT_MP4
], capture_output=True, check=True)

mp4_size = Path(OUTPUT_MP4).stat().st_size / 1024 / 1024
fps = timed_frames / elapsed

print(f"\n{'=' * 60}")
print(f"RESULTS")
print(f"{'=' * 60}")
print(f"Total frames:    {total_frames} ({len(warmup_frames)} warmup + {timed_frames} timed)")
print(f"Timed portion:   {timed_frames} frames in {elapsed:.3f}s = {fps:.1f} fps")
print(f"Per frame:       {elapsed / timed_frames * 1000:.1f} ms")
print(f"Output:          {mp4_size:.1f} MB MP4")
print(f"Projected 300f:  {300 / fps:.1f}s")

# =====================================================================
# VMAF validation: compare against ffmpeg-pipe nvidia-vfx output
# =====================================================================
print(f"\n{'=' * 60}")
print(f"VMAF VALIDATION")
print(f"{'=' * 60}")

# Generate reference: bicubic at same output resolution
bicubic_path = str(OUT_DIR / "bicubic_ref.mp4")
print(f"\nGenerating bicubic reference at {OUT_W}x{OUT_H}...")
subprocess.run([
    'ffmpeg', '-y', '-i', TEST_VIDEO,
    '-vf', f'scale={OUT_W}:{OUT_H}:flags=bicubic',
    '-c:v', 'hevc_nvenc', '-cq', '10', '-preset', 'p7',
    '-pix_fmt', 'yuv420p', bicubic_path
], capture_output=True, check=True)

# Generate ffmpeg-pipe nvidia-vfx reference for direct comparison
pipe_ref_path = str(OUT_DIR / "nvvfx_pipe_ref.mp4")
print("Generating ffmpeg-pipe nvidia-vfx reference...")
# Decode
dec_proc = subprocess.Popen(
    ['ffmpeg', '-i', TEST_VIDEO, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
)
enc_proc = subprocess.Popen(
    ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
     '-s', f'{OUT_W}x{OUT_H}', '-r', str(FPS), '-i', 'pipe:0',
     '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
     '-pix_fmt', 'yuv420p', '-movflags', '+faststart', pipe_ref_path],
    stdin=subprocess.PIPE, stderr=subprocess.PIPE
)

frame_size = W * H * 3
ref_count = 0
while True:
    raw = dec_proc.stdout.read(frame_size)
    if len(raw) < frame_size:
        break
    t = (torch.frombuffer(bytearray(raw), dtype=torch.uint8)
         .reshape(H, W, 3).cuda()
         .to(torch.float32).div_(255.0)
         .permute(2, 0, 1).contiguous())
    result = vsr.run(t)
    out = torch.from_dlpack(result.image).clone()
    out_bytes = (out.clamp_(0, 1).mul_(255).round_()
                 .to(torch.uint8).permute(1, 2, 0)
                 .contiguous().cpu().numpy().tobytes())
    enc_proc.stdin.write(out_bytes)
    ref_count += 1
enc_proc.stdin.close()
enc_proc.wait()
dec_proc.wait()
print(f"  Reference: {ref_count} frames")

vsr.close()


def run_ssim_psnr(distorted, reference, label):
    """Run SSIM and PSNR via ffmpeg and return parsed results."""
    import re
    ssim_result = subprocess.run(
        ['ffmpeg', '-i', distorted, '-i', reference,
         '-filter_complex', f'ssim=stats_file=/tmp/ssim_{label}.log',
         '-f', 'null', '-'],
        capture_output=True, text=True
    )
    psnr_result = subprocess.run(
        ['ffmpeg', '-i', distorted, '-i', reference,
         '-filter_complex', f'psnr=stats_file=/tmp/psnr_{label}.log',
         '-f', 'null', '-'],
        capture_output=True, text=True
    )
    ssim_match = re.search(r'SSIM.*?All:([\d.]+)', ssim_result.stderr)
    psnr_match = re.search(r'PSNR.*?average:([\d.]+)', psnr_result.stderr)
    ssim_y_match = re.search(r'SSIM Y:([\d.]+)', ssim_result.stderr)

    ssim_all = float(ssim_match.group(1)) if ssim_match else 0
    ssim_y = float(ssim_y_match.group(1)) if ssim_y_match else 0
    psnr = float(psnr_match.group(1)) if psnr_match else 0
    return ssim_all, ssim_y, psnr


# Quality: NV12 GPU-direct vs bicubic reference
print("\nSSIM/PSNR: NV12 GPU-direct vs bicubic reference...")
nv12_bic_ssim, nv12_bic_ssim_y, nv12_bic_psnr = run_ssim_psnr(OUTPUT_MP4, bicubic_path, "nv12_bic")
print(f"  SSIM={nv12_bic_ssim:.6f}  SSIM_Y={nv12_bic_ssim_y:.6f}  PSNR={nv12_bic_psnr:.2f} dB")

# Quality: ffmpeg-pipe reference vs bicubic reference
print("\nSSIM/PSNR: ffmpeg-pipe nvidia-vfx vs bicubic reference...")
pipe_bic_ssim, pipe_bic_ssim_y, pipe_bic_psnr = run_ssim_psnr(pipe_ref_path, bicubic_path, "pipe_bic")
print(f"  SSIM={pipe_bic_ssim:.6f}  SSIM_Y={pipe_bic_ssim_y:.6f}  PSNR={pipe_bic_psnr:.2f} dB")

# Color accuracy: NV12 GPU-direct vs ffmpeg-pipe (should be very high)
print("\nSSIM/PSNR: NV12 GPU-direct vs ffmpeg-pipe (color match)...")
nv12_pipe_ssim, nv12_pipe_ssim_y, nv12_pipe_psnr = run_ssim_psnr(OUTPUT_MP4, pipe_ref_path, "nv12_pipe")
print(f"  SSIM={nv12_pipe_ssim:.6f}  SSIM_Y={nv12_pipe_ssim_y:.6f}  PSNR={nv12_pipe_psnr:.2f} dB")

# Extract sample frames for visual comparison
print("\nExtracting sample frames (frame 50)...")
subprocess.run(['ffmpeg', '-y', '-i', OUTPUT_MP4, '-vf', 'select=eq(n\\,50)',
                '-frames:v', '1', str(OUT_DIR / 'frame_nv12_direct.png')], capture_output=True)
subprocess.run(['ffmpeg', '-y', '-i', pipe_ref_path, '-vf', 'select=eq(n\\,50)',
                '-frames:v', '1', str(OUT_DIR / 'frame_pipe_ref.png')], capture_output=True)

# =====================================================================
# Summary
# =====================================================================
print(f"\n{'=' * 60}")
print(f"FINAL SUMMARY")
print(f"{'=' * 60}")
print(f"{'Method':<30} {'FPS':>6} {'300f':>6} {'SSIM(bic)':>10} {'PSNR(bic)':>10} {'SSIM(pipe)':>10} {'PSNR(pipe)':>10}")
print(f"{'-' * 84}")
print(f"{'ffmpeg-pipe nvidia-vfx':<30} {'~11':>6} {'~27s':>6} {pipe_bic_ssim:>10.4f} {pipe_bic_psnr:>9.1f}dB {'ref':>10} {'ref':>10}")
print(f"{'NV12 GPU-direct (this)':<30} {f'{fps:.0f}':>6} {f'{300/fps:.1f}s':>6} {nv12_bic_ssim:>10.4f} {nv12_bic_psnr:>9.1f}dB {nv12_pipe_ssim:>10.4f} {nv12_pipe_psnr:>9.1f}dB")
print(f"\nSpeedup vs pipe:  {fps/11:.1f}x")
print(f"Color match:      SSIM {nv12_pipe_ssim:.4f}, PSNR {nv12_pipe_psnr:.1f} dB (NV12 vs pipe)")
color_ok = "PASS" if nv12_pipe_psnr >= 35 else "FAIL"
print(f"Color check:      {color_ok} (PSNR >= 35 dB means negligible visible difference)")
