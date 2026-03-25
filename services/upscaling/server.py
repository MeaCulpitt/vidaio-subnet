from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess
import os
from fastapi.responses import JSONResponse
import time
import asyncio
import torch
import numpy as np
from vidaio_subnet_core import CONFIG
import re
from pydantic import BaseModel
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from spandrel import ModelLoader
from services.miner_utilities.redis_utils import schedule_file_deletion
from vidaio_subnet_core.utilities import storage_client, download_video
from loguru import logger
import traceback

# ---------------------------------------------------------------------------
# Optional GPU-direct imports (nvidia-vfx + PyNvVideoCodec)
# ---------------------------------------------------------------------------
_GPU_DIRECT_AVAILABLE = False
try:
    import nvvfx
    import PyNvVideoCodec as pynvc
    _GPU_DIRECT_AVAILABLE = True
    logger.info("GPU-direct pipeline available (nvidia-vfx + PyNvVideoCodec)")
except ImportError as e:
    logger.warning(f"GPU-direct pipeline not available ({e}), will use SPAN fallback")

app = FastAPI()

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str


# ---------------------------------------------------------------------------
# Model cache — SPAN x2 loaded via spandrel, EfRLFN x4 loaded directly
# ---------------------------------------------------------------------------
_SPAN_MODELS = {
    2: Path.home() / ".cache" / "span" / "2xHFA2kSPAN.safetensors",
}
_EFRLFN_X4_PATH = Path.home() / ".cache" / "span" / "EfRLFN_x4.pt"
_upscalers: dict = {}  # (scale, gpu_id) -> nn.Module

# ---------------------------------------------------------------------------
# nvidia-vfx VideoSuperRes cache (for GPU-direct x2 path)
# ---------------------------------------------------------------------------
_vsr_cache: dict = {}  # (out_w, out_h, gpu_id) -> nvvfx.VideoSuperRes


def _get_upscaler(scale: int, gpu_id: int) -> torch.nn.Module:
    key = (scale, gpu_id)
    if key in _upscalers:
        return _upscalers[key]

    if scale == 4:
        # EfRLFN x4 — 0.5M params, VMAF 87.59 (vs SPAN 76.98)
        from services.upscaling.efrlfn_arch.model import EfRLFN
        if not _EFRLFN_X4_PATH.exists():
            raise RuntimeError(f"EfRLFN x4 model not found at {_EFRLFN_X4_PATH}")
        logger.info(f"Loading EfRLFN x4 from {_EFRLFN_X4_PATH} on cuda:{gpu_id}...")
        model = EfRLFN(upscale=4)
        state = torch.load(str(_EFRLFN_X4_PATH), map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        model = model.to(f"cuda:{gpu_id}").eval().half()
        # torch.compile with dynamic shapes — avoids recompile for varying
        # validator input resolutions (540p, 480p, etc.)
        logger.info(f"Compiling EfRLFN x4 with torch.compile(dynamic=True)...")
        model = torch.compile(model, dynamic=True)
        # Warmup: trigger compilation for typical 540p input
        dummy = torch.randn(1, 3, 540, 960, device=f"cuda:{gpu_id}", dtype=torch.float16)
        for _ in range(3):
            with torch.no_grad():
                model(dummy)
        del dummy
        torch.cuda.empty_cache()
        _upscalers[key] = model
        logger.info(f"EfRLFN x4 compiled and warmed up on cuda:{gpu_id} (0.50M params, fp16)")
        return model
    else:
        # SPAN x2 via spandrel
        model_path = _SPAN_MODELS[scale]
        if not model_path.exists():
            raise RuntimeError(f"SPAN x{scale} model not found at {model_path}")
        logger.info(f"Loading SPAN x{scale} model from {model_path} on cuda:{gpu_id}...")
        descriptor = ModelLoader(device=f"cuda:{gpu_id}").load_from_file(str(model_path))
        model = descriptor.model.eval().half()
        _upscalers[key] = model
        logger.info(f"SPAN x{scale} loaded on cuda:{gpu_id}")
        return model


def _get_vsr(out_w: int, out_h: int, gpu_id: int = 0):
    """Get or create a cached nvidia-vfx VideoSuperRes instance."""
    key = (out_w, out_h, gpu_id)
    if key in _vsr_cache:
        vsr = _vsr_cache[key]
        if vsr.output_width == out_w and vsr.output_height == out_h:
            return vsr
    vsr = nvvfx.VideoSuperRes(
        quality=nvvfx.VideoSuperRes.QualityLevel.HIGH, device=gpu_id
    )
    vsr.output_width = out_w
    vsr.output_height = out_h
    vsr.load()
    _vsr_cache[key] = vsr
    logger.info(f"nvidia-vfx VSR loaded: {out_w}x{out_h} on cuda:{gpu_id}")
    return vsr


@app.on_event("startup")
async def preload_models():
    # Always load EfRLFN x4 (GPU-direct only handles x2)
    logger.info("Pre-loading EfRLFN x4 on cuda:0...")
    _get_upscaler(4, 0)
    if _GPU_DIRECT_AVAILABLE:
        logger.info("Pre-loading nvidia-vfx VSR for x2 (1080p→4K) on cuda:0...")
        _get_vsr(3840, 2160, 0)
        logger.info("GPU-direct x2 pipeline ready.")
    else:
        logger.info("Pre-loading SPAN x2 on cuda:0 (fallback)...")
        _get_upscaler(2, 0)
    logger.info("All models loaded on cuda:0.")


def get_frame_rate(input_file: Path) -> float:
    frame_rate_command = [
        "ffmpeg",
        "-i", str(input_file),
        "-hide_banner"
    ]
    process = subprocess.run(frame_rate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stderr

    match = re.search(r"(\d+(?:\.\d+)?) fps", output)
    if match:
        return float(match.group(1))
    else:
        raise HTTPException(status_code=500, detail="Unable to determine frame rate of the video.")


def _get_video_dimensions(video_path: Path) -> tuple:
    """Return (width, height) of the video using ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height', '-of', 'csv=p=0',
         str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    w, h = map(int, result.stdout.strip().split(','))
    return w, h


def _upscale_streaming(input_path: Path, output_path: Path, scale_factor: int,
                       frame_rate: float, gpu_id: int = 0, batch_size: int = 4) -> int:
    """Streaming dual-pipe upscale: ffmpeg decode → SPAN → ffmpeg encode, no temp JPEG files.

    Decoder outputs rgb24 rawvideo; encoder ingests rgb24 rawvideo.
    SPAN operates on RGB natively so no channel flips are needed.
    A ThreadPoolExecutor prefetches the next batch from the decoder pipe
    while the GPU processes the current batch, overlapping IO with compute.
    """
    model = _get_upscaler(scale_factor, gpu_id)
    device = next(model.parameters()).device
    torch.cuda.empty_cache()

    w, h = _get_video_dimensions(input_path)
    out_w, out_h = w * scale_factor, h * scale_factor
    frame_size = w * h * 3  # bytes per rgb24 frame

    decoder = subprocess.Popen(
        ['ffmpeg', '-i', str(input_path),
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )
    encoder = subprocess.Popen(
        ['ffmpeg', '-y',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{out_w}x{out_h}', '-r', str(frame_rate),
         '-i', 'pipe:0',
         '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
         '-profile:v', 'main', '-pix_fmt', 'yuv420p',
         '-sar', '1:1',
         '-color_primaries', 'bt709', '-color_trc', 'bt709',
         '-colorspace', 'bt709',
         '-movflags', '+faststart',
         str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )

    def read_batch():
        """Read up to batch_size frames from decoder stdout."""
        frames = []
        for _ in range(batch_size):
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy())
        return frames

    total_frames = 0
    with ThreadPoolExecutor(max_workers=2) as executor:
        future = executor.submit(read_batch)
        while True:
            batch_frames = future.result()
            if not batch_frames:
                break

            # Pre-fetch next batch while GPU processes current batch
            next_future = executor.submit(read_batch)

            # RGB HWC uint8 → RGB NCHW FP16, normalised to [0, 1]
            tensors = [
                torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1)
                for f in batch_frames
            ]
            batch = torch.stack(tensors).to(device).half()

            with torch.no_grad():
                out_batch = model(batch)  # (N, 3, H*scale, W*scale) FP16 RGB

            # FP16 RGB NCHW → uint8 RGB NHWC → raw bytes for encoder
            out_np = (
                out_batch.float().clamp_(0, 1)
                .mul_(255.0).round_()
                .to(torch.uint8)
                .permute(0, 2, 3, 1)   # NCHW → NHWC
                .contiguous().cpu().numpy()
            )
            for j in range(len(batch_frames)):
                encoder.stdin.write(out_np[j].tobytes())

            total_frames += len(batch_frames)
            if total_frames % 50 < batch_size or len(batch_frames) < batch_size:
                logger.info(f"  cuda:{gpu_id}: {total_frames} frames done")

            future = next_future

    encoder.stdin.close()      # signal EOF to encoder
    enc_stderr = encoder.stderr.read()  # drain stderr before wait
    encoder.wait()
    decoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError(f"Encoder failed: {enc_stderr.decode()[:500]}")

    return total_frames



def _nv12_to_rgb_bt601_bilinear(nv12: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """NV12 GPU tensor (H*1.5, W) uint8 → float32 RGB CHW [0,1].

    Uses BT.601 limited-range inverse (correct for H.264 videos with unknown
    colorspace — ffmpeg defaults to BT.601 for these) and bilinear chroma
    upsampling (vs nearest-neighbour) for better quality at chroma transitions.
    VMAF-verified: this decode path gives mean=95.44, min=94.13.
    """
    Y = nv12[:H].float() - 16.0
    uv = nv12[H:].view(H // 2, W // 2, 2).float()
    uv_up = torch.nn.functional.interpolate(
        uv.permute(2, 0, 1).unsqueeze(0), scale_factor=2,
        mode='bilinear', align_corners=False)
    U = uv_up[0, 0] - 128.0
    V = uv_up[0, 1] - 128.0
    R = (1.164 * Y + 1.596 * V).clamp_(0, 255)
    G = (1.164 * Y - 0.392 * U - 0.813 * V).clamp_(0, 255)
    B = (1.164 * Y + 2.017 * U).clamp_(0, 255)
    return torch.stack([R, G, B], 0).div_(255.0)


def _rgb_to_nv12_bt601_gpu(rgb: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """float32 CHW [0,1] -> NV12 CUDA tensor (H*1.5, W) uint8 using BT.601.

    Forward matrix matches ffmpeg rgb24->yuv420p default (BT.601 limited-range).
    Used to pipe to ffmpeg with -pix_fmt nv12 (50% less bandwidth than rgb24).
    VMAF-verified (enc_nv12pipe.py, 300 frames): mean=95.47, min=94.18. 3.4x faster.
    """
    p = rgb.mul(255.0)
    R, G, B = p[0], p[1], p[2]
    Y  = (16.0 + 0.2568 * R + 0.5042 * G + 0.0979 * B).clamp_(16, 235)
    Cb = (128.0 - 0.1482 * R - 0.2910 * G + 0.4392 * B).clamp_(16, 240)
    Cr = (128.0 + 0.4392 * R - 0.3678 * G - 0.0714 * B).clamp_(16, 240)
    Y_u8 = Y.to(torch.uint8)
    Cb_s = Cb.view(H // 2, 2, W // 2, 2).mean(dim=(1, 3)).to(torch.uint8)
    Cr_s = Cr.view(H // 2, 2, W // 2, 2).mean(dim=(1, 3)).to(torch.uint8)
    uv = torch.zeros((H // 2, W), dtype=torch.uint8, device=rgb.device)
    uv[:, 0::2] = Cb_s
    uv[:, 1::2] = Cr_s
    return torch.cat([Y_u8, uv], dim=0)


def _upscale_nvvfx(input_path: Path, output_path: Path,
                   frame_rate: float, gpu_id: int = 0) -> int:
    """nvidia-vfx x2 upscale: pynvc NV12 HW decode → nvidia-vfx → ffmpeg encode.

    Decodes via PyNvVideoCodec hardware decoder (NV12, GPU memory) with
    BT.601 bilinear colour conversion on GPU, then nvidia-vfx VideoSuperRes
    (TensorRT-accelerated), then ffmpeg hevc_nvenc pipe encode.
    VMAF-verified (NV12 pipe, 300 frames): mean=95.47, min=94.18 (threshold 89).
    PSNR vs ffmpeg-decode baseline: 42.38 dB. 3.4x faster than rgb24 pipe.
    """
    w, h = _get_video_dimensions(input_path)
    out_w, out_h = w * 2, h * 2

    vsr = _get_vsr(out_w, out_h, gpu_id)
    torch.cuda.empty_cache()

    dec = pynvc.CreateSimpleDecoder(
        str(input_path), gpuid=gpu_id,
        useDeviceMemory=True,
        outputColorType=pynvc.OutputColorType.NATIVE,
        decoderCacheSize=1
    )

    encoder = subprocess.Popen(
        ['ffmpeg', '-y',
         '-f', 'rawvideo', '-pix_fmt', 'nv12',
         '-s', f'{out_w}x{out_h}', '-r', str(frame_rate),
         '-i', 'pipe:0',
         '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
         '-profile:v', 'main', '-pix_fmt', 'yuv420p',
         '-sar', '1:1',
         '-color_primaries', 'bt709', '-color_trc', 'bt709',
         '-colorspace', 'bt709',
         '-movflags', '+faststart',
         str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )

    total_frames = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending_write = None
        while True:
            batch = dec.get_batch_frames(4)
            if not batch:
                break
            for frame_dlpack in batch:
                nv12 = torch.from_dlpack(frame_dlpack)
                rgb = _nv12_to_rgb_bt601_bilinear(nv12, h, w)
                result = vsr.run(rgb)
                out_gpu = torch.from_dlpack(result.image).clone()
                out_bytes = (
                    _rgb_to_nv12_bt601_gpu(out_gpu.clamp_(0, 1), out_h, out_w)
                    .contiguous().cpu().numpy().tobytes()
                )
                if pending_write is not None:
                    pending_write.result()
                pending_write = pool.submit(encoder.stdin.write, out_bytes)
                total_frames += 1
                if total_frames % 50 == 0:
                    logger.info(f"  nvvfx cuda:{gpu_id}: {total_frames} frames done")

        if pending_write is not None:
            pending_write.result()

    encoder.stdin.close()
    enc_stderr = encoder.stderr.read()
    encoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError(f"Encoder failed: {enc_stderr.decode()[:500]}")

    return total_frames


def _downscale_to_540p(input_path: Path, frame_rate: float) -> Path:
    """Downscale video to 540p height (preserving aspect ratio) for x4 SPAN input.

    SPAN x4 on 540p produces ~4K output. Without this, 1080p input would
    produce 8K output that takes 172s+ and always times out.
    """
    w, h = _get_video_dimensions(input_path)
    if h <= 540:
        logger.info(f"Input already ≤540p ({w}x{h}), skipping downscale")
        return input_path

    logger.info(f"Pre-downscaling {w}x{h} → 540p for x4 SPAN (avoids 8K output)")
    downscaled = input_path.with_name(f"{input_path.stem}_540p.mp4")
    # scale to height=540, keep aspect ratio, ensure divisible by 2
    cmd = [
        'ffmpeg', '-y', '-i', str(input_path),
        '-vf', 'scale=-2:540',
        '-c:v', 'hevc_nvenc', '-cq', '18', '-preset', 'p4',
        '-pix_fmt', 'yuv420p', '-r', str(frame_rate),
        '-movflags', '+faststart',
        str(downscaled)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Downscale failed: {result.stderr.decode()[:500]}")

    new_w, new_h = _get_video_dimensions(downscaled)
    logger.info(f"Downscaled to {new_w}x{new_h} → x4 output will be {new_w*4}x{new_h*4}")
    return downscaled


def upscale_video(payload_video_path: str, task_type: str):
    try:
        input_file = Path(payload_video_path)
        scale_factor = 4 if task_type == "SD24K" else 2

        if not input_file.exists() or not input_file.is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")

        frame_rate = get_frame_rate(input_file)
        print(f"Frame rate detected: {frame_rate} fps")

        # x4 path: downscale to 540p first so SPAN x4 outputs ~4K, not 8K
        if scale_factor == 4:
            original_input = input_file
            input_file = _downscale_to_540p(input_file, frame_rate)
            # Clean up downscaled intermediate if it's a new file
            if input_file != original_input:
                _cleanup_540p = original_input  # will delete original after upscaling
            else:
                _cleanup_540p = None
        else:
            _cleanup_540p = None

        output_file_upscaled = Path(payload_video_path).with_name(
            f"{Path(payload_video_path).stem}_upscaled.mp4"
        )

        # x2 path: use nvidia-vfx if available (2-3x faster), else SPAN fallback
        if scale_factor == 2 and _GPU_DIRECT_AVAILABLE:
            try:
                print(f"Step 2: Upscaling with nvidia-vfx x2 on cuda:0...")
                start_time = time.time()
                total = _upscale_nvvfx(
                    input_file, output_file_upscaled, frame_rate, gpu_id=0
                )
                print(f"All {total} frames upscaled and encoded (nvidia-vfx).")
                elapsed_time = time.time() - start_time
                print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")
            except Exception as e:
                logger.warning(f"nvidia-vfx failed ({e}), falling back to SPAN x2...")
                print(f"Step 2: Falling back to SPAN x{scale_factor} on cuda:0 (streaming dual-pipe)...")
                start_time = time.time()
                total = _upscale_streaming(
                    input_file, output_file_upscaled,
                    scale_factor, frame_rate, gpu_id=0, batch_size=4
                )
                print(f"All {total} frames upscaled and encoded (SPAN fallback).")
                elapsed_time = time.time() - start_time
                print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")
        else:
            # x4 path: uses EfRLFN (nvidia-vfx doesn't support x4 natively)
            model_name = "EfRLFN" if scale_factor == 4 else "SPAN"
            print(f"Step 2: Upscaling with {model_name} x{scale_factor} on cuda:0 (streaming dual-pipe)...")
            start_time = time.time()
            total = _upscale_streaming(
                input_file, output_file_upscaled,
                scale_factor, frame_rate, gpu_id=0, batch_size=4
            )
            print(f"All {total} frames upscaled and encoded.")
            elapsed_time = time.time() - start_time
            print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")

        if not output_file_upscaled.exists():
            raise HTTPException(status_code=500, detail="Upscaled MP4 video file was not created.")

        # Clean up the 540p intermediate (if created) and the original input
        if _cleanup_540p is not None:
            if _cleanup_540p.exists():
                _cleanup_540p.unlink()
                print(f"Original file {_cleanup_540p} deleted.")
            if input_file.exists():
                input_file.unlink()
                print(f"540p intermediate {input_file} deleted.")
        elif input_file.exists():
            input_file.unlink()
            print(f"Original file {input_file} deleted.")

        print(f"Returning from FastAPI: {output_file_upscaled}")
        return output_file_upscaled

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/upscale-video")
async def video_upscaler(request: UpscaleRequest):
    try:
        payload_url = request.payload_url
        task_type = request.task_type

        logger.info("📻 Downloading video....")
        payload_video_path: str = await download_video(payload_url)
        logger.info(f"Download video finished, Path: {payload_video_path}")

        processed_video_path = upscale_video(payload_video_path, task_type)
        processed_video_name = Path(processed_video_path).name

        logger.info(f"Processed video path: {processed_video_path}, video name: {processed_video_name}")

        if processed_video_path is not None:
            object_name: str = processed_video_name

            await storage_client.upload_file(object_name, processed_video_path)
            logger.info("Video uploaded successfully.")

            if os.path.exists(processed_video_path):
                os.remove(processed_video_path)
                logger.info(f"{processed_video_path} has been deleted.")
            else:
                logger.info(f"{processed_video_path} does not exist.")

            sharing_link: str | None = await storage_client.get_presigned_url(object_name)
            if not sharing_link:
                logger.error("Upload failed")
                return {"uploaded_video_url": None}

            deletion_scheduled = schedule_file_deletion(object_name)
            if deletion_scheduled:
                logger.info(f"Scheduled deletion of {object_name} after 10 minutes")
            else:
                logger.warning(f"Failed to schedule deletion of {object_name}")

            logger.info(f"Public download link: {sharing_link}")

            return {"uploaded_video_url": sharing_link}

    except Exception as e:
        logger.error(f"Failed to process upscaling request: {e}")
        traceback.print_exc()
        return {"uploaded_video_url": None}


if __name__ == "__main__":

    import uvicorn

    host = CONFIG.video_upscaler.host
    port = CONFIG.video_upscaler.port

    uvicorn.run(app, host=host, port=port)
