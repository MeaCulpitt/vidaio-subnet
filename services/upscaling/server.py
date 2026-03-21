from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import subprocess
import os
from fastapi.responses import JSONResponse
import time
import asyncio
import tempfile
import urllib.request
import cv2
import torch
import numpy as np
from vidaio_subnet_core import CONFIG
import re
from pydantic import BaseModel
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from services.miner_utilities.redis_utils import schedule_file_deletion
from vidaio_subnet_core.utilities import storage_client, download_video
from loguru import logger
import traceback

app = FastAPI()

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str
    # output_file_upscaled: Optional[str] = None


# ---------------------------------------------------------------------------
# Model cache — two instances per scale (one per GPU), initialised at startup
# Keys: (scale, gpu_id)
# ---------------------------------------------------------------------------
_MODEL_DIR = Path.home() / ".cache" / "realesrgan"
_MODEL_URLS = {
    2: ("RealESRGAN_x2plus.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
    4: ("RealESRGAN_x4plus.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
}
_upscalers: dict = {}  # (scale, gpu_id) -> RealESRGANer


def _get_upscaler(scale: int, gpu_id: int) -> RealESRGANer:
    key = (scale, gpu_id)
    if key in _upscalers:
        return _upscalers[key]

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_name, model_url = _MODEL_URLS[scale]
    model_path = _MODEL_DIR / model_name

    if not model_path.exists():
        logger.info(f"Downloading {model_name}...")
        urllib.request.urlretrieve(model_url, model_path)
        logger.info(f"Downloaded {model_name} to {model_path}")

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    upscaler = RealESRGANer(
        scale=scale,
        model_path=str(model_path),
        model=model,
        tile=0,
        tile_pad=0,
        pre_pad=0,
        half=True,
        device=f"cuda:{gpu_id}",
    )
    logger.info(f"Compiling RealESRGANer x{scale} model with torch.compile (max-autotune)...")
    upscaler.model = torch.compile(upscaler.model, mode="max-autotune")
    _upscalers[key] = upscaler
    logger.info(f"RealESRGANer x{scale} loaded and compiled on cuda:{gpu_id}")
    return upscaler


@app.on_event("startup")
async def preload_models():
    logger.info("Pre-loading RealESRGAN models on cuda:0...")
    for scale in (2, 4):
        _get_upscaler(scale, 0)
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


def _upscale_frames_piped(frame_paths: list, scale_factor: int, gpu_id: int, ffmpeg_proc, batch_size: int = 4) -> None:
    """Upscale frames in batches on the given GPU and pipe raw BGR bytes directly to ffmpeg stdin."""
    upscaler = _get_upscaler(scale_factor, gpu_id)
    device = upscaler.device
    total = len(frame_paths)

    torch.cuda.empty_cache()  # flush fragmented reserved pool before large batch allocation

    for batch_start in range(0, total, batch_size):
        batch_paths = frame_paths[batch_start:batch_start + batch_size]

        # Read JPEG frames and convert: BGR uint8 HWC → RGB float16 CHW
        tensors = []
        for p in batch_paths:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            t = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1]).astype(np.float32) / 255.0)
            tensors.append(t.permute(2, 0, 1))  # HWC → CHW

        batch = torch.stack(tensors).to(device).half()  # (N, 3, H, W) FP16

        with torch.no_grad():
            out_batch = upscaler.model(batch)  # (N, 3, H*scale, W*scale) FP16

        # Post-process on GPU: FP16 RGB → uint8 BGR, then transfer to CPU
        out_np = (
            out_batch.float().clamp_(0, 1)
            .flip(1)                        # RGB → BGR (channel dim)
            .mul_(255.0).round_()
            .to(torch.uint8)
            .permute(0, 2, 3, 1)            # NCHW → NHWC
            .contiguous()
            .cpu()
            .numpy()
        )

        for j in range(len(batch_paths)):
            ffmpeg_proc.stdin.write(out_np[j].tobytes())

        done = min(batch_start + batch_size, total)
        if done % 50 < batch_size or done == total:
            logger.info(f"  cuda:{gpu_id}: {done}/{total} frames done")


def upscale_video(payload_video_path: str, task_type: str):
    try:
        input_file = Path(payload_video_path)
        scale_factor = 4 if task_type == "SD24K" else 2

        if not input_file.exists() or not input_file.is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")

        frame_rate = get_frame_rate(input_file)
        print(f"Frame rate detected: {frame_rate} fps")

        stop_duration = 2 / frame_rate

        output_file_with_extra_frames = input_file.with_name(f"{input_file.stem}_extra_frames.mp4")
        output_file_denoised = input_file.with_name(f"{input_file.stem}_denoised.mp4")
        output_file_upscaled = input_file.with_name(f"{input_file.stem}_upscaled.mp4")

        # Step 1: Duplicate the last frame two times
        print("Step 1: Duplicating the last frame two times...")
        start_time = time.time()

        duplicate_last_frame_command = [
            "ffmpeg",
            "-i", str(input_file),
            "-vf", f"tpad=stop_mode=clone:stop_duration={stop_duration}",
            "-c:v", "libx264",
            "-crf", "28",
            "-preset", "fast",
            str(output_file_with_extra_frames)
        ]

        duplicate_last_frame_process = subprocess.run(
            duplicate_last_frame_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        elapsed_time = time.time() - start_time
        if duplicate_last_frame_process.returncode != 0:
            print(f"Duplicating frames failed: {duplicate_last_frame_process.stderr.strip()}")
            raise HTTPException(status_code=500, detail=f"Duplicating frames failed: {duplicate_last_frame_process.stderr.strip()}")
        if not output_file_with_extra_frames.exists():
            print("MP4 video file with extra frames was not created.")
            raise HTTPException(status_code=500, detail="MP4 video file with extra frames was not created.")
        print(f"Step 1 completed in {elapsed_time:.2f} seconds. File with extra frames: {output_file_with_extra_frames}")

        # Step 1.5: Denoise before upscaling
        print("Step 1.5: Denoising with hqdn3d...")
        start_time = time.time()

        denoise_command = [
            "ffmpeg", "-i", str(output_file_with_extra_frames),
            "-vf", "hqdn3d=3:3:6:6",
            "-c:v", "libx264", "-crf", "18", "-preset", "fast",
            str(output_file_denoised)
        ]
        denoise_process = subprocess.run(
            denoise_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        elapsed_time = time.time() - start_time
        if denoise_process.returncode != 0:
            print(f"Denoising failed: {denoise_process.stderr.strip()}")
            raise HTTPException(status_code=500, detail=f"Denoising failed: {denoise_process.stderr.strip()}")
        if not output_file_denoised.exists():
            raise HTTPException(status_code=500, detail="Denoised video file was not created.")
        print(f"Step 1.5 completed in {elapsed_time:.2f} seconds. Denoised file: {output_file_denoised}")

        # Step 2: Upscale using native PyTorch RealESRGAN on cuda:0, pipe directly to ffmpeg
        print(f"Step 2: Upscaling with RealESRGAN x{scale_factor} on cuda:0 (piped)...")
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_in = Path(tmp_dir) / "frames_in"
            frames_in.mkdir()

            extract_result = subprocess.run(
                ["ffmpeg", "-i", str(output_file_denoised), "-q:v", "2", str(frames_in / "%08d.jpg")],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if extract_result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Frame extraction failed: {extract_result.stderr.strip()}")

            frame_paths = sorted(frames_in.glob("*.jpg"))
            total = len(frame_paths)
            print(f"Extracted {total} frames, upscaling on cuda:0 via pipe...")

            # Determine output dimensions from first frame
            first_img = cv2.imread(str(frame_paths[0]), cv2.IMREAD_UNCHANGED)
            out_h = first_img.shape[0] * scale_factor
            out_w = first_img.shape[1] * scale_factor

            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-s", f"{out_w}x{out_h}",
                "-pix_fmt", "bgr24",
                "-r", str(frame_rate),
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-crf", "20",
                "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-sar", "1:1",
                "-color_primaries", "bt709",
                "-color_trc", "bt709",
                "-colorspace", "bt709",
                "-movflags", "+faststart",
                str(output_file_upscaled),
            ]
            ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            _upscale_frames_piped(frame_paths, scale_factor, 0, ffmpeg_proc)

            _, ffmpeg_stderr = ffmpeg_proc.communicate()
            if ffmpeg_proc.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Reassembly failed: {ffmpeg_stderr.decode().strip()}")

            print(f"All {total} frames upscaled and encoded.")

        elapsed_time = time.time() - start_time
        if not output_file_upscaled.exists():
            raise HTTPException(status_code=500, detail="Upscaled MP4 video file was not created.")
        print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")

        if output_file_denoised.exists():
            output_file_denoised.unlink()
            print(f"Intermediate file {output_file_denoised} deleted.")

        if output_file_with_extra_frames.exists():
            output_file_with_extra_frames.unlink()
            print(f"Intermediate file {output_file_with_extra_frames} deleted.")

        if input_file.exists():
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
