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

app = FastAPI()

class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str


# ---------------------------------------------------------------------------
# Model cache — SPAN models loaded via spandrel, keyed by (scale, gpu_id)
# ---------------------------------------------------------------------------
_SPAN_MODELS = {
    2: Path.home() / ".cache" / "span" / "2xNomosUni_span.safetensors",
    4: Path.home() / ".cache" / "span" / "4xNomosUni_span.safetensors",
}
_upscalers: dict = {}  # (scale, gpu_id) -> nn.Module (SPAN model)


def _get_upscaler(scale: int, gpu_id: int) -> torch.nn.Module:
    key = (scale, gpu_id)
    if key in _upscalers:
        return _upscalers[key]

    model_path = _SPAN_MODELS[scale]
    if not model_path.exists():
        raise RuntimeError(f"SPAN x{scale} model not found at {model_path}")

    logger.info(f"Loading SPAN x{scale} model from {model_path} on cuda:{gpu_id}...")
    descriptor = ModelLoader(device=f"cuda:{gpu_id}").load_from_file(str(model_path))
    model = descriptor.model.eval().half()
    _upscalers[key] = model
    logger.info(f"SPAN x{scale} loaded on cuda:{gpu_id}")
    return model


@app.on_event("startup")
async def preload_models():
    logger.info("Pre-loading SPAN models on cuda:0...")
    for scale in (2, 4):
        _get_upscaler(scale, 0)
    logger.info("All SPAN models loaded on cuda:0.")


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


def upscale_video(payload_video_path: str, task_type: str):
    try:
        input_file = Path(payload_video_path)
        scale_factor = 4 if task_type == "SD24K" else 2

        if not input_file.exists() or not input_file.is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")

        frame_rate = get_frame_rate(input_file)
        print(f"Frame rate detected: {frame_rate} fps")

        output_file_upscaled = input_file.with_name(f"{input_file.stem}_upscaled.mp4")

        # Step 2: Upscale using SPAN streaming dual-pipe (no temp JPEG files)
        # tpad (frame duplication) removed — SPAN preserves all input frames without it.
        print(f"Step 2: Upscaling with SPAN x{scale_factor} on cuda:0 (streaming dual-pipe)...")
        start_time = time.time()

        total = _upscale_streaming(
            input_file, output_file_upscaled,
            scale_factor, frame_rate, gpu_id=0, batch_size=4
        )
        print(f"All {total} frames upscaled and encoded.")

        elapsed_time = time.time() - start_time
        if not output_file_upscaled.exists():
            raise HTTPException(status_code=500, detail="Upscaled MP4 video file was not created.")
        print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")

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
