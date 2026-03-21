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
from vidaio_subnet_core import CONFIG
import re
from pydantic import BaseModel
from typing import Optional
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
# Model cache — initialised once at startup, reused across requests
# ---------------------------------------------------------------------------
_MODEL_DIR = Path.home() / ".cache" / "realesrgan"
_MODEL_URLS = {
    2: ("RealESRGAN_x2plus.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"),
    4: ("RealESRGAN_x4plus.pth", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"),
}
_upscalers: dict = {}


def _get_upscaler(scale: int) -> RealESRGANer:
    if scale in _upscalers:
        return _upscalers[scale]

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
        tile=512,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device="cuda:0",
    )
    _upscalers[scale] = upscaler
    logger.info(f"RealESRGANer x{scale} loaded on cuda:0")
    return upscaler


@app.on_event("startup")
async def preload_models():
    logger.info("Pre-loading RealESRGAN models...")
    _get_upscaler(2)
    _get_upscaler(4)
    logger.info("All models loaded.")


def get_frame_rate(input_file: Path) -> float:
    """
    Extracts the frame rate of the input video using FFmpeg.

    Args:
        input_file (Path): The path to the video file.

    Returns:
        float: The frame rate of the video.
    """
    frame_rate_command = [
        "ffmpeg",
        "-i", str(input_file),
        "-hide_banner"
    ]
    process = subprocess.run(frame_rate_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    output = process.stderr  # Frame rate is usually in stderr

    # Extract frame rate using regex
    match = re.search(r"(\d+(?:\.\d+)?) fps", output)
    if match:
        return float(match.group(1))
    else:
        raise HTTPException(status_code=500, detail="Unable to determine frame rate of the video.")


def upscale_video(payload_video_path: str, task_type: str):
    """
    Upscales a video using native PyTorch RealESRGAN and returns the full path of the upscaled video.

    Args:
        payload_video_path (str): The path to the video to upscale.
        task_type (str): The type of upscaling task to perform.

    Returns:
        str: The full path to the upscaled video.
    """
    try:
        input_file = Path(payload_video_path)
        scale_factor = 4 if task_type == "SD24K" else 2

        # Validate input file
        if not input_file.exists() or not input_file.is_file():
            raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")

        # Get the frame rate of the video
        frame_rate = get_frame_rate(input_file)
        print(f"Frame rate detected: {frame_rate} fps")

        # Calculate the duration to duplicate 2 frames
        stop_duration = 2 / frame_rate

        # Generate output file paths
        output_file_with_extra_frames = input_file.with_name(f"{input_file.stem}_extra_frames.mp4")
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

        # Step 2: Upscale using native PyTorch RealESRGAN
        print(f"Step 2: Upscaling with RealESRGAN x{scale_factor} on cuda:0...")
        start_time = time.time()

        with tempfile.TemporaryDirectory() as tmp_dir:
            frames_in = Path(tmp_dir) / "frames_in"
            frames_out = Path(tmp_dir) / "frames_out"
            frames_in.mkdir()
            frames_out.mkdir()

            # Extract frames as PNG
            extract_result = subprocess.run(
                ["ffmpeg", "-i", str(output_file_with_extra_frames), str(frames_in / "%08d.png")],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if extract_result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Frame extraction failed: {extract_result.stderr.strip()}")

            frame_paths = sorted(frames_in.glob("*.png"))
            print(f"Extracted {len(frame_paths)} frames, upscaling...")

            upscaler = _get_upscaler(scale_factor)
            for i, frame_path in enumerate(frame_paths):
                img = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
                output, _ = upscaler.enhance(img, outscale=scale_factor)
                cv2.imwrite(str(frames_out / frame_path.name), output)
                if (i + 1) % 50 == 0:
                    print(f"  {i + 1}/{len(frame_paths)} frames done")

            print(f"All {len(frame_paths)} frames upscaled, reassembling...")

            # Reassemble with same encoder settings as original video2x pipeline
            reassemble_result = subprocess.run(
                [
                    "ffmpeg",
                    "-framerate", str(frame_rate),
                    "-i", str(frames_out / "%08d.png"),
                    "-c:v", "libx265",
                    "-crf", "20",
                    "-preset", "slow",
                    "-profile:v", "main",
                    "-pix_fmt", "yuv420p",
                    "-sar", "1:1",
                    "-color_primaries", "bt709",
                    "-color_trc", "bt709",
                    "-colorspace", "bt709",
                    "-movflags", "+faststart",
                    str(output_file_upscaled),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            if reassemble_result.returncode != 0:
                raise HTTPException(status_code=500, detail=f"Reassembly failed: {reassemble_result.stderr.strip()}")

        elapsed_time = time.time() - start_time
        if not output_file_upscaled.exists():
            raise HTTPException(status_code=500, detail="Upscaled MP4 video file was not created.")
        print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")

        # Cleanup intermediate files
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
            
            # Delete the local file since we've already uploaded it to MinIO
            if os.path.exists(processed_video_path):
                os.remove(processed_video_path)
                logger.info(f"{processed_video_path} has been deleted.")
            else:
                logger.info(f"{processed_video_path} does not exist.")
                
            sharing_link: str | None = await storage_client.get_presigned_url(object_name)
            if not sharing_link:
                logger.error("Upload failed")
                return {"uploaded_video_url": None}
            
            # Schedule the file for deletion after 10 minutes (600 seconds)
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
