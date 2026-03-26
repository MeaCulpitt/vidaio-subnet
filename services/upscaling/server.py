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
import json
from pydantic import BaseModel
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from spandrel import ModelLoader
from services.miner_utilities.redis_utils import schedule_file_deletion
from vidaio_subnet_core.utilities import storage_client, download_video
from loguru import logger
import traceback
import uuid

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


@app.get("/health")
async def health():
    """Health endpoint for pm2 / load-balancer monitoring."""
    gpu_ok = torch.cuda.is_available()
    return {"status": "ok" if gpu_ok else "degraded", "gpu": gpu_ok}


class UpscaleRequest(BaseModel):
    payload_url: str
    task_type: str


# ---------------------------------------------------------------------------
# Model cache — all models loaded via spandrel
# ---------------------------------------------------------------------------
_MODEL_PATHS = {
    2: Path.home() / ".cache" / "span" / "2xHFA2kSPAN.safetensors",
    4: Path.home() / ".cache" / "span" / "PLKSR_X4_DF2K.pth",             # L1 PLKSR — BONUS tier PieAPP, fast
    "4fast": Path.home() / ".cache" / "span" / "realesr-general-x4v3.pth",  # GAN — fast fallback for large inputs
}
_upscalers: dict = {}  # (scale, gpu_id) -> nn.Module
# PLKSR x4: 66ms/f at 480x270, PieAPP=0.07 (BONUS). Handles up to ~1042x584 in budget.
# realesr-x4v3: 5ms/f at 480x270, PieAPP>2.0 (PENALTY). Fast fallback.
_X4_MAX_PIXELS = 650_000  # ~1042x624 — PLKSR handles this in ~59s. Above: GAN fallback.

# ---------------------------------------------------------------------------
# nvidia-vfx VideoSuperRes cache (for GPU-direct x2 path)
# ---------------------------------------------------------------------------
_vsr_cache: dict = {}  # (out_w, out_h, gpu_id) -> nvvfx.VideoSuperRes


def _get_upscaler(scale: int, gpu_id: int) -> torch.nn.Module:
    key = (scale, gpu_id)
    if key in _upscalers:
        return _upscalers[key]

    model_path = _MODEL_PATHS.get(scale)
    if model_path is None or not model_path.exists():
        raise RuntimeError(f"Model for x{scale} not found at {model_path}")

    model_name = model_path.stem
    logger.info(f"Loading {model_name} x{scale} from {model_path} on cuda:{gpu_id} (spandrel)...")
    descriptor = ModelLoader(device=f"cuda:{gpu_id}").load_from_file(str(model_path))
    model = descriptor.model.eval().half()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6

    if scale == "4fast":
        # Only compile the fast/small GAN model (torch.compile hurts RRDB performance)
        logger.info(f"Compiling {model_name} x4fast with torch.compile(dynamic=True)...")
        model = torch.compile(model, dynamic=True)
        dummy = torch.randn(1, 3, 270, 480, device=f"cuda:{gpu_id}", dtype=torch.float16)
        for _ in range(3):
            with torch.no_grad():
                model(dummy)
        del dummy
        torch.cuda.empty_cache()
        logger.info(f"{model_name} x4fast compiled on cuda:{gpu_id} ({nparams:.2f}M params, fp16)")
    else:
        logger.info(f"{model_name} x{scale} loaded on cuda:{gpu_id} ({nparams:.2f}M params, fp16)")

    _upscalers[key] = model
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
    # Load both x4 models: quality (PLKSR) and fast (realesr-x4v3)
    logger.info("Pre-loading x4 quality model (PLKSR) on cuda:0...")
    _get_upscaler(4, 0)
    logger.info("Pre-loading x4 fast model (realesr-x4v3) on cuda:0...")
    _get_upscaler("4fast", 0)
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


def _get_color_metadata(video_path: Path) -> dict:
    """Extract color_space, color_primaries, color_transfer from input video.

    Returns dict with keys: color_space, color_primaries, color_transfer.
    Values are None if not present in the video metadata.
    Used to mirror the input's color tags on the miner's output so the
    validator's hard-fail colorspace comparison passes.
    """
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
         '-show_entries', 'stream=color_space,color_primaries,color_transfer',
         '-of', 'json',
         str(video_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        info = json.loads(result.stdout)
        stream = info.get("streams", [{}])[0]
        meta = {
            'color_space': stream.get('color_space'),
            'color_primaries': stream.get('color_primaries'),
            'color_transfer': stream.get('color_transfer'),
        }
        logger.info(f"Input color metadata: {meta}")
        return meta
    except (json.JSONDecodeError, IndexError):
        logger.warning("Could not parse color metadata, returning empty")
        return {'color_space': None, 'color_primaries': None, 'color_transfer': None}


def _probe_video(source: str) -> dict:
    """Single ffprobe call to get all video metadata. Works with files and URLs."""
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,color_space,color_primaries,color_transfer',
         '-of', 'json', str(source)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        info = json.loads(result.stdout)
        stream = info.get("streams", [{}])[0]
    except (json.JSONDecodeError, IndexError):
        raise RuntimeError(f"ffprobe failed on {str(source)[:120]}: {result.stderr[:300]}")

    rfr = stream.get('r_frame_rate', '30/1')
    parts = rfr.split('/')
    fps = float(parts[0]) / float(parts[1]) if len(parts) == 2 and float(parts[1]) != 0 else 30.0

    meta = {
        'width': int(stream.get('width', 0)),
        'height': int(stream.get('height', 0)),
        'fps': fps,
        'color_space': stream.get('color_space'),
        'color_primaries': stream.get('color_primaries'),
        'color_transfer': stream.get('color_transfer'),
    }
    logger.info(f"Probed: {meta['width']}x{meta['height']} @ {meta['fps']:.1f}fps, "
                f"color: {meta['color_space']}/{meta['color_primaries']}/{meta['color_transfer']}")
    return meta


def _build_color_flags(color_meta: dict) -> list:
    """Build ffmpeg encoder color metadata flags matching the input video.

    Mirrors the input's tags so the validator's colorspace comparison
    (ref vs dist) won't fail. Omits flags for unknown/absent values.
    """
    flags = []
    cp = color_meta.get('color_primaries')
    if cp and cp != 'unknown':
        flags += ['-color_primaries', cp]
    ct = color_meta.get('color_transfer')
    if ct and ct != 'unknown':
        flags += ['-color_trc', ct]
    cs = color_meta.get('color_space')
    if cs and cs != 'unknown':
        flags += ['-colorspace', cs]
    return flags


def _is_bt709(color_meta: dict) -> bool:
    """Determine if the input video uses BT.709 colorspace.

    Returns True only for explicitly BT.709-tagged content.
    Defaults to BT.601 for absent/unknown tags -- this matches ffmpeg's
    default yuv<->rgb matrix for untagged content and avoids the colour
    shift that caused PieAPP=10+ vs PieAPP=3 with correct BT.601.
    """
    cs = (color_meta or {}).get('color_space')
    return cs == 'bt709'  # None / unknown / absent -> BT.601


def _upscale_streaming(input_source: str, output_path: Path, scale_factor: int,
                       frame_rate: float, gpu_id: int = 0, batch_size: int = 4,
                       color_meta: dict = None, width: int = 0, height: int = 0,
                       model_key=None) -> int:
    """Streaming dual-pipe upscale: ffmpeg decode -> model -> ffmpeg encode.

    input_source can be a local file path or an HTTP(S) URL — ffmpeg handles both.
    Decoder outputs rgb24 rawvideo (input is small, ~1.5 MB/frame at 540p).
    For x4: encoder ingests NV12 (12.4 MB/frame vs 24.9 MB RGB24 -- 2x faster pipe).
    For x2: encoder ingests rgb24 (x2 uses nvidia-vfx path normally; this is fallback).
    RGB->NV12 conversion done on GPU via BT.601/709 matrix (matching input) before CPU transfer.
    A ThreadPoolExecutor prefetches the next batch and overlaps pipe writes.
    """
    # rgb24 pipe for quality model (ffmpeg's chroma subsampling is better than our box filter)
    # NV12 pipe for fast model (halves bandwidth, quality doesn't matter for GAN)
    use_nv12 = (model_key == "4fast")
    model = _get_upscaler(model_key or scale_factor, gpu_id)
    device = next(model.parameters()).device
    torch.cuda.empty_cache()

    w, h = width, height
    out_w, out_h = w * scale_factor, h * scale_factor
    frame_size = w * h * 3  # bytes per rgb24 input frame

    # Select BT.601 or BT.709 NV12 conversion based on input colorspace
    if use_nv12:
        bt709 = _is_bt709(color_meta)
        rgb_to_nv12 = _rgb_to_nv12_bt709_gpu if bt709 else _rgb_to_nv12_bt601_gpu
        cs_label = "BT.709" if bt709 else "BT.601"
        logger.info(f"NV12 encode will use {cs_label} matrix")

    decoder = subprocess.Popen(
        ['ffmpeg', '-i', str(input_source),
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    enc_pix_fmt = 'nv12' if use_nv12 else 'rgb24'
    enc_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', enc_pix_fmt,
        '-s', f'{out_w}x{out_h}', '-r', str(frame_rate),
        '-i', 'pipe:0',
        '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
        '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-sar', '1:1',
        *_build_color_flags(color_meta or {}),
        '-movflags', '+faststart',
        str(output_path),
    ]
    encoder = subprocess.Popen(
        enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
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
        pending_write = None
        future = executor.submit(read_batch)
        while True:
            batch_frames = future.result()
            if not batch_frames:
                break

            # Pre-fetch next batch while GPU processes current batch
            next_future = executor.submit(read_batch)

            # RGB HWC uint8 -> RGB NCHW FP16, normalised to [0, 1]
            tensors = [
                torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
                for f in batch_frames
            ]
            batch = torch.stack(tensors).to(device).half()

            batch = batch.contiguous()
            try:
                with torch.no_grad():
                    out_batch = model(batch)  # (N, 3, H*scale, W*scale) FP16 RGB
            except (AssertionError, RuntimeError) as e:
                err_msg = str(e).lower()
                if "stride" in err_msg or "meta" in err_msg:
                    logger.warning(f"torch.compile stride error, falling back to eager: {e}")
                    eager_model = getattr(model, "_orig_mod", model)
                    with torch.no_grad():
                        out_batch = eager_model(batch)
                elif "input size" in err_msg or "kernel size" in err_msg or "out of memory" in err_msg:
                    logger.error(f"Unrecoverable model error on {batch.shape}: {e}")
                    raise
                else:
                    logger.warning(f"Unexpected model error, falling back to eager: {e}")
                    eager_model = getattr(model, "_orig_mod", model)
                    with torch.no_grad():
                        out_batch = eager_model(batch)

            if use_nv12:
                # RGB->NV12 on GPU, async pipe write (halves 4K pipe bandwidth)
                for j in range(len(batch_frames)):
                    rgb_frame = out_batch[j].float().clamp_(0, 1)
                    nv12_frame = rgb_to_nv12(rgb_frame, out_h, out_w)
                    out_bytes = nv12_frame.contiguous().cpu().numpy().tobytes()
                    if pending_write is not None:
                        pending_write.result()
                    pending_write = executor.submit(encoder.stdin.write, out_bytes)
            else:
                # RGB24 path (x2 fallback)
                out_np = (
                    out_batch.float().clamp_(0, 1)
                    .mul_(255.0).round_()
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)   # NCHW -> NHWC
                    .contiguous().cpu().numpy()
                )
                for j in range(len(batch_frames)):
                    if pending_write is not None:
                        pending_write.result()
                    pending_write = executor.submit(
                        encoder.stdin.write, out_np[j].tobytes())

            total_frames += len(batch_frames)
            if total_frames % 50 < batch_size or len(batch_frames) < batch_size:
                logger.info(f"  cuda:{gpu_id}: {total_frames} frames done")

            future = next_future

        if pending_write is not None:
            pending_write.result()

    encoder.stdin.close()      # signal EOF to encoder
    enc_stderr = encoder.stderr.read()  # drain stderr before wait
    encoder.wait()
    decoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError(f"Encoder failed: {enc_stderr.decode()[:500]}")

    return total_frames


def _nv12_to_rgb_bt601_bilinear(nv12: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """NV12 GPU tensor (H*1.5, W) uint8 -> float32 RGB CHW [0,1].

    Uses BT.601 limited-range inverse (correct for H.264 videos with unknown
    colorspace -- ffmpeg defaults to BT.601 for these) and bilinear chroma
    upsampling (vs nearest-neighbour) for better quality at chroma transitions.
    VMAF-verified: this decode path gives mean=95.44, min=94.13.
    """
    Y = nv12[:H].float() - 16.0
    uv = nv12[H:].view(H // 2, W // 2, 2).float()
    uv_up = torch.nn.functional.interpolate(
        uv.permute(2, 0, 1).contiguous().unsqueeze(0), scale_factor=2,
        mode='bilinear', align_corners=False)
    U = uv_up[0, 0] - 128.0
    V = uv_up[0, 1] - 128.0
    R = (1.164 * Y + 1.596 * V).clamp_(0, 255)
    G = (1.164 * Y - 0.392 * U - 0.813 * V).clamp_(0, 255)
    B = (1.164 * Y + 2.017 * U).clamp_(0, 255)
    return torch.stack([R, G, B], 0).div_(255.0)


def _nv12_to_rgb_bt709_bilinear(nv12: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """NV12 GPU tensor (H*1.5, W) uint8 -> float32 RGB CHW [0,1].

    Uses BT.709 limited-range inverse (correct for HD/4K content tagged as
    BT.709). Same bilinear chroma upsampling as the BT.601 variant.
    """
    Y = nv12[:H].float() - 16.0
    uv = nv12[H:].view(H // 2, W // 2, 2).float()
    uv_up = torch.nn.functional.interpolate(
        uv.permute(2, 0, 1).contiguous().unsqueeze(0), scale_factor=2,
        mode='bilinear', align_corners=False)
    U = uv_up[0, 0] - 128.0
    V = uv_up[0, 1] - 128.0
    R = (1.164 * Y + 1.793 * V).clamp_(0, 255)
    G = (1.164 * Y - 0.213 * U - 0.533 * V).clamp_(0, 255)
    B = (1.164 * Y + 2.112 * U).clamp_(0, 255)
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


def _rgb_to_nv12_bt709_gpu(rgb: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """float32 CHW [0,1] -> NV12 CUDA tensor (H*1.5, W) uint8 using BT.709.

    Forward matrix for BT.709 limited-range. Used when the input video has
    BT.709 color metadata to avoid a color shift vs the validator's reference.
    """
    p = rgb.mul(255.0)
    R, G, B = p[0], p[1], p[2]
    Y  = (16.0 + 0.1826 * R + 0.6142 * G + 0.0620 * B).clamp_(16, 235)
    Cb = (128.0 - 0.1007 * R - 0.3386 * G + 0.4392 * B).clamp_(16, 240)
    Cr = (128.0 + 0.4392 * R - 0.3989 * G - 0.0403 * B).clamp_(16, 240)
    Y_u8 = Y.to(torch.uint8)
    Cb_s = Cb.view(H // 2, 2, W // 2, 2).mean(dim=(1, 3)).to(torch.uint8)
    Cr_s = Cr.view(H // 2, 2, W // 2, 2).mean(dim=(1, 3)).to(torch.uint8)
    uv = torch.zeros((H // 2, W), dtype=torch.uint8, device=rgb.device)
    uv[:, 0::2] = Cb_s
    uv[:, 1::2] = Cr_s
    return torch.cat([Y_u8, uv], dim=0)



def _upscale_nvvfx_streaming(input_source: str, output_path: Path,
                              frame_rate: float, gpu_id: int = 0,
                              color_meta: dict = None, width: int = 0, height: int = 0) -> int:
    """nvidia-vfx x2 via ffmpeg URL decode (no download needed).

    ffmpeg decode (rgb24) -> GPU -> nvvfx VSR -> NV12 -> ffmpeg encode.
    Eliminates the download step that was causing 90s+ timeouts.
    """
    w, h = width, height
    out_w, out_h = w * 2, h * 2

    bt709 = _is_bt709(color_meta)
    rgb_to_nv12 = _rgb_to_nv12_bt709_gpu if bt709 else _rgb_to_nv12_bt601_gpu
    cs_label = "BT.709" if bt709 else "BT.601"
    logger.info(f"nvvfx-stream: NV12 encode will use {cs_label} matrix")

    vsr = _get_vsr(out_w, out_h, gpu_id)
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{gpu_id}")

    frame_size = w * h * 3  # bytes per rgb24 input frame

    decoder = subprocess.Popen(
        ['ffmpeg', '-i', str(input_source),
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    enc_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'nv12',
        '-s', f'{out_w}x{out_h}', '-r', str(frame_rate),
        '-i', 'pipe:0',
        '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
        '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-sar', '1:1',
        *_build_color_flags(color_meta or {}),
        '-movflags', '+faststart',
        str(output_path),
    ]
    encoder = subprocess.Popen(
        enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
    )

    total_frames = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending_write = None
        while True:
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size:
                break

            # rgb24 HWC uint8 -> float32 CHW [0,1] on GPU
            frame_np = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            rgb = (
                torch.from_numpy(frame_np.astype(np.float32) / 255.0)
                .permute(2, 0, 1).contiguous()
                .to(device)
            )

            result = vsr.run(rgb)
            out_gpu = torch.from_dlpack(result.image).clone()

            out_bytes = (
                rgb_to_nv12(out_gpu.clamp_(0, 1), out_h, out_w)
                .contiguous().cpu().numpy().tobytes()
            )
            if pending_write is not None:
                pending_write.result()
            pending_write = pool.submit(encoder.stdin.write, out_bytes)

            total_frames += 1
            if total_frames % 50 == 0:
                logger.info(f"  nvvfx-stream cuda:{gpu_id}: {total_frames} frames done")

        if pending_write is not None:
            pending_write.result()

    encoder.stdin.close()
    enc_stderr = encoder.stderr.read()
    encoder.wait()
    decoder.wait()

    if encoder.returncode != 0:
        raise RuntimeError(f"Encoder failed: {enc_stderr.decode()[:500]}")

    return total_frames


def _upscale_nvvfx(input_path: Path, output_path: Path,
                   frame_rate: float, gpu_id: int = 0,
                   color_meta: dict = None, width: int = 0, height: int = 0) -> int:
    """nvidia-vfx x2 upscale: pynvc NV12 HW decode -> nvidia-vfx -> ffmpeg encode.

    Decodes via PyNvVideoCodec hardware decoder (NV12, GPU memory) with
    BT.601/709 bilinear colour conversion on GPU (matching input colorspace),
    then nvidia-vfx VideoSuperRes (TensorRT-accelerated), then ffmpeg
    hevc_nvenc pipe encode with input-matching color metadata tags.
    """
    w, h = width, height
    out_w, out_h = w * 2, h * 2

    # Select BT.601 or BT.709 conversion based on input colorspace
    bt709 = _is_bt709(color_meta)
    nv12_to_rgb = _nv12_to_rgb_bt709_bilinear if bt709 else _nv12_to_rgb_bt601_bilinear
    rgb_to_nv12 = _rgb_to_nv12_bt709_gpu if bt709 else _rgb_to_nv12_bt601_gpu
    cs_label = "BT.709" if bt709 else "BT.601"
    logger.info(f"nvvfx: NV12 decode/encode will use {cs_label} matrix")

    vsr = _get_vsr(out_w, out_h, gpu_id)
    torch.cuda.empty_cache()

    dec = pynvc.CreateSimpleDecoder(
        str(input_path), gpuid=gpu_id,
        useDeviceMemory=True,
        outputColorType=pynvc.OutputColorType.NATIVE,
        decoderCacheSize=1
    )

    enc_cmd = [
        'ffmpeg', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'nv12',
        '-s', f'{out_w}x{out_h}', '-r', str(frame_rate),
        '-i', 'pipe:0',
        '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
        '-profile:v', 'main', '-pix_fmt', 'yuv420p',
        '-sar', '1:1',
        *_build_color_flags(color_meta or {}),
        '-movflags', '+faststart',
        str(output_path),
    ]
    encoder = subprocess.Popen(
        enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE
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
                rgb = nv12_to_rgb(nv12, h, w)
                result = vsr.run(rgb)
                out_gpu = torch.from_dlpack(result.image).clone()
                out_bytes = (
                    rgb_to_nv12(out_gpu.clamp_(0, 1), out_h, out_w)
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


def _downscale_to_540p(input_source: str, frame_rate: float,
                       width: int, height: int) -> tuple:
    """Downscale video to 540p for x4 EfRLFN input. Accepts file path or URL.

    Returns (output_source, new_w, new_h, created_local_file).
    When input is already ≤540p, returns it unchanged.
    When input is a URL, ffmpeg downloads+downscales in one pass.
    """
    if height <= 540:
        logger.info(f"Input already ≤540p ({width}x{height}), skipping downscale")
        return input_source, width, height, False

    logger.info(f"Pre-downscaling {width}x{height} → 540p for x4 EfRLFN (avoids 8K output)")
    downscaled = Path("tmp") / f"{uuid.uuid4()}_540p.mp4"
    Path("tmp").mkdir(exist_ok=True)
    cmd = [
        'ffmpeg', '-y', '-i', str(input_source),
        '-vf', 'scale=-2:540',
        '-c:v', 'hevc_nvenc', '-cq', '18', '-preset', 'p4',
        '-pix_fmt', 'yuv420p', '-r', str(frame_rate),
        '-movflags', '+faststart',
        str(downscaled)
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(f"Downscale failed: {result.stderr.decode()[:500]}")

    new_w, new_h = _get_video_dimensions(str(downscaled))
    logger.info(f"Downscaled to {new_w}x{new_h} → x4 output will be {new_w*4}x{new_h*4}")
    return str(downscaled), new_w, new_h, True


def upscale_video(input_source: str, task_type: str, is_url: bool = False):
    """Upscale a video from a local file path or URL.

    For x4 (SD24K): input_source can be a URL (ffmpeg reads directly, skips download).
    For x2 (SD2HD): input_source must be a local file (pynvc HW decoder needs file).
    """
    try:
        scale_factor = 4 if task_type == "SD24K" else 2

        if not is_url:
            input_file = Path(input_source)
            if not input_file.exists() or not input_file.is_file():
                raise HTTPException(status_code=400, detail="Input file does not exist or is not a valid file.")

        # Single ffprobe call for all metadata (works on both files and URLs)
        t_probe = time.time()
        probe = _probe_video(input_source)
        frame_rate = probe['fps']
        w, h = probe['width'], probe['height']
        color_meta = {
            'color_space': probe['color_space'],
            'color_primaries': probe['color_primaries'],
            'color_transfer': probe['color_transfer'],
        }
        logger.info(f"Probe took {time.time()-t_probe:.2f}s")
        print(f"Frame rate detected: {frame_rate} fps")

        # Reject degenerate inputs (0x0, or too small for conv kernels)
        _MIN_DIM = 8  # PLKSR's 3x3 convs need at least 8px after feature extraction
        if w < _MIN_DIM or h < _MIN_DIM:
            raise HTTPException(
                status_code=400,
                detail=f"Input too small: {w}x{h} (minimum {_MIN_DIM}x{_MIN_DIM})"
            )

        # Generate output path with UUID (works for both URL and file inputs)
        output_file_upscaled = Path("tmp") / f"{uuid.uuid4()}_upscaled.mp4"
        Path("tmp").mkdir(exist_ok=True)
        cleanup_files = []  # local intermediates to delete

        # Pass input directly to upscaler (no downscale — validator expects output = input × scale_factor)
        upscale_source, upscale_w, upscale_h = input_source, w, h

        # x2 path: use nvidia-vfx if available (2-3x faster), else SPAN fallback
        if scale_factor == 2 and _GPU_DIRECT_AVAILABLE:
            try:
                print(f"Step 2: Upscaling with nvidia-vfx x2 on cuda:0...")
                start_time = time.time()
                if is_url:
                    total = _upscale_nvvfx_streaming(
                        upscale_source, output_file_upscaled, frame_rate, gpu_id=0,
                        color_meta=color_meta, width=upscale_w, height=upscale_h
                    )
                else:
                    total = _upscale_nvvfx(
                        Path(upscale_source), output_file_upscaled, frame_rate, gpu_id=0,
                        color_meta=color_meta, width=upscale_w, height=upscale_h
                    )
                print(f"All {total} frames upscaled and encoded (nvidia-vfx).")
                elapsed_time = time.time() - start_time
                print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")
            except Exception as e:
                logger.warning(f"nvidia-vfx failed ({e}), falling back to SPAN x2...")
                print(f"Step 2: Falling back to SPAN x{scale_factor} on cuda:0 (streaming dual-pipe)...")
                start_time = time.time()
                total = _upscale_streaming(
                    upscale_source, output_file_upscaled,
                    scale_factor, frame_rate, gpu_id=0, batch_size=4,
                    color_meta=color_meta, width=upscale_w, height=upscale_h
                )
                print(f"All {total} frames upscaled and encoded (SPAN fallback).")
                elapsed_time = time.time() - start_time
                print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")
        else:
            x4_key = ("4fast" if upscale_w * upscale_h > _X4_MAX_PIXELS else 4) if scale_factor == 4 else None
            x4_label = "PLKSR" if (x4_key == 4) else ("realesr-x4v3" if x4_key == "4fast" else "SPAN")
            print(f"Step 2: Upscaling with {x4_label} x{scale_factor} on cuda:0 (streaming dual-pipe)...")
            logger.info(f"x4 model selection: {upscale_w}x{upscale_h} = {upscale_w*upscale_h} pixels, model_key={x4_key}")
            start_time = time.time()
            total = _upscale_streaming(
                upscale_source, output_file_upscaled,
                scale_factor, frame_rate, gpu_id=0, batch_size=4,
                color_meta=color_meta, width=upscale_w, height=upscale_h,
                model_key=x4_key
            )
            print(f"All {total} frames upscaled and encoded.")
            elapsed_time = time.time() - start_time
            print(f"Step 2 completed in {elapsed_time:.2f} seconds. Upscaled MP4 file: {output_file_upscaled}")

        if not output_file_upscaled.exists():
            raise HTTPException(status_code=500, detail="Upscaled MP4 video file was not created.")

        # Clean up intermediates
        for f in cleanup_files:
            if f.exists():
                f.unlink()
                print(f"Intermediate {f} deleted.")

        # Clean up input file only if it's a local file (not a URL)
        if not is_url:
            p = Path(input_source)
            if p.exists():
                p.unlink()
                print(f"Original file {p} deleted.")

        print(f"Returning from FastAPI: {output_file_upscaled}")
        return output_file_upscaled

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/upscale-video")
async def video_upscaler(request: UpscaleRequest):
    try:
        payload_url = request.payload_url
        task_type = request.task_type

        if not payload_url or not payload_url.startswith(("http://", "https://")):
            logger.error(f"Invalid payload URL (missing protocol): {payload_url!r}")
            return {"uploaded_video_url": None}

        scale_factor = 4 if task_type == "SD24K" else 2
        t_total = time.time()

        # Both x4 and x2 now stream from URL directly (no download)
        logger.info(f"⚡ {'x4' if scale_factor == 4 else 'x2'} path: passing URL directly to ffmpeg (no download)")
        processed_video_path = upscale_video(payload_url, task_type, is_url=True)

        processed_video_name = Path(processed_video_path).name
        logger.info(f"Processed video path: {processed_video_path}, video name: {processed_video_name}")

        if processed_video_path is not None:
            object_name: str = processed_video_name

            t_upload = time.time()
            await storage_client.upload_file(object_name, processed_video_path)
            logger.info(f"Upload took {time.time()-t_upload:.2f}s")

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

            logger.info(f"Total e2e: {time.time()-t_total:.2f}s | link: {sharing_link}")

            return {"uploaded_video_url": sharing_link}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to process upscaling request: {e}")
        traceback.print_exc()
        return {"uploaded_video_url": None}
    finally:
        torch.cuda.empty_cache()


if __name__ == "__main__":

    import uvicorn

    host = CONFIG.video_upscaler.host
    port = CONFIG.video_upscaler.port

    uvicorn.run(app, host=host, port=port)
