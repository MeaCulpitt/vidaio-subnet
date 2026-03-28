#!/usr/bin/env python3
"""Quick smoke test of the two-stage pipeline."""
import sys, os, time
sys.path.insert(0, '/root/vidaio-subnet')
sys.stdout.reconfigure(line_buffering=True)

# Must run with venv for nvvfx
import torch
import numpy as np

# Test the two-stage function directly
from services.upscaling.server import _upscale_two_stage, _get_upscaler, _get_vsr
from pathlib import Path

input_path = '/root/pipeline_audit/test_960x540_5f.mp4'
output_path = Path('/root/pipeline_audit/test_two_stage_output.mp4')

print("Testing two-stage pipeline: 960x540 → 3840x2160 (5 frames)")
t0 = time.time()
total = _upscale_two_stage(
    input_path, output_path,
    frame_rate=30.0, gpu_id=0,
    color_meta={'color_space': None, 'color_primaries': None, 'color_transfer': None},
    width=960, height=540
)
elapsed = time.time() - t0
print(f"Done: {total} frames in {elapsed:.2f}s ({total/elapsed:.1f} FPS)")

# Verify output dimensions
import subprocess, json
r = subprocess.run(
    ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
     '-show_entries', 'stream=width,height,nb_frames', '-of', 'json',
     str(output_path)],
    stdout=subprocess.PIPE, text=True)
info = json.loads(r.stdout)['streams'][0]
print(f"Output: {info['width']}x{info['height']}, frames: {info.get('nb_frames', '?')}")
expected = "3840x2160"
actual = f"{info['width']}x{info['height']}"
print(f"Resolution match: {actual} == {expected}: {actual == expected}")

if output_path.exists():
    size_mb = output_path.stat().st_size / 1e6
    print(f"File size: {size_mb:.1f} MB")
