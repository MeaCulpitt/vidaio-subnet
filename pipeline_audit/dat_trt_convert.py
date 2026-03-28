#!/usr/bin/env python3
"""Export DAT x4 to ONNX (on CPU), then convert to TensorRT FP16 engine."""
import sys, os, time, gc
sys.path.insert(0, '/root/vidaio-subnet')
sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np

MODEL_PATH = os.path.expanduser('~/.cache/span/DAT_x4.safetensors')
ONNX_PATH = '/root/pipeline_audit/dat_x4.onnx'
TRT_PATH = '/root/pipeline_audit/dat_x4_fp16.engine'

# Input: 480x270 (standard 270p for x4→1080p)
H, W = 270, 480
# DAT needs padding to multiple of 64
PAD_H = (64 - H % 64) % 64  # 270 % 64 = 14, pad = 50
PAD_W = (64 - W % 64) % 64  # 480 % 64 = 0, pad = 0
H_PAD, W_PAD = H + PAD_H, W + PAD_W

print(f"Input: {W}x{H}, Padded: {W_PAD}x{H_PAD}, Pad: ({PAD_W},{PAD_H})")
print(f"Output: {W*4}x{H*4}")

# Step 1: Export to ONNX on CPU (GPU tracing uses too much VRAM)
print("\n=== Step 1: ONNX Export (CPU tracing) ===")
from spandrel import ModelLoader

loader = ModelLoader(device="cpu")
descriptor = loader.load_from_file(MODEL_PATH)
model = descriptor.model.eval().float().cpu()
nparams = sum(p.numel() for p in model.parameters()) / 1e6
print(f"DAT: {nparams:.1f}M params loaded on CPU")

dummy = torch.randn(1, 3, H_PAD, W_PAD, dtype=torch.float32)

print(f"Exporting to ONNX ({ONNX_PATH})... (this takes a few minutes on CPU)")
t0 = time.time()
torch.onnx.export(
    model,
    dummy,
    ONNX_PATH,
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None,  # Fixed shape for max TRT optimization
)
export_time = time.time() - t0
print(f"ONNX export: {export_time:.1f}s, size: {os.path.getsize(ONNX_PATH)/1e6:.1f}MB")

del model, dummy, descriptor, loader
gc.collect()

# Step 2: Build TensorRT engine
print("\n=== Step 2: TensorRT FP16 Engine ===")
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

print(f"Parsing ONNX...")
with open(ONNX_PATH, 'rb') as f:
    if not parser.parse(f.read()):
        for i in range(parser.num_errors):
            print(f"  ERROR: {parser.get_error(i)}")
        sys.exit(1)
print(f"ONNX parsed: {network.num_inputs} inputs, {network.num_outputs} outputs, {network.num_layers} layers")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)  # 8GB workspace
config.set_flag(trt.BuilderFlag.FP16)

# Timing cache for faster rebuilds
cache_path = '/root/pipeline_audit/trt_timing_cache.bin'
if os.path.exists(cache_path):
    with open(cache_path, 'rb') as f:
        cache = config.create_timing_cache(f.read())
    config.set_timing_cache(cache, ignore_mismatch=True)
else:
    cache = config.create_timing_cache(b'')
    config.set_timing_cache(cache, ignore_mismatch=True)

print(f"Building TRT engine (FP16, this may take 10-30 minutes)...")
t0 = time.time()
engine_bytes = builder.build_serialized_network(network, config)
build_time = time.time() - t0

if engine_bytes is None:
    print("FAILED to build TRT engine!")
    sys.exit(1)

# Save timing cache
cache = config.get_timing_cache()
with open(cache_path, 'wb') as f:
    f.write(memoryview(cache.serialize()))

with open(TRT_PATH, 'wb') as f:
    f.write(engine_bytes)

print(f"Engine built in {build_time:.0f}s, size: {os.path.getsize(TRT_PATH)/1e6:.1f}MB")
print(f"Saved to {TRT_PATH}")
print("\nDone! Run benchmark_dat_trt.py next.")
