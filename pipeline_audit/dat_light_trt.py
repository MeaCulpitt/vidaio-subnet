#!/usr/bin/env python3
"""Export DAT-light x4 to ONNX → TRT FP16, then benchmark."""
import sys, os, time, gc
sys.path.insert(0, '/root/vidaio-subnet')
sys.stdout.reconfigure(line_buffering=True)

import torch
import numpy as np

MODEL_PATH = os.path.expanduser('~/.cache/span/DAT_light_x4.pth')
ONNX_PATH = '/root/pipeline_audit/dat_light_x4.onnx'
TRT_PATH = '/root/pipeline_audit/dat_light_x4_fp16.engine'

H, W = 270, 480
PAD_H = (64 - H % 64) % 64
PAD_W = (64 - W % 64) % 64
H_PAD, W_PAD = H + PAD_H, W + PAD_W

print(f"Input: {W}x{H}, Padded: {W_PAD}x{H_PAD}")

# Step 1: ONNX export (GPU — DAT-light is small enough)
print("\n=== Step 1: ONNX Export ===")
from spandrel import ModelLoader
model = ModelLoader(device="cuda:0").load_from_file(MODEL_PATH).model.eval().float().cuda()
nparams = sum(p.numel() for p in model.parameters()) / 1e6
print(f"DAT-light: {nparams:.2f}M params")

dummy = torch.randn(1, 3, H_PAD, W_PAD, device='cuda', dtype=torch.float32)
with torch.no_grad():
    out = model(dummy)
    print(f"Output shape: {list(out.shape)}")

print(f"Exporting ONNX (legacy JIT tracer)...")
t0 = time.time()
# Force legacy JIT-based exporter (torch 2.11 dynamo exporter has bugs with DAT)
torch.onnx.utils.export(
    model, dummy, ONNX_PATH,
    opset_version=17,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes=None,
)
print(f"ONNX: {time.time()-t0:.1f}s, {os.path.getsize(ONNX_PATH)/1e6:.1f}MB")
del model, dummy; torch.cuda.empty_cache(); gc.collect()

# Step 2: TRT FP16 engine
print("\n=== Step 2: TRT FP16 Engine ===")
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(TRT_LOGGER)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, TRT_LOGGER)

with open(ONNX_PATH, 'rb') as f:
    ok = parser.parse(f.read())
    if not ok:
        for i in range(parser.num_errors):
            print(f"  ERROR: {parser.get_error(i)}")
        sys.exit(1)
print(f"Parsed: {network.num_layers} layers")

config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)
config.set_flag(trt.BuilderFlag.FP16)

print(f"Building TRT engine...")
t0 = time.time()
engine_bytes = builder.build_serialized_network(network, config)
if engine_bytes is None:
    print("FAILED!"); sys.exit(1)
with open(TRT_PATH, 'wb') as f:
    f.write(engine_bytes)
print(f"Built in {time.time()-t0:.0f}s, {os.path.getsize(TRT_PATH)/1e6:.1f}MB")

# Step 3: Benchmark TRT vs PyTorch
print("\n=== Step 3: Benchmark ===")

# TRT inference
runtime = trt.Runtime(TRT_LOGGER)
engine = runtime.deserialize_cuda_engine(engine_bytes)
context = engine.create_execution_context()
in_name = engine.get_tensor_name(0)
out_name = engine.get_tensor_name(1)
in_shape = engine.get_tensor_shape(in_name)
out_shape = engine.get_tensor_shape(out_name)
print(f"TRT: {in_name}{list(in_shape)} → {out_name}{list(out_shape)}")

in_buf = torch.empty(list(in_shape), dtype=torch.float32, device='cuda')
out_buf = torch.empty(list(out_shape), dtype=torch.float32, device='cuda')
context.set_tensor_address(in_name, in_buf.data_ptr())
context.set_tensor_address(out_name, out_buf.data_ptr())
stream = torch.cuda.Stream()

def trt_infer(x):
    in_buf.copy_(x)
    with torch.cuda.stream(stream):
        context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()
    return out_buf

# Warmup TRT
dummy = torch.randn(1, 3, H_PAD, W_PAD, device='cuda', dtype=torch.float32)
for _ in range(10):
    _ = trt_infer(dummy)

# Benchmark TRT
times_trt = []
for _ in range(50):
    t0 = time.time()
    _ = trt_infer(dummy)
    times_trt.append(time.time() - t0)
fps_trt = 1.0 / np.mean(times_trt)
print(f"TRT FP16: {fps_trt:.1f} FPS ({np.mean(times_trt)*1000:.1f} ms/frame)")

# Benchmark PyTorch FP16 eager for comparison
model = ModelLoader(device="cuda:0").load_from_file(MODEL_PATH).model.eval().half().cuda()
dummy_fp16 = torch.randn(1, 3, H_PAD, W_PAD, device='cuda', dtype=torch.float16)
with torch.no_grad():
    for _ in range(10): _ = model(dummy_fp16); torch.cuda.synchronize()
times_pt = []
with torch.no_grad():
    for _ in range(50):
        t0 = time.time()
        _ = model(dummy_fp16)
        torch.cuda.synchronize()
        times_pt.append(time.time() - t0)
fps_pt = 1.0 / np.mean(times_pt)
print(f"PyTorch FP16: {fps_pt:.1f} FPS ({np.mean(times_pt)*1000:.1f} ms/frame)")

# Numerical comparison
with torch.no_grad():
    out_pt = model(dummy_fp16).float()
out_trt_f = trt_infer(dummy.float()).clone()
diff = (out_trt_f - out_pt).abs()
print(f"Max diff: {diff.max().item():.6f}, Mean: {diff.mean().item():.6f}")

print(f"\nSpeedup: {fps_trt/fps_pt:.2f}x")
ext_trt = 300 / fps_trt; ext_pt = 300 / fps_pt
print(f"300 frames: TRT={ext_trt:.0f}s  PyTorch={ext_pt:.0f}s")
print(f"Target: ≥15 FPS (300f in <20s)")
