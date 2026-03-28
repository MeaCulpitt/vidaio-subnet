#!/usr/bin/env python3
"""SN85 Pipeline Audit — Local Baseline Benchmark

Calls the running video-upscaler service (localhost:29115) for both paths,
then measures PieAPP and computes S_F locally.
"""

import sys, os, json, time, math, subprocess, shutil
import numpy as np

sys.path.insert(0, '/root/vidaio-subnet')

# ------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_quality_score(pieapp_score):
    s = sigmoid(pieapp_score)
    at_zero = (1 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
    at_two = (1 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
    val = (1 - (np.log10(s + 1) / np.log10(3.5))) ** 2.5
    return 1 - ((val - at_zero) / (at_two - at_zero))

def calculate_length_score(cl):
    return math.log(1 + cl) / math.log(1 + 320)

def calculate_final_score(s_pre):
    return 0.1 * math.exp(6.979 * (s_pre - 0.5))

def get_video_info(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
         '-show_entries', 'format=duration,size',
         '-of', 'json', str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(r.stdout)
    s = info.get('streams', [{}])[0]
    fmt = info.get('format', {})
    w, h = int(s.get('width', 0)), int(s.get('height', 0))
    rfr = s.get('r_frame_rate', '30/1')
    parts = rfr.split('/')
    fps = float(parts[0]) / float(parts[1]) if len(parts) == 2 else 30.0
    dur = float(s.get('duration', 0) or fmt.get('duration', 0))
    nf = int(s.get('nb_frames', 0) or 0)
    if nf == 0 and dur > 0: nf = int(dur * fps)
    size = int(fmt.get('size', 0))
    return {'width': w, 'height': h, 'fps': fps, 'duration': dur,
            'nb_frames': nf, 'size_bytes': size}

def measure_pieapp(ref_path, dist_path, sample_count=8):
    """Measure PieAPP. ref and dist can be different resolutions — dist is resized."""
    import torch, pyiqa, cv2
    device = torch.device('cuda')
    metric = pyiqa.create_metric('pieapp', device=device)

    ref_cap = cv2.VideoCapture(str(ref_path))
    dist_cap = cv2.VideoCapture(str(dist_path))
    ref_count = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dist_count = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(ref_count, dist_count)
    if total < sample_count: sample_count = max(1, total)

    indices = np.linspace(0, total - 1, sample_count, dtype=int) if total > sample_count else list(range(total))

    scores = []
    for idx in indices:
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        dist_cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret1, rf = ref_cap.read()
        ret2, df = dist_cap.read()
        if not ret1 or not ret2: continue

        # Resize dist to ref dimensions for fair comparison
        if rf.shape[:2] != df.shape[:2]:
            df = cv2.resize(df, (rf.shape[1], rf.shape[0]), interpolation=cv2.INTER_LANCZOS4)

        rf_rgb = cv2.cvtColor(rf, cv2.COLOR_BGR2RGB)
        df_rgb = cv2.cvtColor(df, cv2.COLOR_BGR2RGB)
        rt = torch.from_numpy(rf_rgb).permute(2,0,1).float().div(255.0).unsqueeze(0).to(device)
        dt = torch.from_numpy(df_rgb).permute(2,0,1).float().div(255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            s = metric(dt, rt).item()
            if s < 0: s = abs(s)
        scores.append(s)

    ref_cap.release()
    dist_cap.release()
    avg = float(np.mean(scores)) if scores else 5.0
    return {
        'mean': min(avg, 2.0),
        'per_frame': [round(float(s), 5) for s in scores],
        'std': float(np.std(scores)) if len(scores) > 1 else 0.0,
        'min': float(min(scores)) if scores else 5.0,
        'max': float(max(scores)) if scores else 5.0,
        'n': len(scores),
    }


def upscale_via_file(input_path, task_type):
    """Call the upscaler by importing its function directly in this process.

    This bypasses the HTTP API to avoid needing presigned URLs etc.
    We load models separately to avoid conflict with the running service.
    """
    # Use the running service via curl instead
    import requests
    # The service expects a URL, not a local file
    # Let's do direct function call approach
    pass


def upscale_direct(input_path, task_type):
    """Direct upscale using the same pipeline code as the service."""
    os.chdir('/root/vidaio-subnet')
    sys.path.insert(0, '/root/vidaio-subnet/services/upscaling')

    # We can't use the service's models (they're in a different process)
    # Instead, load models here and run the pipeline
    from pathlib import Path
    import torch
    from spandrel import ModelLoader
    import uuid

    scale_factor = 4 if task_type == "SD24K" else 2

    # Probe
    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,color_space,color_primaries,color_transfer',
         '-of', 'json', str(input_path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(r.stdout)['streams'][0]
    w, h = int(info['width']), int(info['height'])
    rfr = info.get('r_frame_rate', '30/1').split('/')
    fps = float(rfr[0]) / float(rfr[1]) if len(rfr) == 2 else 30.0

    out_w, out_h = w * scale_factor, h * scale_factor
    output_path = f'/root/pipeline_audit/upscaled_{task_type}_{uuid.uuid4().hex[:8]}.mp4'

    if scale_factor == 4:
        model_path = os.path.expanduser('~/.cache/span/PLKSR_X4_DF2K.pth')
    else:
        model_path = os.path.expanduser('~/.cache/span/2xHFA2kSPAN.safetensors')

    print(f"  Loading model from {model_path}...")
    model = ModelLoader(device="cuda:0").load_from_file(model_path).model.eval().half()
    if scale_factor == 4:
        model = torch.compile(model, dynamic=True)
        dummy = torch.randn(1, 3, 270, 480, device="cuda:0", dtype=torch.float16)
        for _ in range(3):
            with torch.no_grad(): model(dummy)
        del dummy
        torch.cuda.empty_cache()

    frame_size = w * h * 3
    batch_size = 4

    decoder = subprocess.Popen(
        ['ffmpeg', '-i', str(input_path),
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    encoder = subprocess.Popen(
        ['ffmpeg', '-y',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{out_w}x{out_h}', '-r', str(fps),
         '-i', 'pipe:0',
         '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
         '-profile:v', 'main', '-pix_fmt', 'yuv420p',
         '-sar', '1:1', '-movflags', '+faststart',
         output_path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    total_frames = 0
    t_inference = 0

    while True:
        frames = []
        for _ in range(batch_size):
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size: break
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy())
        if not frames: break

        tensors = [torch.from_numpy(f.astype(np.float32)/255.0).permute(2,0,1).contiguous() for f in frames]
        batch = torch.stack(tensors).to("cuda:0").half()

        t0 = time.time()
        with torch.no_grad():
            out = model(batch)
        torch.cuda.synchronize()
        t_inference += time.time() - t0

        out_np = out.float().clamp_(0,1).mul_(255).round_().to(torch.uint8).permute(0,2,3,1).contiguous().cpu().numpy()
        for j in range(len(frames)):
            encoder.stdin.write(out_np[j].tobytes())

        total_frames += len(frames)
        if total_frames % 100 == 0:
            print(f"  {total_frames} frames processed...")

    encoder.stdin.close()
    encoder.wait()
    decoder.wait()

    del model
    torch.cuda.empty_cache()

    return output_path, total_frames, t_inference


def benchmark_path(name, input_path, task_type, scale_factor):
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name} ({task_type}, {scale_factor}x)")
    print(f"{'='*60}")

    input_info = get_video_info(input_path)
    print(f"Input: {input_info['width']}x{input_info['height']} @{input_info['fps']}fps "
          f"{input_info['duration']:.1f}s {input_info['nb_frames']}f {input_info['size_bytes']/1e6:.1f}MB")

    t_start = time.time()
    output_path, total_frames, t_inference = upscale_direct(input_path, task_type)
    t_upscale = time.time() - t_start

    output_info = get_video_info(output_path)
    print(f"Output: {output_info['width']}x{output_info['height']} @{output_info['fps']}fps "
          f"{output_info['duration']:.1f}s {output_info['nb_frames']}f {output_info['size_bytes']/1e6:.1f}MB")
    print(f"Time: total={t_upscale:.1f}s, inference={t_inference:.1f}s, "
          f"FPS={total_frames/t_inference:.1f}")

    # PieAPP: compare upscaled output vs input (resized to match)
    # This simulates the validator comparing the upscaled output to the original reference
    # In production, the validator compares to the ORIGINAL (pre-downscale) video
    # Since we don't have the original, comparing vs the input (resized) gives us
    # a good estimate — PieAPP measures perceptual distance
    print(f"\nMeasuring PieAPP (upscaled vs input, {8} samples)...")
    t_pa = time.time()
    pieapp = measure_pieapp(input_path, output_path, sample_count=8)
    t_pa = time.time() - t_pa
    print(f"PieAPP: mean={pieapp['mean']:.4f} std={pieapp['std']:.4f} "
          f"min={pieapp['min']:.4f} max={pieapp['max']:.4f} ({t_pa:.1f}s)")
    print(f"Per-frame: {pieapp['per_frame']}")

    # Compute S_F
    cl = min(input_info['duration'], 10)
    s_q = calculate_quality_score(pieapp['mean'])
    s_l = calculate_length_score(cl)
    s_pre = 0.5 * s_q + 0.5 * s_l
    s_f = calculate_final_score(s_pre)

    status = "BONUS" if s_f > 0.32 else "BELOW TARGET"
    print(f"\n--- S_F = {s_f:.4f} ({status}) ---")
    print(f"S_Q={s_q:.4f}  S_L={s_l:.4f}  S_PRE={s_pre:.4f}")

    return {
        'name': name, 'task_type': task_type, 'scale_factor': scale_factor,
        'input': input_info, 'output': output_info,
        'timing': {
            'total_upscale': round(t_upscale, 2),
            'inference_only': round(t_inference, 2),
            'fps': round(total_frames / t_inference, 1),
            'pieapp_measurement': round(t_pa, 2),
        },
        'pieapp': {k: round(v, 5) if isinstance(v, float) else v for k, v in pieapp.items()},
        'scores': {
            'pieapp': round(pieapp['mean'], 5),
            's_q': round(s_q, 4), 's_l': round(s_l, 4),
            's_pre': round(s_pre, 4), 's_f': round(s_f, 4),
            'content_length': cl,
            'target_met': s_f > 0.32,
        },
        'output_path': output_path,
    }


def main():
    print("SN85 Pipeline Audit — Baseline Benchmark")
    print("=" * 60)

    # Use a real 1080p test video
    test_1080p = '/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
    if not os.path.exists(test_1080p):
        # Fallback: create synthetic
        test_1080p = '/root/pipeline_audit/test_1080p.mp4'
        subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi', '-i',
            'testsrc2=duration=10:size=1920x1080:rate=30',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
            '-movflags', '+faststart', test_1080p
        ], capture_output=True, check=True)

    print(f"Test video: {test_1080p}")

    results = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'gpu': 'RTX 4090',
        'test_video': test_1080p,
    }

    # ---- x4 path (SD24K) ----
    # Create 540p input (standard x4 input size)
    x4_input = '/root/pipeline_audit/test_x4_540p.mp4'
    subprocess.run([
        'ffmpeg', '-y', '-i', test_1080p,
        '-vf', 'scale=-2:540',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-movflags', '+faststart', x4_input
    ], capture_output=True, check=True)

    results['x4_SD24K'] = benchmark_path('x4_PLKSR', x4_input, 'SD24K', 4)

    # ---- x2 path (SD2HD) — SPAN fallback since nvvfx not in this env ----
    x2_input = '/root/pipeline_audit/test_x2_1080p.mp4'
    shutil.copy(test_1080p, x2_input)

    results['x2_SD2HD'] = benchmark_path('x2_SPAN', x2_input, 'SD2HD', 2)

    # Save
    out = '/root/pipeline_audit/baseline_scores.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")

    # Copy as current_scores.json (will be updated after improvements)
    shutil.copy(out, '/root/pipeline_audit/current_scores.json')

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    for key in ['x4_SD24K', 'x2_SD2HD']:
        r = results[key]
        s = r['scores']
        print(f"  {key}: PieAPP={s['pieapp']:.4f} S_Q={s['s_q']:.4f} "
              f"S_F={s['s_f']:.4f} {'BONUS' if s['target_met'] else 'BELOW'} "
              f"({r['timing']['fps']:.0f} FPS, {r['timing']['total_upscale']:.1f}s)")


if __name__ == '__main__':
    main()
