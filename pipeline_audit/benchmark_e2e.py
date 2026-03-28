#!/usr/bin/env python3
"""SN85 Pipeline Audit — Baseline E2E Benchmark

Tests both x2 and x4 upscaling paths locally:
1. Uses an existing 1080p test video from tmp/
2. Runs upscaling through the actual pipeline functions
3. Measures: total time, per-step times, PieAPP, VMAF
4. Computes S_Q, S_L, S_PRE, S_F
5. Saves results to baseline_scores.json
"""

import sys
import os
import json
import time
import math
import subprocess
import tempfile
import numpy as np

# Add vidaio-subnet to path
sys.path.insert(0, '/root/vidaio-subnet')
sys.path.insert(0, '/root/vidaio-subnet/services/scoring')

# ------------------------------------------------------------------
# Scoring functions (copied from server.py to avoid import issues)
# ------------------------------------------------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_quality_score(pieapp_score):
    s = sigmoid(pieapp_score)
    at_zero = (1 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
    at_two = (1 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
    val = (1 - (np.log10(s + 1) / np.log10(3.5))) ** 2.5
    return 1 - ((val - at_zero) / (at_two - at_zero))

def calculate_length_score(content_length):
    return math.log(1 + content_length) / math.log(1 + 320)

def calculate_final_score(s_pre):
    return 0.1 * math.exp(6.979 * (s_pre - 0.5))


def get_video_info(path):
    """Get video dimensions, fps, duration, frame count via ffprobe."""
    result = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
         '-show_entries', 'format=duration',
         '-of', 'json', str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    info = json.loads(result.stdout)
    stream = info.get('streams', [{}])[0]
    fmt = info.get('format', {})

    w = int(stream.get('width', 0))
    h = int(stream.get('height', 0))
    rfr = stream.get('r_frame_rate', '30/1')
    parts = rfr.split('/')
    fps = float(parts[0]) / float(parts[1]) if len(parts) == 2 else 30.0

    duration = float(stream.get('duration', 0) or fmt.get('duration', 0))
    nb_frames = int(stream.get('nb_frames', 0) or 0)
    if nb_frames == 0 and duration > 0:
        nb_frames = int(duration * fps)

    return {'width': w, 'height': h, 'fps': fps, 'duration': duration,
            'nb_frames': nb_frames}


def measure_pieapp(ref_path, dist_path, sample_count=4):
    """Measure PieAPP between reference and distorted videos."""
    import torch
    import pyiqa
    import cv2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    metric = pyiqa.create_metric('pieapp', device=device)

    ref_cap = cv2.VideoCapture(str(ref_path))
    dist_cap = cv2.VideoCapture(str(dist_path))

    ref_count = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dist_count = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(ref_count, dist_count)

    if total < sample_count:
        sample_count = total

    # Sample frames evenly across the video
    if total > sample_count:
        indices = np.linspace(0, total - 1, sample_count, dtype=int)
    else:
        indices = list(range(total))

    scores = []
    for idx in indices:
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        dist_cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

        ret1, ref_frame = ref_cap.read()
        ret2, dist_frame = dist_cap.read()

        if not ret1 or not ret2:
            continue

        # Resize dist to match ref if dimensions differ (for fair comparison)
        if ref_frame.shape != dist_frame.shape:
            dist_frame = cv2.resize(dist_frame, (ref_frame.shape[1], ref_frame.shape[0]),
                                     interpolation=cv2.INTER_LANCZOS4)

        ref_rgb = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2RGB)
        dist_rgb = cv2.cvtColor(dist_frame, cv2.COLOR_BGR2RGB)

        ref_t = torch.from_numpy(ref_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        dist_t = torch.from_numpy(dist_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)

        with torch.no_grad():
            score = metric(dist_t, ref_t).item()
            if score < 0:
                score = abs(score)
        scores.append(score)

    ref_cap.release()
    dist_cap.release()

    avg = np.mean(scores) if scores else 5.0
    return {
        'mean': float(min(avg, 2.0)),
        'per_frame': [float(s) for s in scores],
        'std': float(np.std(scores)) if len(scores) > 1 else 0.0,
        'sample_count': len(scores),
    }


def measure_vmaf(ref_path, dist_path):
    """Measure VMAF using ffmpeg libvmaf filter."""
    # Try ffmpeg with libvmaf
    cmd = [
        'ffmpeg', '-i', str(dist_path), '-i', str(ref_path),
        '-lavfi', f'[0:v]scale={{}},{{}}[dist];[1:v][dist]libvmaf=log_fmt=json:log_path=/dev/stdout',
        '-f', 'null', '-'
    ]

    # Simpler approach: use ffmpeg with libvmaf, scale dist to ref size
    ref_info = get_video_info(ref_path)
    dist_info = get_video_info(dist_path)

    # If dimensions differ, scale dist down to ref for VMAF comparison
    if ref_info['width'] != dist_info['width'] or ref_info['height'] != dist_info['height']:
        scale_filter = f"[0:v]scale={ref_info['width']}:{ref_info['height']}:flags=bicubic[dist];[1:v][dist]libvmaf=log_fmt=json:log_path=/dev/stderr"
    else:
        scale_filter = "[0:v][1:v]libvmaf=log_fmt=json:log_path=/dev/stderr"

    cmd = [
        'ffmpeg', '-i', str(dist_path), '-i', str(ref_path),
        '-lavfi', scale_filter,
        '-f', 'null', '-'
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        # Parse VMAF from stderr (json log)
        stderr = result.stderr
        # Find the JSON portion
        json_start = stderr.rfind('{"version"')
        if json_start >= 0:
            vmaf_data = json.loads(stderr[json_start:])
            vmaf_mean = vmaf_data.get('pooled_metrics', {}).get('vmaf', {}).get('mean', 0)
            vmaf_min = vmaf_data.get('pooled_metrics', {}).get('vmaf', {}).get('min', 0)
            return {'mean': vmaf_mean, 'min': vmaf_min}
    except Exception as e:
        print(f"VMAF measurement failed: {e}")

    return {'mean': 0, 'min': 0, 'error': 'ffmpeg libvmaf not available'}


def find_test_video():
    """Find a suitable 1080p test video."""
    tmp_dir = '/root/vidaio-subnet/tmp'
    candidates = []
    for f in os.listdir(tmp_dir):
        if not f.endswith('.mp4'):
            continue
        # Skip upscaled/denoised files
        if '_upscaled' in f or '_denoised' in f or '_extra' in f or '_540p' in f:
            continue
        path = os.path.join(tmp_dir, f)
        info = get_video_info(path)
        if info['width'] >= 960 and info['height'] >= 540 and info['duration'] >= 3:
            candidates.append((path, info))

    if not candidates:
        print("No suitable test videos found in tmp/")
        return None, None

    # Prefer 1080p
    for path, info in candidates:
        if info['height'] == 1080:
            return path, info

    return candidates[0]


def create_540p_test_video(source_path):
    """Create a 540p version for x4 testing."""
    out_path = '/root/pipeline_audit/test_input_540p.mp4'
    cmd = [
        'ffmpeg', '-y', '-i', source_path,
        '-vf', 'scale=-2:540',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
        '-t', '10',  # limit to 10s
        '-movflags', '+faststart',
        out_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return out_path


def run_upscale(input_path, scale_factor, task_type):
    """Run upscaling via the server's upscale_video function."""
    os.chdir('/root/vidaio-subnet')
    sys.path.insert(0, '/root/vidaio-subnet/services/upscaling')

    from server import upscale_video

    t_start = time.time()
    output_path = upscale_video(input_path, task_type, is_url=False)
    elapsed = time.time() - t_start

    return str(output_path), elapsed


def benchmark_path(name, input_path, task_type, scale_factor):
    """Benchmark a single upscaling path."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name} ({task_type}, {scale_factor}x)")
    print(f"{'='*60}")

    input_info = get_video_info(input_path)
    print(f"Input: {input_info['width']}x{input_info['height']} @ {input_info['fps']}fps, "
          f"{input_info['duration']:.1f}s, {input_info['nb_frames']} frames")

    # Run upscaling
    print(f"\nRunning upscale...")
    t_total = time.time()
    try:
        output_path, upscale_time = run_upscale(input_path, scale_factor, task_type)
    except Exception as e:
        print(f"ERROR: Upscale failed: {e}")
        return {'name': name, 'error': str(e)}

    output_info = get_video_info(output_path)
    print(f"Output: {output_info['width']}x{output_info['height']} @ {output_info['fps']}fps, "
          f"{output_info['duration']:.1f}s, {output_info['nb_frames']} frames")
    print(f"Upscale time: {upscale_time:.2f}s")

    # Measure PieAPP (dist vs ref — we compare upscaled output to input upscaled with bicubic)
    print(f"\nMeasuring PieAPP...")
    t_pieapp = time.time()

    # Create bicubic reference (what a "perfect" x2/x4 would look like)
    # Actually, the validator compares upscaled output to the ORIGINAL high-res reference
    # For local testing, we can compare to bicubic or use the input as pseudo-reference
    # The validator scores PieAPP by comparing output frames to the original (pre-downscale) reference
    # Since we don't have the original, we'll measure PieAPP vs bicubic upscale

    bicubic_ref = f'/root/pipeline_audit/bicubic_ref_{name}.mp4'
    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f'scale={output_info["width"]}:{output_info["height"]}:flags=bicubic',
        '-c:v', 'libx264', '-crf', '1', '-preset', 'ultrafast',
        '-movflags', '+faststart',
        bicubic_ref
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    pieapp = measure_pieapp(bicubic_ref, output_path, sample_count=8)
    pieapp_time = time.time() - t_pieapp
    print(f"PieAPP (vs bicubic): mean={pieapp['mean']:.4f}, std={pieapp['std']:.4f}")
    print(f"PieAPP per frame: {[f'{s:.4f}' for s in pieapp['per_frame']]}")
    print(f"PieAPP measurement took {pieapp_time:.2f}s")

    # Also measure PieAPP with input directly (after resize to match)
    print(f"\nMeasuring PieAPP (vs input resized)...")
    pieapp_vs_input = measure_pieapp(input_path, output_path, sample_count=8)
    print(f"PieAPP (vs input resized): mean={pieapp_vs_input['mean']:.4f}")

    # Compute scores
    content_length = min(input_info['duration'], 10)
    s_q = calculate_quality_score(pieapp['mean'])
    s_l = calculate_length_score(content_length)
    s_pre = 0.5 * s_q + 0.5 * s_l
    s_f = calculate_final_score(s_pre)

    # Also compute with PieAPP vs input
    s_q_input = calculate_quality_score(pieapp_vs_input['mean'])
    s_pre_input = 0.5 * s_q_input + 0.5 * s_l
    s_f_input = calculate_final_score(s_pre_input)

    total_time = time.time() - t_total

    result = {
        'name': name,
        'task_type': task_type,
        'scale_factor': scale_factor,
        'input': input_info,
        'output': output_info,
        'timing': {
            'upscale': round(upscale_time, 3),
            'pieapp_measurement': round(pieapp_time, 3),
            'total': round(total_time, 3),
        },
        'pieapp_vs_bicubic': pieapp,
        'pieapp_vs_input': pieapp_vs_input,
        'scores': {
            's_q': round(s_q, 4),
            's_l': round(s_l, 4),
            's_pre': round(s_pre, 4),
            's_f': round(s_f, 4),
            'content_length': content_length,
        },
        'scores_vs_input': {
            's_q': round(s_q_input, 4),
            's_f': round(s_f_input, 4),
        },
        'output_path': output_path,
    }

    print(f"\n--- Scoring Summary ---")
    print(f"PieAPP (vs bicubic): {pieapp['mean']:.4f}")
    print(f"S_Q: {s_q:.4f}, S_L: {s_l:.4f}, S_PRE: {s_pre:.4f}")
    print(f"S_F: {s_f:.4f} {'BONUS (>0.32)' if s_f > 0.32 else 'BELOW TARGET'}")

    # Clean up bicubic ref
    if os.path.exists(bicubic_ref):
        os.unlink(bicubic_ref)

    return result


def main():
    print("SN85 Pipeline Audit — Baseline E2E Benchmark")
    print("=" * 60)

    # Find test video
    test_path, test_info = find_test_video()
    if test_path is None:
        print("No test video found. Creating synthetic test video...")
        test_path = '/root/pipeline_audit/test_input_1080p.mp4'
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'testsrc2=duration=10:size=1920x1080:rate=30',
            '-c:v', 'libx264', '-crf', '18', '-preset', 'fast',
            '-movflags', '+faststart',
            test_path
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        test_info = get_video_info(test_path)

    print(f"Test video: {test_path}")
    print(f"Info: {test_info}")

    results = {'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'), 'test_video': test_path}

    # Benchmark x2 path (SD2HD)
    # Copy test video to avoid deletion by upscale_video
    x2_input = '/root/pipeline_audit/test_input_x2.mp4'
    subprocess.run(['cp', test_path, x2_input], check=True)
    results['x2_SD2HD'] = benchmark_path('x2_nvvfx', x2_input, 'SD2HD', 2)

    # Benchmark x4 path (SD24K) — create 540p input
    print("\n\nCreating 540p input for x4 test...")
    x4_input = create_540p_test_video(test_path)
    # Copy to avoid deletion
    x4_input_copy = '/root/pipeline_audit/test_input_x4.mp4'
    subprocess.run(['cp', x4_input, x4_input_copy], check=True)
    results['x4_SD24K'] = benchmark_path('x4_plksr', x4_input_copy, 'SD24K', 4)

    # Save results
    output_path = '/root/pipeline_audit/baseline_scores.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {output_path}")

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY")
    print(f"{'='*60}")
    for key in ['x2_SD2HD', 'x4_SD24K']:
        r = results.get(key, {})
        if 'error' in r:
            print(f"{key}: ERROR — {r['error']}")
        elif 'scores' in r:
            s = r['scores']
            p = r.get('pieapp_vs_bicubic', {}).get('mean', '?')
            print(f"{key}: PieAPP={p:.4f}, S_Q={s['s_q']:.4f}, S_F={s['s_f']:.4f} "
                  f"{'BONUS' if s['s_f'] > 0.32 else 'BELOW'}")


if __name__ == '__main__':
    main()
