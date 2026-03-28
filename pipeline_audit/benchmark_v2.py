#!/usr/bin/env python3
"""SN85 Pipeline Audit — Correct Baseline Benchmark v2

Proper methodology matching the validator's scoring:
1. Start with a 1080p "ground truth" video
2. Downscale to create the miner's input (simulating validator payload)
3. Upscale back to 1080p using our models
4. Compare upscaled output to original ground truth at SAME resolution
5. Compute PieAPP → S_Q → S_F

This exactly matches how the validator scores:
- ref_path = original high-res video (ground truth)
- dist_path = miner's upscaled output (same resolution as ref)
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
    """Measure PieAPP between same-resolution ref and dist videos.

    Matches the validator's methodology exactly:
    - Both videos at same resolution
    - pyiqa PieAPP metric on CUDA
    - Sample frames evenly
    """
    import torch, pyiqa, cv2
    device = torch.device('cuda')
    metric = pyiqa.create_metric('pieapp', device=device)

    ref_cap = cv2.VideoCapture(str(ref_path))
    dist_cap = cv2.VideoCapture(str(dist_path))
    ref_count = int(ref_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dist_count = int(dist_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = min(ref_count, dist_count)
    n = min(sample_count, max(1, total))
    indices = np.linspace(0, total - 1, n, dtype=int)

    scores = []
    for idx in indices:
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        dist_cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret1, rf = ref_cap.read()
        ret2, df = dist_cap.read()
        if not ret1 or not ret2:
            continue

        # Both should be same size. Assert to catch bugs.
        assert rf.shape == df.shape, f"Size mismatch: ref={rf.shape} dist={df.shape}"

        rf_rgb = cv2.cvtColor(rf, cv2.COLOR_BGR2RGB)
        df_rgb = cv2.cvtColor(df, cv2.COLOR_BGR2RGB)
        rt = torch.from_numpy(rf_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        dt = torch.from_numpy(df_rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
        with torch.no_grad():
            s = metric(dt, rt).item()
            if s < 0: s = abs(s)
        scores.append(s)
        print(f"    frame {int(idx)}: PieAPP={s:.5f}")

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


def downscale_video(input_path, output_path, target_height):
    """Downscale video to target height, maintaining aspect ratio."""
    subprocess.run([
        'ffmpeg', '-y', '-i', input_path,
        '-vf', f'scale=-2:{target_height}',
        '-c:v', 'libx264', '-crf', '12', '-preset', 'medium',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart', output_path
    ], capture_output=True, check=True)
    return get_video_info(output_path)


def upscale_with_model(input_path, output_path, model_path, scale_factor):
    """Upscale video using spandrel model (PLKSR or SPAN)."""
    import torch
    from spandrel import ModelLoader

    info = get_video_info(input_path)
    w, h = info['width'], info['height']
    fps = info['fps']
    out_w, out_h = w * scale_factor, h * scale_factor
    frame_size = w * h * 3
    batch_size = 4

    print(f"  Loading model {os.path.basename(model_path)}...")
    model = ModelLoader(device="cuda:0").load_from_file(model_path).model.eval().half()

    if scale_factor == 4:
        model = torch.compile(model, dynamic=True)
        dummy = torch.randn(1, 3, min(h, 270), min(w, 480), device="cuda:0", dtype=torch.float16)
        for _ in range(3):
            with torch.no_grad():
                model(dummy)
        del dummy
        torch.cuda.empty_cache()

    decoder = subprocess.Popen(
        ['ffmpeg', '-i', str(input_path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    encoder = subprocess.Popen(
        ['ffmpeg', '-y',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{out_w}x{out_h}', '-r', str(fps), '-i', 'pipe:0',
         '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
         '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-sar', '1:1',
         '-movflags', '+faststart', output_path],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    total_frames = 0
    t_inference = 0

    while True:
        frames = []
        for _ in range(batch_size):
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy())
        if not frames:
            break

        tensors = [torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
                   for f in frames]
        batch = torch.stack(tensors).to("cuda:0").half()

        t0 = time.time()
        with torch.no_grad():
            out = model(batch)
        torch.cuda.synchronize()
        t_inference += time.time() - t0

        out_np = (out.float().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
                  .permute(0, 2, 3, 1).contiguous().cpu().numpy())
        for j in range(len(frames)):
            encoder.stdin.write(out_np[j].tobytes())
        total_frames += len(frames)

    encoder.stdin.close()
    encoder.wait()
    decoder.wait()

    del model
    import torch as _torch
    _torch.cuda.empty_cache()

    return total_frames, t_inference


def benchmark_path(name, ground_truth, payload_height, scale_factor, model_path, task_type):
    """Full benchmark: downscale ground_truth → upscale → compare."""
    print(f"\n{'='*60}")
    print(f"BENCHMARK: {name} ({task_type}, {scale_factor}x)")
    print(f"{'='*60}")

    gt_info = get_video_info(ground_truth)
    print(f"Ground truth: {gt_info['width']}x{gt_info['height']} @{gt_info['fps']}fps "
          f"{gt_info['duration']:.1f}s {gt_info['nb_frames']}f")

    # Step 1: Downscale to create miner input
    payload_path = f'/root/pipeline_audit/payload_{name}.mp4'
    print(f"\n1. Downscaling to {payload_height}p...")
    t0 = time.time()
    payload_info = downscale_video(ground_truth, payload_path, payload_height)
    t_downscale = time.time() - t0
    print(f"   Payload: {payload_info['width']}x{payload_info['height']} ({t_downscale:.1f}s)")

    # Step 2: Upscale
    output_path = f'/root/pipeline_audit/upscaled_{name}.mp4'
    print(f"\n2. Upscaling {scale_factor}x with {os.path.basename(model_path)}...")
    t0 = time.time()
    total_frames, t_inference = upscale_with_model(payload_path, output_path, model_path, scale_factor)
    t_upscale = time.time() - t0
    output_info = get_video_info(output_path)
    fps = total_frames / t_inference if t_inference > 0 else 0
    print(f"   Output: {output_info['width']}x{output_info['height']} "
          f"({t_upscale:.1f}s total, {t_inference:.1f}s inference, {fps:.1f} FPS)")

    # Step 3: Resize ground truth to match output resolution for PieAPP comparison
    # The validator's reference IS at the output resolution (original high-res video)
    # But our ground truth might be different resolution from the output
    gt_w, gt_h = gt_info['width'], gt_info['height']
    out_w, out_h = output_info['width'], output_info['height']

    if gt_w != out_w or gt_h != out_h:
        print(f"\n   Note: Ground truth {gt_w}x{gt_h} != output {out_w}x{out_h}")
        print(f"   Resizing ground truth to {out_w}x{out_h} for fair PieAPP comparison")
        gt_resized = f'/root/pipeline_audit/gt_resized_{name}.mp4'
        subprocess.run([
            'ffmpeg', '-y', '-i', ground_truth,
            '-vf', f'scale={out_w}:{out_h}:flags=lanczos',
            '-c:v', 'libx264', '-crf', '1', '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', gt_resized
        ], capture_output=True, check=True)
        ref_for_pieapp = gt_resized
    else:
        ref_for_pieapp = ground_truth

    # Step 4: Measure PieAPP (upscaled vs ground truth, same resolution)
    print(f"\n3. Measuring PieAPP ({output_info['width']}x{output_info['height']} output vs ground truth)...")
    t0 = time.time()
    pieapp = measure_pieapp(ref_for_pieapp, output_path, sample_count=8)
    t_pieapp = time.time() - t0
    print(f"   PieAPP: mean={pieapp['mean']:.5f} std={pieapp['std']:.5f} "
          f"range=[{pieapp['min']:.5f}, {pieapp['max']:.5f}] ({t_pieapp:.1f}s)")

    # Step 5: Compute S_F
    cl = min(gt_info['duration'], 10)
    s_q = calculate_quality_score(pieapp['mean'])
    s_l = calculate_length_score(cl)
    s_pre = 0.5 * s_q + 0.5 * s_l
    s_f = calculate_final_score(s_pre)

    bonus = s_f > 0.32
    print(f"\n   === S_F = {s_f:.4f} ({'BONUS' if bonus else 'BELOW TARGET'}) ===")
    print(f"   PieAPP={pieapp['mean']:.5f} → S_Q={s_q:.4f}, S_L={s_l:.4f}, S_PRE={s_pre:.4f}")

    # Cleanup temp files
    for f in [payload_path, f'/root/pipeline_audit/gt_resized_{name}.mp4']:
        if os.path.exists(f):
            os.unlink(f)

    return {
        'name': name, 'task_type': task_type, 'scale_factor': scale_factor,
        'model': os.path.basename(model_path),
        'ground_truth': gt_info,
        'payload_height': payload_height,
        'output': output_info,
        'timing': {
            'downscale': round(t_downscale, 2),
            'upscale_total': round(t_upscale, 2),
            'inference_only': round(t_inference, 2),
            'fps': round(fps, 1),
            'pieapp_measurement': round(t_pieapp, 2),
        },
        'pieapp': {k: round(v, 5) if isinstance(v, float) else v for k, v in pieapp.items()},
        'scores': {
            'pieapp': round(pieapp['mean'], 5),
            's_q': round(s_q, 4), 's_l': round(s_l, 4),
            's_pre': round(s_pre, 4), 's_f': round(s_f, 4),
            'content_length': cl, 'target_met': bonus,
        },
        'output_path': output_path,
    }


def main():
    print("SN85 Pipeline Audit — Correct Baseline Benchmark v2")
    print("=" * 60)
    print("Methodology: downscale ground truth → upscale → compare to ground truth")
    print("This matches the validator's scoring exactly.\n")

    # Use real 1080p test video as ground truth
    ground_truth = '/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
    if not os.path.exists(ground_truth):
        print("ERROR: No test video found")
        return

    gt_info = get_video_info(ground_truth)
    print(f"Ground truth: {ground_truth}")
    print(f"  {gt_info['width']}x{gt_info['height']} @{gt_info['fps']}fps {gt_info['duration']:.1f}s\n")

    results = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'methodology': 'downscale_gt → upscale → compare_to_gt (same as validator)',
        'gpu': 'RTX 4090',
        'ground_truth': ground_truth,
    }

    # ---- x4 path (SD24K): GT 1080p → downscale to 270p → PLKSR x4 → 1080p → compare to GT ----
    results['x4_SD24K'] = benchmark_path(
        name='x4_PLKSR',
        ground_truth=ground_truth,
        payload_height=270,  # 1080/4 = 270
        scale_factor=4,
        model_path=os.path.expanduser('~/.cache/span/PLKSR_X4_DF2K.pth'),
        task_type='SD24K',
    )

    # ---- x2 path (SD2HD): GT 1080p → downscale to 540p → SPAN x2 → 1080p → compare to GT ----
    results['x2_SD2HD_SPAN'] = benchmark_path(
        name='x2_SPAN',
        ground_truth=ground_truth,
        payload_height=540,  # 1080/2 = 540
        scale_factor=2,
        model_path=os.path.expanduser('~/.cache/span/2xHFA2kSPAN.safetensors'),
        task_type='SD2HD',
    )

    # Save
    out = '/root/pipeline_audit/baseline_scores.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    shutil.copy(out, '/root/pipeline_audit/current_scores.json')
    print(f"\nResults saved to {out}")

    # Summary
    print(f"\n{'='*60}")
    print("BASELINE SUMMARY (validator-equivalent methodology)")
    print(f"{'='*60}")
    for key in ['x4_SD24K', 'x2_SD2HD_SPAN']:
        r = results.get(key, {})
        if 'error' in r:
            print(f"  {key}: ERROR — {r.get('error')}")
        else:
            s = r['scores']
            print(f"  {key}: PieAPP={s['pieapp']:.5f} S_Q={s['s_q']:.4f} "
                  f"S_F={s['s_f']:.4f} {'BONUS' if s['target_met'] else 'BELOW'} "
                  f"({r['timing']['fps']:.0f} FPS)")

    print(f"\nTarget: S_F > 0.32 (need PieAPP < 0.092)")


if __name__ == '__main__':
    main()
