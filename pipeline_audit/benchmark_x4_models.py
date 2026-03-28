#!/usr/bin/env python3
"""Benchmark all cached x4 SR models — PieAPP quality + inference speed.

Methodology (matches validator):
  1080p GT → downscale to 270p → x4 upscale → 1080p → PieAPP vs GT

Measures: PieAPP (mean, std, min, max, per-frame), inference FPS, VRAM peak.
"""

import sys, os, json, time, math, subprocess, gc
import numpy as np

sys.path.insert(0, '/root/vidaio-subnet')

# ── Scoring helpers ──────────────────────────────────────────────────
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calc_sq(pieapp):
    s = sigmoid(pieapp)
    at_zero = (1 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
    at_two  = (1 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
    val     = (1 - (np.log10(s + 1) / np.log10(3.5))) ** 2.5
    return 1 - ((val - at_zero) / (at_two - at_zero))

def calc_sf(pieapp, content_length=10):
    s_q = calc_sq(pieapp)
    s_l = math.log(1 + content_length) / math.log(1 + 320)
    s_pre = 0.5 * s_q + 0.5 * s_l
    return 0.1 * math.exp(6.979 * (s_pre - 0.5)), s_q, s_l, s_pre

# ── Video helpers ────────────────────────────────────────────────────
def get_info(path):
    r = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
         '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
         '-show_entries', 'format=duration', '-of', 'json', str(path)],
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
    return w, h, fps, dur, nf


def downscale(src, dst, target_h):
    subprocess.run([
        'ffmpeg', '-y', '-i', src,
        '-vf', f'scale=-2:{target_h}',
        '-c:v', 'libx264', '-crf', '12', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', dst
    ], capture_output=True, check=True)


def measure_pieapp(ref, dist, n_samples=16):
    """PieAPP on n_samples evenly-spaced frames. Returns dict."""
    import torch, pyiqa, cv2
    device = torch.device('cuda')
    metric = pyiqa.create_metric('pieapp', device=device)

    rc = cv2.VideoCapture(str(ref))
    dc = cv2.VideoCapture(str(dist))
    total = min(int(rc.get(cv2.CAP_PROP_FRAME_COUNT)),
                int(dc.get(cv2.CAP_PROP_FRAME_COUNT)))
    n = min(n_samples, total)
    indices = np.linspace(0, total - 1, n, dtype=int)

    scores = []
    for idx in indices:
        rc.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        dc.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok1, rf = rc.read()
        ok2, df = dc.read()
        if not ok1 or not ok2:
            continue
        if rf.shape != df.shape:
            # resize dist to ref size for comparison
            df = cv2.resize(df, (rf.shape[1], rf.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        rt = torch.from_numpy(cv2.cvtColor(rf, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
        dt = torch.from_numpy(cv2.cvtColor(df, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
        with torch.no_grad():
            s = metric(dt, rt).item()
            if s < 0: s = abs(s)
        scores.append(s)
    rc.release()
    dc.release()
    del metric
    torch.cuda.empty_cache()

    if not scores:
        return {'mean': 5.0, 'std': 0, 'min': 5.0, 'max': 5.0, 'per_frame': [], 'n': 0}
    return {
        'mean': float(np.mean(scores)),
        'std': float(np.std(scores)),
        'min': float(min(scores)),
        'max': float(max(scores)),
        'per_frame': [round(s, 5) for s in scores],
        'n': len(scores),
    }


# ── Upscale with arbitrary spandrel model ────────────────────────────
def _pad_to(h, w, multiple=64):
    """Return (pad_h, pad_w) so h+pad_h and w+pad_w are divisible by multiple."""
    ph = (multiple - h % multiple) % multiple
    pw = (multiple - w % multiple) % multiple
    return ph, pw

def upscale_model(input_path, output_path, model_path, scale=4, batch_size=4, compile_model=True):
    """Upscale video, return (n_frames, inference_seconds, vram_peak_mb)."""
    import torch
    import torch.nn.functional as F
    from spandrel import ModelLoader

    w, h, fps, dur, nf = get_info(input_path)
    out_w, out_h = w * scale, h * scale
    frame_size = w * h * 3

    # Pad for transformer window sizes (pad to multiple of 64 covers 8/16/32/48)
    pad_h, pad_w = _pad_to(h, w, 64)
    padded_h, padded_w = h + pad_h, w + pad_w
    need_pad = pad_h > 0 or pad_w > 0
    if need_pad:
        print(f"    Padding {w}x{h} → {padded_w}x{padded_h} (window alignment)")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"    Loading {os.path.basename(model_path)}...")
    try:
        model = ModelLoader(device="cuda:0").load_from_file(str(model_path)).model.eval().half()
    except Exception as e:
        print(f"    FAILED to load: {e}")
        return 0, 0, 0

    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"    Params: {nparams:.2f}M")

    # Skip torch.compile for large transformer models — they often fail with dynamic shapes
    if compile_model and nparams < 10:
        try:
            model = torch.compile(model, dynamic=True)
            dh, dw = (padded_h, padded_w) if need_pad else (min(h, 270), min(w, 480))
            dummy = torch.randn(1, 3, dh, dw, device="cuda:0", dtype=torch.float16)
            for _ in range(3):
                with torch.no_grad():
                    model(dummy)
            del dummy
            torch.cuda.empty_cache()
            print(f"    torch.compile OK")
        except Exception as e:
            print(f"    torch.compile failed, using eager")
            model = getattr(model, '_orig_mod', model)
    elif nparams >= 10:
        print(f"    Skipping torch.compile (large model, {nparams:.1f}M params)")

    # For large models, reduce batch size to avoid OOM
    effective_bs = min(batch_size, 1 if nparams > 12 else (2 if nparams > 8 else batch_size))
    if effective_bs != batch_size:
        print(f"    Reduced batch_size {batch_size} → {effective_bs} (large model)")

    decoder = subprocess.Popen(
        ['ffmpeg', '-i', str(input_path), '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    encoder = subprocess.Popen(
        ['ffmpeg', '-y',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{out_w}x{out_h}', '-r', str(fps), '-i', 'pipe:0',
         '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
         '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-sar', '1:1',
         '-movflags', '+faststart', str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    total_frames = 0
    t_inference = 0.0

    try:
        while True:
            frames = []
            for _ in range(effective_bs):
                raw = decoder.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break
                frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy())
            if not frames:
                break

            tensors = [torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1).contiguous()
                       for f in frames]
            batch = torch.stack(tensors).to("cuda:0").half()

            # Reflect-pad if needed for window-based models
            if need_pad:
                batch = F.pad(batch, (0, pad_w, 0, pad_h), mode='reflect')

            t0 = time.time()
            with torch.no_grad():
                try:
                    out = model(batch)
                except Exception as e:
                    err = str(e)
                    if 'view' in err or 'shape' in err or 'stride' in err:
                        eager = getattr(model, '_orig_mod', model)
                        with torch.no_grad():
                            out = eager(batch)
                    else:
                        raise
            torch.cuda.synchronize()
            t_inference += time.time() - t0

            # Crop padding from output
            if need_pad:
                crop_h = h * scale
                crop_w = w * scale
                out = out[:, :, :crop_h, :crop_w]

            out_np = (out.float().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
                      .permute(0, 2, 3, 1).contiguous().cpu().numpy())
            for j in range(len(frames)):
                encoder.stdin.write(out_np[j].tobytes())
            total_frames += len(frames)

            if total_frames % 50 < effective_bs:
                print(f"    {total_frames} frames done...")

    except Exception as e:
        print(f"    INFERENCE FAILED at frame {total_frames}: {e}")
        # Clean up subprocesses
        try:
            decoder.kill()
            encoder.kill()
        except:
            pass
        del model
        torch.cuda.empty_cache()
        gc.collect()
        return 0, 0, 0

    encoder.stdin.close()
    encoder.wait()
    decoder.wait()

    vram_peak = torch.cuda.max_memory_allocated() / 1e6  # MB

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return total_frames, t_inference, vram_peak


# ── Main ─────────────────────────────────────────────────────────────
def main():
    GT = '/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
    PAYLOAD = '/root/pipeline_audit/payload_270p.mp4'
    OUTDIR = '/root/pipeline_audit'

    # Models to benchmark (path, name, compile?)
    SPAN_DIR = os.path.expanduser('~/.cache/span')
    models = [
        ('PLKSR_X4_DF2K.pth',              'PLKSR-x4 (current prod)', True),
        ('DRCT_SRx4_DF2K.pth',             'DRCT-x4 DF2K',           True),
        ('SwinIR_classicalSR_x4.pth',      'SwinIR-classical-x4',    True),
        ('RealESRNet_x4plus.pth',           'RealESRNet-x4plus (L1)', True),
        ('EfRLFN_x4.pt',                   'EfRLFN-x4',              True),
        ('4xNomos8k_span_otf_strong.pth',  'SPAN-Nomos8k-x4',        True),
        ('4xNomosUni_span.safetensors',     'SPAN-NomosUni-x4',       True),
    ]

    # Filter to existing, non-empty files
    valid = []
    for fname, name, comp in models:
        fpath = os.path.join(SPAN_DIR, fname)
        if os.path.exists(fpath) and os.path.getsize(fpath) > 100:
            valid.append((fpath, name, comp))
        else:
            print(f"SKIP {fname} (missing or empty)")
    models = valid

    print(f"{'='*70}")
    print(f"x4 SR Model Benchmark — {len(models)} models")
    print(f"GT: 1080p/30fps/10s → 270p → x4 → 1080p → PieAPP vs GT")
    print(f"{'='*70}\n")

    # Step 1: Create 270p payload
    if not os.path.exists(PAYLOAD):
        print("Creating 270p payload from GT...")
        downscale(GT, PAYLOAD, 270)
    pw, ph, pfps, pdur, pnf = get_info(PAYLOAD)
    print(f"Payload: {pw}x{ph} @{pfps}fps {pdur:.1f}s {pnf}f\n")

    # Step 2: Resize GT to match expected output (270p * 4 = 1080p, should match)
    gw, gh, gfps, gdur, gnf = get_info(GT)
    expected_w, expected_h = pw * 4, ph * 4
    if gw != expected_w or gh != expected_h:
        print(f"Resizing GT from {gw}x{gh} to {expected_w}x{expected_h} for fair comparison...")
        GT_RESIZED = f'{OUTDIR}/gt_resized_1080p.mp4'
        subprocess.run([
            'ffmpeg', '-y', '-i', GT,
            '-vf', f'scale={expected_w}:{expected_h}:flags=lanczos',
            '-c:v', 'libx264', '-crf', '1', '-preset', 'ultrafast',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', GT_RESIZED
        ], capture_output=True, check=True)
        ref_path = GT_RESIZED
    else:
        ref_path = GT

    results = []

    for model_path, model_name, do_compile in models:
        print(f"\n{'─'*70}")
        print(f"  MODEL: {model_name}")
        print(f"  File:  {os.path.basename(model_path)} ({os.path.getsize(model_path)/1e6:.1f}MB)")
        print(f"{'─'*70}")

        output_path = f'{OUTDIR}/bench_{os.path.basename(model_path).replace(".", "_")}.mp4'

        # Upscale
        t_total_start = time.time()
        nf, t_inf, vram = upscale_model(PAYLOAD, output_path, model_path,
                                          scale=4, batch_size=4, compile_model=do_compile)
        t_total = time.time() - t_total_start

        if nf == 0:
            print(f"    FAILED — skipping")
            results.append({'model': model_name, 'error': 'load/inference failed'})
            continue

        fps = nf / t_inf if t_inf > 0 else 0

        # PieAPP
        print(f"    Upscaled {nf}f in {t_inf:.1f}s ({fps:.1f} FPS), VRAM peak={vram:.0f}MB")
        print(f"    Measuring PieAPP (16 frames)...")
        pieapp = measure_pieapp(ref_path, output_path, n_samples=16)

        # Scoring
        sf, sq, sl, spre = calc_sf(pieapp['mean'])
        bonus = sf > 0.32

        tag = "BONUS" if bonus else "BELOW"
        print(f"\n    PieAPP: mean={pieapp['mean']:.5f}  std={pieapp['std']:.5f}  "
              f"range=[{pieapp['min']:.5f}, {pieapp['max']:.5f}]")
        print(f"    S_Q={sq:.4f}  S_F={sf:.4f}  [{tag}]")
        print(f"    Speed: {fps:.1f} FPS, {t_inf:.1f}s inference, {t_total:.1f}s total")

        results.append({
            'model': model_name,
            'file': os.path.basename(model_path),
            'size_mb': round(os.path.getsize(model_path) / 1e6, 1),
            'pieapp_mean': round(pieapp['mean'], 5),
            'pieapp_std': round(pieapp['std'], 5),
            'pieapp_min': round(pieapp['min'], 5),
            'pieapp_max': round(pieapp['max'], 5),
            'pieapp_per_frame': pieapp['per_frame'],
            's_q': round(sq, 4),
            's_f': round(sf, 4),
            'bonus': bonus,
            'fps': round(fps, 1),
            'inference_s': round(t_inf, 2),
            'total_s': round(t_total, 2),
            'vram_peak_mb': round(vram, 0),
            'n_frames': nf,
        })

        # Cleanup upscaled file to save disk
        if os.path.exists(output_path):
            os.unlink(output_path)

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n\n{'='*90}")
    print(f"SUMMARY — x4 Model Comparison (sorted by PieAPP)")
    print(f"{'='*90}")
    valid_results = [r for r in results if 'error' not in r]
    valid_results.sort(key=lambda r: r['pieapp_mean'])

    print(f"{'Model':<30} {'PieAPP':>8} {'Std':>7} {'Max':>7} {'S_F':>6} {'FPS':>6} {'VRAM':>6} {'Status':<8}")
    print(f"{'─'*30} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
    for r in valid_results:
        tag = "BONUS" if r['bonus'] else "below"
        print(f"{r['model']:<30} {r['pieapp_mean']:8.5f} {r['pieapp_std']:7.5f} "
              f"{r['pieapp_max']:7.5f} {r['s_f']:6.4f} {r['fps']:6.1f} {r['vram_peak_mb']:5.0f}M {tag:<8}")

    for r in results:
        if 'error' in r:
            print(f"{r['model']:<30} {'FAILED':>8}")

    print(f"\nTarget: PieAPP < 0.092 → S_F > 0.32")
    print(f"Budget: <15s inference for 300 frames (>20 FPS)")

    # Save results
    out_file = f'{OUTDIR}/x4_model_benchmark.json'
    with open(out_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'methodology': '1080p GT → 270p → x4 → 1080p → PieAPP vs GT (16 frames)',
            'ground_truth': GT,
            'payload': f'{pw}x{ph}',
            'results': results,
        }, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    main()
