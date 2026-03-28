#!/usr/bin/env python3
"""Benchmark: DAT full x4 and HAT-L bf16 — lossless vs yuv420p PieAPP.

Key question: Is the yuv420p PieAPP penalty constant (~0.133) regardless of model,
or does it scale with model quality?

DAT full x4 has PieAPP=0.064 lossless — well below 0.092 threshold.
DAT-light x4 loses ~0.133 PieAPP from yuv420p encoding.
If DAT full after yuv420p still < 0.092, it's viable.
"""
import sys, os, json, time, math, subprocess, gc, traceback
import numpy as np
sys.path.insert(0, '/root/vidaio-subnet')
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── scoring ──
def sigmoid(x): return 1/(1+np.exp(-x))
def calc_sq(p):
    s=sigmoid(p); a0=(1-(np.log10(sigmoid(0)+1)/np.log10(3.5)))**2.5
    a2=(1-(np.log10(sigmoid(2.0)+1)/np.log10(3.5)))**2.5; v=(1-(np.log10(s+1)/np.log10(3.5)))**2.5
    return 1-((v-a0)/(a2-a0))
def calc_sf(p,cl=10):
    sq=calc_sq(p);sl=math.log(1+cl)/math.log(1+320);sp=0.5*sq+0.5*sl
    return 0.1*math.exp(6.979*(sp-0.5)),sq

# ── paths ──
GT = '/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
PAYLOAD = '/root/pipeline_audit/payload_270p.mp4'
SPAN_DIR = os.path.expanduser('~/.cache/span')
OUT_DIR = '/root/pipeline_audit/encoding_bench_v2'
os.makedirs(OUT_DIR, exist_ok=True)

NUM_FRAMES = 30
PAD_MULT = 64

def get_info(path):
    r = subprocess.run(['ffprobe','-v','quiet','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate','-of','json',str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(r.stdout); s = info.get('streams',[{}])[0]
    w, h = int(s.get('width',0)), int(s.get('height',0))
    rfr = s.get('r_frame_rate','30/1'); parts = rfr.split('/')
    fps = float(parts[0])/float(parts[1]) if len(parts)==2 else 30.0
    return w, h, fps

def upscale_and_save(input_path, model_path, dtype_str, n_frames, lossless_path, yuv420_path):
    """Upscale n_frames and write BOTH lossless (ffv1 rgb24) and yuv420p hevc outputs."""
    import torch, torch.nn.functional as F
    from spandrel import ModelLoader

    w, h, fps = get_info(input_path)
    out_w, out_h = w*4, h*4
    frame_size = w*h*3
    pad_h = (PAD_MULT - h % PAD_MULT) % PAD_MULT
    pad_w = (PAD_MULT - w % PAD_MULT) % PAD_MULT
    need_pad = pad_h > 0 or pad_w > 0

    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    print(f"    Loading model ({dtype_str})...", flush=True)
    descriptor = ModelLoader(device="cuda:0").load_from_file(str(model_path))
    model = descriptor.model.eval()
    nparams = sum(p.numel() for p in model.parameters()) / 1e6

    if dtype_str == 'bf16':
        model = model.bfloat16(); dtype = torch.bfloat16
    elif dtype_str == 'fp16':
        model = model.half(); dtype = torch.float16
    else:
        dtype = torch.float32
    print(f"    {nparams:.1f}M params, input={w}x{h}, output={out_w}x{out_h}, pad={'yes' if need_pad else 'no'}", flush=True)

    # Decode input
    decoder = subprocess.Popen(
        ['ffmpeg','-i',str(input_path),'-frames:v',str(n_frames),'-f','rawvideo','-pix_fmt','rgb24','-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Lossless encoder (ffv1 rgb24)
    enc_lossless = subprocess.Popen(
        ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{out_w}x{out_h}','-r',str(fps),'-i','pipe:0',
         '-c:v','ffv1','-level','3','-slicecrc','1','-pix_fmt','rgb24',
         str(lossless_path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # HEVC yuv420p encoder
    enc_yuv420 = subprocess.Popen(
        ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{out_w}x{out_h}','-r',str(fps),'-i','pipe:0',
         '-c:v','hevc_nvenc','-cq','20','-preset','p4','-tune','hq',
         '-rc','vbr','-b:v','0',
         '-profile:v','main','-pix_fmt','yuv420p','-sar','1:1',
         '-movflags','+faststart',str(yuv420_path)],
        stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    total = 0; t_inf = 0.0
    try:
        while total < n_frames:
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            f = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            t = torch.from_numpy(f.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to("cuda:0", dtype=dtype)
            if need_pad:
                t = F.pad(t, (0, pad_w, 0, pad_h), mode='reflect')

            t0 = time.time()
            with torch.no_grad():
                out = model(t)
            torch.cuda.synchronize()
            t_inf += time.time() - t0

            if need_pad:
                out = out[:, :, :h*4, :w*4]

            out_np = out.float().clamp_(0,1).mul_(255).round_().to(torch.uint8).permute(0,2,3,1).contiguous().cpu().numpy()
            frame_bytes = out_np[0].tobytes()

            enc_lossless.stdin.write(frame_bytes)
            enc_yuv420.stdin.write(frame_bytes)

            total += 1
            if total % 5 == 0:
                fps_now = total / t_inf if t_inf > 0 else 0
                print(f"    {total}/{n_frames} frames ({fps_now:.2f} FPS)...", flush=True)

    except Exception as e:
        print(f"    FAILED at frame {total}: {e}", flush=True)
        traceback.print_exc()
        try: decoder.kill(); enc_lossless.kill(); enc_yuv420.kill()
        except: pass
        del model; torch.cuda.empty_cache(); gc.collect()
        return 0, 0, 0

    enc_lossless.stdin.close(); enc_lossless.wait()
    enc_yuv420.stdin.close(); enc_yuv420.wait()
    decoder.wait()

    vram = torch.cuda.max_memory_allocated() / 1e6
    del model; torch.cuda.empty_cache(); gc.collect()
    return total, t_inf, vram


def measure_pieapp(ref, dist, n_samples=16):
    import torch, pyiqa, cv2
    device = torch.device('cuda')
    metric = pyiqa.create_metric('pieapp', device=device)
    rc = cv2.VideoCapture(str(ref)); dc = cv2.VideoCapture(str(dist))
    total = min(int(rc.get(cv2.CAP_PROP_FRAME_COUNT)), int(dc.get(cv2.CAP_PROP_FRAME_COUNT)))
    n = min(n_samples, total)
    indices = np.linspace(0, total-1, n, dtype=int)
    scores = []
    for idx in indices:
        rc.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        dc.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok1, rf = rc.read(); ok2, df = dc.read()
        if not ok1 or not ok2:
            continue
        if rf.shape != df.shape:
            df = cv2.resize(df, (rf.shape[1], rf.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        rt = torch.from_numpy(cv2.cvtColor(rf, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
        dt = torch.from_numpy(cv2.cvtColor(df, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
        with torch.no_grad():
            s = metric(dt, rt).item()
            if s < 0: s = abs(s)
        scores.append(s)
    rc.release(); dc.release()
    del metric; torch.cuda.empty_cache()
    return {'mean': float(np.mean(scores)), 'std': float(np.std(scores)),
            'min': float(min(scores)), 'max': float(max(scores)),
            'per_frame': [round(s,5) for s in scores], 'n': len(scores)}


def main():
    models = [
        (f'{SPAN_DIR}/DAT_x4.safetensors', 'DAT-full x4', 'bf16'),
        (f'{SPAN_DIR}/HAT-L_SRx4_ImageNet-pretrain.pth', 'HAT-L x4', 'bf16'),
    ]

    print(f"{'='*80}", flush=True)
    print(f"DAT-full & HAT-L x4: Lossless vs yuv420p PieAPP ({NUM_FRAMES} frames)", flush=True)
    print(f"Question: Is yuv420p penalty ~0.133 constant or proportional to model quality?", flush=True)
    print(f"{'='*80}", flush=True)

    all_results = []

    for model_path, name, dtype_str in models:
        if not os.path.exists(model_path):
            print(f"\nSKIP {name} — not found at {model_path}", flush=True)
            continue

        print(f"\n{'─'*80}", flush=True)
        print(f"  {name} ({os.path.getsize(model_path)/1e6:.0f}MB, {dtype_str})", flush=True)
        print(f"{'─'*80}", flush=True)

        safe_name = name.replace(' ', '_').replace('-', '_')
        lossless_path = os.path.join(OUT_DIR, f'{safe_name}_lossless.mkv')
        yuv420_path = os.path.join(OUT_DIR, f'{safe_name}_yuv420p_cq20.mp4')

        t0 = time.time()
        nf, t_inf, vram = upscale_and_save(PAYLOAD, model_path, dtype_str, NUM_FRAMES,
                                             lossless_path, yuv420_path)
        t_total = time.time() - t0

        if nf == 0:
            print(f"    FAILED", flush=True)
            all_results.append({'model': name, 'error': 'failed'})
            continue

        fps = nf / t_inf if t_inf > 0 else 0
        print(f"    Done: {nf}f in {t_inf:.1f}s ({fps:.2f} FPS), VRAM={vram:.0f}MB", flush=True)

        # Measure PieAPP for lossless
        print(f"    Measuring PieAPP: lossless vs ground truth...", flush=True)
        n_samples = min(16, nf)
        pa_lossless = measure_pieapp(GT, lossless_path, n_samples=n_samples)
        sf_ll, sq_ll = calc_sf(pa_lossless['mean'])
        print(f"    Lossless PieAPP = {pa_lossless['mean']:.5f} +/- {pa_lossless['std']:.5f}  "
              f"[{pa_lossless['min']:.5f}..{pa_lossless['max']:.5f}]", flush=True)
        print(f"    Lossless S_Q={sq_ll:.4f}  S_F={sf_ll:.4f}", flush=True)

        # Measure PieAPP for yuv420p
        print(f"    Measuring PieAPP: yuv420p CQ=20 vs ground truth...", flush=True)
        pa_yuv420 = measure_pieapp(GT, yuv420_path, n_samples=n_samples)
        sf_yv, sq_yv = calc_sf(pa_yuv420['mean'])
        print(f"    yuv420p  PieAPP = {pa_yuv420['mean']:.5f} +/- {pa_yuv420['std']:.5f}  "
              f"[{pa_yuv420['min']:.5f}..{pa_yuv420['max']:.5f}]", flush=True)
        print(f"    yuv420p  S_Q={sq_yv:.4f}  S_F={sf_yv:.4f}", flush=True)

        delta = pa_yuv420['mean'] - pa_lossless['mean']
        ratio = pa_yuv420['mean'] / pa_lossless['mean'] if pa_lossless['mean'] > 0 else 0
        print(f"    Delta (yuv420p - lossless) = +{delta:.5f}", flush=True)
        print(f"    Ratio (yuv420p / lossless) = {ratio:.3f}x", flush=True)

        viable = pa_yuv420['mean'] < 0.092
        print(f"    yuv420p PieAPP < 0.092 threshold? {'YES' if viable else 'NO'} ({pa_yuv420['mean']:.5f})", flush=True)

        lossless_size = os.path.getsize(lossless_path) / 1024 / 1024
        yuv420_size = os.path.getsize(yuv420_path) / 1024 / 1024
        print(f"    File sizes: lossless={lossless_size:.1f}MB, yuv420p={yuv420_size:.1f}MB", flush=True)

        all_results.append({
            'model': name, 'dtype': dtype_str, 'n_frames': nf,
            'fps': round(fps, 2), 'vram_mb': round(vram, 0),
            'lossless': {
                'pieapp_mean': round(pa_lossless['mean'], 5),
                'pieapp_std': round(pa_lossless['std'], 5),
                'pieapp_min': round(pa_lossless['min'], 5),
                'pieapp_max': round(pa_lossless['max'], 5),
                's_q': round(sq_ll, 4), 's_f': round(sf_ll, 4),
                'size_mb': round(lossless_size, 1),
            },
            'yuv420p_cq20': {
                'pieapp_mean': round(pa_yuv420['mean'], 5),
                'pieapp_std': round(pa_yuv420['std'], 5),
                'pieapp_min': round(pa_yuv420['min'], 5),
                'pieapp_max': round(pa_yuv420['max'], 5),
                's_q': round(sq_yv, 4), 's_f': round(sf_yv, 4),
                'size_mb': round(yuv420_size, 1),
            },
            'delta_pieapp': round(delta, 5),
            'ratio_pieapp': round(ratio, 3),
            'viable_under_092': viable,
        })

    # ── Summary ──
    print(f"\n{'='*80}", flush=True)
    print(f"SUMMARY: Lossless vs yuv420p PieAPP", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Model':<20} {'Lossless':>10} {'yuv420p':>10} {'Delta':>10} {'Ratio':>8} {'<0.092?':>8}", flush=True)
    print(f"{'-'*80}", flush=True)

    for r in all_results:
        if 'error' in r:
            print(f"{r['model']:<20} FAILED", flush=True)
            continue
        print(f"{r['model']:<20} {r['lossless']['pieapp_mean']:>10.5f} {r['yuv420p_cq20']['pieapp_mean']:>10.5f} "
              f"+{r['delta_pieapp']:>9.5f} {r['ratio_pieapp']:>7.3f}x {'YES' if r['viable_under_092'] else 'NO':>8}", flush=True)

    print(f"\nFor reference (DAT-light x4 from previous benchmark):", flush=True)
    print(f"  DAT-light lossless: ~0.064   yuv420p CQ=20: ~0.197   delta: ~0.133", flush=True)
    print(f"\nThreshold for bonus: PieAPP < 0.092  (S_F > 0.32)", flush=True)

    # Scoring details
    print(f"\nScoring details:", flush=True)
    for r in all_results:
        if 'error' in r: continue
        print(f"  {r['model']}:", flush=True)
        print(f"    Lossless: S_Q={r['lossless']['s_q']:.4f}  S_F={r['lossless']['s_f']:.4f}", flush=True)
        print(f"    yuv420p:  S_Q={r['yuv420p_cq20']['s_q']:.4f}  S_F={r['yuv420p_cq20']['s_f']:.4f}", flush=True)
        print(f"    FPS={r['fps']:.2f}  VRAM={r['vram_mb']:.0f}MB", flush=True)

    # Save JSON
    out_json = os.path.join(OUT_DIR, 'dat_full_yuv420_results.json')
    with open(out_json, 'w') as f:
        json.dump({'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'), 'results': all_results}, f, indent=2)
    print(f"\nResults saved to {out_json}", flush=True)


if __name__ == '__main__':
    main()
