#!/usr/bin/env python3
"""Benchmark transformer x4 SR models in FP32 (they don't support FP16).

Tests: DRCT, SwinIR + PLKSR baseline for comparison.
"""

import sys, os, json, time, math, subprocess, gc
import numpy as np

sys.path.insert(0, '/root/vidaio-subnet')

# ── Scoring ──────────────────────────────────────────────────────────
def sigmoid(x): return 1 / (1 + np.exp(-x))
def calc_sq(p):
    s = sigmoid(p)
    a0 = (1 - (np.log10(sigmoid(0)+1)/np.log10(3.5)))**2.5
    a2 = (1 - (np.log10(sigmoid(2.0)+1)/np.log10(3.5)))**2.5
    v  = (1 - (np.log10(s+1)/np.log10(3.5)))**2.5
    return 1 - ((v - a0) / (a2 - a0))
def calc_sf(p, cl=10):
    sq = calc_sq(p); sl = math.log(1+cl)/math.log(1+320)
    sp = 0.5*sq + 0.5*sl; sf = 0.1*math.exp(6.979*(sp-0.5))
    return sf, sq, sl, sp

def get_info(path):
    r = subprocess.run(['ffprobe','-v','quiet','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate,nb_frames,duration',
        '-show_entries','format=duration','-of','json',str(path)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    info = json.loads(r.stdout); s = info.get('streams',[{}])[0]; fmt = info.get('format',{})
    w,h = int(s.get('width',0)), int(s.get('height',0))
    rfr = s.get('r_frame_rate','30/1'); parts = rfr.split('/')
    fps = float(parts[0])/float(parts[1]) if len(parts)==2 else 30.0
    dur = float(s.get('duration',0) or fmt.get('duration',0))
    nf = int(s.get('nb_frames',0) or 0)
    if nf==0 and dur>0: nf=int(dur*fps)
    return w,h,fps,dur,nf

def measure_pieapp(ref, dist, n_samples=16):
    import torch, pyiqa, cv2
    device = torch.device('cuda')
    metric = pyiqa.create_metric('pieapp', device=device)
    rc = cv2.VideoCapture(str(ref)); dc = cv2.VideoCapture(str(dist))
    total = min(int(rc.get(cv2.CAP_PROP_FRAME_COUNT)), int(dc.get(cv2.CAP_PROP_FRAME_COUNT)))
    n = min(n_samples, total); indices = np.linspace(0, total-1, n, dtype=int)
    scores = []
    for idx in indices:
        rc.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); dc.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok1,rf = rc.read(); ok2,df = dc.read()
        if not ok1 or not ok2: continue
        if rf.shape != df.shape: df = cv2.resize(df, (rf.shape[1],rf.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        rt = torch.from_numpy(cv2.cvtColor(rf,cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
        dt = torch.from_numpy(cv2.cvtColor(df,cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
        with torch.no_grad():
            s = metric(dt, rt).item()
            if s < 0: s = abs(s)
        scores.append(s)
    rc.release(); dc.release(); del metric; torch.cuda.empty_cache()
    if not scores: return {'mean':5,'std':0,'min':5,'max':5,'per_frame':[],'n':0}
    return {'mean':float(np.mean(scores)),'std':float(np.std(scores)),
            'min':float(min(scores)),'max':float(max(scores)),
            'per_frame':[round(s,5) for s in scores],'n':len(scores)}

def _pad(h,w,mult=64):
    return (mult-h%mult)%mult, (mult-w%mult)%mult

def upscale(input_path, output_path, model_path, scale=4, batch_size=1, use_fp16=False):
    import torch, torch.nn.functional as F
    from spandrel import ModelLoader

    w,h,fps,dur,nf = get_info(input_path)
    out_w, out_h = w*scale, h*scale
    frame_size = w*h*3
    pad_h, pad_w = _pad(h, w, 64)
    padded_h, padded_w = h+pad_h, w+pad_w
    need_pad = pad_h>0 or pad_w>0

    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    dtype_label = "fp16" if use_fp16 else "fp32"
    print(f"    Loading {os.path.basename(model_path)} ({dtype_label})...")

    try:
        model = ModelLoader(device="cuda:0").load_from_file(str(model_path)).model.eval()
        if use_fp16:
            model = model.half()
    except Exception as e:
        print(f"    FAILED to load: {e}"); return 0,0,0

    nparams = sum(p.numel() for p in model.parameters())/1e6
    print(f"    Params: {nparams:.2f}M, pad: {w}x{h}→{padded_w}x{padded_h}" if need_pad else f"    Params: {nparams:.2f}M")

    decoder = subprocess.Popen(
        ['ffmpeg','-i',str(input_path),'-f','rawvideo','-pix_fmt','rgb24','-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen(
        ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{out_w}x{out_h}','-r',str(fps),'-i','pipe:0',
         '-c:v','hevc_nvenc','-cq','20','-preset','p4',
         '-profile:v','main','-pix_fmt','yuv420p','-sar','1:1',
         '-movflags','+faststart',str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    total_frames = 0; t_inference = 0.0
    dtype = torch.float16 if use_fp16 else torch.float32

    try:
        while True:
            frames = []
            for _ in range(batch_size):
                raw = decoder.stdout.read(frame_size)
                if len(raw) < frame_size: break
                frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(h,w,3).copy())
            if not frames: break

            tensors = [torch.from_numpy(f.astype(np.float32)/255.0).permute(2,0,1).contiguous() for f in frames]
            batch = torch.stack(tensors).to("cuda:0", dtype=dtype)
            if need_pad: batch = F.pad(batch, (0,pad_w,0,pad_h), mode='reflect')

            t0 = time.time()
            with torch.no_grad(): out = model(batch)
            torch.cuda.synchronize()
            t_inference += time.time()-t0

            if need_pad: out = out[:,:,:h*scale,:w*scale]
            out_np = (out.float().clamp_(0,1).mul_(255).round_().to(torch.uint8)
                      .permute(0,2,3,1).contiguous().cpu().numpy())
            for j in range(len(frames)): encoder.stdin.write(out_np[j].tobytes())
            total_frames += len(frames)
            if total_frames % 50 < batch_size: print(f"    {total_frames}f...")

    except Exception as e:
        print(f"    FAILED at frame {total_frames}: {e}")
        try: decoder.kill(); encoder.kill()
        except: pass
        del model; torch.cuda.empty_cache(); gc.collect()
        return 0,0,0

    encoder.stdin.close(); encoder.wait(); decoder.wait()
    vram = torch.cuda.max_memory_allocated()/1e6
    del model; torch.cuda.empty_cache(); gc.collect()
    return total_frames, t_inference, vram


def bench_one(name, model_path, ref_path, payload_path, use_fp16=False, batch_size=1):
    print(f"\n{'─'*70}")
    print(f"  {name}")
    print(f"  {os.path.basename(model_path)} ({os.path.getsize(model_path)/1e6:.1f}MB)")
    print(f"{'─'*70}")

    outdir = '/root/pipeline_audit'
    output = f'{outdir}/bench_{name.replace(" ","_")}.mp4'

    t0 = time.time()
    nf, t_inf, vram = upscale(payload_path, output, model_path, batch_size=batch_size, use_fp16=use_fp16)
    t_total = time.time() - t0

    if nf == 0:
        print(f"    FAILED"); return None

    fps = nf/t_inf if t_inf>0 else 0
    print(f"    {nf}f in {t_inf:.1f}s ({fps:.1f} FPS), VRAM={vram:.0f}MB")
    print(f"    Measuring PieAPP...")
    pieapp = measure_pieapp(ref_path, output, n_samples=16)
    sf, sq, sl, sp = calc_sf(pieapp['mean'])
    tag = "BONUS" if sf>0.32 else "BELOW"

    print(f"    PieAPP: {pieapp['mean']:.5f} ±{pieapp['std']:.5f} [{pieapp['min']:.5f}..{pieapp['max']:.5f}]")
    print(f"    S_Q={sq:.4f}  S_F={sf:.4f}  [{tag}]")

    if os.path.exists(output): os.unlink(output)

    return {
        'model': name, 'file': os.path.basename(model_path),
        'size_mb': round(os.path.getsize(model_path)/1e6,1),
        'fp16': use_fp16, 'batch_size': batch_size,
        'pieapp_mean': round(pieapp['mean'],5), 'pieapp_std': round(pieapp['std'],5),
        'pieapp_min': round(pieapp['min'],5), 'pieapp_max': round(pieapp['max'],5),
        'pieapp_per_frame': pieapp['per_frame'],
        's_q': round(sq,4), 's_f': round(sf,4), 'bonus': sf>0.32,
        'fps': round(fps,1), 'inference_s': round(t_inf,2),
        'total_s': round(t_total,2), 'vram_mb': round(vram,0), 'n_frames': nf,
    }


def main():
    GT = '/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
    PAYLOAD = '/root/pipeline_audit/payload_270p.mp4'
    SPAN = os.path.expanduser('~/.cache/span')

    if not os.path.exists(PAYLOAD):
        print("Creating 270p payload...")
        subprocess.run(['ffmpeg','-y','-i',GT,'-vf','scale=-2:270',
            '-c:v','libx264','-crf','12','-preset','medium',
            '-pix_fmt','yuv420p','-movflags','+faststart',PAYLOAD],
            capture_output=True, check=True)

    print(f"{'='*70}")
    print(f"x4 Transformer Model Benchmark (FP32)")
    print(f"GT: 1080p → 270p → x4 → 1080p → PieAPP (16 frames)")
    print(f"{'='*70}")

    results = []

    # PLKSR baseline (FP16, batch=4 — production config)
    r = bench_one("PLKSR-x4 (prod, fp16, b4)",
                  f"{SPAN}/PLKSR_X4_DF2K.pth", GT, PAYLOAD, use_fp16=True, batch_size=4)
    if r: results.append(r)

    # DRCT in FP32 batch=1
    r = bench_one("DRCT-x4 (fp32, b1)",
                  f"{SPAN}/DRCT_SRx4_DF2K.pth", GT, PAYLOAD, use_fp16=False, batch_size=1)
    if r: results.append(r)

    # SwinIR in FP32 batch=1
    r = bench_one("SwinIR-classical-x4 (fp32, b1)",
                  f"{SPAN}/SwinIR_classicalSR_x4.pth", GT, PAYLOAD, use_fp16=False, batch_size=1)
    if r: results.append(r)

    # RealESRNet FP32 batch=1
    r = bench_one("RealESRNet-x4plus (fp32, b1)",
                  f"{SPAN}/RealESRNet_x4plus.pth", GT, PAYLOAD, use_fp16=False, batch_size=1)
    if r: results.append(r)

    # Also try PLKSR in FP32 to see if precision matters
    r = bench_one("PLKSR-x4 (fp32, b4)",
                  f"{SPAN}/PLKSR_X4_DF2K.pth", GT, PAYLOAD, use_fp16=False, batch_size=4)
    if r: results.append(r)

    # Summary
    print(f"\n\n{'='*90}")
    print(f"SUMMARY — sorted by PieAPP")
    print(f"{'='*90}")
    results.sort(key=lambda r: r['pieapp_mean'])
    print(f"{'Model':<35} {'PieAPP':>8} {'Std':>7} {'Max':>7} {'S_F':>6} {'FPS':>6} {'VRAM':>7} {'OK':<5}")
    print(f"{'─'*35} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*5}")
    for r in results:
        print(f"{r['model']:<35} {r['pieapp_mean']:8.5f} {r['pieapp_std']:7.5f} "
              f"{r['pieapp_max']:7.5f} {r['s_f']:6.4f} {r['fps']:6.1f} {r['vram_mb']:6.0f}M "
              f"{'YES' if r['bonus'] else 'no'}")
    print(f"\nTarget: PieAPP < 0.092 → S_F > 0.32, Budget: >20 FPS")

    with open('/root/pipeline_audit/x4_fp32_benchmark.json','w') as f:
        json.dump({'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),'results':results}, f, indent=2)
    print(f"Saved to x4_fp32_benchmark.json")


if __name__ == '__main__':
    main()
