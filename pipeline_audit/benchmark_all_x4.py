#!/usr/bin/env python3
"""Comprehensive x4 SR model benchmark — all cached models.

Handles FP16 vs FP32/bfloat16 per model, padding for transformers,
batch size scaling, and proper error recovery.

Usage: PYTHONUNBUFFERED=1 python benchmark_all_x4.py
"""

import sys, os, json, time, math, subprocess, gc, traceback
import numpy as np

sys.path.insert(0, '/root/vidaio-subnet')
os.environ['PYTHONUNBUFFERED'] = '1'

# Force unbuffered
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Scoring ──────────────────────────────────────────────────────────
def sigmoid(x): return 1 / (1 + np.exp(-x))
def calc_sq(p):
    s = sigmoid(p)
    a0 = (1-(np.log10(sigmoid(0)+1)/np.log10(3.5)))**2.5
    a2 = (1-(np.log10(sigmoid(2.0)+1)/np.log10(3.5)))**2.5
    v  = (1-(np.log10(s+1)/np.log10(3.5)))**2.5
    return 1 - ((v-a0)/(a2-a0))
def calc_sf(p, cl=10):
    sq=calc_sq(p); sl=math.log(1+cl)/math.log(1+320)
    sp=0.5*sq+0.5*sl; sf=0.1*math.exp(6.979*(sp-0.5))
    return sf,sq,sl,sp

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
        if rf.shape != df.shape:
            df = cv2.resize(df, (rf.shape[1], rf.shape[0]), interpolation=cv2.INTER_LANCZOS4)
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


def upscale(input_path, output_path, model_path, scale=4, batch_size=4, dtype_str='fp16'):
    """Upscale video. dtype_str: 'fp16', 'bf16', 'fp32'."""
    import torch, torch.nn.functional as F
    from spandrel import ModelLoader

    w,h,fps,dur,nf = get_info(input_path)
    out_w, out_h = w*scale, h*scale
    frame_size = w*h*3

    # Pad to multiple of 64 for window-based transformers
    pad_h = (64 - h%64) % 64
    pad_w = (64 - w%64) % 64
    need_pad = pad_h > 0 or pad_w > 0

    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    print(f"    Loading {os.path.basename(model_path)} ({dtype_str}, bs={batch_size})...", flush=True)

    try:
        descriptor = ModelLoader(device="cuda:0").load_from_file(str(model_path))
        model = descriptor.model.eval()
    except Exception as e:
        print(f"    LOAD FAILED: {e}", flush=True); return 0,0,0

    nparams = sum(p.numel() for p in model.parameters())/1e6

    if dtype_str == 'fp16':
        model = model.half(); dtype = torch.float16
    elif dtype_str == 'bf16':
        model = model.bfloat16(); dtype = torch.bfloat16
    else:
        dtype = torch.float32

    # Only torch.compile small convnet models (PLKSR, SPAN) — transformers choke
    if nparams < 10 and dtype_str == 'fp16':
        try:
            model = torch.compile(model, dynamic=True)
            dh, dw = (h+pad_h, w+pad_w) if need_pad else (h, w)
            dummy = torch.randn(1, 3, dh, dw, device="cuda:0", dtype=dtype)
            for _ in range(3):
                with torch.no_grad(): model(dummy)
            del dummy; torch.cuda.empty_cache()
            print(f"    {nparams:.1f}M params, compiled", flush=True)
        except:
            model = getattr(model, '_orig_mod', model)
            print(f"    {nparams:.1f}M params, eager (compile failed)", flush=True)
    else:
        print(f"    {nparams:.1f}M params, eager", flush=True)

    if need_pad:
        print(f"    Padding {w}x{h} → {w+pad_w}x{h+pad_h}", flush=True)

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
            with torch.no_grad():
                try:
                    out = model(batch)
                except Exception:
                    eager = getattr(model, '_orig_mod', model)
                    with torch.no_grad(): out = eager(batch)
            torch.cuda.synchronize()
            t_inference += time.time()-t0

            if need_pad: out = out[:,:,:h*scale,:w*scale]
            out_np = (out.float().clamp_(0,1).mul_(255).round_().to(torch.uint8)
                      .permute(0,2,3,1).contiguous().cpu().numpy())
            for j in range(len(frames)): encoder.stdin.write(out_np[j].tobytes())
            total_frames += len(frames)
            if total_frames % 100 < batch_size:
                print(f"    {total_frames}f...", flush=True)

    except Exception as e:
        print(f"    INFERENCE FAILED at {total_frames}f: {e}", flush=True)
        traceback.print_exc()
        try: decoder.kill(); encoder.kill()
        except: pass
        del model; torch.cuda.empty_cache(); gc.collect()
        return 0,0,0

    encoder.stdin.close(); encoder.wait(); decoder.wait()
    vram = torch.cuda.max_memory_allocated()/1e6
    del model; torch.cuda.empty_cache(); gc.collect()
    return total_frames, t_inference, vram


def bench(name, path, ref, payload, dtype_str='fp16', batch_size=4):
    print(f"\n{'─'*70}", flush=True)
    print(f"  {name}", flush=True)
    sz = os.path.getsize(path)/1e6 if os.path.exists(path) else 0
    print(f"  {os.path.basename(path)} ({sz:.1f}MB) [{dtype_str}, bs={batch_size}]", flush=True)
    print(f"{'─'*70}", flush=True)

    output = f'/root/pipeline_audit/bench_{name.replace(" ","_").replace("(","").replace(")","")}.mp4'

    t0 = time.time()
    nf, t_inf, vram = upscale(payload, output, path, batch_size=batch_size, dtype_str=dtype_str)
    t_total = time.time()-t0

    if nf == 0:
        print(f"    FAILED", flush=True); return None

    fps = nf/t_inf if t_inf>0 else 0
    print(f"    Done: {nf}f, {t_inf:.1f}s inference ({fps:.1f} FPS), VRAM={vram:.0f}MB", flush=True)
    print(f"    PieAPP...", flush=True)
    pieapp = measure_pieapp(ref, output, n_samples=16)
    sf,sq,sl,sp = calc_sf(pieapp['mean'])
    tag = "BONUS!" if sf>0.32 else "below"

    print(f"    PieAPP={pieapp['mean']:.5f} ±{pieapp['std']:.5f} [{pieapp['min']:.5f}..{pieapp['max']:.5f}]", flush=True)
    print(f"    S_Q={sq:.4f}  S_F={sf:.4f}  [{tag}]", flush=True)

    if os.path.exists(output): os.unlink(output)

    return {
        'model':name, 'file':os.path.basename(path), 'size_mb':round(sz,1),
        'dtype':dtype_str, 'batch_size':batch_size,
        'pieapp_mean':round(pieapp['mean'],5), 'pieapp_std':round(pieapp['std'],5),
        'pieapp_min':round(pieapp['min'],5), 'pieapp_max':round(pieapp['max'],5),
        'pieapp_per_frame':pieapp['per_frame'],
        's_q':round(sq,4), 's_f':round(sf,4), 'bonus':sf>0.32,
        'fps':round(fps,1), 'inference_s':round(t_inf,2),
        'total_s':round(t_total,2), 'vram_mb':round(vram,0), 'n_frames':nf,
    }


def main():
    GT = '/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
    PAYLOAD = '/root/pipeline_audit/payload_270p.mp4'
    S = os.path.expanduser('~/.cache/span')

    if not os.path.exists(PAYLOAD):
        print("Creating 270p payload...", flush=True)
        subprocess.run(['ffmpeg','-y','-i',GT,'-vf','scale=-2:270',
            '-c:v','libx264','-crf','12','-preset','medium',
            '-pix_fmt','yuv420p','-movflags','+faststart',PAYLOAD],
            capture_output=True, check=True)

    pw,ph,_,_,_ = get_info(PAYLOAD)
    print(f"Payload: {pw}x{ph}", flush=True)

    # Model list: (path, name, dtype, batch_size)
    models = []

    # Production baseline
    models.append((f'{S}/PLKSR_X4_DF2K.pth', 'PLKSR (prod fp16 b4)', 'fp16', 4))

    # Transformer models — FP32, batch=1 (no fp16 support)
    if os.path.exists(f'{S}/DRCT_SRx4_DF2K.pth'):
        models.append((f'{S}/DRCT_SRx4_DF2K.pth', 'DRCT (fp32 b1)', 'fp32', 1))
    if os.path.exists(f'{S}/SwinIR_classicalSR_x4.pth'):
        models.append((f'{S}/SwinIR_classicalSR_x4.pth', 'SwinIR (fp32 b1)', 'fp32', 1))

    # HAT models (new downloads)
    if os.path.exists(f'{S}/HAT_SRx4_ImageNet-pretrain.pth') and os.path.getsize(f'{S}/HAT_SRx4_ImageNet-pretrain.pth') > 1000:
        models.append((f'{S}/HAT_SRx4_ImageNet-pretrain.pth', 'HAT (fp32 b1)', 'fp32', 1))
    # HAT bfloat16 test
    if os.path.exists(f'{S}/HAT_SRx4_ImageNet-pretrain.pth') and os.path.getsize(f'{S}/HAT_SRx4_ImageNet-pretrain.pth') > 1000:
        models.append((f'{S}/HAT_SRx4_ImageNet-pretrain.pth', 'HAT (bf16 b1)', 'bf16', 1))

    # DAT models
    if os.path.exists(f'{S}/DAT_x4.pth') and os.path.getsize(f'{S}/DAT_x4.pth') > 1000:
        models.append((f'{S}/DAT_x4.pth', 'DAT (fp32 b1)', 'fp32', 1))
    if os.path.exists(f'{S}/DAT_S_x4.pth') and os.path.getsize(f'{S}/DAT_S_x4.pth') > 1000:
        models.append((f'{S}/DAT_S_x4.pth', 'DAT-S (fp32 b1)', 'fp32', 1))

    # RealESRNet — large convnet, FP32
    models.append((f'{S}/RealESRNet_x4plus.pth', 'RealESRNet (fp32 b1)', 'fp32', 1))

    # Filter valid
    models = [(p,n,d,b) for p,n,d,b in models if os.path.exists(p) and os.path.getsize(p) > 100]

    print(f"\n{'='*70}", flush=True)
    print(f"x4 SR Benchmark — {len(models)} configurations", flush=True)
    print(f"270p→1080p, 300 frames, PieAPP vs 1080p GT (16 samples)", flush=True)
    print(f"{'='*70}", flush=True)

    results = []
    for path, name, dtype_str, bs in models:
        r = bench(name, path, GT, PAYLOAD, dtype_str=dtype_str, batch_size=bs)
        if r: results.append(r)

    # Summary
    print(f"\n\n{'='*90}", flush=True)
    print(f"RESULTS — sorted by PieAPP (lower=better)", flush=True)
    print(f"{'='*90}", flush=True)
    results.sort(key=lambda r: r['pieapp_mean'])
    print(f"{'Model':<30} {'PieAPP':>8} {'±Std':>7} {'Max':>7} {'S_F':>6} {'FPS':>6} {'VRAM':>7} {'OK'}", flush=True)
    print(f"{'─'*30} {'─'*8} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*4}", flush=True)
    for r in results:
        tag = "YES" if r['bonus'] else "no"
        print(f"{r['model']:<30} {r['pieapp_mean']:8.5f} {r['pieapp_std']:7.5f} "
              f"{r['pieapp_max']:7.5f} {r['s_f']:6.4f} {r['fps']:6.1f} {r['vram_mb']:6.0f}M {tag}", flush=True)

    print(f"\nTarget: PieAPP < 0.092 → S_F > 0.32", flush=True)

    with open('/root/pipeline_audit/x4_all_benchmark.json','w') as f:
        json.dump({'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),'results':results}, f, indent=2)
    print(f"Saved to x4_all_benchmark.json", flush=True)


if __name__ == '__main__':
    main()
