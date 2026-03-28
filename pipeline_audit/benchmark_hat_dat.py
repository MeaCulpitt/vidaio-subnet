#!/usr/bin/env python3
"""Benchmark HAT-L and DAT x4 — the SOTA transformer models."""
import sys, os, json, time, math, subprocess, gc, traceback
import numpy as np
sys.path.insert(0, '/root/vidaio-subnet')
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

def sigmoid(x): return 1/(1+np.exp(-x))
def calc_sq(p):
    s=sigmoid(p); a0=(1-(np.log10(sigmoid(0)+1)/np.log10(3.5)))**2.5
    a2=(1-(np.log10(sigmoid(2.0)+1)/np.log10(3.5)))**2.5; v=(1-(np.log10(s+1)/np.log10(3.5)))**2.5
    return 1-((v-a0)/(a2-a0))
def calc_sf(p,cl=10):
    sq=calc_sq(p);sl=math.log(1+cl)/math.log(1+320);sp=0.5*sq+0.5*sl
    return 0.1*math.exp(6.979*(sp-0.5)),sq

def get_info(path):
    r=subprocess.run(['ffprobe','-v','quiet','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate,nb_frames,duration',
        '-show_entries','format=duration','-of','json',str(path)],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    info=json.loads(r.stdout);s=info.get('streams',[{}])[0];fmt=info.get('format',{})
    w,h=int(s.get('width',0)),int(s.get('height',0))
    rfr=s.get('r_frame_rate','30/1');parts=rfr.split('/')
    fps=float(parts[0])/float(parts[1]) if len(parts)==2 else 30.0
    dur=float(s.get('duration',0) or fmt.get('duration',0))
    nf=int(s.get('nb_frames',0) or 0)
    if nf==0 and dur>0: nf=int(dur*fps)
    return w,h,fps,dur,nf

def measure_pieapp(ref,dist,n_samples=16):
    import torch,pyiqa,cv2
    device=torch.device('cuda');metric=pyiqa.create_metric('pieapp',device=device)
    rc=cv2.VideoCapture(str(ref));dc=cv2.VideoCapture(str(dist))
    total=min(int(rc.get(cv2.CAP_PROP_FRAME_COUNT)),int(dc.get(cv2.CAP_PROP_FRAME_COUNT)))
    n=min(n_samples,total);indices=np.linspace(0,total-1,n,dtype=int)
    scores=[]
    for idx in indices:
        rc.set(cv2.CAP_PROP_POS_FRAMES,int(idx));dc.set(cv2.CAP_PROP_POS_FRAMES,int(idx))
        ok1,rf=rc.read();ok2,df=dc.read()
        if not ok1 or not ok2: continue
        if rf.shape!=df.shape: df=cv2.resize(df,(rf.shape[1],rf.shape[0]),interpolation=cv2.INTER_LANCZOS4)
        rt=torch.from_numpy(cv2.cvtColor(rf,cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
        dt=torch.from_numpy(cv2.cvtColor(df,cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
        with torch.no_grad():
            s=metric(dt,rt).item()
            if s<0: s=abs(s)
        scores.append(s)
    rc.release();dc.release();del metric;torch.cuda.empty_cache()
    return {'mean':float(np.mean(scores)),'std':float(np.std(scores)),
            'min':float(min(scores)),'max':float(max(scores)),
            'per_frame':[round(s,5) for s in scores],'n':len(scores)}

def upscale_first_n(input_path, output_path, model_path, n_frames=60, dtype_str='fp32'):
    """Upscale only first n_frames for faster benchmarking."""
    import torch, torch.nn.functional as F
    from spandrel import ModelLoader

    w,h,fps,dur,nf = get_info(input_path)
    out_w,out_h = w*4, h*4
    frame_size = w*h*3
    pad_h=(64-h%64)%64; pad_w=(64-w%64)%64; need_pad=pad_h>0 or pad_w>0

    torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    print(f"    Loading ({dtype_str})...", flush=True)
    descriptor = ModelLoader(device="cuda:0").load_from_file(str(model_path))
    model = descriptor.model.eval()
    nparams = sum(p.numel() for p in model.parameters())/1e6

    if dtype_str=='bf16': model=model.bfloat16(); dtype=torch.bfloat16
    elif dtype_str=='fp16': model=model.half(); dtype=torch.float16
    else: dtype=torch.float32
    print(f"    {nparams:.1f}M params, pad={'yes' if need_pad else 'no'}", flush=True)

    decoder = subprocess.Popen(
        ['ffmpeg','-i',str(input_path),'-frames:v',str(n_frames),'-f','rawvideo','-pix_fmt','rgb24','-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen(
        ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{out_w}x{out_h}','-r',str(fps),'-i','pipe:0',
         '-c:v','hevc_nvenc','-cq','20','-preset','p4',
         '-profile:v','main','-pix_fmt','yuv420p','-sar','1:1',
         '-movflags','+faststart',str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    total=0; t_inf=0.0
    try:
        while total < n_frames:
            raw = decoder.stdout.read(frame_size)
            if len(raw)<frame_size: break
            f = np.frombuffer(raw, dtype=np.uint8).reshape(h,w,3).copy()
            t = torch.from_numpy(f.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to("cuda:0",dtype=dtype)
            if need_pad: t = F.pad(t,(0,pad_w,0,pad_h),mode='reflect')
            t0=time.time()
            with torch.no_grad(): out=model(t)
            torch.cuda.synchronize()
            t_inf+=time.time()-t0
            if need_pad: out=out[:,:,:h*4,:w*4]
            out_np=(out.float().clamp_(0,1).mul_(255).round_().to(torch.uint8).permute(0,2,3,1).contiguous().cpu().numpy())
            encoder.stdin.write(out_np[0].tobytes())
            total+=1
            if total%20==0: print(f"    {total}f ({total/t_inf:.1f} FPS)...", flush=True)
    except Exception as e:
        print(f"    FAILED at {total}f: {e}", flush=True)
        traceback.print_exc()
        try: decoder.kill();encoder.kill()
        except: pass
        del model;torch.cuda.empty_cache();gc.collect()
        return 0,0,0

    encoder.stdin.close();encoder.wait();decoder.wait()
    vram=torch.cuda.max_memory_allocated()/1e6
    del model;torch.cuda.empty_cache();gc.collect()
    return total, t_inf, vram


def main():
    GT='/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
    PAYLOAD='/root/pipeline_audit/payload_270p.mp4'
    S=os.path.expanduser('~/.cache/span')
    N=60  # test 60 frames (2s) for speed, full 16-sample PieAPP

    models = [
        (f'{S}/HAT-L_SRx4_ImageNet-pretrain.pth', 'HAT-L x4', 'fp32'),
        (f'{S}/HAT-L_SRx4_ImageNet-pretrain.pth', 'HAT-L x4 bf16', 'bf16'),
        (f'{S}/DAT_x4.safetensors',               'DAT x4', 'fp32'),
        (f'{S}/DAT_x4.safetensors',               'DAT x4 bf16', 'bf16'),
    ]

    print(f"{'='*70}", flush=True)
    print(f"HAT-L + DAT x4 Benchmark ({N} frames, PieAPP on 16 samples)", flush=True)
    print(f"{'='*70}", flush=True)

    results = []
    for path, name, dtype_str in models:
        if not os.path.exists(path) or os.path.getsize(path)<1000:
            print(f"\nSKIP {name} — not found", flush=True); continue

        print(f"\n{'─'*70}", flush=True)
        print(f"  {name} ({os.path.getsize(path)/1e6:.0f}MB, {dtype_str})", flush=True)
        print(f"{'─'*70}", flush=True)

        output = f'/root/pipeline_audit/bench_{name.replace(" ","_")}.mp4'
        t0=time.time()
        nf, t_inf, vram = upscale_first_n(PAYLOAD, output, path, n_frames=N, dtype_str=dtype_str)
        t_total=time.time()-t0

        if nf==0:
            print(f"    FAILED", flush=True)
            results.append({'model':name,'error':'failed'}); continue

        fps = nf/t_inf if t_inf>0 else 0
        extrapolated_300f = 300/fps if fps>0 else 99999

        print(f"    {nf}f in {t_inf:.1f}s ({fps:.1f} FPS), VRAM={vram:.0f}MB", flush=True)
        print(f"    Extrapolated 300f: {extrapolated_300f:.0f}s", flush=True)

        # PieAPP on the partial output
        print(f"    PieAPP...", flush=True)
        pieapp = measure_pieapp(GT, output, n_samples=min(16, nf))
        sf, sq = calc_sf(pieapp['mean'])
        tag = "BONUS!" if sf>0.32 else "below"

        print(f"    PieAPP={pieapp['mean']:.5f} ±{pieapp['std']:.5f} [{pieapp['min']:.5f}..{pieapp['max']:.5f}]", flush=True)
        print(f"    S_Q={sq:.4f}  S_F={sf:.4f}  [{tag}]", flush=True)

        if os.path.exists(output): os.unlink(output)
        results.append({
            'model':name,'dtype':dtype_str,'n_frames':nf,
            'pieapp_mean':round(pieapp['mean'],5),'pieapp_std':round(pieapp['std'],5),
            'pieapp_min':round(pieapp['min'],5),'pieapp_max':round(pieapp['max'],5),
            's_q':round(sq,4),'s_f':round(sf,4),'bonus':sf>0.32,
            'fps':round(fps,2),'inference_s':round(t_inf,2),
            'extrapolated_300f_s':round(extrapolated_300f,1),
            'vram_mb':round(vram,0),
        })

    # Summary
    print(f"\n{'='*70}", flush=True)
    valid=[r for r in results if 'error' not in r]
    valid.sort(key=lambda r: r['pieapp_mean'])
    for r in valid:
        tag="YES" if r['bonus'] else "no"
        print(f"{r['model']:<20} PieAPP={r['pieapp_mean']:.5f} S_F={r['s_f']:.4f} "
              f"FPS={r['fps']:.1f} 300f≈{r['extrapolated_300f_s']:.0f}s VRAM={r['vram_mb']:.0f}MB [{tag}]", flush=True)

    # Also include previous results for comparison
    print(f"\nPrevious results for reference:", flush=True)
    print(f"  SwinIR:    PieAPP=0.32575  S_F=0.1643  0.5 FPS  1835M", flush=True)
    print(f"  PLKSR:     PieAPP=0.41020  S_F=0.1319  19.2 FPS  694M", flush=True)
    print(f"  DRCT:      PieAPP=0.44922  S_F=0.1196  0.1 FPS  4094M", flush=True)
    print(f"  RealESRNet:PieAPP=0.47863  S_F=0.1113  1.8 FPS  2122M", flush=True)
    print(f"\nTarget: PieAPP < 0.092 → S_F > 0.32", flush=True)

    with open('/root/pipeline_audit/x4_hat_dat_benchmark.json','w') as f:
        json.dump({'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),'results':results}, f, indent=2)

if __name__=='__main__':
    main()
