#!/usr/bin/env python3
"""Benchmark DAT x4 TensorRT FP16 engine vs PyTorch bf16 baseline."""
import sys, os, json, time, math, subprocess, gc
import numpy as np
sys.path.insert(0, '/root/vidaio-subnet')
sys.stdout.reconfigure(line_buffering=True)

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


class TRTInference:
    """TensorRT engine inference wrapper."""
    def __init__(self, engine_path):
        import tensorrt as trt
        import torch
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Get I/O tensor info
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.input_shape = self.engine.get_tensor_shape(self.input_name)
        self.output_shape = self.engine.get_tensor_shape(self.output_name)

        print(f"  TRT Input:  {self.input_name} {list(self.input_shape)}")
        print(f"  TRT Output: {self.output_name} {list(self.output_shape)}")

        # Pre-allocate GPU buffers
        self.input_buf = torch.empty(list(self.input_shape), dtype=torch.float32, device='cuda')
        self.output_buf = torch.empty(list(self.output_shape), dtype=torch.float32, device='cuda')

        # Set tensor addresses
        self.context.set_tensor_address(self.input_name, self.input_buf.data_ptr())
        self.context.set_tensor_address(self.output_name, self.output_buf.data_ptr())

        # Create CUDA stream
        self.stream = torch.cuda.Stream()

    def __call__(self, x):
        """Run inference. x: (1,3,H,W) float32 CUDA tensor."""
        self.input_buf.copy_(x)
        with torch.cuda.stream(self.stream):
            self.context.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return self.output_buf.clone()

    def __del__(self):
        del self.context, self.engine


def upscale_trt(input_path, output_path, engine_path, n_frames=300):
    """Upscale video using TRT engine."""
    import torch
    w, h, fps, dur, nf = get_info(input_path)
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    need_pad = pad_h > 0 or pad_w > 0
    out_w, out_h = w * 4, h * 4
    frame_size = w * h * 3
    h_pad, w_pad = h + pad_h, w + pad_w

    print(f"  Input: {w}x{h}, Pad: {w_pad}x{h_pad}, Output: {out_w}x{out_h}")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    print(f"  Loading TRT engine...")
    trt_model = TRTInference(engine_path)

    # Warmup
    dummy = torch.randn(1, 3, h_pad, w_pad, device='cuda', dtype=torch.float32)
    for _ in range(3):
        _ = trt_model(dummy)
    del dummy
    print(f"  Warmup done")

    decoder = subprocess.Popen(
        ['ffmpeg', '-i', str(input_path), '-frames:v', str(n_frames),
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen(
        ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{out_w}x{out_h}', '-r', str(fps), '-i', 'pipe:0',
         '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
         '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-sar', '1:1',
         '-movflags', '+faststart', str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    total = 0
    t_inf = 0.0
    try:
        while total < n_frames:
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            f = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            t = torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).cuda()
            if need_pad:
                t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='reflect')

            t0 = time.time()
            out = trt_model(t)
            t_inf += time.time() - t0

            if need_pad:
                out = out[:, :, :h * 4, :w * 4]
            out_np = (out.clamp_(0, 1).mul_(255).round_().to(torch.uint8)
                      .permute(0, 2, 3, 1).contiguous().cpu().numpy())
            encoder.stdin.write(out_np[0].tobytes())
            total += 1
            if total % 30 == 0:
                print(f"  {total}f ({total/t_inf:.1f} FPS)...", flush=True)
    except Exception as e:
        print(f"  FAILED at {total}f: {e}", flush=True)
        import traceback; traceback.print_exc()
        try:
            decoder.kill(); encoder.kill()
        except:
            pass
        del trt_model; torch.cuda.empty_cache(); gc.collect()
        return 0, 0, 0

    encoder.stdin.close(); encoder.wait(); decoder.wait()
    vram = torch.cuda.max_memory_allocated() / 1e6
    del trt_model; torch.cuda.empty_cache(); gc.collect()
    return total, t_inf, vram


def upscale_pytorch(input_path, output_path, model_path, n_frames=300, dtype_str='bf16'):
    """Upscale using PyTorch for comparison."""
    import torch
    from spandrel import ModelLoader
    w, h, fps, dur, nf = get_info(input_path)
    pad_h = (64 - h % 64) % 64
    pad_w = (64 - w % 64) % 64
    need_pad = pad_h > 0 or pad_w > 0
    out_w, out_h = w * 4, h * 4
    frame_size = w * h * 3
    h_pad, w_pad = h + pad_h, w + pad_w

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    descriptor = ModelLoader(device="cuda:0").load_from_file(str(model_path))
    model = descriptor.model.eval()
    if dtype_str == 'bf16':
        model = model.bfloat16(); dtype = torch.bfloat16
    elif dtype_str == 'fp16':
        model = model.half(); dtype = torch.float16
    else:
        dtype = torch.float32

    # Warmup
    dummy = torch.randn(1, 3, h_pad, w_pad, device='cuda', dtype=dtype)
    with torch.no_grad():
        _ = model(dummy)
    torch.cuda.synchronize()
    del dummy

    decoder = subprocess.Popen(
        ['ffmpeg', '-i', str(input_path), '-frames:v', str(n_frames),
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen(
        ['ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
         '-s', f'{out_w}x{out_h}', '-r', str(fps), '-i', 'pipe:0',
         '-c:v', 'hevc_nvenc', '-cq', '20', '-preset', 'p4',
         '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-sar', '1:1',
         '-movflags', '+faststart', str(output_path)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    total = 0; t_inf = 0.0
    try:
        while total < n_frames:
            raw = decoder.stdout.read(frame_size)
            if len(raw) < frame_size: break
            f = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            t = torch.from_numpy(f.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to("cuda:0", dtype=dtype)
            if need_pad: t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
            t0 = time.time()
            with torch.no_grad(): out = model(t)
            torch.cuda.synchronize()
            t_inf += time.time() - t0
            if need_pad: out = out[:, :, :h*4, :w*4]
            out_np = (out.float().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
                      .permute(0, 2, 3, 1).contiguous().cpu().numpy())
            encoder.stdin.write(out_np[0].tobytes())
            total += 1
            if total % 10 == 0: print(f"  {total}f ({total/t_inf:.1f} FPS)...", flush=True)
    except Exception as e:
        print(f"  FAILED at {total}f: {e}", flush=True)
        import traceback; traceback.print_exc()
        try: decoder.kill(); encoder.kill()
        except: pass
        del model; torch.cuda.empty_cache(); gc.collect()
        return 0, 0, 0

    encoder.stdin.close(); encoder.wait(); decoder.wait()
    vram = torch.cuda.max_memory_allocated() / 1e6
    del model; torch.cuda.empty_cache(); gc.collect()
    return total, t_inf, vram


def main():
    GT = '/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
    PAYLOAD = '/root/pipeline_audit/payload_270p.mp4'
    MODEL_PATH = os.path.expanduser('~/.cache/span/DAT_x4.safetensors')
    ENGINE_PATH = '/root/pipeline_audit/dat_x4_fp16.engine'
    N = 60  # frames for speed benchmark

    print(f"{'='*70}")
    print(f"DAT x4 TensorRT FP16 vs PyTorch bf16 Benchmark")
    print(f"{'='*70}")

    results = []

    # --- TRT FP16 ---
    if os.path.exists(ENGINE_PATH):
        print(f"\n{'─'*70}")
        print(f"  DAT x4 TensorRT FP16")
        print(f"{'─'*70}")
        output = '/root/pipeline_audit/bench_DAT_TRT_FP16.mp4'
        t0 = time.time()
        nf, t_inf, vram = upscale_trt(PAYLOAD, output, ENGINE_PATH, n_frames=N)
        t_total = time.time() - t0

        if nf > 0:
            fps = nf / t_inf
            ext300 = 300 / fps
            print(f"  {nf}f in {t_inf:.1f}s ({fps:.1f} FPS), VRAM={vram:.0f}MB")
            print(f"  Extrapolated 300f: {ext300:.0f}s")

            print(f"  PieAPP...", flush=True)
            pieapp = measure_pieapp(GT, output, n_samples=min(16, nf))
            sf, sq = calc_sf(pieapp['mean'])
            tag = "BONUS!" if sf > 0.32 else "below"
            print(f"  PieAPP={pieapp['mean']:.5f} ±{pieapp['std']:.5f} [{pieapp['min']:.5f}..{pieapp['max']:.5f}]")
            print(f"  S_Q={sq:.4f}  S_F={sf:.4f}  [{tag}]")

            results.append({
                'model': 'DAT x4 TRT FP16', 'n_frames': nf,
                'pieapp_mean': round(pieapp['mean'], 5), 'pieapp_std': round(pieapp['std'], 5),
                'pieapp_min': round(pieapp['min'], 5), 'pieapp_max': round(pieapp['max'], 5),
                's_q': round(sq, 4), 's_f': round(sf, 4), 'bonus': sf > 0.32,
                'fps': round(fps, 2), 'inference_s': round(t_inf, 2),
                'extrapolated_300f_s': round(ext300, 1), 'vram_mb': round(vram, 0),
            })
            if os.path.exists(output): os.unlink(output)
        else:
            print(f"  TRT FAILED")
            results.append({'model': 'DAT x4 TRT FP16', 'error': 'failed'})
    else:
        print(f"\nSKIP TRT — engine not found at {ENGINE_PATH}")

    # --- PyTorch bf16 baseline (30 frames only, it's slow) ---
    print(f"\n{'─'*70}")
    print(f"  DAT x4 PyTorch bf16 (baseline, 30 frames)")
    print(f"{'─'*70}")
    output = '/root/pipeline_audit/bench_DAT_bf16.mp4'
    t0 = time.time()
    nf, t_inf, vram = upscale_pytorch(PAYLOAD, output, MODEL_PATH, n_frames=30, dtype_str='bf16')
    t_total = time.time() - t0

    if nf > 0:
        fps = nf / t_inf
        ext300 = 300 / fps
        print(f"  {nf}f in {t_inf:.1f}s ({fps:.1f} FPS), VRAM={vram:.0f}MB")
        print(f"  Extrapolated 300f: {ext300:.0f}s")

        print(f"  PieAPP...", flush=True)
        pieapp = measure_pieapp(GT, output, n_samples=min(16, nf))
        sf, sq = calc_sf(pieapp['mean'])
        tag = "BONUS!" if sf > 0.32 else "below"
        print(f"  PieAPP={pieapp['mean']:.5f} ±{pieapp['std']:.5f} [{pieapp['min']:.5f}..{pieapp['max']:.5f}]")
        print(f"  S_Q={sq:.4f}  S_F={sf:.4f}  [{tag}]")

        results.append({
            'model': 'DAT x4 bf16 (baseline)', 'n_frames': nf,
            'pieapp_mean': round(pieapp['mean'], 5), 'pieapp_std': round(pieapp['std'], 5),
            'pieapp_min': round(pieapp['min'], 5), 'pieapp_max': round(pieapp['max'], 5),
            's_q': round(sq, 4), 's_f': round(sf, 4), 'bonus': sf > 0.32,
            'fps': round(fps, 2), 'inference_s': round(t_inf, 2),
            'extrapolated_300f_s': round(ext300, 1), 'vram_mb': round(vram, 0),
        })
        if os.path.exists(output): os.unlink(output)

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    valid = [r for r in results if 'error' not in r]
    for r in valid:
        tag = "BONUS!" if r.get('bonus') else "below"
        print(f"  {r['model']:<25} PieAPP={r['pieapp_mean']:.5f} S_F={r['s_f']:.4f} "
              f"FPS={r['fps']:.1f} 300f≈{r['extrapolated_300f_s']:.0f}s VRAM={r['vram_mb']:.0f}MB [{tag}]")

    print(f"\n  Target: PieAPP < 0.092, FPS ≥ 15, S_F > 0.32")
    print(f"  PLKSR (current prod): PieAPP=0.41 FPS=19.2 S_F=0.13")

    with open('/root/pipeline_audit/dat_trt_benchmark.json', 'w') as f:
        json.dump({'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'), 'results': results}, f, indent=2)
    print(f"\nResults saved to dat_trt_benchmark.json")


if __name__ == '__main__':
    main()
