#!/usr/bin/env python3
"""Final quality comparison: PLKSR vs DAT-light vs two-stage vs DAT-full.
All at 480x270→1920x1080 with HEVC CQ=20 yuv420p encoding, PieAPP vs GT."""
import sys, os, json, time, math, subprocess, gc, traceback
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
sys.path.insert(0, '/root/vidaio-subnet')
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn.functional as F
import numpy as np
from spandrel import ModelLoader

GT = '/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
PAYLOAD = '/root/pipeline_audit/payload_270p.mp4'
S = os.path.expanduser('~/.cache/span')
N_FRAMES = 60

def sigmoid(x): return 1/(1+np.exp(-x))
def calc_sq(p):
    s=sigmoid(p); a0=(1-(np.log10(sigmoid(0)+1)/np.log10(3.5)))**2.5
    a2=(1-(np.log10(sigmoid(2.0)+1)/np.log10(3.5)))**2.5; v=(1-(np.log10(s+1)/np.log10(3.5)))**2.5
    return 1-((v-a0)/(a2-a0))
def calc_sf(p,cl=10):
    sq=calc_sq(p);sl=math.log(1+cl)/math.log(1+320);sp=0.5*sq+0.5*sl
    return 0.1*math.exp(6.979*(sp-0.5)),sq

def measure_pieapp(ref, dist, n_samples=16):
    import pyiqa, cv2
    device = torch.device('cuda')
    metric = pyiqa.create_metric('pieapp', device=device)
    rc = cv2.VideoCapture(str(ref)); dc = cv2.VideoCapture(str(dist))
    total = min(int(rc.get(cv2.CAP_PROP_FRAME_COUNT)), int(dc.get(cv2.CAP_PROP_FRAME_COUNT)))
    n = min(n_samples, total); indices = np.linspace(0, total-1, n, dtype=int)
    scores = []
    for idx in indices:
        rc.set(cv2.CAP_PROP_POS_FRAMES, int(idx)); dc.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok1, rf = rc.read(); ok2, df = dc.read()
        if not ok1 or not ok2: continue
        if rf.shape != df.shape:
            df = cv2.resize(df, (rf.shape[1], rf.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        rt = torch.from_numpy(cv2.cvtColor(rf, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
        dt = torch.from_numpy(cv2.cvtColor(df, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
        with torch.no_grad():
            s = metric(dt, rt).item()
            if s < 0: s = abs(s)
        scores.append(s)
    rc.release(); dc.release(); del metric; torch.cuda.empty_cache()
    return float(np.mean(scores)), float(np.std(scores))

def get_info(path):
    r = subprocess.run(['ffprobe','-v','quiet','-select_streams','v:0',
        '-show_entries','stream=width,height,r_frame_rate','-of','json',str(path)],
        stdout=subprocess.PIPE, text=True)
    s = json.loads(r.stdout)['streams'][0]
    w, h = int(s['width']), int(s['height'])
    rfr = s.get('r_frame_rate','30/1'); parts = rfr.split('/')
    fps = float(parts[0])/float(parts[1]) if len(parts)==2 else 30.0
    return w, h, fps

def upscale_and_encode(model, payload, output, n_frames, w, h, fps, pad_mult=64, dtype=torch.float16):
    """Upscale n_frames and encode to HEVC yuv420p CQ=20."""
    pad_h = (pad_mult - h % pad_mult) % pad_mult
    pad_w = (pad_mult - w % pad_mult) % pad_mult
    need_pad = pad_h > 0 or pad_w > 0
    out_w, out_h = w * 4, h * 4
    frame_size = w * h * 3

    decoder = subprocess.Popen(
        ['ffmpeg','-i',str(payload),'-frames:v',str(n_frames),
         '-f','rawvideo','-pix_fmt','rgb24','-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen(
        ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{out_w}x{out_h}','-r',str(fps),'-i','pipe:0',
         '-c:v','hevc_nvenc','-cq','20','-preset','p4',
         '-profile:v','main','-pix_fmt','yuv420p','-sar','1:1',
         '-movflags','+faststart',str(output)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    total = 0; t_inf = 0
    while total < n_frames:
        raw = decoder.stdout.read(frame_size)
        if len(raw) < frame_size: break
        f = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
        t = torch.from_numpy(f.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to("cuda:0", dtype=dtype)
        if need_pad: t = F.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
        t0 = time.time()
        with torch.no_grad(): out = model(t)
        torch.cuda.synchronize()
        t_inf += time.time() - t0
        if need_pad: out = out[:,:,:h*4,:w*4]
        out_np = (out.float().clamp_(0,1).mul_(255).round_().to(torch.uint8)
                  .permute(0,2,3,1).contiguous().cpu().numpy())
        encoder.stdin.write(out_np[0].tobytes())
        total += 1
    encoder.stdin.close(); encoder.wait(); decoder.wait()
    return total, t_inf

def upscale_two_stage_1080(payload, output, n_frames, w, h, fps, dat_model, vsr):
    """Two-stage: 480x270 → 240x135 → DAT-light 4x → 960x540 → nvvfx 2x → 1920x1080."""
    half_w = (w // 2) & ~1  # 240
    half_h = (h // 2) & ~1  # 134 (even)
    pad_h = (64 - half_h % 64) % 64
    pad_w = (64 - half_w % 64) % 64
    need_pad = pad_h > 0 or pad_w > 0
    mid_w, mid_h = half_w * 4, half_h * 4
    final_w, final_h = w * 4, h * 4

    from services.upscaling.server import _rgb_to_nv12_bt601_gpu
    half_frame_size = half_w * half_h * 3

    decoder = subprocess.Popen(
        ['ffmpeg','-i',str(payload),'-frames:v',str(n_frames),
         '-vf',f'scale={half_w}:{half_h}',
         '-f','rawvideo','-pix_fmt','rgb24','-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    encoder = subprocess.Popen(
        ['ffmpeg','-y','-f','rawvideo','-pix_fmt','nv12',
         '-s',f'{final_w}x{final_h}','-r',str(fps),'-i','pipe:0',
         '-c:v','hevc_nvenc','-cq','20','-preset','p4',
         '-profile:v','main','-pix_fmt','yuv420p','-sar','1:1',
         '-movflags','+faststart',str(output)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    total = 0; t_inf = 0
    while total < n_frames:
        raw = decoder.stdout.read(half_frame_size)
        if len(raw) < half_frame_size: break
        f = np.frombuffer(raw, dtype=np.uint8).reshape(half_h, half_w, 3).copy()
        inp = torch.from_numpy(f.astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to("cuda:0", dtype=torch.float16)
        if need_pad: inp = F.pad(inp, (0, pad_w, 0, pad_h), mode='reflect')
        t0 = time.time()
        with torch.no_grad(): mid = dat_model(inp)
        torch.cuda.synchronize()
        if need_pad: mid = mid[:,:,:half_h*4,:half_w*4]
        mid_rgb = mid[0].float().clamp_(0,1).contiguous().detach()
        result = vsr.run(mid_rgb)
        out_gpu = torch.from_dlpack(result.image).clone()
        torch.cuda.synchronize()
        t_inf += time.time() - t0
        nv12 = _rgb_to_nv12_bt601_gpu(out_gpu.clamp_(0,1), final_h, final_w)
        encoder.stdin.write(nv12.contiguous().cpu().numpy().tobytes())
        total += 1
    encoder.stdin.close(); encoder.wait(); decoder.wait()
    return total, t_inf

# ========================================
w, h, fps = get_info(PAYLOAD)
print(f"Payload: {w}x{h} @ {fps}fps")
print(f"GT: 1920x1080")
print(f"{'='*70}")

results = []

# Config 1: PLKSR 4x direct
print(f"\n--- PLKSR 4x direct (current prod) ---")
torch.cuda.empty_cache(); gc.collect()
model = ModelLoader(device="cuda:0").load_from_file(f'{S}/PLKSR_X4_DF2K.pth').model.eval().half().cuda()
out_path = '/root/pipeline_audit/qual_plksr.mp4'
nf, t = upscale_and_encode(model, PAYLOAD, out_path, N_FRAMES, w, h, fps, pad_mult=4)
fps_val = nf/t if t > 0 else 0
print(f"  {nf}f in {t:.1f}s ({fps_val:.1f} FPS)")
print(f"  PieAPP...", flush=True)
pa_mean, pa_std = measure_pieapp(GT, out_path)
sf, sq = calc_sf(pa_mean)
tag = "BONUS!" if sf > 0.32 else "below"
print(f"  PieAPP={pa_mean:.5f}±{pa_std:.5f}  S_F={sf:.4f} [{tag}]")
results.append(('PLKSR 4x', nf, fps_val, pa_mean, pa_std, sq, sf, sf>0.32))
del model; torch.cuda.empty_cache(); gc.collect()
os.unlink(out_path)

# Config 2: DAT-light 4x direct
print(f"\n--- DAT-light 4x direct ---")
torch.cuda.empty_cache(); gc.collect()
model = ModelLoader(device="cuda:0").load_from_file(f'{S}/DAT_light_x4.pth').model.eval().half().cuda()
out_path = '/root/pipeline_audit/qual_datlight.mp4'
nf, t = upscale_and_encode(model, PAYLOAD, out_path, N_FRAMES, w, h, fps, pad_mult=64)
fps_val = nf/t if t > 0 else 0
print(f"  {nf}f in {t:.1f}s ({fps_val:.1f} FPS)")
print(f"  PieAPP...", flush=True)
pa_mean, pa_std = measure_pieapp(GT, out_path)
sf, sq = calc_sf(pa_mean)
tag = "BONUS!" if sf > 0.32 else "below"
print(f"  PieAPP={pa_mean:.5f}±{pa_std:.5f}  S_F={sf:.4f} [{tag}]")
results.append(('DAT-light 4x', nf, fps_val, pa_mean, pa_std, sq, sf, sf>0.32))
del model; torch.cuda.empty_cache(); gc.collect()
os.unlink(out_path)

# Config 3: Two-stage (480x270 → 240x134 → DAT-light 4x → 960x536 → nvvfx 2x → 1920x1072)
# Note: this tests the two-stage approach at 1080p for PieAPP comparison
print(f"\n--- Two-stage (downscale→DAT-light→nvvfx 2x) ---")
torch.cuda.empty_cache(); gc.collect()
try:
    import nvvfx
    dat_model = ModelLoader(device="cuda:0").load_from_file(f'{S}/DAT_light_x4.pth').model.eval().half().cuda()
    from services.upscaling.server import _get_vsr
    vsr = _get_vsr(1920, 1080, 0)  # final output = 1920x1080

    out_path = '/root/pipeline_audit/qual_twostage.mp4'
    nf, t = upscale_two_stage_1080(PAYLOAD, out_path, N_FRAMES, w, h, fps, dat_model, vsr)
    fps_val = nf/t if t > 0 else 0
    print(f"  {nf}f in {t:.1f}s ({fps_val:.1f} FPS)")
    print(f"  PieAPP...", flush=True)
    pa_mean, pa_std = measure_pieapp(GT, out_path)
    sf, sq = calc_sf(pa_mean)
    tag = "BONUS!" if sf > 0.32 else "below"
    print(f"  PieAPP={pa_mean:.5f}±{pa_std:.5f}  S_F={sf:.4f} [{tag}]")
    results.append(('Two-stage', nf, fps_val, pa_mean, pa_std, sq, sf, sf>0.32))
    del dat_model; torch.cuda.empty_cache(); gc.collect()
    os.unlink(out_path)
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

# Config 4: DAT-full bf16 direct (30 frames only — slow)
print(f"\n--- DAT-full bf16 4x direct (30 frames) ---")
torch.cuda.empty_cache(); gc.collect()
model = ModelLoader(device="cuda:0").load_from_file(f'{S}/DAT_x4.safetensors').model.eval().bfloat16().cuda()
out_path = '/root/pipeline_audit/qual_datfull.mp4'
nf, t = upscale_and_encode(model, PAYLOAD, out_path, 30, w, h, fps, pad_mult=64, dtype=torch.bfloat16)
fps_val = nf/t if t > 0 else 0
print(f"  {nf}f in {t:.1f}s ({fps_val:.1f} FPS)")
print(f"  PieAPP...", flush=True)
pa_mean, pa_std = measure_pieapp(GT, out_path, n_samples=min(16, nf))
sf, sq = calc_sf(pa_mean)
tag = "BONUS!" if sf > 0.32 else "below"
print(f"  PieAPP={pa_mean:.5f}±{pa_std:.5f}  S_F={sf:.4f} [{tag}]")
results.append(('DAT-full bf16', nf, fps_val, pa_mean, pa_std, sq, sf, sf>0.32))
del model; torch.cuda.empty_cache(); gc.collect()
os.unlink(out_path)

# Summary
print(f"\n{'='*70}")
print(f"FINAL QUALITY COMPARISON (480x270→1080p, HEVC CQ=20 yuv420p)")
print(f"{'='*70}")
for name, nf, fps_val, pa, pa_std, sq, sf, bonus in results:
    tag = "BONUS!" if bonus else "below"
    print(f"  {name:<22} PieAPP={pa:.5f}±{pa_std:.5f}  S_Q={sq:.4f}  S_F={sf:.4f}  FPS={fps_val:>5.1f}  [{tag}]")
print(f"\n  Target: PieAPP < 0.092 → S_F > 0.32")

with open('/root/pipeline_audit/final_quality_results.json', 'w') as f:
    json.dump({'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
               'results': [{'name':r[0],'pieapp':r[3],'s_f':r[6],'fps':r[2],'bonus':r[7]} for r in results]}, f, indent=2)
