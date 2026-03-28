#!/usr/bin/env python3
"""Verify x2 quality modes + bt709 fix with consistent conditions."""
import sys
sys.path.insert(0, '/usr/local/lib/python3.12/dist-packages')
sys.path.insert(0, '/root/vidaio-subnet')
sys.stdout.reconfigure(line_buffering=True)

import torch, numpy as np, subprocess, json, time, math, os, gc

GT = '/root/vidaio-subnet/tmp/1693996b-ebb9-44ce-bdf7-ee74f63bc78a.mp4'
PAYLOAD = '/root/pipeline_audit/payload_x2_clean.mp4'
OUT_DIR = '/root/pipeline_audit/x2_verify'
os.makedirs(OUT_DIR, exist_ok=True)

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

import nvvfx

# Get payload info
r = subprocess.run(['ffprobe','-v','quiet','-select_streams','v:0',
    '-show_entries','stream=width,height,r_frame_rate','-of','json', PAYLOAD],
    stdout=subprocess.PIPE, text=True)
s = json.loads(r.stdout)['streams'][0]
w, h = int(s['width']), int(s['height'])
rfr = s.get('r_frame_rate','30/1'); parts = rfr.split('/')
fps = float(parts[0])/float(parts[1])
out_w, out_h = 1920, 1080
N = 60

print(f"Input: {w}x{h} → nvvfx 2x → {out_w}x{out_h}")
print(f"GT: {GT}")

def run_nvvfx_encode(quality_name, quality_val, extra_encode_flags=None, label=None):
    """Run nvvfx x2 and encode with optional extra flags."""
    if label is None:
        label = quality_name

    vsr = nvvfx.VideoSuperRes(quality=quality_val, device=0)
    vsr.output_width = out_w
    vsr.output_height = out_h
    vsr.load()

    frame_size = w * h * 3
    out_path = f'{OUT_DIR}/{label}.mp4'

    decoder = subprocess.Popen(
        ['ffmpeg','-i', PAYLOAD,'-frames:v',str(N),
         '-f','rawvideo','-pix_fmt','rgb24','-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    enc_cmd = ['ffmpeg','-y','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{out_w}x{out_h}','-r',str(fps),'-i','pipe:0',
        '-c:v','hevc_nvenc','-cq','20','-preset','p4',
        '-profile:v','main','-pix_fmt','yuv420p','-sar','1:1',
        '-movflags','+faststart']
    if extra_encode_flags:
        enc_cmd.extend(extra_encode_flags)
    enc_cmd.append(out_path)

    encoder = subprocess.Popen(enc_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    total = 0; t_inf = 0
    while total < N:
        raw = decoder.stdout.read(frame_size)
        if len(raw) < frame_size: break
        f = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
        rgb = torch.from_numpy(f.astype(np.float32)/255.0).permute(2,0,1).contiguous().to('cuda')
        t0 = time.time()
        result = vsr.run(rgb)
        out_gpu = torch.from_dlpack(result.image).clone()
        torch.cuda.synchronize()
        t_inf += time.time() - t0
        out_np = (out_gpu.clamp_(0,1).mul_(255).round_().to(torch.uint8)
                  .permute(1,2,0).contiguous().cpu().numpy())
        encoder.stdin.write(out_np.tobytes())
        total += 1

    encoder.stdin.close(); encoder.wait(); decoder.wait()
    del vsr

    if total == 0:
        return None

    fps_val = total / t_inf
    pa_mean, pa_std = measure_pieapp(GT, out_path)
    sf, sq = calc_sf(pa_mean)
    bonus = "BONUS!" if sf > 0.32 else ("CLOSE" if sf > 0.28 else "below")
    print(f"  {label:<30} PieAPP={pa_mean:.5f}±{pa_std:.5f}  S_F={sf:.4f}  {fps_val:.0f}FPS  [{bonus}]")
    os.unlink(out_path)
    return {'label': label, 'pieapp': pa_mean, 'sf': sf, 'fps': fps_val, 'bonus': sf > 0.32}

ql = nvvfx.VideoSuperRes.QualityLevel

print(f"\n{'='*70}")
print(f"1. Quality mode comparison (same payload, same encode)")
print(f"{'='*70}")

for name, val in [('HIGH', ql.HIGH), ('ULTRA', ql.ULTRA),
                   ('HIGHBITRATE_HIGH', ql.HIGHBITRATE_HIGH),
                   ('HIGHBITRATE_ULTRA', ql.HIGHBITRATE_ULTRA)]:
    run_nvvfx_encode(name, val)

print(f"\n{'='*70}")
print(f"2. bt709 colorspace tagging (HIGHBITRATE_ULTRA + bt709 flags)")
print(f"{'='*70}")

# Baseline: no color flags (current production)
run_nvvfx_encode('HIGHBITRATE_ULTRA', ql.HIGHBITRATE_ULTRA, label='HBU_no_color_flags')

# With bt709 colorspace flags on encode
run_nvvfx_encode('HIGHBITRATE_ULTRA', ql.HIGHBITRATE_ULTRA,
    extra_encode_flags=['-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709'],
    label='HBU_bt709_tags')

# With full colorspace conversion filter + tags
run_nvvfx_encode('HIGHBITRATE_ULTRA', ql.HIGHBITRATE_ULTRA,
    extra_encode_flags=['-vf','colorspace=all=bt709:iall=bt709',
                        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709'],
    label='HBU_bt709_convert')

# Same bt709 tests with ULTRA (best quality mode)
run_nvvfx_encode('ULTRA', ql.ULTRA, label='ULTRA_no_color_flags')
run_nvvfx_encode('ULTRA', ql.ULTRA,
    extra_encode_flags=['-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709'],
    label='ULTRA_bt709_tags')
run_nvvfx_encode('ULTRA', ql.ULTRA,
    extra_encode_flags=['-vf','colorspace=all=bt709:iall=bt709',
                        '-colorspace','bt709','-color_primaries','bt709','-color_trc','bt709'],
    label='ULTRA_bt709_convert')

print(f"\nTarget: PieAPP ≤ 0.092, S_F > 0.32")
