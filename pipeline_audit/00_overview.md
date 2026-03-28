# SN85 Vidaio Miner — Pipeline Audit Overview

**Target:** S_F > 0.32 consistently on both x2 (SD2HD) and x4 (SD24K) paths
**Date started:** 2026-03-26
**Miner:** LukeTao, RTX 4090, netuid 85

## Scoring Formula (Upscaling)

```
PieAPP → S_Q (sigmoid + log transform)
S_L = log(1 + content_length) / log(1 + 320)    # 10s → 0.4155, 5s → 0.3105
S_PRE = 0.5 * S_Q + 0.5 * S_L
S_F = 0.1 * exp(6.979 * (S_PRE - 0.5))
```

### Key Insight: PieAPP is the ONLY lever

| Content | S_L    | PieAPP needed for S_F>0.32 | S_Q needed |
|---------|--------|----------------------------|------------|
| 10s     | 0.4155 | < 0.092                    | > 0.918    |
| 5s      | 0.3105 | impossible (even 0.0→0.295)| > 1.029    |

**5-second content can NEVER reach S_F=0.32.** Only 10s content can hit bonus.

### PieAPP → S_F Lookup (10s content)

| PieAPP | S_Q    | S_F    | Status  |
|--------|--------|--------|---------|
| 0.00   | 1.0000 | 0.4262 | BONUS++ |
| 0.02   | 0.9819 | 0.4001 | BONUS++ |
| 0.05   | 0.9550 | 0.3643 | BONUS+  |
| 0.07   | 0.9373 | 0.3425 | BONUS   |
| 0.092  | 0.9179 | 0.3200 | THRESHOLD|
| 0.10   | 0.9111 | 0.3125 | below   |
| 0.15   | 0.8683 | 0.2691 | below   |
| 0.20   | 0.8266 | 0.2327 | below   |

## Pipeline Steps (Priority Order)

| # | Step | Impact | Status |
|---|------|--------|--------|
| 1 | 08_scoring | Understand formula, measure baseline | [x] analyzed |
| 2 | 05_upscale_x4 | PLKSR eager deployed, torch.compile removed | [x] fixed |
| 3 | 04_upscale_x2 | nvvfx HIGHBITRATE_ULTRA (PieAPP 0.113→S_F 0.301) | [x] improved |
| 4 | 06_encode | CQ irrelevant (yuv420p dominates), no change | [x] confirmed |
| 5 | 07_upload | S3 multipart configured, 4.7s/25MB | [x] confirmed |
| 6 | 01_download | Streaming via ffmpeg, no download step | [x] confirmed |
| 7 | 02_decode | CPU 182fps, NVDEC slower — no change | [x] confirmed |
| 8 | 03_colorspace | BT.601/709 matrices verified correct | [x] confirmed |

## Baseline Measurements (2026-03-26)

| Path | Model | PieAPP | S_Q | S_F | FPS | Status |
|------|-------|--------|-----|-----|-----|--------|
| x4 SD24K | PLKSR_X4_DF2K | 0.335 | 0.720 | 0.160 | 23 | BELOW |
| x2 SD2HD | SPAN x2 (fallback) | 2.000 | 0.000 | 0.013 | 44 | CATASTROPHIC |
| x2 SD2HD | nvidia-vfx HIGH (old) | 0.249 | 0.787 | 0.203 | 193 | BELOW |
| x2 SD2HD | nvidia-vfx HIGHBITRATE_ULTRA (new) | 0.113 | 0.900 | 0.301 | 186 | CLOSE (0.019 gap) |

**Key insight:** x4 PLKSR has HUGE per-frame variance: PieAPP ranges from 0.034 to 0.940.
The worst frames (high-texture, high-motion) drag the average way up.

**Gap to target:** Need PieAPP from 0.335 → 0.092 (3.6x improvement needed for x4).

## Change Log

| Date | Step | Change | PieAPP Before | PieAPP After | S_F Impact |
|------|------|--------|---------------|--------------|------------|
| 03-26 | baseline | Initial measurement | — | x4: 0.335, x2: 2.0 | x4: 0.16, x2: 0.01 |
| 03-27 | 05_x4 | Benchmarked 10 models + TRT/compile | PLKSR: 0.41 | best: DAT-light 0.165 | 0.13→0.26 |
| 03-27 | 05_x4 | TRT conversion attempts (all failed) | — | — | — |
| 03-27 | 06_encode | CQ sweep: yuv420p adds +0.13 PieAPP, CQ irrelevant | — | — | — |
| 03-27 | 05_x4 | DAT-full after yuv420p: PieAPP=0.049 (BONUS!) | PLKSR: 0.41 | DAT-full: 0.049 | 0.13→0.37 |
| 03-27 | 05_x4 | Two-stage pipeline implemented: DAT-light→nvvfx 2x | — | — | 4.66 FPS, 64s/300f |
| 03-27 | misc | Compression mining gets 60% emissions vs 40% upscaling | — | — | strategic |
| 03-27 | 04_x2 | nvvfx HIGH→HIGHBITRATE_ULTRA | 0.249 | 0.113 | 0.203→0.301 |
| 03-27 | 06-03 | Encode/upload/download/decode/colorspace audited | — | — | no changes needed |
| 03-27 | 04_x2 | Exhaustive x2 lever search (9 modes, filters, encoding) | 0.113 | 0.113 | ceiling confirmed |

## Pipeline Audit Complete (2026-03-27)

### Final Production S_F Scores
| Path | PieAPP | S_F | vs Before | Status |
|------|--------|-----|-----------|--------|
| x4 PLKSR eager | 0.190 | 0.240 | was 0 (timeout) | FIXED |
| x2 nvvfx HIGHBITRATE_ULTRA | 0.113 | 0.301 | was 0.203 (HIGH) | +48% improved |

### Gap to S_F > 0.32
| Path | Current | Needed PieAPP | Gap | Blocker |
|------|---------|---------------|-----|---------|
| x4 | S_F=0.240 | 0.092 | 0.098 | Only DAT-full (0.049) hits target but at 1.8 FPS |
| x2 | S_F=0.301 | 0.092 | 0.021 | nvvfx ceiling — all modes/filters/encoding tested |

### Exhaustive x2 Lever Search
All tested, none close the 0.021 gap:
- 9 nvvfx quality modes (HIGHBITRATE_ULTRA is best)
- Pre-filters: sharpen, denoise, bilateral (all hurt PieAPP)
- Post-filters: sharpen, CAS, blur (all hurt PieAPP)
- Encoding: CQ irrelevant, bt709 tags hurt HIGHBITRATE_ULTRA
- yuv444p: validator rejects (requires yuv420p)

## Implemented: Two-Stage x4 Pipeline (2026-03-27)

**downscale→DAT-light 4x→nvvfx 2x**
- Input: 960x540 → downscale to 480x270 → DAT-light 4x → 1920x1080 → nvvfx 2x → 3840x2160
- Speed: **4.66 FPS, 300f in 64s** (fits 82s inference budget)
- Output: 3840x2160 (exact match for validator)
- Quality: TBD — needs PieAPP measurement with proper 4K ground truth
- File: `services/upscaling/server.py` — `_upscale_two_stage()`
- Enabled via `_TWO_STAGE_ENABLED = True`
- Falls back to PLKSR on failure

**Key discoveries:**
- Venv torch 2.8 has 14x regression for DAT attention ops (0.8 FPS vs 11.7 FPS)
- Fixed by preferring system torch 2.7 in sys.path
- yuv420p chroma subsampling adds ~0.13 PieAPP (CQ is irrelevant)
- DAT-full after yuv420p achieves PieAPP=0.049 (BONUS!) but too slow (1.6 FPS)
- Compression mining gets 60% emissions vs 40% for upscaling — strategic consideration

## Current Scores

See `baseline_scores.json` and `current_scores.json` for raw data.
