# 08 — Scoring Analysis

## Current Implementation

**File:** `/root/vidaio-subnet/services/scoring/server.py` (lines 981-994)

### Quality Score (S_Q) from PieAPP

```python
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_quality_score(pieapp_score):
    sigmoid_normalized_score = sigmoid(pieapp_score)
    original_at_zero = (1 - (np.log10(sigmoid(0) + 1) / np.log10(3.5))) ** 2.5
    original_at_two = (1 - (np.log10(sigmoid(2.0) + 1) / np.log10(3.5))) ** 2.5
    original_value = (1 - (np.log10(sigmoid_normalized_score + 1) / np.log10(3.5))) ** 2.5
    scaled_value = 1 - ((original_value - original_at_zero) / (original_at_two - original_at_zero))
    return scaled_value
```

Constants:
- `original_at_zero` ≈ 0.3761 (PieAPP=0 → S_Q=1.0)
- `original_at_two` ≈ 0.1732 (PieAPP=2.0 → S_Q=0.0)

### Length Score (S_L)

```python
S_L = log(1 + content_length) / log(1 + 320)
```

- 5s → 0.3105
- 10s → 0.4155 (our MAX_CONTENT_LEN)

### Preliminary Score

```python
S_PRE = 0.5 * S_Q + 0.5 * S_L
```

### Final Score (Exponential)

```python
S_F = 0.1 * exp(6.979 * (S_PRE - 0.5))
```

This exponential creates sharp discrimination:
- S_PRE=0.5 → S_F=0.1
- S_PRE=0.667 → S_F=0.32 (our target)
- S_PRE=0.7 → S_F=0.41
- S_PRE=1.0 → S_F=108.85 (capped in practice)

### VMAF Gate

```python
if vmaf_score / 100 < VMAF_THRESHOLD:  # VMAF_THRESHOLD=0.5 → need VMAF>50
    score = 0
```

VMAF > 50 is required. Our nvidia-vfx achieves 91.55 mean — well above gate.

### PieAPP Clamp

```python
return min(avg_score, 2.0)  # pieapp_metric.py line 86
```

PieAPP is clamped to [0, 2.0] range.

### PieAPP Sampling

```python
PIEAPP_SAMPLE_COUNT = CONFIG.score.pieapp_sample_count  # default: 4
```

Only 4 frames sampled from a random starting position. High variance possible.

## Key Findings

1. **PieAPP is the ONLY quality lever** — VMAF is just a pass/fail gate
2. **Content length is fixed** at 10s (MAX_CONTENT_LEN), giving S_L=0.4155
3. **Need PieAPP < 0.092** for S_F > 0.32
4. **4-frame sampling** means individual frame quality variance matters a lot
5. **5s content can never reach bonus** — max S_F is 0.295 even with perfect PieAPP

## Improvement Opportunities

| Opportunity | Est. Score Gain | Ease | Priority |
|-------------|----------------|------|----------|
| Lower PieAPP via better SR model | +0.05-0.15 S_F | medium | **1** |
| Lower PieAPP via encoding quality | +0.01-0.03 S_F | easy | **2** |
| Reduce frame-to-frame variance | consistency | medium | **3** |
| Increase MAX_CONTENT_LEN to 20s+ | +S_L but timeout risk | hard | **4** |

## Status: [x] analyzed
