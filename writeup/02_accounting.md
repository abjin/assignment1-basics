# Resource Accounting

Written answers for **Problem (transformer_accounting)** and **Problem (adamw_accounting)**.

Notation: `V` = vocab_size, `T` = context_length, `L` = num_layers, `d` = d_model,
`h` = num_heads, `d_ff` = feed-forward dimension. Our architecture (see
`cs336_basics/model.py`) uses bias-free `Linear` layers, RMSNorm (one learned gain vector
of size `d` each), a SwiGLU FFN with three weight matrices `w1, w3: d -> d_ff` and
`w2: d_ff -> d`, RoPE (no learned parameters), no weight tying between the token embedding
and `lm_head`, and no biases anywhere. Matrix-multiply FLOPs follow the rule
`(m x n) @ (n x p) = 2mnp` FLOPs.

All numbers below were verified with a Python script; its output is reproduced in the
appendix at the end of this document.

---

## Problem (transformer_accounting)

GPT-2 XL configuration: `V = 50,257`, `T = 1,024`, `L = 48`, `d = 1,600`, `h = 25`,
`d_ff = 4,288` (nearest multiple of 64 to (8/3) x 1,600).

### (a) Parameter count and memory

Per-component counts:

| Component | Formula | Count |
|---|---|---:|
| Token embedding | `V*d` | 80,411,200 |
| Attention (per block) | `4*d^2` (Q, K, V, output projections) | 10,240,000 |
| SwiGLU FFN (per block) | `3*d*d_ff` (w1, w2, w3) | 20,582,400 |
| RMSNorms (per block) | `2*d` | 3,200 |
| One block total | `4*d^2 + 3*d*d_ff + 2*d` | 30,825,600 |
| All 48 blocks | | 1,479,628,800 |
| Final RMSNorm | `d` | 1,600 |
| LM head | `V*d` | 80,411,200 |
| **Total** | `2*V*d + L*(4*d^2 + 3*d*d_ff + 2*d) + d` | **1,640,452,800** |

**Answer:** The model has **1,640,452,800 ≈ 1.64 B trainable parameters**. At 4 bytes per
float32 parameter, just loading the weights takes `4 x 1,640,452,800 =`
**6,561,811,200 bytes ≈ 6.56 GB (6.11 GiB)**.

### (b) Matrix multiplies in one forward pass (batch of one sequence of `T` tokens)

Per transformer block:

| Matmul | Shapes | FLOPs formula | FLOPs (XL) |
|---|---|---|---:|
| Q projection | `(T x d) @ (d x d)` | `2*T*d^2` | 5.24e9 |
| K projection | `(T x d) @ (d x d)` | `2*T*d^2` | 5.24e9 |
| V projection | `(T x d) @ (d x d)` | `2*T*d^2` | 5.24e9 |
| Attention scores `Q K^T` | `h` x `(T x d/h) @ (d/h x T)` | `2*T^2*d` | 3.36e9 |
| Weighted values `P V` | `h` x `(T x T) @ (T x d/h)` | `2*T^2*d` | 3.36e9 |
| Output projection | `(T x d) @ (d x d)` | `2*T*d^2` | 5.24e9 |
| FFN `w1` (gate) | `(T x d) @ (d x d_ff)` | `2*T*d*d_ff` | 1.40e10 |
| FFN `w3` (up) | `(T x d) @ (d x d_ff)` | `2*T*d*d_ff` | 1.40e10 |
| FFN `w2` (down) | `(T x d_ff) @ (d_ff x d)` | `2*T*d*d_ff` | 1.40e10 |
| Per-block total | | `8*T*d^2 + 4*T^2*d + 6*T*d*d_ff` | 6.98e10 |

Outside the blocks:

| Matmul | Shapes | FLOPs formula | FLOPs (XL) |
|---|---|---|---:|
| LM head (logits) | `(T x d) @ (d x V)` | `2*T*d*V` | 1.647e11 |

(The token embedding is a lookup, RoPE and RMSNorm are element-wise/reduction ops, and the
softmaxes are not matmuls, so none of them count here.)

Total:

```
FLOPs = L * (8*T*d^2 + 4*T^2*d + 6*T*d*d_ff) + 2*T*d*V
      = 48 * (2.097e10 + 6.71e9 + 4.215e10) + 1.647e11
```

**Answer: ≈ 3.517e12 FLOPs (about 3.52 TFLOPs) per forward pass**, broken down as:

| Component | FLOPs | Share |
|---|---:|---:|
| QKV projections (all layers) | 7.550e11 | 21.5% |
| Attention `Q K^T` (all layers) | 1.611e11 | 4.6% |
| Attention `P V` (all layers) | 1.611e11 | 4.6% |
| Output projections (all layers) | 2.517e11 | 7.2% |
| SwiGLU FFN (all layers) | 2.023e12 | 57.5% |
| LM head | 1.647e11 | 4.7% |
| **Total** | **3.5168e12** | 100% |

### (c) Which parts dominate?

The **feed-forward (SwiGLU) layers dominate at ~57.5%** of all FLOPs, and together with the
attention Q/K/V/output projections (~28.6%) the position-wise linear layers account for
~86% of the total; the actual attention matmuls `Q K^T` and `P V` (~9.2%) and the LM head
(~4.7%) are comparatively small at `T = 1,024`.

### (d) GPT-2 small / medium / large breakdown

Using `d_ff` = nearest multiple of 64 to (8/3)*d for each model (small: 2,048; medium:
2,752; large: 3,392), share of total forward FLOPs:

| Component | small (12L, 768d) | medium (24L, 1024d) | large (36L, 1280d) | XL (48L, 1600d) |
|---|---:|---:|---:|---:|
| QKV projections | 14.9% | 18.6% | 20.5% | 21.5% |
| Attention `Q K^T` | 6.6% | 6.2% | 5.5% | 4.6% |
| Attention `P V` | 6.6% | 6.2% | 5.5% | 4.6% |
| Output projections | 5.0% | 6.2% | 6.8% | 7.2% |
| FFN (SwiGLU) | 39.8% | 50.0% | 54.3% | 57.5% |
| LM head | 27.1% | 12.7% | 7.4% | 4.7% |
| **Total FLOPs** | **2.92e11** | **8.30e11** | **1.77e12** | **3.52e12** |

**Answer:** As the model grows, the FLOPs of the position-wise linear layers (FFN and
attention projections) scale as `L*d^2` and take up an increasingly large share (FFN goes
from ~40% to ~58%), while the LM head (fixed `2*T*d*V`, from 27% down to 5%) and the
attention score/value matmuls (`L*T^2*d`, which grow only linearly in `d`) shrink
proportionally.

### (e) GPT-2 XL with context length 16,384

Increasing `T` from 1,024 to 16,384 (16x) raises the total forward FLOPs from 3.52e12 to
**1.336e14 FLOPs — about a 38x increase** (super-linear, because the `Q K^T` and `P V`
terms scale with `T^2` while everything else scales with `T`). The relative contributions
flip: the attention score/value matmuls go from ~9% to **~62%** of total FLOPs (30.9% each),
while the FFN drops to ~24%, the QKVO projections to ~12%, and the LM head to ~2% — at long
context, attention itself becomes the dominant cost.

---

## Problem (adamw_accounting)

Everything in float32 (4 bytes per element). Let `B` = batch_size and assume
`d_ff = (8/3)*d`, so `3*d*d_ff = 8*d^2`.

### (a) Peak memory, decomposed

**Parameters.** From part (a) above:

```
P = 2*V*d + L*(4*d^2 + 3*d*d_ff + 2*d) + d
  = 2*V*d + 12*L*d^2 + 2*L*d + d          [using d_ff = 8d/3]
mem_params = 4*P bytes
```

**Gradients.** One float per parameter: `mem_grads = 4*P` bytes.

**Optimizer state.** AdamW keeps first and second moments `m, v`, each the shape of the
parameters: `mem_opt = 2 * 4*P = 8*P` bytes.

**Activations.** Counting only the components listed in the handout, each of shape
`(B, T, ...)`; float counts per batch element:

| Activation | Floats |
|---|---|
| RMSNorm outputs (2 per block) | `2*T*d` |
| Q, K, V projections | `3*T*d` |
| `Q K^T` attention scores | `h*T^2` |
| softmax output | `h*T^2` |
| weighted sum of values | `T*d` |
| output projection | `T*d` |
| FFN `w1` output (gate branch) | `T*d_ff` |
| FFN `w3` output (up branch) | `T*d_ff` |
| SiLU output | `T*d_ff` |
| element-wise product | `T*d_ff` |
| FFN `w2` output | `T*d` |
| **Per block** | `8*T*d + 4*T*d_ff + 2*h*T^2 = (56/3)*T*d + 2*h*T^2` |
| Final RMSNorm | `T*d` |
| Output embedding (logits) | `T*V` |
| Cross-entropy on logits (softmax kept for backward) | `T*V` |

```
A = B * [ L*((56/3)*T*d + 2*h*T^2) + T*d + 2*T*V ]   floats
mem_activations = 4*A bytes
```

**Total peak memory:**

```
mem_total = 4*A + 16*P
          = 4*B*[ L*((56/3)*T*d + 2*h*T^2) + T*d + 2*T*V ]
            + 16*[ 2*V*d + 12*L*d^2 + 2*L*d + d ]        bytes
```

### (b) Instantiated for GPT-2 XL

Plugging in `V = 50,257`, `T = 1,024`, `L = 48`, `d = 1,600`, `h = 25` (and the model's
actual `d_ff = 4,288`; using exactly `8d/3` changes the numbers by < 0.5%):

- Activations per batch element: `48 * 83,099,648 + 1,638,400 + 102,926,336`
  `= 4,093,347,840` floats = `16.373e9` bytes ≈ **15.25 GiB**
- Parameters + gradients + optimizer state: `16 * 1,640,452,800 = 26.247e9` bytes ≈
  **24.44 GiB**

```
mem(B) ≈ 16.37 GB * B + 26.25 GB        (a = 1.6373e10 bytes, b = 2.6247e10 bytes)
       ≈ 15.25 GiB * B + 24.44 GiB
```

Setting `mem(B) <= 80 GiB` gives `B <= (85.90e9 - 26.25e9) / 16.37e9 = 3.64`
(or `B <= 3.28` if "80 GB" means `80e9` bytes).

**Answer: the maximum batch size is 3.**

### (c) FLOPs for one AdamW step

The optimizer update is purely element-wise over the `P` parameters; per element
(treating scalar quantities like `alpha*lambda`, `alpha_t`, `1-beta_1`, `1-beta_2` as
precomputed once):

- weight decay `theta <- theta - alpha*lambda*theta`: 2 FLOPs
- `m <- beta_1*m + (1-beta_1)*g`: 3 FLOPs
- `v <- beta_2*v + (1-beta_2)*g^2`: 4 FLOPs
- `theta <- theta - alpha_t * m / (sqrt(v) + eps)`: 5 FLOPs

**Answer: ≈ 14*P FLOPs per step, i.e. Theta(P)** — about `2.3e10` FLOPs for GPT-2 XL,
which is negligible (~0.0002%) compared to the `~1e16` FLOPs of a forward+backward pass at
batch size 1024.

### (d) Training time at 50% MFU

Per training step at batch size 1024 and `T = 1,024`, assuming (Kaplan/Hoffmann) the
backward pass costs twice the forward pass:

```
FLOPs/step  = 1024 * 3 * 3.5168e12 = 1.080e16
Total FLOPs = 400,000 * 1.080e16   = 4.321e21
```

The handout specifies an **H100 with a theoretical peak of 495 TFLOP/s** for float32
(TF32); at 50% MFU the sustained throughput is `0.5 * 495e12 = 247.5e12` FLOP/s:

```
time = 4.321e21 / 2.475e14 = 1.746e7 s ≈ 4,850 hours ≈ 202 days
```

**Answer: ≈ 4,850 hours, i.e. about 202 days (~6.6 months) on a single H100 at 50% MFU.**

(For reference, on an **A100 at 19.5 TFLOP/s FP32** peak — the figure used in an earlier
version of this problem — the same computation gives `4.321e21 / 9.75e12 = 4.432e8 s ≈`
**123,100 hours ≈ 5,130 days ≈ 14 years**, which is why nobody trains GPT-2 XL in true
FP32 on one GPU.)

---

## Appendix: verification script output

Script: computes every number above from the formulas
(`scratchpad/accounting.py`, python3).

```
(a) GPT-2 XL: d_ff = 4288
    token embedding :      80,411,200
    per block       :      30,825,600  (attn 10,240,000 + ffn 20,582,400 + norms 3,200)
    blocks x48      :   1,479,628,800
    final RMSNorm   :           1,600
    lm_head         :      80,411,200
    TOTAL           :   1,640,452,800  = 1.6405 B params
    fp32 memory     : 6,561,811,200 bytes = 6.562 GB = 6.111 GiB

GPT-2 small: L=12 d=768 h=12 d_ff=2048  total fwd FLOPs = 2.9165e+11
    qkv_proj  14.91% | QK^T 6.63% | PV 6.63% | out_proj 4.97% | ffn 39.76% | lm_head 27.10%
GPT-2 medium: L=24 d=1024 h=16 d_ff=2752  total fwd FLOPs = 8.3017e+11
    qkv_proj  18.62% | QK^T 6.21% | PV 6.21% | out_proj 6.21% | ffn 50.05% | lm_head 12.70%
GPT-2 large: L=36 d=1280 h=20 d_ff=3392  total fwd FLOPs = 1.7685e+12
    qkv_proj  20.49% | QK^T 5.46% | PV 5.46% | out_proj 6.83% | ffn 54.30% | lm_head  7.45%
GPT-2 XL: L=48 d=1600 h=25 d_ff=4288  total fwd FLOPs = 3.5168e+12
    qkv_proj  21.47% | QK^T 4.58% | PV 4.58% | out_proj 7.16% | ffn 57.53% | lm_head  4.68%

(e) XL @ T=16384: total = 1.3358e+14 FLOPs  (38.0x the T=1024 total)
    qkv_proj   9.04% | QK^T 30.87% | PV 30.87% | out_proj 3.01% | ffn 24.24% | lm_head 1.97%
    attention QK^T+PV share: 61.73%  (was 9.16% x 2 = 18.32% at T=1024)

adamw (b) [actual d_ff=4288] P = 1,640,452,800
    activations/sample: 4,093,347,840 floats = 15.249 GiB = 16.373 GB
    params+grads+opt (16P bytes): 24.445 GiB = 26.247 GB
    max batch size within 80 GiB: 3.643 -> 3   (within 80e9 bytes: 3.283 -> 3)

adamw (c) AdamW step ~= 14 FLOPs/param -> 2.297e+10 FLOPs for XL (negligible)

adamw (d) fwd FLOPs/seq = 3.5168e+12; total training FLOPs = 4.3214e+21
    H100 495 TFLOP/s (handout): 1.7460e+07 s = 4,850 hours = 202.1 days
    A100 19.5 TFLOP/s FP32:     4.4322e+08 s = 123,117 hours = 5,129.9 days = 14.05 years
```
