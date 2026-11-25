# StreamState

Optimizer state streamed through the backward pass, with no gradient buffer. The flagship optimizer is
**StreamMuon**, from the paper *Eliminating the Gradient Buffer: Algebraically Exact In-Backward Momentum with
Two-Thirds Less Memory for Dense and Routed Models* ([`paper/paper.pdf`](paper/paper.pdf)).

With gradient accumulation, training keeps a full-size gradient buffer next to the weights and the optimizer state.
That costs 16 bytes per parameter with AdamW, or 12 with Muon, before any activation is stored. Momentum optimizers
don't need that buffer. StreamMuon folds each micro-batch's gradient into an int8 momentum during the backward pass and
frees it at once. The micro-batch weights make the result equal to accumulate-then-update.

One extra byte per parameter recovers Nesterov momentum, the global gradient norm and clipping exactly. That byte is
either a snapshot of the previous momentum or (recommended) an int8 running mean of the update's gradient. Weights are
bf16 with stochastic rounding, so a weight matrix trains on **4 bytes per parameter**. The lighter variant,
StreamMuon-Lite, uses 3.

| bytes / parameter (weight matrices) | weights | gradients | optimizer state | total | vs Muon |
|---|---|---|---|---|---|
| AdamW | 4 | 4 | 8 | 16 | +33% |
| Muon | 4 | 4 | 4 | 12 | - |
| 8-bit Muon | 4 | 4 | 1 | 9 | -25% |
| **StreamMuon** | 2 | 0 | 2 | **4** | **-67%** |
| **StreamMuon-Lite** | 2 | 0 | 1 | **3** | **-75%** |

## Results

All results are from one 8 GB laptop GPU. See the paper for setup and caveats.

- **Dense 113M Transformer, 50M tokens:** over three seeds, Muon 1.145 ± 0.007 bits/byte and StreamMuon (one-byte
  mean) 1.148 ± 0.007; paired by seed the gap is 0.003 ± 0.0005. Streaming costs no throughput. Where Muon has to recompute activations to fit, StreamMuon is 1.18-1.22x faster.
- **Standard MoE Transformer** (SwiGLU experts, a shared expert, top-2): Muon 1.126, StreamMuon 1.131 (one-byte mean)
  and 1.135 (snapshot). Token-clock momentum cuts weight drift of rarely used experts from 214% to 21%.
- **CIFAR-10 with SGD-Nesterov** (the method is independent of the update rule): fp32 SGD 86.4%, streamed 85.9%
  (3 seeds, final accuracy).
- **Largest model that trains on 8 GB:**
  - dense: AdamW, Muon and 8-bit Muon reach 0.28B; StreamMuon 1.46B; Lite 2.0B;
  - MoE: 8-bit Muon 0.51B; StreamMuon 1.31B; Lite at least 1.71B (the next size could not be built in host memory).

Paper: [`paper/paper.pdf`](paper/paper.pdf).

## Install

Requires an NVIDIA GPU, PyTorch and Triton. Tested only with PyTorch 2.13 and Triton 3.7.1 on an RTX 4060 (sm_89).
fp8 Newton-Schulz runs on sm_89+ GPUs; older ones fall back to bf16. On sm_90+ the MoE layer uses PyTorch's own grouped
matmul, so experts take the hook path instead of the in-kernel absorb (untested).

```bash
pip install -e .
```

## Verify (about 2 minutes)

```bash
pip install -e ".[test]"
pytest -q tests/          # 10 tests, about 30 s on an RTX 4060 (most of it compiling the kernels)
```

The tests check the method's claims against a plain reference built from the same gradients:
- **One-byte gradient mean (`gbar=True`):** the clip coefficient and the update direction match an fp32
  accumulate-then-update reference (exact global norm and clip, Nesterov direction).
- **Routed experts:** the streamed global norm equals the norm of the plainly accumulated gradient, in both forms.
  The two forms give the same expert updates, and an expert that gets no tokens doesn't move.
- **Snapshot form (`gbar=False`):** its norm is exact under slowly changing gradient scale, and it overshoots after an
  abrupt 50x drop (the paper's documented limitation), while the one-byte mean stays exact.
- **MoE layer:** the absorb inside the grouped weight-gradient kernel trains like the hook path.
- **Fast path:** the dense one-launch absorb gives bit-identical momentum to the general path.

Separately, the authors checked that this optimizer trains **bit-identically** to the research version that produced
the paper's numbers: same seeds give the same weights and int8 momentum. That covers every reported configuration:
both StreamMuon forms and Lite, dense and expert matrices with and without the grouped-kernel absorb, and the SGD rule.
That check needs the research code, so it isn't part of this repository.

## Usage

```python
import torch
from streamstate import StreamMuon, StreamMuonLite

opt = StreamMuon(model, lr=2e-3, weight_decay=0.05, accum_steps=4)    # or StreamMuonLite(model, ...)
for step in range(steps):
    for g in opt.param_groups:
        g["lr"] = schedule(step)                   # any schedule (the paper: warmup + cosine to 10%)
    for micro in range(4):
        with torch.autocast("cuda", torch.bfloat16):
            loss = model(next_batch()) / 4         # each backward uses loss / accum_steps
        loss.backward()                            # matrix gradients are absorbed into int8 momentum and freed here
    opt.stream_clip(model.parameters(), 1.0)       # exact global-norm clipping (replaces clip_grad_norm_)
    opt.step()
    opt.zero_grad(set_to_none=True)
```

- **The one-byte gradient mean** is on by default (`gbar=True`). `StreamMuon(..., gbar=False)` uses the one-byte
  snapshot of the previous momentum instead, which holds no gradient-shaped state at all. Its int8 norm estimate
  overshoots after abrupt drops in gradient scale. Both forms use 4 bytes per parameter.
- **`StreamMuonLite`** saves the extra byte (3 bytes per parameter). It has no Nesterov momentum, and its clip is
  applied one update late.
- **Which parameters are streamed:** weight matrices with both dimensions >= 64, except embeddings, heads and
  routers. Pass `matrix_names=[...]` to choose them yourself. All other parameters use the built-in AdamW.
- **Weights:** matrices are cast to bf16 in place, with stochastic-rounded updates. `bf16_weights=False` keeps fp32
  weights. With bf16 weights, run every forward pass (evaluation too) under `torch.autocast("cuda", torch.bfloat16)`.
- **Odd sizes:** the one-byte gradient mean works in int8 blocks of 128 values, so with `gbar=True` a matrix whose
  size is not a multiple of 128 is trained by AdamW instead (with a warning). `gbar=False` and Lite stream every
  matrix.
- **Mixture of experts:** `streamstate.moe.StandardMoE` is the layer used in the paper. Any expert tensor of shape
  `(experts, in, out)` that carries a `_region` attribute takes the expert path: token-clock momentum, with the absorb
  inside the grouped weight-gradient kernel. The `_region` attribute must expose `dispatch_acc`, `tokens_acc`,
  `active` and `k`.
- **SGD-Nesterov instead of Muon:** `StreamMuonOptimizer(..., mode="gbar", rule="sgd")` streams the same state and
  applies the Nesterov direction without Newton-Schulz (see `benchmarks/cifar_bench.py`).

## Layout

| Path | Contents |
|---|---|
| `src/streamstate/__init__.py` | `StreamMuon(model, ...)` and `StreamMuonLite(model, ...)`: pick the matrices and build the optimizer |
| `src/streamstate/optim.py` | `StreamMuonOptimizer` (modes `gbar`, `snapshot`, `lite`) and the Muon baseline |
| `src/streamstate/kernels.py` | Triton absorb kernels |
| `src/streamstate/grouped_gemm.py` | grouped GEMMs, with the absorb in the weight-gradient epilogue |
| `src/streamstate/fp8_polar.py` | Newton-Schulz with fp8 GEMMs and a divergence guard |
| `src/streamstate/moe.py` | the standard MoE layer used in the paper |
| `tests/test_streammuon.py` | one-byte mean against an fp32 accumulate-then-update reference; exact expert norm; snapshot vs one-byte mean; MoE kernel absorb vs hook path; fast vs general absorb path |
| `benchmarks/cifar_bench.py` | the paper's CIFAR-10 SGD-Nesterov comparison |

The kernels, the MoE layer and the optimizer's compiled update functions are copied verbatim from the code that
produced the paper's numbers. The optimizer class keeps only the code paths the paper measured (bit-identical to the
original; see Verify).

## Citation

Paper: [doi:10.5281/zenodo.24544000](https://doi.org/10.5281/zenodo.24544000). Code (v0.1.0):
[doi:10.5281/zenodo.24544217](https://doi.org/10.5281/zenodo.24544217).

```bibtex
@misc{patel2026streammuon,
  title     = {Eliminating the Gradient Buffer: Algebraically Exact In-Backward Momentum with Two-Thirds Less Memory
               for Dense and Routed Models},
  author    = {Patel, Ammar},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24544000},
  url       = {https://doi.org/10.5281/zenodo.24544000}
}

@software{patel2026streamstate,
  title     = {StreamState: optimizer state streamed through the backward pass},
  author    = {Patel, Ammar},
  year      = {2026},
  version   = {0.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24544217},
  url       = {https://github.com/amzker-labs/streamstate}
}
```

## License

Code: Apache-2.0 (see [`LICENSE`](LICENSE)). Paper (`paper/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Acknowledgments

StreamMuon came out of a larger architecture project trained on a single 8 GB GPU. See the paper's acknowledgments for
the AI systems that contributed.
