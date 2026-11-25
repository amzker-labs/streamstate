"""StreamMuon: Muon-style training with no gradient buffer.

Each weight matrix's gradient is folded into its int8 momentum by a post-accumulate hook during the backward pass and
freed at once; the optimizer step then needs no matrix gradients. Three forms (paper, Section 3):

  "gbar"      (recommended) the micro-batch gradients go into a one-byte running mean Gbar of this update; at the step
              M_t = a M_{t-1} + (1 - a) kappa Gbar, direction a M_t + (1 - a) kappa Gbar (Nesterov). Exact global norm
              ||Gbar|| and clip kappa. 4 bytes per parameter (bf16 weight + int8 momentum + int8 Gbar).
  "snapshot"  the micro-batches go straight into M with token-count weights (equal to accumulate-then-update); a
              one-byte snapshot of M_{t-1} gives the exact norm ||M_t - a M_{t-1}|| / (1 - a), the clip and Nesterov
              (1 + a) M_t - a M_{t-1}. 4 bytes per parameter, no gradient-shaped state.
  "lite"      the micro-batches go into M with token-count weights, no snapshot: clip from the summed micro-batch
              norms with one update of lag, no Nesterov. 3 bytes per parameter.

Update: W <- W - lr (0.2 sqrt(max(m, n)) NS(direction) + wd W), Newton-Schulz (Muon's quintic, first iterations in
fp8 where supported); bf16 weights get stochastic rounding. rule="sgd": the direction itself (SGD-Nesterov) instead of
NS. Every other parameter (embeddings, head, norms, biases) is trained by AdamW.

Routed experts (3-D tensors (experts, in, out) whose `_region` exposes dispatch_acc, tokens_acc, active and k, as
streamstate.moe.StandardMoE): token-clock momentum, M_s decays by tokens the expert receives, and the absorb runs in the
grouped weight-gradient kernel's epilogue.

The compiled kernels below are verbatim copies of the research code that produced the paper's numbers; the class keeps
only the code paths the paper measured (the research version's ablation options are removed).
"""

from __future__ import annotations

import functools
import weakref

import numpy as np
import torch

from streamstate import kernels as SK


def optimizer_state_bytes(opt, model):
    """Measured training-state memory: weights (+ any fp32 master copies), optimizer state, by dtype; bytes / param."""
    def tensors(o):
        for st in o.state.values():
            for v in st.values():
                if torch.is_tensor(v):
                    yield v
        for inner in (o.inner() if hasattr(o, "inner") else []):
            yield from tensors(inner)
    seen, state_b = set(), 0
    for t in tensors(opt):
        if t.data_ptr() not in seen:
            seen.add(t.data_ptr())
            state_b += t.numel() * t.element_size()
    n = sum(p.numel() for p in model.parameters())
    w_b = sum(p.numel() * p.element_size() for p in model.parameters())
    return {"params": n, "weight_bytes": w_b, "state_bytes": state_b,
            "weights_per_param": round(w_b / n, 3), "state_per_param": round(state_b / n, 3),
            "total_per_param": round((w_b + state_b) / n, 3)}


@torch.no_grad()
def orthogonalize(g, steps=5):
    """Newton-Schulz iteration towards the nearest semi-orthogonal matrix U V^T of g = U S V^T (the quintic of the
    Muon optimizer, coefficients (3.4445, -4.7750, 2.0315)); batched over leading dimensions, in bf16."""
    a, b, c = 3.4445, -4.7750, 2.0315
    x = g.bfloat16()
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.transpose(-2, -1)
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = x @ x.transpose(-2, -1)
        x = a * x + (b * A + c * A @ A) @ x
    return x.transpose(-2, -1) if tall else x


def load_optimizer_exact(opt, sd):
    """opt.load_state_dict(sd), then every saved state tensor put back with its own dtype: torch casts floating
    state to the parameter's dtype, which with bf16 parameters (bf16 matrices) silently
    turns fp32 accumulators / statistics into bf16."""
    opt.load_state_dict(sd)
    params = [p for g in opt.param_groups for p in g["params"]]
    for i, saved in sd.get("state", {}).items():
        st = opt.state[params[i]]
        for k, v in saved.items():
            if torch.is_tensor(v):
                st[k] = v.to(params[i].device)


@torch.compile(dynamic=True)
def _swo_post_fused(p, o, dL, dR, step, lam, use_sign, sign_alpha, gs, bf16_out, fixed_scale=False, cautious=True):
    """Update direction U and the step, in one pass:
      use_sign = 0: U = 0.2 sqrt(max(m, n)) polar (fixed_scale; the released optimizer always uses this form);
      use_sign = 2: U = the direction itself (SGD-Nesterov). Other arguments are fixed by the released optimizer.
      p <- p + step (U + lam [U p >= 0] p); bf16_out: stochastic rounding to bf16, also returns the rounding error."""
    pf = p.float()
    if use_sign == 2:                         # rule "sgd": the (Nesterov) momentum direction itself, no polar
        u = o.float()
    elif use_sign:
        u = 0.2 * sign_alpha * torch.sign(o.float())
    else:
        ut = o.float() / (dL.unsqueeze(-1) * dR.unsqueeze(-2))
        if fixed_scale:                       # (Muon: constant 0.2 sqrt(max(m, n)); no per-matrix RMS rescaling)
            u = 0.2 * max(ut.shape[-2], ut.shape[-1]) ** 0.5 * ut
        else:
            u = 0.2 * ut / (ut.pow(2).mean((1, 2), keepdim=True).sqrt() + 1e-7)
    if gs:
        u = u - (u * pf).sum((1, 2), keepdim=True) / ((pf * pf).sum((1, 2), keepdim=True) + 1e-12) * pf
    mask = ((u * pf) >= 0).to(pf.dtype) if cautious else torch.ones_like(pf)
    new = pf + step.view(-1, 1, 1) * (u + lam * mask * pf)
    if bf16_out:
        xi = new.view(torch.int32)
        rounded = ((xi + torch.randint_like(xi, 0, 1 << 16)) & -65536).view(torch.float32)
        return rounded.bfloat16(), new - rounded
    return new, new


@torch.compile(dynamic=True)
def _q8(x):
    """Blockwise linear int8 (blocks of 128, bf16 scale per block), stochastic rounding: 1.016 B / value."""
    flat = x.reshape(-1, 128)
    scale = (flat.abs().amax(-1, keepdim=True) * (1.0078125 / 127)).bfloat16().float().clamp_min(1e-30)
    qv = torch.floor(flat / scale + torch.rand_like(flat)).clamp(-127, 127)
    return qv.to(torch.int8).view(x.shape), scale.bfloat16()


@torch.compile(dynamic=True)
def _snap_diff_sq_w(q, s, qp, sp, a, w2):
    """_snap_diff_sq with a per-slab weight: sum_s w2_s ||M_s - a_s M_prev,s||^2 / (1 - a_s)^2 (w2 = 1 for dense
    matrices; (n_s / N)^2 for routed experts, whose momentum is in token-clock units)."""
    S = a.shape[0]
    m = (q.float().reshape(-1, 128) * s.float()).reshape(S, -1)
    mp = (qp.float().reshape(-1, 128) * sp.float()).reshape(S, -1)
    d = m - a.view(-1, 1) * mp
    return (w2 * d.pow(2).sum(1) / (1 - a).clamp_min(1e-6) ** 2).sum()


@torch.compile(dynamic=True)
def _snap_step(q, s, qp, sp, a, coef):
    """exact_stream end of update, fused: M <- a M_prev + coef (M - a M_prev) requantized to int8 (stochastic
    rounding, same format as _q8), and the Nesterov direction (1 + a) M - a M_prev."""
    S = a.shape[0]
    shape = q.shape
    m = (q.float().reshape(-1, 128) * s.float()).reshape(S, -1)
    mp = (qp.float().reshape(-1, 128) * sp.float()).reshape(S, -1)
    av = a.view(-1, 1)
    m = av * mp + coef * (m - av * mp)
    direction = (1 + av) * m - av * mp
    flat = m.reshape(-1, 128)
    scale = (flat.abs().amax(-1, keepdim=True) * (1.0078125 / 127)).bfloat16().float().clamp_min(1e-30)
    qv = torch.floor(flat / scale + torch.rand_like(flat)).clamp(-127, 127)
    return qv.to(torch.int8).view(shape), scale.bfloat16(), direction


@torch.compile(dynamic=True)
def _gbar_step(qm, sm, qg, sg, a, coef, c_m, c_g):
    """gbar update end, fused: M <- a M + (1 - a) coef Gbar requantized (stochastic rounding, _q8 format), and the
    direction c_m M_{t-1} + c_g coef Gbar (Nesterov for the polar rule: a M_t + (1 - a) coef G = a^2 M_{t-1} +
    (a (1 - a) + 1 - a) coef G)."""
    shape = qm.shape
    m = (qm.float().reshape(-1, 128) * sm.float())
    g = (qg.float().reshape(-1, 128) * sg.float()) * coef
    direction = (c_m * m + c_g * g).view(shape)
    m = a * m + (1 - a) * g
    scale = (m.abs().amax(-1, keepdim=True) * (1.0078125 / 127)).bfloat16().float().clamp_min(1e-30)
    qv = torch.floor(m / scale + torch.rand_like(m)).clamp(-127, 127)
    return qv.to(torch.int8).view(shape), scale.bfloat16(), direction


@torch.compile(dynamic=True)
def _gbar_step_slots(qm, sm, qg, sg, a, coef, c_m, c_g):
    """_gbar_step with per-slab (per-expert) coefficients a, c_m, c_g of shape (S,): M_s <- a_s M_s + (1 - a_s) coef
    Gbar_s requantized, direction c_m,s M_s + c_g,s coef Gbar_s (both on M_{t-1})."""
    S = a.shape[0]
    shape = qm.shape
    m = (qm.float().reshape(-1, 128) * sm.float()).reshape(S, -1)
    g = (qg.float().reshape(-1, 128) * sg.float()).reshape(S, -1) * coef
    direction = (c_m.view(-1, 1) * m + c_g.view(-1, 1) * g).view(shape)
    m = a.view(-1, 1) * m + (1 - a.view(-1, 1)) * g
    flat = m.reshape(-1, 128)
    scale = (flat.abs().amax(-1, keepdim=True) * (1.0078125 / 127)).bfloat16().float().clamp_min(1e-30)
    qv = torch.floor(flat / scale + torch.rand_like(flat)).clamp(-127, 127)
    return qv.to(torch.int8).view(shape), scale.bfloat16(), direction


@torch.compile(dynamic=True)
def _dq8_sq_w(qv, scale, w2):
    """sum_s w2_s ||dequant(qv)_s||^2 over S slabs (experts)."""
    S = w2.shape[0]
    return (w2 * (qv.float().reshape(-1, 128) * scale.float()).reshape(S, -1).pow(2).sum(1)).sum()


@torch.compile(dynamic=True)
def _dq8_sq(qv, scale):
    return (qv.float().reshape(-1, 128) * scale.float()).pow(2).sum()


@torch.compile(dynamic=True)
def _dq8(qv, scale):
    return (qv.float().reshape(-1, 128) * scale.float()).view(qv.shape)


@torch.compile(dynamic=True)
def _swo_absorb(g, m, seen, r, c, live, b_t, gscale, b2, store_bf16):
    """Streaming (backward-into-M): fold one micro-batch's gradient into the momentum and the r / c
    statistics right after it is accumulated, so full gradients never coexist. Token-time momentum per micro-batch
    (b_t from this micro-batch's token counts); first visit M = G~."""
    gt = g.float() * gscale.view(-1, 1, 1)
    mo = m.float()
    fresh = live & ~seen
    bb = b_t.view(-1, 1, 1)
    m_new = torch.where(fresh.view(-1, 1, 1), gt, bb * mo + (1 - bb) * gt)
    m_new = torch.where(live.view(-1, 1, 1), m_new, mo)
    if store_bf16:
        xi = m_new.view(torch.int32)
        m_out = ((xi + torch.randint_like(xi, 0, 1 << 16)) & -65536).view(torch.float32).bfloat16()
    else:
        m_out = m_new
    g2 = gt * gt
    r_new, c_new = g2.mean(-1), g2.mean(-2)
    fv, lv = fresh.view(-1, 1), live.view(-1, 1)
    r = torch.where(lv, torch.where(fv, r_new, b2 * r + (1 - b2) * r_new), r)
    c = torch.where(lv, torch.where(fv, c_new, b2 * c + (1 - b2) * c_new), c)
    return m_out, seen | live, r, c


@torch.compile(dynamic=True)
def _swo_absorb_q8(g, qv, qs, seen, r, c, live, b_t, gscale, b2, K):
    """_swo_absorb for int8 momentum in one compiled pass: dequantize, fold the micro-batch in, requantize
    (blockwise int8, stochastic rounding), r / c EMAs, and this gradient's squared norm (for the streaming clip)."""
    S, m_, n_ = g.shape
    gf = g.float()
    sq = (gf * K).pow(2).sum()
    gt = gf * gscale.view(-1, 1, 1)
    mo = (qv.float().reshape(-1, 128) * qs.float()).view(S, m_, n_)
    fresh = live & ~seen
    bb = b_t.view(-1, 1, 1)
    m_new = torch.where(fresh.view(-1, 1, 1), gt, bb * mo + (1 - bb) * gt)
    m_new = torch.where(live.view(-1, 1, 1), m_new, mo)
    flat = m_new.reshape(-1, 128)
    scale = (flat.abs().amax(-1, keepdim=True) * (1.0078125 / 127)).bfloat16().float().clamp_min(1e-30)
    q_new = torch.floor(flat / scale + torch.rand_like(flat)).clamp(-127, 127).to(torch.int8)
    g2 = gt * gt
    r_new, c_new = g2.mean(-1), g2.mean(-2)
    fv, lv = fresh.view(-1, 1), live.view(-1, 1)
    r = torch.where(lv, torch.where(fv, r_new, b2 * r + (1 - b2) * r_new), r)
    c = torch.where(lv, torch.where(fv, c_new, b2 * c + (1 - b2) * c_new), c)
    return q_new.view(S, m_, n_), scale.bfloat16(), seen | live, r, c, sq


@torch.compile(dynamic=True)
def _swo_vec_fused(p, g, m1, v, hp, decay):
    """AdamW + cautious decay in one pass; hp = [1 / bc1, 1 / bc2, lr, eps, beta2] (one device tensor per step)."""
    g = g.float()
    m1 = 0.9 * m1 + 0.1 * g
    v = hp[4] * v + (1 - hp[4]) * g * g
    u = (m1 * hp[0]) / ((v * hp[1]).sqrt() + hp[3])
    return p - hp[2] * (u + decay * ((u * p) >= 0).to(p.dtype) * p), m1, v


class Muon(torch.optim.Optimizer):
    """Muon (momentum + orthogonalised update) for weight matrices, the baseline: fp32 momentum, gradient buffer.
    3-D tensors (stacked expert matrices) are orthogonalised per matrix. Update scaled by 0.2 * sqrt(max(rows, cols))
    so its RMS matches AdamW's (Moonshot, "Muon is Scalable for LLM Training", 2025). fast_ns=True: StreamMuon's
    Newton-Schulz path for Muon (same-shape matrices batched, first 3 iterations in fp8, divergence guard)."""

    def __init__(self, params, lr=6e-4, momentum=0.95, weight_decay=0.0, fast_ns=False):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay))
        self.fast_ns = fast_ns

    @torch.no_grad()
    def step(self):
        work = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                buf = st.setdefault("momentum", torch.zeros_like(p))
                buf.mul_(group["momentum"]).add_(p.grad)
                work.append((group, p, p.grad.add(buf, alpha=group["momentum"])))   # (Nesterov)
        # stacked expert matrices: skip empty ones (no tokens routed); which ones are empty is read back in ONE
        # device-to-host copy for all parameters
        stacked = [g.flatten(1).abs().amax(1) > 0 for _, p, g in work if p.ndim == 3]
        live_all = torch.cat(stacked).cpu().tolist() if stacked else []
        at = 0
        fast = {}
        if self.fast_ns:
            from streamstate.fp8_polar import orthogonalize_quintic_fp8
            by_shape = {}
            for i, (group, p, g) in enumerate(work):
                if p.ndim == 2:
                    by_shape.setdefault(tuple(p.shape), []).append(i)
            for shape, idx in by_shape.items():
                step = max(1, (1 << 25) // (shape[0] * shape[1]))
                for c0 in range(0, len(idx), step):
                    chunk = idx[c0:c0 + step]
                    o = orthogonalize_quintic_fp8(torch.stack([work[i][2] for i in chunk]), fp8_steps=3)
                    for k, i in enumerate(chunk):
                        fast[i] = o[k]
        for wi, (group, p, g) in enumerate(work):
            scale = 0.2 * max(p.shape[-2], p.shape[-1]) ** 0.5
            p.mul_(1 - group["lr"] * group["weight_decay"])
            if wi in fast:
                p.add_(fast[wi].to(p.dtype), alpha=-group["lr"] * scale)
                continue
            if p.ndim == 3:
                live = [i for i, ok in enumerate(live_all[at:at + p.shape[0]]) if ok]
                at += p.shape[0]
                if live:
                    li = torch.tensor(live, device=p.device)
                    p.index_add_(0, li, orthogonalize(g[li]).to(p.dtype), alpha=-group["lr"] * scale)
            else:
                p.add_(orthogonalize(g).to(p.dtype), alpha=-group["lr"] * scale)


def default_ortho():
    """Muon's quintic Newton-Schulz: the first 3 iterations in fp8 on GPUs with fp8 tensor cores (sm >= 8.9),
    else all in bf16."""
    if torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9):
        from streamstate.fp8_polar import orthogonalize_quintic_fp8
        return functools.partial(orthogonalize_quintic_fp8, fp8_steps=3)
    return orthogonalize


class StreamMuonOptimizer(torch.optim.Optimizer):
    """See the module docstring. Usually built with streamstate.StreamMuon(model, ...).

    matrix_params: weight matrices (2-D) and expert stacks (3-D) to stream; vector_params: everything else (AdamW,
    betas (0.9, vec_beta2), decoupled weight decay with a sign mask on >= 2-D tensors, none on 1-D ones).
    accum_steps: micro-batches per update (each backward must use loss / accum_steps). clip: global gradient-norm
    clip (call stream_clip() where clip_grad_norm_ would go; None: off). Momentum decay beta1 per update (it is
    spread over the micro-batches); beta2 only feeds row / column statistics the absorb kernels maintain."""

    def __init__(self, matrix_params, vector_params=(), lr=2e-3, weight_decay=0.05, mode="gbar", rule="polar",
                 accum_steps=1, clip=1.0, beta1=0.95, beta2=0.99, vec_beta2=0.95, eps=1e-8, ortho=None,
                 fused_dw=True, fast_absorb=True):
        assert mode in ("gbar", "snapshot", "lite"), mode
        assert rule in ("polar", "sgd") and (rule == "polar" or mode == "gbar"), "rule='sgd' needs mode='gbar'"
        groups = [{"params": list(matrix_params), "kind": "matrix"}, {"params": list(vector_params), "kind": "vector"}]
        super().__init__([g for g in groups if g["params"]], dict(lr=lr, weight_decay=weight_decay))
        snapshot = mode == "snapshot"
        self.hp = dict(beta1=beta1, beta2=beta2, eps=eps, vec_beta2=vec_beta2, rule=rule, gbar=mode == "gbar",
                       nesterov=snapshot, exact_stream=snapshot, equal_weights=mode != "gbar")
        self.ortho = ortho or default_ortho()
        # 1-D vectors (no decay) go through torch's multi-tensor fused AdamW (same update) - a few launches
        vec1d = [q for g in self.param_groups if g["kind"] == "vector" for q in g["params"] if q.ndim < 2]
        self.vec_adam = torch.optim.AdamW(vec1d, lr=lr, betas=(0.9, vec_beta2), eps=eps, weight_decay=0.0,
                                          fused=True) if vec1d else None
        self._vec1d = {id(q) for q in vec1d} if self.vec_adam is not None else set()
        self.t = 0
        self.stream, self.K = True, accum_steps
        self._dense_c = {}
        # global-norm clip: the hooks keep what stream_clip() needs (gbar / snapshot: exact norm at the update;
        # lite: each micro-batch's ||g||^2 summed, the coefficient applied to the NEXT update's micro-batches)
        dev = self.param_groups[0]["params"][0].device
        self.clip = clip
        self._sq = torch.zeros((), device=dev)
        self._upd_sq = torch.zeros((), device=dev)
        self._coef = torch.ones((), device=dev)
        self._cb = False
        # dense fast path (_absorb_dense_fast): host-side per-matrix flags / coefficients, cached constant tensors
        self._fast_absorb = fast_absorb
        self._host, self._consts, self._gs_src, self._gs_coef = {}, {}, None, None
        for group in self.param_groups:
            if group["kind"] == "matrix":
                for p in group["params"]:
                    p.register_post_accumulate_grad_hook(self._absorb)
                    if fused_dw and getattr(p, "_region", None) is not None and p.ndim == 3 and \
                            p.dtype == torch.bfloat16:
                        SK.FUSED_DW[p.data_ptr()] = (weakref.ref(p), weakref.WeakMethod(self._fused_dw_ctx))

    def inner(self):
        return [self.vec_adam] if self.vec_adam is not None else []

    def state_dict(self):
        sd = super().state_dict()
        sd["swo_t"] = self.t
        sd["inner"] = {"vec": self.vec_adam.state_dict()} if self.vec_adam is not None else {}
        return sd

    def load_state_dict(self, sd):
        sd = dict(sd)
        self.t, inner = sd.pop("swo_t", 0), sd.pop("inner", {})
        super().load_state_dict(sd)
        # torch casts floating state to the parameter dtype (bf16 matrices); put every saved tensor back with its own
        # dtype (int8 momentum, bf16 scales, fp32 statistics, bool flags). Per-micro-batch transients are dropped.
        params = [p for g in self.param_groups for p in g["params"]]
        for i, saved in sd.get("state", {}).items():
            st = self.state[params[i]]
            for k, v in saved.items():
                if torch.is_tensor(v):
                    st[k] = v.to(params[i].device)
            for k in ("n_snap", "N_snap", "n_acc", "N_acc", "live_acc", "_fused", "_begun", "m_prev", "ms_prev", "a_acc"):
                st.pop(k, None)
        self._host = {}                           # (host-side seen flags re-read from the loaded state)
        if self.vec_adam is not None and "vec" in inner:
            load_optimizer_exact(self.vec_adam, inner["vec"])

    @torch.no_grad()
    def _absorb_begin(self, p):
        """Per micro-batch and matrix: the token clock (b_t), load scale (gscale, x clip coefficient), live slots,
        lazily made state. Returns (g-shaped lead S, coefficients) for the absorb kernels."""
        h, K = self.hp, self.K
        st = self.state[p]
        dev = p.device
        S = p.shape[0] if p.ndim == 3 else 1
        region = getattr(p, "_region", None) if p.ndim == 3 else None
        b1 = h["beta1"]
        if self.clip is not None and not self._cb:
            self._cb = True
            torch.autograd.Variable._execution_engine.queue_callback(self._end_backward)
        if region is not None and h["gbar"]:
            # gbar for routed experts: each micro-batch's expert gradient (token-clock units, x K N / n_s as below)
            # goes into a per-update int8 running mean Gbar_s weighted by tokens (first visit of the update
            # overwrites); the momentum is updated once per update in _stream_matrix / _gbar_dir
            cum, Ncum = region.dispatch_acc, region.tokens_acc
            n = cum - st.get("n_snap", torch.zeros_like(cum))
            N = (Ncum - st.get("N_snap", torch.zeros_like(Ncum))).clamp_min(1.0)
            st["n_snap"], st["N_snap"] = cum.clone(), Ncum.clone()
            live = n > 0
            st["live_acc"] = st.get("live_acc", torch.zeros_like(live)) | live
            st["n_acc"] = st.get("n_acc", torch.zeros_like(n)) + n
            st["N_acc"] = st.get("N_acc", torch.zeros_like(N)) + N
            lead = (S,)
            if "m" not in st:
                for key, sk in (("m", "ms"), ("g", "gs")):
                    st[key] = torch.zeros_like(p, dtype=torch.int8)
                    st[sk] = torch.zeros(p.numel() // 128, 1, dtype=torch.bfloat16, device=dev)
                st["seen"] = torch.zeros(lead, dtype=torch.bool, device=dev)       # expert has a momentum yet
                st["r"] = torch.zeros(p.shape[:-1], device=dev)
                st["c"] = torch.zeros(p.shape[:-2] + p.shape[-1:], device=dev)
            if "g_seen" not in st:                                                  # (per update; reset by the step)
                st["g_seen"] = torch.zeros(lead, dtype=torch.bool, device=dev)
                st["g_nb"] = torch.zeros(lead, device=dev)
            nb = st["g_nb"]
            b_t = torch.where(live, nb / (nb + n).clamp_min(1e-12), torch.ones_like(nb)).float()
            st["g_nb"] = nb + n
            gscale = (K * N / n.clamp_min(1.0)).float()
            st["_gtarget"] = True
            return S, lead, True, region, live, b_t, gscale
        if region is not None:
            cum, Ncum = region.dispatch_acc, region.tokens_acc
            n = cum - st.get("n_snap", torch.zeros_like(cum))
            N = (Ncum - st.get("N_snap", torch.zeros_like(Ncum))).clamp_min(1.0)
            st["n_snap"], st["N_snap"] = cum.clone(), Ncum.clone()
            live = n > 0
            f_ref = region.k / region.active.sum().clamp_min(1)
            tau = b1 / (1 - b1) * f_ref * N * K        # momentum horizon in tokens of one whole update
            b_t = (tau / (tau + n)).float()
            gscale = (K * N / n.clamp_min(1.0)).float()
            st["live_acc"] = st.get("live_acc", torch.zeros_like(live)) | live
            st["n_acc"] = st.get("n_acc", torch.zeros_like(n)) + n
            st["N_acc"] = st.get("N_acc", torch.zeros_like(N)) + N      # (tokens through the layer this update)
        else:
            if id(p) not in self._dense_c:   # (constant per tensor: made once; kept out of the state_dict, whose
                self._dense_c[id(p)] = (     #  load would cast it to the parameter dtype)
                                 torch.ones(S, dtype=torch.bool, device=dev),
                                 torch.full((S,), b1 * K / (b1 * K + 1 - b1), device=dev),   # compounds to ~b1
                                 torch.full((S,), float(K), device=dev))
            live, b_t, gscale = self._dense_c[id(p)]
        lead = (S,) if p.ndim == 3 else ()
        q8 = p.numel() % 128 == 0
        if "m" not in st:
            if q8:
                st["m"] = torch.zeros_like(p, dtype=torch.int8)
                st["ms"] = torch.zeros(p.numel() // 128, 1, dtype=torch.bfloat16, device=dev)
            else:
                st["m"] = torch.zeros_like(p, dtype=torch.bfloat16)
            st["seen"] = torch.zeros(lead, dtype=torch.bool, device=dev)
            st["r"] = torch.zeros(p.shape[:-1], device=dev)
            st["c"] = torch.zeros(p.shape[:-2] + p.shape[-1:], device=dev)
        if h["equal_weights"]:
            if "k_in" not in st:                  # (first micro-batch of this update)
                st["fresh_upd"] = ~st["seen"].reshape(S)
                st["k_in"] = 0
                st["n_before"] = torch.zeros(S, device=dev)
            st["k_in"] += 1
            fr = st["fresh_upd"]
            if region is not None:
                nn = n.float().reshape(S)
                t_eff = torch.where(fr, torch.zeros_like(nn), (tau * torch.ones_like(n)).float().reshape(S))
                den = t_eff + st["n_before"] + nn
                b_t = torch.where(live.reshape(S) & (den > 0), (t_eff + st["n_before"]) / den.clamp_min(1e-12),
                                  torch.ones_like(den))
                st["n_before"] = st["n_before"] + nn
            else:
                k, tp = float(st["k_in"]), b1 / (1 - b1) * K
                b_t = torch.where(fr, torch.full((S,), (k - 1) / k, device=dev),
                                  torch.full((S,), (tp + k - 1) / (tp + k), device=dev))
        if h["nesterov"]:
            if "m_prev" not in st:                # (first micro-batch of this update: M_{t-1})
                st["m_prev"] = st["m"].clone()
                st["ms_prev"] = st["ms"].clone() if q8 else None
                st["a_acc"] = torch.ones(S, device=dev)
            st["a_acc"] = st["a_acc"] * torch.where(live.reshape(S), b_t.reshape(S).float(), torch.ones_like(st["a_acc"]))
        if self.clip is not None and not h["exact_stream"]:
            gscale = gscale * self._coef
        return S, lead, q8, region, live, b_t, gscale

    @torch.no_grad()
    def _fused_dw_ctx(self, p):
        """Called by the grouped-GEMM backward (kernels.FUSED_DW) just before p's dW tiles: the absorb is done in the
        dW kernel's epilogue; the post-accumulate hook then only finishes (r, c, seen, norms)."""
        st = self.state[p]
        S, lead, q8, region, live, b_t, gscale = self._absorb_begin(p)
        mk, sk, seenk = ("g", "gs", "g_seen") if st.get("_gtarget") else ("m", "ms", "seen")
        if not (q8 and st[mk].is_contiguous()):
            st["_begun"] = (S, lead, q8, region, live, b_t, gscale)
            return None
        ctx = SK.AbsorbCtx(st[mk], st[sk], st[seenk].reshape(S), live, b_t, gscale, self.K)
        st["_fused"] = (ctx, S, lead, region)
        return ctx

    @torch.no_grad()
    def _absorb(self, p):
        h, K = self.hp, self.K
        st = self.state[p]
        fused = st.pop("_fused", None)
        if fused is None and "_begun" not in st and self._dense_fast_ok(p, st):
            return self._absorb_dense_fast(p, st)
        if fused is not None:                 # dW already folded into the momentum by the grouped-GEMM kernel
            ctx, S, lead, region = fused
            seenk = "g_seen" if st.get("_gtarget") else "seen"
            seen, r, c, sq, _ = SK.absorb_finish(ctx, st[seenk].reshape(S), st["r"].reshape(S, -1),
                                                 st["c"].reshape(S, -1), h["beta2"] ** (1 / K))
        else:
            begun = st.pop("_begun", None)
            S, lead, q8, region, live, b_t, gscale = begun if begun is not None else self._absorb_begin(p)
            mk, sk, seenk = ("g", "gs", "g_seen") if st.get("_gtarget") else ("m", "ms", "seen")
            g = p.grad if p.ndim == 3 else p.grad.unsqueeze(0)
            if q8 and SK.supported(g) and st[mk].is_contiguous():     # hand-written Triton: one pass, in place
                seen, r, c, sq, _ = SK.absorb_q8(g, st[mk].view(g.shape), st[sk], st[seenk].reshape(S),
                                                 st["r"].reshape(S, -1), st["c"].reshape(S, -1), live, b_t,
                                                 gscale, h["beta2"] ** (1 / K), K)
            elif q8:                              # one fused compiled pass (dequantize, absorb, requantize, norm)
                qv, st[sk], seen, r, c, sq = _swo_absorb_q8(
                    g, st[mk].view(g.shape), st[sk], st[seenk].reshape(S), st["r"].reshape(S, -1),
                    st["c"].reshape(S, -1), live, b_t, gscale, h["beta2"] ** (1 / K), float(K))
                st[mk] = qv.view(p.shape)
            else:                                 # (size not a multiple of 128: bf16 momentum)
                sq = (g.float() * K).pow(2).sum()
                m_out, seen, r, c = _swo_absorb(g, st["m"].view(g.shape), st["seen"].reshape(S),
                                                st["r"].reshape(S, -1), st["c"].reshape(S, -1), live, b_t,
                                                gscale, h["beta2"] ** (1 / K), True)
                st["m"] = m_out.view(p.shape)
        gtarget = st.pop("_gtarget", False)
        if self.clip is not None and not h["exact_stream"] and not gtarget:
            self._sq += sq                    # (snapshot: every matrix's norm comes from the snapshot in stream_clip;
            #                                    gbar experts: from Gbar)
        st["g_seen" if gtarget else "seen"] = seen.reshape(lead)
        st["r"], st["c"] = r.view(st["r"].shape), c.view(st["c"].shape)
        st["absorbed"] = True
        p.grad = None

    def _dense_fast_ok(self, p, st):
        """Dense matrix, int8 momentum, kernel-friendly shape."""
        h, g = self.hp, p.grad
        if h["gbar"] and p.ndim == 2:          # (routed experts: the general path, _absorb_begin's gbar branch)
            assert self._fast_absorb and p.numel() % 128 == 0, \
                "mode='gbar' needs matrices whose size is a multiple of 128 (StreamMuon(...) sends others to AdamW; " \
                "or use mode='snapshot')"
        return (self._fast_absorb and p.ndim == 2 and g is not None and g.is_contiguous() and p.numel() % 128 == 0
                and ("m" not in st or st["m"].is_contiguous()))

    def _const(self, v):
        """Cached one-element fp32 device tensor holding v (coefficients that repeat every update)."""
        t = self._consts.get(v)
        if t is None:
            t = self._consts[v] = torch.full((1,), v, dtype=torch.float32,
                                             device=self.param_groups[0]["params"][0].device)
        return t

    @torch.no_grad()
    def _absorb_dense_fast(self, p, st):
        """_absorb_begin + absorb_q8 for a dense matrix, with the per-micro-batch coefficients (token-count weights,
        Nesterov's a, first visit) computed on the host: one kernel launch per matrix and micro-batch. Same momentum
        bits as the general path for the same seed; r / c are not maintained (unused)."""
        h, K, dev = self.hp, self.K, p.device
        b1 = h["beta1"]
        if self.clip is not None and not self._cb:
            self._cb = True
            torch.autograd.Variable._execution_engine.queue_callback(self._end_backward)
        hs = self._host.setdefault(id(p), {})
        if h["gbar"]:
            if "g" not in st:
                st["g"] = torch.zeros_like(p, dtype=torch.int8)
                st["gs"] = torch.zeros(p.numel() // 128, 1, dtype=torch.bfloat16, device=dev)
            if "m" not in st:
                st["m"] = torch.zeros_like(p, dtype=torch.int8)
                st["ms"] = torch.zeros(p.numel() // 128, 1, dtype=torch.bfloat16, device=dev)
                st["seen"] = torch.zeros((), dtype=torch.bool, device=dev)
                hs["seen"] = False
            if "seen" not in hs:
                hs["seen"] = bool(st["seen"].all())
            k = hs["gk"] = hs.get("gk", 0) + 1            # (micro-batch index in this update; reset by the step)
            SK.absorb_q8_dense(p.grad, st["g"], st["gs"], self._const(float(K)), float(np.float32((k - 1) / k)),
                               k == 1, K, None)
            st["absorbed"] = True
            p.grad = None
            return
        if "m" not in st:
            st["m"] = torch.zeros_like(p, dtype=torch.int8)
            st["ms"] = torch.zeros(p.numel() // 128, 1, dtype=torch.bfloat16, device=dev)
            st["seen"] = torch.zeros((), dtype=torch.bool, device=dev)
            st["r"] = torch.zeros(p.shape[:-1], device=dev)
            st["c"] = torch.zeros(p.shape[-1:], device=dev)
            hs["seen"] = False
        if "seen" not in hs:                      # (once per process, e.g. after a resume: one device read)
            hs["seen"] = bool(st["seen"].all())
        fresh = not hs["seen"]
        b = b1 * K / (b1 * K + 1 - b1)
        if h["equal_weights"]:
            if "k_in" not in st:                  # (first micro-batch of this update; popped with the others)
                st["fresh_upd"] = fresh
                st["k_in"] = 0
            st["k_in"] += 1
            k, tp = float(st["k_in"]), b1 / (1 - b1) * K
            b = (k - 1) / k if st["fresh_upd"] else (tp + k - 1) / (tp + k)
        b = float(np.float32(b))
        if h["nesterov"]:
            if "m_prev" not in st:                # (first micro-batch of this update: M_{t-1})
                st["m_prev"] = st["m"].clone()
                st["ms_prev"] = st["ms"].clone()
                hs["a"] = np.float32(1.0)
            hs["a"] = np.float32(hs["a"] * np.float32(b))
            st["a_acc"] = self._const(float(hs["a"]))
        sq_acc = None
        if self.clip is not None and not h["exact_stream"]:
            if self._gs_src is not self._coef:    # (K x this update's clip coefficient, once per update)
                self._gs_src, self._gs_coef = self._coef, self._const(float(K)) * self._coef
            gs, sq_acc = self._gs_coef, self._sq
        else:
            gs = self._const(float(K))
        SK.absorb_q8_dense(p.grad, st["m"], st["ms"], gs, b, fresh, K, sq_acc)
        if fresh:
            st["seen"] = torch.ones((), dtype=torch.bool, device=dev)
            hs["seen"] = True
        st["absorbed"] = True
        p.grad = None

    def _end_backward(self):
        self._upd_sq = self._upd_sq + self._sq / (self.K * self.K)     # (hooks summed ||K g||^2; g = grad / K)
        self._sq = torch.zeros_like(self._sq)
        self._cb = False

    @torch.no_grad()
    def stream_clip(self, params, max_norm=None):
        """Replacement for clip_grad_norm_(params, max_norm): clips the gradients still present (vectors) by the
        global norm, sets the matrices' coefficient, returns the global norm (device scalar)."""
        max_norm = self.clip if max_norm is None else max_norm
        grads = [p.grad for p in params if p.grad is not None and not p.grad.is_sparse]
        vec_sq = sum((g.float().pow(2).sum() for g in grads), torch.zeros((), device=self._sq.device))
        if self.hp["gbar"]:                   # ||G_bar|| directly from the per-update mean buffer
            for group in self.param_groups:
                if group["kind"] == "matrix":
                    for p in group["params"]:
                        st = self.state.get(p, {})
                        if st.get("absorbed"):
                            if getattr(p, "_region", None) is not None and p.ndim == 3 and "n_acc" in st:
                                w = (st["n_acc"].float() / st["N_acc"].float().clamp_min(1.0)).reshape(-1)
                                vec_sq = vec_sq + _dq8_sq_w(st["g"], st["gs"], w * w)
                            else:
                                vec_sq = vec_sq + _dq8_sq(st["g"], st["gs"])
        if self.hp["exact_stream"]:           # ||G_bar|| = ||M_t - a M_{t-1}|| / (1 - a), exact
            for group in self.param_groups:
                if group["kind"] != "matrix":
                    continue
                for p in group["params"]:
                    st = self.state.get(p, {})
                    if "m_prev" not in st:
                        continue
                    a = st["a_acc"].reshape(-1).float()
                    # routed experts: the snapshot gives each expert's token-weighted mean gradient (token-clock units,
                    # N/n_s times the gradient of the global mean loss); w_s = n_s / N maps it back, so the global
                    # norm is that of the gradient of the mean loss, as for dense matrices (w = 1)
                    if getattr(p, "_region", None) is not None and p.ndim == 3 and "n_acc" in st:
                        w = (st["n_acc"].float() / st["N_acc"].float().clamp_min(1.0)).reshape(-1)
                    else:
                        w = torch.ones_like(a)
                    if st["m"].dtype == torch.int8:     # (no fp32 copies of M, M_prev)
                        vec_sq = vec_sq + _snap_diff_sq_w(st["m"], st["ms"], st["m_prev"], st["ms_prev"], a, w * w)
                        continue
                    m = st["m"].float()
                    mp = st["m_prev"].float()
                    d = (m.reshape(a.shape[0], -1) - a.reshape(-1, 1) * mp.reshape(a.shape[0], -1))
                    vec_sq = vec_sq + (w * w * d.pow(2).sum(1) / (1 - a).clamp_min(1e-6) ** 2).sum()
        total = (vec_sq + self._upd_sq).sqrt()
        coef = (max_norm / (total + 1e-6)).clamp(max=1.0)
        if grads:
            torch._foreach_mul_(grads, coef)
        self._coef = coef
        self._upd_sq = torch.zeros_like(self._upd_sq)
        return total

    @torch.no_grad()
    def step(self, closure=None):
        h = self.hp
        self.t += 1
        dev = self.param_groups[0]["params"][0].device
        jobs, plan, vec_hp = {}, [], {}
        for group in self.param_groups:
            lr, lam = group["lr"], group["weight_decay"]
            for p in group["params"]:
                if group["kind"] == "matrix":
                    st = self.state.get(p, {})
                    if st.pop("absorbed", False):
                        if h["rule"] == "sgd":    # (no Newton-Schulz: the direction itself, in _post)
                            pl = self._stream_matrix(p, st, lr, lam)
                            d_, dL_, dR_ = self._direction(self._gbar_dir(p), st, 1)
                            plan.append((pl[0], dL_, dR_, pl[3], pl[4], False, d_))
                            continue
                        plan.append(self._stream_matrix(p, st, lr, lam))
                        jobs.setdefault(tuple(p.shape[-2:]), []).append((p, None))
                    continue
                if p.grad is None:
                    continue
                if id(p) in self._vec1d:
                    continue
                st = self.state[p]
                st["t"] = st.get("t", 0) + 1
                if "m1" not in st:
                    st["m1"], st["v"] = torch.zeros_like(p, dtype=torch.float32), torch.zeros_like(p, dtype=torch.float32)
                key = (st["t"], lr)
                if key not in vec_hp:
                    vec_hp[key] = torch.tensor([1 / (1 - 0.9 ** st["t"]), 1 / (1 - h["vec_beta2"] ** st["t"]), lr, h["eps"],
                                                h["vec_beta2"]],
                                               device=dev)
                new, st["m1"], st["v"] = _swo_vec_fused(p, p.grad, st["m1"], st["v"], vec_hp[key],
                                                         lam if p.ndim >= 2 else 0.0)
                p.copy_(new)
        if self.vec_adam is not None:
            for g_ in self.vec_adam.param_groups:
                g_["lr"] = next(g["lr"] for g in self.param_groups if g["kind"] == "vector")
            self.vec_adam.step()
        # Newton-Schulz per shape group, in chunks of <= CHUNK elements: temporaries stay bounded whatever the model
        # size (each matrix builds its direction here, chunk by chunk, from the momentum)
        CHUNK = 1 << 25
        info = {id(it[0]): it for it in plan}
        done = set()
        for shape, items in jobs.items():
            i = 0
            while i < len(items):
                chunk, tot = [], 0
                while i < len(items) and (not chunk or tot + items[i][0].numel() <= CHUNK):
                    chunk.append(items[i])
                    tot += items[i][0].numel()
                    i += 1
                dirs, mats = {}, []
                for p, mt in chunk:
                    if mt is None:
                        mt, dL, dR = self._stream_dir(p)
                        dirs[id(p)] = (dL, dR)
                    mats.append(mt)
                o = self.ortho(torch.cat(mats))
                del mats
                at = 0
                for p, mt in chunk:
                    S = p.shape[0] if p.ndim == 3 else 1
                    _, dL, dR, step, lam, _, _ = info[id(p)]
                    if id(p) in dirs:
                        dL, dR = dirs[id(p)]
                    self._post(p, o[at:at + S], dL, dR, step, lam, True)
                    done.add(id(p))
                    at += S
                del o
        for p, dL, dR, step, lam, spectral, mt in plan:
            if id(p) not in done:
                self._post(p, mt, dL, dR, step, lam, spectral)
        self._finish()

    def _post(self, p, o, dL, dR, step, lam, spectral):
        p3 = p if p.ndim == 3 else p.unsqueeze(0)
        bf = p.dtype == torch.bfloat16
        mode = 0 if spectral else 2               # Newton-Schulz direction (polar) / the direction itself (sgd)
        new, _ = _swo_post_fused(p3, o, dL, dR, step, lam, mode, 1.0, False, bf, True, False)
        p.copy_(new.view_as(p))

    def _direction(self, m, st, S):
        """The direction as fp32 and unit row / column scales (the update kernel's per-row / per-column factors)."""
        key = ("ones", S, m.shape[-2], m.shape[-1])
        if key not in self._consts:
            self._consts[key] = (torch.ones(S, m.shape[-2], device=m.device),
                                 torch.ones(S, m.shape[-1], device=m.device))
        dL, dR = self._consts[key]
        return m.float(), dL, dR

    def _gbar_dir(self, p):
        """gbar: M_t = a M_{t-1} + (1 - a) kappa Gbar (requantized), and the Nesterov direction
        a M_t + (1 - a) kappa Gbar = a^2 M_{t-1} + (a (1 - a) + 1 - a) kappa Gbar."""
        h, st = self.hp, self.state[p]
        if p.ndim == 3 and "gbar_slots" in st:   # routed experts: per-expert token-clock coefficients (_stream_matrix)
            a, c_m, c_g = st.pop("gbar_slots")
            qn, sn, d = _gbar_step_slots(st["m"], st["ms"], st["g"], st["gs"], a, self._coef.float().reshape(1),
                                         c_m, c_g)
            st["m"], st["ms"] = qn, sn
            st["g_seen"].zero_()
            st["g_nb"].zero_()
            return d.reshape(p.shape[0], *p.shape[-2:])
        hs = self._host[id(p)]
        hs.pop("gk", None)
        first = not hs["seen"]
        a = 0.0 if first else h["beta1"]
        c_m, c_g = a * a, a * (1 - a) + (1 - a)
        qn, sn, d = _gbar_step(st["m"], st["ms"], st["g"], st["gs"], self._const(a), self._coef.float().reshape(1),
                               self._const(c_m), self._const(c_g))
        st["m"], st["ms"] = qn, sn
        if first:
            st["seen"] = torch.ones((), dtype=torch.bool, device=p.device)
            hs["seen"] = True
        return d.reshape(1, *p.shape[-2:])

    def _stream_dir(self, p):
        h, st = self.hp, self.state[p]
        S = p.shape[0] if p.ndim == 3 else 1
        if h["gbar"]:
            return self._direction(self._gbar_dir(p), st, S)
        q8 = st["m"].dtype == torch.int8
        if h["nesterov"] and "m_prev" in st and q8:      # snapshot, fused
            for k_ in ("fresh_upd", "k_in", "n_before"):
                st.pop(k_, None)
            qn, sn, d = _snap_step(st["m"], st["ms"], st.pop("m_prev"), st.pop("ms_prev"),
                                   st.pop("a_acc").reshape(S).float(), self._coef.float())
            st["m"], st["ms"] = qn, sn
            return self._direction(d.reshape(S, *p.shape[-2:]), st, S)
        m = (_dq8(st["m"], st["ms"]) if q8 else st["m"]).reshape(S, *p.shape[-2:])
        if h["nesterov"] and "m_prev" in st:     # snapshot, bf16 momentum (size not a multiple of 128)
            mp, a = st.pop("m_prev"), st.pop("a_acc").view(S, 1, 1)
            st.pop("ms_prev")
            mp = mp.reshape(S, *p.shape[-2:]).float()
            m = m.float()
            for k_ in ("fresh_upd", "k_in", "n_before"):
                st.pop(k_, None)
            m = a * mp + self._coef * (m - a * mp)   # absorbed unclipped: this update's coefficient once, factor <= 1
            st["m"] = m.reshape(p.shape).to(st["m"].dtype)
            m = (1 + a) * m - a * mp
        return self._direction(m, st, S)

    def _stream_matrix(self, p, st, lr, lam):
        """Step size for one matrix (experts: the token clock's per-expert factor), then reset the per-update
        counters. The direction is built later, in the Newton-Schulz chunks (_stream_dir)."""
        h = self.hp
        S = p.shape[0] if p.ndim == 3 else 1
        region = getattr(p, "_region", None) if p.ndim == 3 else None
        scale = torch.ones(S, device=p.device)
        if not h["nesterov"]:                     # (equal-weight counters; with the snapshot _stream_dir pops them)
            for k_ in ("fresh_upd", "k_in", "n_before"):
                st.pop(k_, None)
        if region is not None:
            n, live = st.pop("n_acc"), st.pop("live_acc")
            st.pop("N_acc", None)
            N = region.tokens_acc.clamp_min(1.0)
            f_ref = region.k / region.active.sum().clamp_min(1)
            tau = h["beta1"] / (1 - h["beta1"]) * f_ref * N
            b_t = tau / (tau + n)
            scale = torch.clamp((1 - b_t) / (1 - h["beta1"]), max=1.0) * live
            if h["gbar"]:                    # M_s <- b_s M_s + (1 - b_s) kappa Gbar_s (first ever: M_s = Gbar_s)
                b = torch.where(st["seen"].reshape(S), b_t.float().reshape(S), torch.zeros(S, device=p.device))
                lv = live.reshape(S)
                st["gbar_slots"] = (torch.where(lv, b, torch.ones_like(b)), torch.where(lv, b * b, torch.zeros_like(b)),
                                    torch.where(lv, b * (1 - b) + (1 - b), torch.zeros_like(b)))
                st["seen"] = st["seen"] | live.reshape(st["seen"].shape)
            st.pop("n_snap", None), st.pop("N_snap", None)
        return (p, None, None, scale * (-lr * 1.0), lam, True, None)

    def _finish(self):
        for group in self.param_groups:                               # routed slot counts: read, now reset
            for p in group["params"]:
                region = getattr(p, "_region", None)
                if region is not None:
                    region.dispatch_acc.zero_()
                    region.tokens_acc.zero_()


def muon_split(named):
    """Names of the parameters StreamMuon (or Muon) trains: weight matrices with both sides >= 64, except embeddings,
    output heads, routers, convolution taps, decays, positions and scales (those stay on AdamW)."""
    skip = ("emb", "head", "router", "conv_w", "decay", ".pos", "scale")
    return [n for n, p in named.items() if p.requires_grad and p.ndim >= 2 and min(p.shape[-2:]) >= 64
            and not any(s in n for s in skip)]
