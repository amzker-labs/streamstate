"""Newton-Schulz with fp8 (e4m3) GEMMs for the first iterations (on the RTX 4060 fp8 GEMMs run ~1.9x
bf16; rounding X to e4m3 perturbs it by a few percent, far below the stochastic noise already in the momentum, and the
later bf16 iterations correct it). Batched Triton kernel: C[b] = s_p[b] s_q[b] (P[b] @ Q[b]) with fp8 inputs, fp32
accumulation, optional fused epilogue C = alpha * Z[b] + beta * C (the Newton-Schulz update a X + b (X X^T) X).
"""

import torch
import triton
import triton.language as tl

E4M3_MAX = 448.0


@triton.jit
def _bmm_fp8_kernel(p_ptr, q_ptr, c_ptr, c8_ptr, z_ptr, M, N, K,
                    spb, spm, spk, sqb, sqk, sqn, scb, scm, scn, szb, szm, szn,
                    in_scale, out_inv, alpha, beta,
                    HAS_Z: tl.constexpr, STORE_BF16: tl.constexpr, STORE_FP8: tl.constexpr, Q_IS_PT: tl.constexpr,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C = in_scale * (P @ Q) (or alpha Z + beta in_scale (P @ Q)); Q_IS_PT: Q = P^T read row-major from P (X X^T
    with coalesced loads). Stores bf16 C and / or fp8 C * out_inv (clamped: e4m3 overflow is NaN, not saturating)."""
    b = tl.program_id(2)
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(p_ptr + b * spb + rm[:, None] * spm + rk[None, :] * spk,
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        if Q_IS_PT:
            bt = tl.load(p_ptr + b * spb + rn[:, None] * spm + rk[None, :] * spk,
                         mask=(rn[:, None] < N) & (rk[None, :] < K), other=0.0)
            acc += tl.dot(a, tl.trans(bt))
        else:
            bb = tl.load(q_ptr + b * sqb + rk[:, None] * sqk + rn[None, :] * sqn,
                         mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
            acc += tl.dot(a, bb)
    acc = acc * in_scale
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    if HAS_Z:
        z = tl.load(z_ptr + b * szb + rm[:, None] * szm + rn[None, :] * szn, mask=mask, other=0.0).to(tl.float32)
        acc = alpha * z + beta * acc
    off = b * scb + rm[:, None] * scm + rn[None, :] * scn
    if STORE_BF16:
        tl.store(c_ptr + off, acc.to(tl.bfloat16), mask=mask)
    if STORE_FP8:
        tl.store(c8_ptr + off, tl.clamp(acc * out_inv, -448.0, 448.0).to(tl.float8e4nv), mask=mask)


# Fixed fp8 scales (no amax pass): |x_ij| <= sigma_max(X). X starts Frobenius-normalised (sigma <= 1) and the
# Newton-Schulz polynomials keep sigma below ~1.3, so |X| < 2 and |X X^T| <= sigma_max^2 < 2 -> one scale of 2 / 448 for both.
SCALE = 2.0 / E4M3_MAX


def _launch(p8, q8, M, N, K, in_scale, z=None, alpha=0.0, beta=1.0, store_bf16=True, store_fp8=False,
            q_is_pt=False, out_scale=None):
    Bt = p8.shape[0]
    c = torch.empty(Bt, M, N, device=p8.device, dtype=torch.bfloat16) if store_bf16 else None
    c8 = torch.empty(Bt, M, N, device=p8.device, dtype=torch.float8_e4m3fn) if store_fp8 else None
    ref = c if c is not None else c8
    zz = z if z is not None else ref
    q = q8 if q8 is not None else p8
    BM, BN, BK = (64, 64, 64) if q_is_pt else (64, 128, 64)       # (tile sweep, RTX 4060: AX 0.78 -> 0.62 ms)
    _bmm_fp8_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN), Bt)](
        p8, q, c if c is not None else c8, c8 if c8 is not None else ref, zz, M, N, K,
        p8.stride(0), p8.stride(1), p8.stride(2), q.stride(0), q.stride(1), q.stride(2),
        ref.stride(0), ref.stride(1), ref.stride(2), zz.stride(0), zz.stride(1), zz.stride(2),
        float(in_scale), 1.0 / (out_scale or SCALE), float(alpha), float(beta), HAS_Z=z is not None, STORE_BF16=store_bf16,
        STORE_FP8=store_fp8, Q_IS_PT=q_is_pt, BM=BM, BN=BN, BK=BK, num_warps=4, num_stages=3)
    return c, c8


QUINTIC = (3.4445, -4.7750, 2.0315)
GUARD_HITS = 0      # (matrices recomputed in bf16 by the divergence guard, for logging)
SCALE_B = 16.0 / E4M3_MAX      # B = b A + c A^2: |A| <= sigma^2 <= 1.44 (quintic keeps sigma <= ~1.2) -> |B| < 11


def orthogonalize_quintic_fp8(g, steps=5, fp8_steps=3):
    """Muon's quintic Newton-Schulz, X <- a X + (b A + c A^2) X with A = X X^T, the first `fp8_steps` iterations'
    GEMMs in fp8 (e4m3, fixed scales: X and A with 2 / 448, B with 16 / 448); the rest in bf16."""
    a, b, c = QUINTIC
    x = g.bfloat16()
    flat = x.ndim == 2
    if flat:
        x = x.unsqueeze(0)
    tall = x.shape[-2] > x.shape[-1]
    if tall:
        x = x.transpose(-2, -1)
    x = (x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)).contiguous()
    m, n = x.shape[-2:]
    x8 = (x * (1.0 / SCALE)).clamp(-448, 448).to(torch.float8_e4m3fn) if fp8_steps else None
    for i in range(steps):
        if i < fp8_steps:
            A, A8 = _launch(x8, None, m, m, n, SCALE * SCALE, store_bf16=True, store_fp8=True, q_is_pt=True)
            _, B8 = _launch(A8, A8, m, m, m, SCALE * SCALE, z=A, alpha=b, beta=c, store_bf16=False, store_fp8=True,
                            out_scale=SCALE_B)                                   # b A + c A A
            x, x8 = _launch(B8, x8, m, n, m, SCALE_B * SCALE, z=x, alpha=a, beta=1.0, store_bf16=True,
                            store_fp8=i + 1 < fp8_steps)                         # a X + B X
        else:
            A = torch.bmm(x, x.transpose(-2, -1))
            B = torch.baddbmm(A, A, A, beta=b, alpha=c)
            x = torch.baddbmm(x, B, x, beta=a, alpha=1.0)
    if fp8_steps:
        # Guard (found in fine-tuning Qwen2.5-0.5B, 2026-11-24): e4m3 keeps 3 mantissa bits, and near sigma ~ 1.2 the
        # quintic amplifies errors ~4x per step; a third fp8 iteration can push the top singular value past ~1.3, where
        # the quintic diverges (1.43 -> 3.2 -> blow-up, steps 1000x too large). A converged X has ||X||_F^2 <= ~1.45 m
        # (singular values <= ~1.2); any matrix above 2 m (or non-finite) is recomputed in bf16. One reduction and
        # one host read per call; a no-op when nothing diverged.
        n2 = x.float().pow(2).sum((-2, -1))
        bad = ~(n2 <= 2.0 * m)
        if bool(bad.any()):
            gb = g.bfloat16() if g.ndim == 3 else g.bfloat16().unsqueeze(0)
            ob = orthogonalize_quintic_fp8(gb[bad], steps=steps, fp8_steps=0)
            x[bad] = (ob.transpose(-2, -1) if tall else ob).to(x.dtype)
            global GUARD_HITS
            GUARD_HITS += int(bad.sum())
    x = x.transpose(-2, -1) if tall else x
    return x[0] if flat else x
