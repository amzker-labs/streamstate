"""Hand-written Triton kernel for StreamMuon's streaming step (backward-into-M with int8 momentum).

One pass per micro-batch and matrix: read the gradient (bf16) and the int8 momentum + its per-128 bf16 scales, fold
the gradient into the momentum (token-time EMA, first visit M = G~), requantize with in-kernel stochastic rounding,
and accumulate this micro-batch's row / column means of G~^2 and the gradient's squared norm. ~4 B / parameter of
memory traffic instead of the compiled version's several full fp32 passes. Same math as optim._swo_absorb_q8.
"""

import random

import torch
import triton
import triton.language as tl


@triton.jit
def _dither(off, seed):
    """Uniform [0, 1) per element for stochastic rounding: a 32-bit integer hash of (offset, seed) (lowbias32,
    ~8 integer ops) instead of tl.rand's Philox (~10 rounds; it made the absorb compute-bound: 42 G elem/s). 24-bit
    resolution, far finer than needed to dither int8 / bf16 rounding (bias <= 2^-24 of a step)."""
    x = off.to(tl.uint32) * 0x9E3779B1 + seed.to(tl.uint32)
    x = x ^ (x >> 16)
    x = x * 0x7FEB352D
    x = x ^ (x >> 15)
    x = x * 0x846CA68B
    x = x ^ (x >> 16)
    return (x >> 8).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _absorb_q8_kernel(g_ptr, q_ptr, s_ptr, b_ptr, gs_ptr, upd_ptr, fresh_ptr, rsum_ptr, csum_ptr, sq_ptr,
                      M, N, K, seed, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    """One program per (slot, block of ROWS rows, 128-column block); column sums are reduced over the ROWS rows
    in registers before one atomic (one atomic per element contended 1024-way and capped the old kernel at ~68 GB/s);
    the squared norm goes to a per-program partial (a single scalar atomic contends badly)."""
    pid = tl.program_id(0)
    jb = tl.program_id(1)
    slot = pid // (M // ROWS)
    upd = tl.load(upd_ptr + slot)          # live this micro-batch (int8 flag)
    if upd == 0:                           # slot got no tokens: momentum untouched (not even requantized)
        tl.store(sq_ptr + pid * (N // BLOCK) + jb, 0.0)
        return
    fresh = tl.load(fresh_ptr + slot)      # live and never seen: M = G~
    b = tl.load(b_ptr + slot)
    gsc = tl.load(gs_ptr + slot)
    rr = pid * ROWS + tl.arange(0, ROWS)
    cols = jb * BLOCK + tl.arange(0, BLOCK)
    off = rr[:, None] * N + cols[None, :]
    g = tl.load(g_ptr + off).to(tl.float32)
    sidx = rr * (N // BLOCK) + jb
    m = tl.load(q_ptr + off).to(tl.float32) * tl.load(s_ptr + sidx).to(tl.float32)[:, None]
    gt = g * gsc
    m_new = tl.where(fresh != 0, gt, b * m + (1 - b) * gt)
    # scale rounded UP to bf16 (x (1 + 2^-7)): bf16(amax / 127) rounds down half the time, then amax / scale reaches
    # 127.5 and the clamp shrank the block maximum (measured 5.8% after 8 micro-batch requantizations)
    amax = tl.max(tl.abs(m_new), 1)
    scale = tl.maximum((amax * (1.0078125 / 127.0)).to(tl.bfloat16).to(tl.float32), 1e-30)
    qn = tl.minimum(tl.maximum(tl.floor(m_new / scale[:, None] + _dither(off, seed)), -127.0), 127.0)
    tl.store(q_ptr + off, qn.to(tl.int8))
    tl.store(s_ptr + sidx, scale.to(tl.bfloat16))
    g2 = gt * gt
    tl.atomic_add(rsum_ptr + rr, tl.sum(g2, 1) / N)
    tl.atomic_add(csum_ptr + slot * N + cols, tl.sum(g2, 0))
    tl.store(sq_ptr + pid * (N // BLOCK) + jb, tl.sum(tl.sum(g * g, 1), 0) * K * K)


ROWS = 32


def supported(g):
    return g.dim() == 3 and g.shape[-1] % 128 == 0 and g.shape[-2] % ROWS == 0 and g.is_contiguous()


def absorb_q8(g, qv, qs, seen, r, c, live, b_t, gscale, b2, K):
    """In place on qv (int8, (S, M, N)) and qs (bf16 scales, (S*M*N/128, 1)); returns new seen, r, c and the
    squared full-scale gradient norm (a device scalar). Requires supported(g) (N % 128 == 0, M % 32 == 0)."""
    S, M, N = g.shape
    fresh = live & ~seen
    rsum = torch.zeros(S * M, device=g.device, dtype=torch.float32)
    csum = torch.zeros(S, N, device=g.device, dtype=torch.float32)
    grid = (S * M // ROWS, N // 128)
    sq = torch.empty(grid[0] * grid[1], device=g.device, dtype=torch.float32)
    _absorb_q8_kernel[grid](g, qv, qs, b_t.float().contiguous(), gscale.float().contiguous(), live.to(torch.int8),
                            fresh.to(torch.int8), rsum, csum, sq, M, N, float(K), random.getrandbits(31),
                            ROWS=ROWS, BLOCK=128, num_warps=4)
    r_new, c_new = rsum.view(S, M), csum / M
    fv, lv = fresh.view(-1, 1), live.view(-1, 1)
    r = torch.where(lv, torch.where(fv, r_new, b2 * r + (1 - b2) * r_new), r)
    c = torch.where(lv, torch.where(fv, c_new, b2 * c + (1 - b2) * c_new), c)
    g2_slot = rsum.view(S, M).sum(1) * N / gscale.float().pow(2).clamp_min(1e-30)      # per-slot ||g||^2
    return seen | live, r, c, sq.sum(), g2_slot


# ---------------------------------------------------------------------------------------------------------------
# Fused weight gradient -> int8 momentum (Plan A): the routed slots' dW tile is folded into the momentum in the
# grouped-GEMM epilogue, so the gradient is never written to memory nor read back (saves 4 B / parameter / micro-
# batch). Same arithmetic as _absorb_q8_kernel on bf16(dW) (the gradient is rounded to bf16 first, as the stored
# gradient was); only the stochastic-rounding draws differ.


@triton.jit
def _absorb_q8_dense_kernel(g_ptr, q_ptr, s_ptr, gs_ptr, sq_ptr, N, MROWS, K, b, fresh, seed,
                            SQ: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr):
    """_absorb_q8_kernel for one dense matrix without row / column statistics: the momentum
    coefficient b and the first-visit flag come from the host (no per-call coefficient tensors), the squared norm
    (only when SQ) is added straight into the optimizer's accumulator. Same arithmetic and same dither as
    _absorb_q8_kernel, so the int8 momentum is bit-identical for the same seed."""
    pid = tl.program_id(0)
    jb = tl.program_id(1)
    gsc = tl.load(gs_ptr)
    rr = pid * ROWS + tl.arange(0, ROWS)
    rm = rr < MROWS
    cols = jb * BLOCK + tl.arange(0, BLOCK)
    off = rr[:, None] * N + cols[None, :]
    g = tl.load(g_ptr + off, mask=rm[:, None], other=0.0).to(tl.float32)
    sidx = rr * (N // BLOCK) + jb
    gt = g * gsc
    if fresh != 0:
        m_new = gt
    else:
        m = tl.load(q_ptr + off, mask=rm[:, None], other=0).to(tl.float32) * \
            tl.load(s_ptr + sidx, mask=rm, other=0.0).to(tl.float32)[:, None]
        m_new = b * m + (1 - b) * gt
    amax = tl.max(tl.abs(m_new), 1)
    scale = tl.maximum((amax * (1.0078125 / 127.0)).to(tl.bfloat16).to(tl.float32), 1e-30)
    qn = tl.minimum(tl.maximum(tl.floor(m_new / scale[:, None] + _dither(off, seed)), -127.0), 127.0)
    tl.store(q_ptr + off, qn.to(tl.int8), mask=rm[:, None])
    tl.store(s_ptr + sidx, scale.to(tl.bfloat16), mask=rm)
    if SQ:
        tl.atomic_add(sq_ptr, tl.sum(tl.sum(g * g, 1), 0) * K * K)


def absorb_q8_dense(g, qv, qs, gscale, b, fresh, K, sq_acc=None):
    """In place on qv (int8, (M, N) or (1, M, N)) and qs; b: python float momentum coefficient; fresh: python bool
    (first visit: M = G~); gscale: device fp32 tensor with one element; sq_acc: device fp32 scalar the squared
    full-scale gradient norm is added to (None: not computed). Requires contiguous g / qv and numel % 128 == 0."""
    if g.shape[-1] % 128 == 0:                 # (row layout: same offsets / dither as _absorb_q8_kernel)
        M, N = g.numel() // g.shape[-1], g.shape[-1]
    else:                                      # (no row statistics: rows do not matter; the flat tensor in rows of
        M, N = g.numel() // 128, 128           #  128 = the int8 scale blocks, same flat offsets and block index)
    grid = (triton.cdiv(M, ROWS), N // 128)
    _absorb_q8_dense_kernel[grid](g, qv, qs, gscale, sq_acc if sq_acc is not None else gscale, N, M, float(K),
                                  float(b), int(bool(fresh)), random.getrandbits(31), SQ=sq_acc is not None,
                                  ROWS=ROWS, BLOCK=128, num_warps=4)


@triton.jit
def _dw_absorb_kernel(x_ptr, dy_ptr, offs_ptr, q_ptr, s_ptr, b_ptr, gs_ptr, upd_ptr, fresh_ptr, rsum_ptr,
                      csum_ptr, sq_ptr, K, N, NF, col0, sxm, sxk, sym, syn, seed, KA,
                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """dW[g][:, col0:col0+N] = X_g^T dY_g (rows of group g), one (BK x BN) tile per program, then absorb as
    _absorb_q8_kernel (BN = 128 = the int8 scale block; NF = the full weight width, col0 = this half's offset)."""
    g = tl.program_id(0)
    kb = tl.program_id(1)
    nb = tl.program_id(2)
    pidlin = (g * tl.num_programs(1) + kb) * tl.num_programs(2) + nb
    upd = tl.load(upd_ptr + g)
    if upd == 0:                           # slot got no tokens: dW = 0 and the momentum is untouched
        tl.store(sq_ptr + pidlin, 0.0)
        return
    ks = kb * BK + tl.arange(0, BK)
    cols = nb * BN + tl.arange(0, BN)
    start = tl.where(g > 0, tl.load(offs_ptr + g - 1, mask=g > 0, other=0), 0)
    end = tl.load(offs_ptr + g)
    acc = tl.zeros((BK, BN), dtype=tl.float32)
    for r0 in range(start, end, BM):
        rows = r0 + tl.arange(0, BM)
        rmask = rows < end
        a = tl.load(x_ptr + rows[None, :] * sxm + ks[:, None] * sxk, mask=rmask[None, :], other=0.0)
        b = tl.load(dy_ptr + rows[:, None] * sym + cols[None, :] * syn, mask=rmask[:, None], other=0.0)
        acc += tl.dot(a, b)
    gq = acc.to(tl.bfloat16).to(tl.float32)
    fresh = tl.load(fresh_ptr + g)
    bt = tl.load(b_ptr + g)
    gsc = tl.load(gs_ptr + g)
    fc = col0 + cols
    off = g * K * NF + ks[:, None] * NF + fc[None, :]
    sidx = (g * K + ks) * (NF // BN) + (col0 + nb * BN) // BN
    m = tl.load(q_ptr + off).to(tl.float32) * tl.load(s_ptr + sidx).to(tl.float32)[:, None]
    gt = gq * gsc
    m_new = tl.where(fresh != 0, gt, bt * m + (1 - bt) * gt)
    amax = tl.max(tl.abs(m_new), 1)
    scale = tl.maximum((amax * (1.0078125 / 127.0)).to(tl.bfloat16).to(tl.float32), 1e-30)
    qn = tl.minimum(tl.maximum(tl.floor(m_new / scale[:, None] + _dither(off, seed)), -127.0), 127.0)
    tl.store(q_ptr + off, qn.to(tl.int8))
    tl.store(s_ptr + sidx, scale.to(tl.bfloat16))
    g2 = gt * gt
    tl.atomic_add(rsum_ptr + g * K + ks, tl.sum(g2, 1) / NF)
    tl.atomic_add(csum_ptr + g * NF + fc, tl.sum(g2, 0))
    tl.store(sq_ptr + pidlin, tl.sum(tl.sum(gq * gq, 1), 0) * KA * KA)


def dw_supported(x, dy, w_shape):
    G, K, N = w_shape
    return (x.dtype == torch.bfloat16 and dy.dtype == torch.bfloat16 and K % 32 == 0 and N % 128 == 0)


class AbsorbCtx:
    """One micro-batch's fused absorb of one routed weight (made by the optimizer before the first dW tile of that weight;
    finished by the optimizer's post-accumulate hook). Holds the state views and the per-slot coefficients."""

    def __init__(self, qv, qs, seen, live, b_t, gscale, K):
        S = qv.shape[0]
        self.qv, self.qs, self.K = qv, qs, K
        self.live, self.fresh = live, live & ~seen
        self.b_t, self.gscale = b_t.float().contiguous(), gscale.float().contiguous()
        self.live8, self.fresh8 = live.to(torch.int8), self.fresh.to(torch.int8)
        self.rsum = torch.zeros(S * qv.shape[1], device=qv.device, dtype=torch.float32)
        self.csum = torch.zeros(S, qv.shape[2], device=qv.device, dtype=torch.float32)
        self.sq = []


DW_CFG = (32, 32, 4, 2)        # (BM rows per step, BK, warps, stages)


def dw_absorb(x, dy, offs, ctx, col0=0, cfg=None):
    """Fused dW + absorb for the columns [col0, col0 + dy.shape[1]) of the weight held by ctx."""
    S, K, NF = ctx.qv.shape
    N = dy.shape[1]
    BM_, BK_, nw, ns = cfg or DW_CFG
    grid = (S, K // BK_, N // 128)
    sq = torch.empty(grid[0] * grid[1] * grid[2], device=x.device, dtype=torch.float32)
    _dw_absorb_kernel[grid](x, dy, offs, ctx.qv, ctx.qs, ctx.b_t, ctx.gscale, ctx.live8, ctx.fresh8, ctx.rsum,
                            ctx.csum, sq, K, N, NF, col0, x.stride(0), x.stride(1), dy.stride(0), dy.stride(1),
                            random.getrandbits(31), float(ctx.K), BM=BM_, BN=128, BK=BK_, num_warps=nw,
                            num_stages=ns)
    ctx.sq.append(sq)


def absorb_finish(ctx, seen, r, c, b2):
    """The tail of absorb_q8 for a fused absorb: r / c EMAs, seen, squared norm, per-slot ||g||^2."""
    S, M, N = ctx.qv.shape
    r_new, c_new = ctx.rsum.view(S, M), ctx.csum / M
    fv, lv = ctx.fresh.view(-1, 1), ctx.live.view(-1, 1)
    r = torch.where(lv, torch.where(fv, r_new, b2 * r + (1 - b2) * r_new), r)
    c = torch.where(lv, torch.where(fv, c_new, b2 * c + (1 - b2) * c_new), c)
    g2_slot = ctx.rsum.view(S, M).sum(1) * N / ctx.gscale.pow(2).clamp_min(1e-30)
    sq = sum(s.sum() for s in ctx.sq)
    return seen | ctx.live, r, c, sq, g2_slot


# registry: weight data_ptr -> callable returning that weight's AbsorbCtx for the current micro-batch (set by the optimizer)
FUSED_DW = {}
