"""Grouped matrix multiply for routed experts, in Triton: Y[rows of group g] = X[rows of g] @ W[g].

Replaces torch._grouped_mm on GPUs without a native grouped kernel (< sm_90, e.g. the RTX 4060), where PyTorch
falls back to a Python loop over groups with one GPU->CPU sync per group (~48 per call). Here everything stays
on the device: a tile table (group, tile-within-group) is built from the offsets with tensor ops, every kernel
launches a static grid and idle tiles exit early. Same result as torch._grouped_mm (fp32 accumulation).

  X (T, K) rows sorted by group, W (G, K, N), offs (G,) int32 inclusive cumulative counts (as torch._grouped_mm).
"""

import torch
import triton
import triton.language as tl

BM, BN, BK = 64, 64, 32


@triton.jit
def _gmm_kernel(x_ptr, w_ptr, y_ptr, tg_ptr, tj_ptr, offs_ptr, ntiles_ptr, K, N,
                sxm, sxk, swg, swk, swn, sym, syn, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                IEEE: tl.constexpr):
    t = tl.program_id(0)
    if t >= tl.load(ntiles_ptr):
        return
    g = tl.load(tg_ptr + t)
    j = tl.load(tj_ptr + t)
    start = tl.where(g > 0, tl.load(offs_ptr + g - 1, mask=g > 0, other=0), 0)
    end = tl.load(offs_ptr + g)
    rows = start + j * BM + tl.arange(0, BM)
    rmask = rows < end
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    cmask = cols < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        ks = k0 + tl.arange(0, BK)
        kmask = ks < K
        a = tl.load(x_ptr + rows[:, None] * sxm + ks[None, :] * sxk, mask=rmask[:, None] & kmask[None, :], other=0.0)
        b = tl.load(w_ptr + g * swg + ks[:, None] * swk + cols[None, :] * swn, mask=kmask[:, None] & cmask[None, :],
                    other=0.0)
        if IEEE:                          # fp32 inputs: exact (tl.dot would use TF32)
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc += tl.dot(a, b)
    tl.store(y_ptr + rows[:, None] * sym + cols[None, :] * syn, acc.to(y_ptr.dtype.element_ty),
             mask=rmask[:, None] & cmask[None, :])


@triton.jit
def _gmm_dw_kernel(x_ptr, dy_ptr, dw_ptr, offs_ptr, K, N, sxm, sxk, sym, syn, swg, swk, swn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, IEEE: tl.constexpr):
    g = tl.program_id(0)
    ks = tl.program_id(1) * BK + tl.arange(0, BK)
    cols = tl.program_id(2) * BN + tl.arange(0, BN)
    kmask, cmask = ks < K, cols < N
    start = tl.where(g > 0, tl.load(offs_ptr + g - 1, mask=g > 0, other=0), 0)
    end = tl.load(offs_ptr + g)
    acc = tl.zeros((BK, BN), dtype=tl.float32)
    for r0 in range(start, end, BM):
        rows = r0 + tl.arange(0, BM)
        rmask = rows < end
        a = tl.load(x_ptr + rows[None, :] * sxm + ks[:, None] * sxk, mask=kmask[:, None] & rmask[None, :], other=0.0)
        b = tl.load(dy_ptr + rows[:, None] * sym + cols[None, :] * syn, mask=rmask[:, None] & cmask[None, :], other=0.0)
        if IEEE:                          # fp32 inputs: exact (tl.dot would use TF32)
            acc += tl.dot(a, b, input_precision="ieee")
        else:
            acc += tl.dot(a, b)
    tl.store(dw_ptr + g * swg + ks[:, None] * swk + cols[None, :] * swn, acc.to(dw_ptr.dtype.element_ty),
             mask=kmask[:, None] & cmask[None, :])


def _tiles(offs, T):
    """(group, tile-within-group) per row tile, on the device; static size T // BM + G."""
    G = offs.shape[0]
    counts = torch.diff(offs, prepend=offs.new_zeros(1))
    nt = (counts + BM - 1) // BM
    tend = nt.cumsum(0)
    maxt = T // BM + G
    t = torch.arange(maxt, device=offs.device, dtype=torch.int32)
    tg = torch.searchsorted(tend, t, right=True).clamp(max=G - 1).to(torch.int32)
    tj = (t - (tend[tg] - nt[tg])).to(torch.int32)
    return tg, tj, tend[-1:].to(torch.int32), maxt


def _gmm(x, w, offs, tiles, transpose_w=False, y=None):
    tg, tj, ntiles, maxt = tiles
    T, K = x.shape
    N = w.shape[1] if transpose_w else w.shape[2]
    if y is None:
        y = torch.empty(T, N, device=x.device, dtype=x.dtype)
    if transpose_w:                       # use W[g]^T: (N, K) view of (K, N) storage
        swk, swn = w.stride(2), w.stride(1)
    else:
        swk, swn = w.stride(1), w.stride(2)
    _gmm_kernel[(maxt, triton.cdiv(N, BN))](x, w, y, tg, tj, offs, ntiles, K, N, x.stride(0), x.stride(1),
                                           w.stride(0), swk, swn, y.stride(0), y.stride(1), BM=BM, BN=BN, BK=BK,
                                           IEEE=x.dtype == torch.float32)
    return y


def _dw_tiles(K, N):
    """dW tile config (BM rows per step, BN, BK, warps, stages), measured on the RTX 4060 at 1B shapes:
    (K 1024, N 256) 136 -> 104 us with 32x64x64; (K 512, N 1024) 310 -> 238 us with
    32x128x32 (both 8 warps, 2 stages). Small shapes keep the generic config."""
    if K * N < (1 << 16):
        return BM, BN, BK, 4, 2
    return (32, 128, 32, 8, 2) if N >= 512 else (32, 64, 64, 8, 2)


def _dw(x, dy, offs, dw):
    G, K, N = dw.shape
    bm, bn, bk, warps, stages = _dw_tiles(K, N)
    _gmm_dw_kernel[(G, triton.cdiv(K, bk), triton.cdiv(N, bn))](
        x, dy, dw, offs, K, N, x.stride(0), x.stride(1), dy.stride(0), dy.stride(1), dw.stride(0), dw.stride(1),
        dw.stride(2), BM=bm, BN=bn, BK=bk, IEEE=x.dtype == torch.float32, num_warps=warps, num_stages=stages)


class _GroupedMM(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, offs, tiles=None):
        tiles = tiles if tiles is not None else _tiles(offs, x.shape[0])
        ctx.save_for_backward(x, w, offs)
        ctx.tiles = tiles
        return _gmm(x, w, offs, tiles)

    @staticmethod
    def backward(ctx, dy):
        x, w, offs = ctx.saved_tensors
        dy = dy.contiguous()
        dx = _gmm(dy, w, offs, ctx.tiles, transpose_w=True) if ctx.needs_input_grad[0] else None
        dw = None
        if ctx.needs_input_grad[1]:
            G, K, N = w.shape
            dw = torch.empty(G, K, N, device=w.device, dtype=w.dtype)
            _dw(x, dy, offs, dw)
        return dx, dw, None, None


@torch.compiler.disable
def tiles_for(offs, T):
    """The tile table for T grouped rows (compute once per MoE layer forward; shared by all its grouped matmuls)."""
    return _tiles(offs, T)


@torch.compiler.disable
def grouped_mm(x, w, offs, tiles=None):
    """Drop-in for torch._grouped_mm(x, w, offs=offs) with 2-D x, 3-D w (sync-free, all GPUs)."""
    return _GroupedMM.apply(x, w, offs, tiles)


# ---------------------------------------------------------------------------------------------------------------
# torch.library custom ops: the same kernels, visible to torch.compile as opaque ops with autograd (no graph breaks
# around them, so the surrounding norms / casts / routing glue can fuse)

@torch.library.custom_op("streamstate::gmm_tiles", mutates_args=())
def gmm_tiles_op(offs: torch.Tensor, T: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tg, tj, ntiles, _ = _tiles(offs, T)
    return tg, tj, ntiles


@gmm_tiles_op.register_fake
def _(offs, T):
    maxt = T // BM + offs.shape[0]
    return (offs.new_empty(maxt, dtype=torch.int32), offs.new_empty(maxt, dtype=torch.int32),
            offs.new_empty(1, dtype=torch.int32))


@torch.library.custom_op("streamstate::gmm", mutates_args=())
def gmm_op(x: torch.Tensor, w: torch.Tensor, offs: torch.Tensor, tg: torch.Tensor, tj: torch.Tensor,
           ntiles: torch.Tensor, transpose_w: bool) -> torch.Tensor:
    return _gmm(x, w, offs, (tg, tj, ntiles, tg.shape[0]), transpose_w=transpose_w)


@gmm_op.register_fake
def _(x, w, offs, tg, tj, ntiles, transpose_w):
    return x.new_empty(x.shape[0], w.shape[1] if transpose_w else w.shape[2])


def _fused_ctx(w, pairs):
    """StreamMuon streaming hook (kernels.FUSED_DW): when the optimizer registered this weight, its gradient is folded
    into the int8 momentum inside the dW kernel and the returned 'gradient' is an unused placeholder (the optimizer's
    post-accumulate hook finishes the update and drops it). pairs: [(x, dy), ...] per column block."""
    from streamstate import kernels as SK
    ent = SK.FUSED_DW.get(w.data_ptr()) if SK.FUSED_DW else None
    if ent is None:
        return None
    p, fn = ent[0](), ent[1]()
    if p is None or fn is None or p.data_ptr() != w.data_ptr() or p.shape != w.shape:   # (stale entry)
        SK.FUSED_DW.pop(w.data_ptr(), None)
        return None
    if not all(SK.dw_supported(x, dy, w.shape) for x, dy in pairs):
        return None
    return fn(p)


@torch.library.custom_op("streamstate::gmm_dw", mutates_args=())
def gmm_dw_op(x: torch.Tensor, dy: torch.Tensor, offs: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    ctx = _fused_ctx(w, [(x, dy)])
    if ctx is not None:
        from streamstate import kernels as SK
        SK.dw_absorb(x, dy, offs, ctx)
        return w.new_empty(w.shape)
    dw = x.new_empty(offs.shape[0], x.shape[1], dy.shape[1])
    _dw(x, dy, offs, dw)
    return dw


@gmm_dw_op.register_fake
def _(x, dy, offs, w):
    return x.new_empty(offs.shape[0], x.shape[1], dy.shape[1])


def _gmm_setup(ctx, inputs, output):
    x, w, offs, tg, tj, ntiles, transpose_w = inputs
    ctx.save_for_backward(x, w, offs, tg, tj, ntiles)


def _gmm_backward(ctx, dy):
    x, w, offs, tg, tj, ntiles = ctx.saved_tensors
    dy = dy.contiguous()
    dx = gmm_op(dy, w, offs, tg, tj, ntiles, True)
    dw = gmm_dw_op(x, dy, offs, w)
    return dx, dw, None, None, None, None, None


gmm_op.register_autograd(_gmm_backward, setup_context=_gmm_setup)


