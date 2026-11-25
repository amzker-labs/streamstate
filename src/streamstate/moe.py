"""A standard mixture-of-experts layer (SwiGLU experts, shared expert, top-k) whose expert weights carry the `_region` tag StreamMuon's expert path reads.

Copied verbatim (import paths and comments aside) from the code that produced the paper's numbers.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class _Permute(torch.autograd.Function):
    """y = x[idx] for a permutation idx; backward g[inv] (a gather, no accumulate)."""

    @staticmethod
    def forward(ctx, x, idx, inv):
        ctx.save_for_backward(inv)
        return x[idx]

    @staticmethod
    def backward(ctx, g):
        inv, = ctx.saved_tensors
        return g[inv], None, None


class _GatherK(torch.autograd.Function):
    """Routed dispatch: y[p] = x[tok[p]].to(dt) where every token appears in exactly k pairs; backward
    dx[t] = sum_j g[inv[t*k + j]] in fp32 (deterministic gather-sum instead of an index_put scatter)."""

    @staticmethod
    def forward(ctx, x, tok, inv, k, dt):
        ctx.save_for_backward(inv)
        ctx.k, ctx.xdt = k, x.dtype
        return x[tok].to(dt)

    @staticmethod
    def backward(ctx, g):
        inv, = ctx.saved_tensors
        return g[inv].float().view(-1, ctx.k, g.shape[-1]).sum(1).to(ctx.xdt), None, None, None, None


def _triton_gmm(x):
    return x.is_cuda and torch.cuda.get_device_capability(x.device)[0] < 9


def _gmm(x, w, offs, tiles=None):
    """Grouped matmul for routed experts: torch._grouped_mm where the GPU has a native grouped kernel (>= sm_90);
    else the sync-free Triton kernel (torch's fallback loops over groups with a GPU->CPU sync per group), as a
    torch.library custom op (compiles without graph breaks). tiles: (tg, tj, ntiles) from streamstate::gmm_tiles."""
    if _triton_gmm(x):
        from streamstate import grouped_gemm as GG
        tg, tj, nt = tiles if tiles is not None else GG.gmm_tiles_op(offs, x.shape[0])
        return GG.gmm_op(x, w, offs, tg, tj, nt, False)
    return torch._grouped_mm(x, w, offs=offs)


class StandardMoE(nn.Module):
    def __init__(self, d, experts=8, hidden=512, top_k=2, shared_hidden=512, layers=1):
        super().__init__()
        self.d, self.E, self.h, self.k = d, experts, hidden, top_k
        std, out_std = 0.02, 0.02 / math.sqrt(2 * layers)
        self.W1 = nn.Parameter(torch.randn(experts, d, hidden) * std)       # gate
        self.W3 = nn.Parameter(torch.randn(experts, d, hidden) * std)       # up
        self.W2 = nn.Parameter(torch.randn(experts, hidden, d) * out_std)   # down
        self.router = nn.Linear(d, experts, bias=False)
        nn.init.normal_(self.router.weight, std=std)
        self.shared_gate_up = nn.Linear(d, 2 * shared_hidden, bias=False) if shared_hidden else None
        self.shared_down = nn.Linear(shared_hidden, d, bias=False) if shared_hidden else None
        if shared_hidden:
            nn.init.normal_(self.shared_gate_up.weight, std=std)
            nn.init.normal_(self.shared_down.weight, std=out_std)
        # what StreamMuon's expert path reads: tokens per expert and through the layer
        # since the optimizer last read them, top-k, active experts
        self.register_buffer("dispatch_acc", torch.zeros(experts), persistent=False)
        self.register_buffer("tokens_acc", torch.zeros(()), persistent=False)
        self.register_buffer("active", torch.ones(experts, dtype=torch.bool))
        self.register_buffer("load", torch.zeros(experts))
        for prm in (self.W1, self.W3, self.W2):
            prm._region = self

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, self.d)
        T, k = x.shape[0], self.k
        probs = self.router(x).float().softmax(-1)
        topv, topi = probs.topk(k, -1)
        gw = topv / topv.sum(-1, keepdim=True)
        flat = topi.flatten()
        cnt = torch.zeros(self.E, dtype=torch.long, device=x.device).index_add_(0, flat, torch.ones_like(flat))
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                self.dispatch_acc += cnt.float()
                self.tokens_acc += T
        order = flat.argsort(stable=True)
        inv = torch.empty_like(order).scatter_(0, order, torch.arange(order.numel(), device=x.device))
        tok = order // k
        offs = cnt.cumsum(0).to(torch.int32)
        dt = torch.bfloat16 if (x.is_cuda and torch.is_autocast_enabled()) else x.dtype
        W1, W3, W2 = self.W1.to(dt), self.W3.to(dt), self.W2.to(dt)
        tiles = None
        if _triton_gmm(x):
            from streamstate.grouped_gemm import gmm_tiles_op
            tiles = gmm_tiles_op(offs, flat.shape[0])
        xs = _GatherK.apply(x, tok, inv, k, dt)
        hcat = (F.silu(_gmm(xs, W1, offs, tiles)) * _gmm(xs, W3, offs, tiles)).to(dt).contiguous()
        out = _gmm(hcat, W2, offs, tiles)
        outp = _Permute.apply(out, inv, order).view(T, k, -1)
        y = (outp.float() * gw.unsqueeze(-1)).sum(1).to(x.dtype)
        frac = cnt.float() / T
        aux = self.E * (frac * probs.mean(0)).sum()
        with torch.no_grad():
            self.load.mul_(0.99).add_(0.01 * frac)
        if self.shared_gate_up is not None:
            g, u = self.shared_gate_up(x).chunk(2, -1)
            y = y + self.shared_down(F.silu(g) * u).to(y.dtype)
        return y.view(shape), aux
