"""StreamMuon tests (CUDA).

    pytest -q tests/

1. gbar: the streamed clip coefficient and update direction match an fp32 accumulate-then-update reference built from
   the same gradients (exact global norm, exact clip, Nesterov direction).
2. Routed experts: the streamed global norm equals the norm of the plainly accumulated gradient (gbar and snapshot);
   gbar and snapshot give the same expert updates; an expert that receives no tokens does not move.
3. Dense snapshot norm: exact under slowly varying gradient scale; under an abrupt 50x drop the int8 snapshot norm
   inflates (the paper's documented limitation) while gbar stays exact.
4. The standard MoE layer: the absorb inside the grouped weight-gradient kernel trains like the hook path.
5. The dense fast absorb path gives bit-identical momentum to the general path.
"""

import random

import pytest
import torch

from streamstate import StreamMuonOptimizer

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


# ---------------------------------------------------------------- 1. gbar vs an fp32 reference
def gbar_vs_reference(K=4, updates=6, d=256, h=768, clip=1.0, beta=0.95):
    torch.manual_seed(0)
    random.seed(0)
    mats = [torch.nn.Parameter((torch.randn(o, i, device="cuda") / i ** 0.5).bfloat16()) for o, i in [(h, d), (d, h)]]
    vec = torch.nn.Parameter(torch.ones(d, device="cuda"))
    acc = {}
    for p in mats:                                   # (registered before the optimizer's hook, which frees p.grad)
        p.register_post_accumulate_grad_hook(lambda p: acc.__setitem__(id(p), acc.get(id(p), 0) + p.grad.float() * K))
    opt = StreamMuonOptimizer(mats, [vec], lr=0.0, weight_decay=0.0, mode="gbar", accum_steps=K, clip=clip, beta1=beta)
    captured = {}
    orig = opt._gbar_dir

    def cap(p):
        out = orig(p)
        captured[id(p)] = out.float().clone()
        return out
    opt._gbar_dir = cap
    ref_m = {id(p): None for p in mats}
    errs, kerr = [], []
    gen = torch.Generator(device="cuda").manual_seed(7)
    for _ in range(updates):
        acc.clear()
        for k in range(K):
            x = torch.randn(64, d, device="cuda", generator=gen).bfloat16() * (1 + 3 * (k % 2))
            y = x * vec.bfloat16()
            y = y + torch.relu(y @ mats[0].t()) @ mats[1].t()
            (y.float().pow(2).mean() / K).backward()
        ref_norm = (vec.grad.float().pow(2).sum() + sum((acc[id(p)] / K).pow(2).sum() for p in mats)).sqrt()
        opt.stream_clip(mats + [vec], clip)
        kappa = float(opt._coef)
        kerr.append(abs(kappa - min(1.0, clip / (float(ref_norm) + 1e-6))) / kappa)
        opt.step()
        opt.zero_grad(set_to_none=True)
        for p in mats:
            G = acc[id(p)] / K
            M0 = ref_m[id(p)]
            if M0 is None:
                M1, D = kappa * G, kappa * G
            else:
                M1 = beta * M0 + (1 - beta) * kappa * G
                D = beta * M1 + (1 - beta) * kappa * G
            ref_m[id(p)] = M1
            errs.append(float((captured[id(p)].reshape(D.shape) - D).norm() / D.norm()))
    return max(kerr), max(errs)


@cuda
def test_gbar_matches_fp32_reference():
    kappa_err, dir_err = gbar_vs_reference()
    assert kappa_err < 0.01, kappa_err           # the clip coefficient: exact up to int8 rounding of Gbar
    assert dir_err < 0.05, dir_err               # the direction: int8 momentum + int8 Gbar vs fp32


# ---------------------------------------------------------------- 2. routed experts
class FakeRegion:
    """What the optimizer reads from a routed layer: cumulative tokens per expert and through the layer, top-k, active."""

    def __init__(self, S, k=2, device="cuda"):
        self.dispatch_acc = torch.zeros(S, device=device)
        self.tokens_acc = torch.zeros((), device=device)
        self.k = k
        self.active = torch.ones(S, dtype=torch.bool, device=device)


def run_routed(mode, K=4, S=8, updates=6, seed=0):
    """Routed weight with a FakeRegion, `updates` updates; -> (streamed norms, true norms, final weight delta)."""
    torch.manual_seed(seed)
    W0 = torch.randn(S, 128, 256, device="cuda") * 0.02
    W = torch.nn.Parameter(W0.clone())
    region = FakeRegion(S)
    W._region = region
    opt = StreamMuonOptimizer([W], lr=1e-3, weight_decay=0.0, mode=mode, accum_steps=K, clip=1e9, fused_dw=False)
    N = 1024                                                          # tokens through the layer per micro-batch
    g_rng = torch.Generator(device="cuda").manual_seed(seed + 7)      # same gradients for both modes
    got, want = [], []
    for u in range(updates):
        total = torch.zeros_like(W)
        for k in range(K):
            share = torch.rand(S, device="cuda", generator=g_rng) + 0.2
            share[7] = 0                                              # expert 7 never receives tokens
            if k % 2:
                share[3] = 0                                          # expert 3 idle in odd micro-batches
            n = torch.floor(share / share.sum() * 2 * N)
            common = torch.randn(S, 128, 256, device="cuda", generator=g_rng)
            noise = torch.randn(S, 128, 256, device="cuda", generator=g_rng)
            g = (common + 0.5 * noise) * (n > 0).float().view(S, 1, 1) * n.view(S, 1, 1) * (0.5 + u) / (K * N)
            region.dispatch_acc += n
            region.tokens_acc += N
            (W * g).sum().backward()
            total += g
        got.append(float(opt.stream_clip([])))
        want.append(float(total.norm()))
        opt.step()
        opt.zero_grad()
    return got, want, (W.detach() - W0)


@cuda
@pytest.mark.parametrize("mode,tol", [("gbar", 0.02), ("snapshot", 0.06)])
def test_expert_norm_is_exact(mode, tol):
    got, want, _ = run_routed(mode)
    rel = [abs(a - b) / b for a, b in zip(got, want)]
    assert max(rel) < tol, list(zip(got, want))


@cuda
def test_gbar_expert_update_matches_snapshot_and_idle_expert_stays():
    _, _, dg = run_routed("gbar")
    _, _, ds = run_routed("snapshot")
    assert float(dg[7].abs().max()) == 0.0 and float(ds[7].abs().max()) == 0.0
    rel = float((dg[:7] - ds[:7]).norm() / ds[:7].norm())
    assert rel < 0.12, rel      # (measured 0.095 with the paper's settings: same algebra, different int8 roundings)


# ---------------------------------------------------------------- 3. dense norm
def dense_norms(scales, mode, seed=1, K=4):
    torch.manual_seed(seed)
    W = torch.nn.Parameter(torch.randn(128, 256, device="cuda") * 0.02)
    opt = StreamMuonOptimizer([W], lr=1e-2, weight_decay=0.0, mode=mode, accum_steps=K, clip=1e9)
    ratios = []
    for s in scales:
        common = torch.randn(128, 256, device="cuda")
        gs = [(common + 0.3 * torch.randn(128, 256, device="cuda")) * s / 100 for _ in range(K)]
        for g in gs:
            (W * g / K).sum().backward()
        ratios.append(float(opt.stream_clip([])) / float(sum(gs).div(K).norm()))
        opt.step()
        opt.zero_grad()
    return ratios


@cuda
def test_dense_snapshot_norm_slowly_varying_scale():
    r = dense_norms([1.0, 0.95, 1.05, 0.9, 1.0, 0.92, 1.08, 1.0], "snapshot")
    assert max(abs(x - 1) for x in r) < 0.08, r


@cuda
def test_snapshot_norm_inflates_on_abrupt_drop_gbar_does_not():
    snap = dense_norms([50, 20, 5, 1, 3], "snapshot")
    gbar = dense_norms([50, 20, 5, 1, 3], "gbar")
    assert max(snap) > 2.0, snap
    assert max(abs(x - 1) for x in gbar) < 0.02, gbar


# ---------------------------------------------------------------- 4. MoE layer: grouped-kernel absorb vs hook path
@cuda
@pytest.mark.parametrize("mode", ["gbar", "snapshot"])
def test_moe_layer_fused_dw_matches_hook_path(mode):
    from streamstate.moe import StandardMoE

    def train(fused, sr_seed=0):
        random.seed(sr_seed)
        torch.manual_seed(0)
        layer = StandardMoE(128, experts=4, hidden=128, top_k=2, shared_hidden=0).cuda()
        for prm in (layer.W1, layer.W3, layer.W2):
            prm.data = prm.data.bfloat16()
        w0 = [p.detach().float().clone() for p in (layer.W1, layer.W3, layer.W2)]
        opt = StreamMuonOptimizer([layer.W1, layer.W3, layer.W2], [layer.router.weight], lr=1e-2, weight_decay=0.0,
                                  mode=mode, accum_steps=2, clip=1.0, fused_dw=fused)
        g = torch.Generator(device="cuda").manual_seed(1)
        for _ in range(4):
            for _ in range(2):
                x = torch.randn(256, 128, device="cuda", generator=g)
                with torch.autocast("cuda", torch.bfloat16):
                    y, aux = layer(x)
                ((y.float() ** 2).mean() / 2).backward()
            opt.stream_clip(list(layer.parameters()))
            opt.step()
            opt.zero_grad()
        return [p.detach().float() - w for p, w in zip((layer.W1, layer.W3, layer.W2), w0)]

    # stochastic rounding draws differ between the two paths and routing feeds back into the next forward, so the
    # yardstick is the spread between two hook-path runs that differ only in their rounding draws
    fused, hook, hook2 = train(True), train(False), train(False, sr_seed=1)
    for da, db, db2 in zip(fused, hook, hook2):
        assert float(da.norm()) > 0
        noise = float((db2 - db).norm() / db.norm())
        diff = float((da - db).norm() / db.norm())
        assert diff < 2 * noise + 0.02, (diff, noise)


# ---------------------------------------------------------------- 5. dense fast absorb vs general path
def fast_vs_general(mode, fast, K=4, updates=5, d=256, h=768):
    torch.manual_seed(0)
    random.seed(0)
    mats = [torch.nn.Parameter((torch.randn(o, i, device="cuda") / i ** 0.5).bfloat16())
            for o, i in [(h, d), (d, h), (h, d), (d, h)]]
    vecs = [torch.nn.Parameter(torch.ones(d, device="cuda"))]
    opt = StreamMuonOptimizer(mats, vecs, lr=2e-3, weight_decay=0.0, mode=mode, accum_steps=K, fast_absorb=fast)
    gen = torch.Generator(device="cuda").manual_seed(123)
    for _ in range(updates):
        for _ in range(K):
            y = torch.randn(64, d, device="cuda", generator=gen).bfloat16() * vecs[0].bfloat16()
            for i in range(0, 4, 2):
                y = y + torch.relu(y @ mats[i].t()) @ mats[i + 1].t()
            (y.float().pow(2).mean() / K).backward()
        opt.stream_clip(mats + vecs, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return [m.detach().float().clone() for m in mats], [opt.state[m]["m"].clone() for m in mats]


@cuda
@pytest.mark.parametrize("mode", ["snapshot", "lite"])
def test_fast_absorb_matches_general_path(mode):
    """Widths % 128 == 0: both paths use the same Triton arithmetic and dither, so the momentum is bit-identical;
    weights agree up to the clip-norm summation order (lite)."""
    w0, m0 = fast_vs_general(mode, fast=False)
    w1, m1 = fast_vs_general(mode, fast=True)
    assert all(torch.equal(a, b) for a, b in zip(m0, m1))
    assert max(float((a - b).norm() / a.norm()) for a, b in zip(w0, w1)) < 1e-3
