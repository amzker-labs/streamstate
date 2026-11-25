"""StreamState: optimizer state streamed through the backward pass, with no gradient buffer.

Flagship: StreamMuon, Muon-style training at 4 bytes per parameter of training state (StreamMuonLite: 3).

    from streamstate import StreamMuon

    opt = StreamMuon(model, lr=2e-3, accum_steps=4)             # StreamMuonLite(model, ...) for 3 bytes / parameter
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = schedule(step)                            # any schedule; set it before the micro-batches
        for micro in range(4):
            with torch.autocast("cuda", torch.bfloat16):
                loss = model(batch()) / 4                       # each backward uses loss / accum_steps
            loss.backward()                                     # matrix gradients are absorbed and freed here
        opt.stream_clip(model.parameters(), 1.0)                # global-norm clip (replaces clip_grad_norm_)
        opt.step()
        opt.zero_grad(set_to_none=True)
"""

from __future__ import annotations

import warnings

from streamstate.optim import Muon, StreamMuonOptimizer, muon_split, optimizer_state_bytes

__all__ = ["StreamMuon", "StreamMuonLite", "StreamMuonOptimizer", "Muon", "muon_split", "optimizer_state_bytes"]
__version__ = "0.1.0"


def _build(model, mode, lr, weight_decay, accum_steps, clip, bf16_weights, matrix_names, kw):
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    mats = muon_split(named) if matrix_names is None else list(matrix_names)
    if mode == "gbar":                    # the one-byte gradient mean needs whole int8 blocks of 128 values
        odd = [n for n in mats if named[n].numel() % 128]
        if odd:
            warnings.warn(f"StreamMuon(gbar=True): {len(odd)} matrices whose size is not a multiple of 128 are trained "
                          f"by AdamW instead: {odd[:5]}{' ...' if len(odd) > 5 else ''} (gbar=False streams them)")
            mats = [n for n in mats if n not in odd]
    vecs = [n for n in named if n not in mats]
    if bf16_weights:
        for n in mats:
            named[n].data = named[n].data.bfloat16()
    return StreamMuonOptimizer([named[n] for n in mats], [named[n] for n in vecs], lr=lr, weight_decay=weight_decay,
                               mode=mode, accum_steps=accum_steps, clip=clip, **kw)


def StreamMuon(model, lr, weight_decay=0.05, accum_steps=1, gbar=True, clip=1.0, bf16_weights=True,
               matrix_names=None, **kw):
    """StreamMuon for `model`: exact global norm, clip and Nesterov momentum, 4 bytes per parameter.

    gbar=True (recommended): a one-byte running mean of the update's gradient (the norm stays exact even when the
    gradient scale changes abruptly). gbar=False: a one-byte snapshot of the previous momentum instead (no
    gradient-shaped state at all; the int8 snapshot overestimates the norm after abrupt gradient-scale drops).

    Weight matrices (`muon_split` by default, or the parameter names in `matrix_names`) are streamed; every other
    parameter is trained by AdamW. bf16_weights: the streamed matrices are cast to bf16 in place (updates are
    stochastic-rounded); False keeps their dtype. Expert stacks of a routed layer (3-D tensors tagged `_region`, as in
    streamstate.moe.StandardMoE) take the token-clock expert path. kw: StreamMuonOptimizer options.
    """
    return _build(model, "gbar" if gbar else "snapshot", lr, weight_decay, accum_steps, clip, bf16_weights,
                  matrix_names, kw)


def StreamMuonLite(model, lr, weight_decay=0.05, accum_steps=1, clip=1.0, bf16_weights=True, matrix_names=None, **kw):
    """StreamMuon-Lite: 3 bytes per parameter (no extra byte), so no Nesterov momentum and a clip from the summed
    micro-batch norms applied one update late. Trails StreamMuon by ~0.03 bits per byte at 113M (paper, Section 5)."""
    return _build(model, "lite", lr, weight_decay, accum_steps, clip, bf16_weights, matrix_names, kw)
