"""CIFAR-10 benchmark of the paper's rule-independence test (appendix, "A non-Muon update rule"): a ResNet-style CNN
(9.2M parameters at --width 128) whose convolution matrices are trained with SGD + Nesterov momentum 0.9 and decoupled
weight decay, in two forms:
  sgdw       the reference: PyTorch SGD, fp32 weights, fp32 momentum, gradient buffer
  streamsgd  StreamMuonOptimizer(mode="gbar", rule="sgd"): bf16 weights (stochastic rounding), int8 momentum, the
             2 micro-batches streamed into the one-byte gradient mean; lr and decay rescaled for the EMA form of
             momentum
Everything else (stem conv, head, norms, biases) uses PyTorch SGD-Nesterov with L2 5e-4 in both arms. Baselines on the
same model: sgd, adamw, muon. Conv weights are stored as 2-D parameters (out, in*k*k).
Data: the CIFAR-10 training set (torchvision), put in the order of the copy the paper used, and the paper's fixed
45,000 / 5,000 train / held-out split. The paper reports final held-out accuracy (val_acc). The sgdw arm is
deterministic for a seed; the streamed arm is not (its stochastic-rounding draws depend on kernel compilation), so
repeated runs of one seed differ by about a point, as the seeds do.
    python benchmarks/cifar_bench.py --opt streamsgd --lr 0.1 --seed 0     (the paper: seeds 0-2, lr 0.1, 10 epochs)
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--opt", choices=["sgd", "adamw", "muon", "sgdw", "streamsgd"],
                required=True)
ap.add_argument("--lr", type=float, required=True)
ap.add_argument("--epochs", type=int, default=10)
ap.add_argument("--width", type=int, default=128)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--wd", type=float, default=None)
ap.add_argument("--data", default="data", help="torchvision CIFAR-10 root (downloaded if missing)")
a = ap.parse_args()
dev = "cuda"
torch.manual_seed(a.seed)
np.random.seed(a.seed)

# ------------------------------------------------------------------ data
import torchvision                                     # noqa: E402

_ds = torchvision.datasets.CIFAR10(a.data, train=True, download=True)
# the paper read the training set from the Hugging Face copy (uoft-cs/cifar10), which holds the same 50,000 images in
# the order of torchvision's batches 4, 3, 2, 5, 1; reorder so the fixed split below is exactly the paper's
_order = np.concatenate([np.arange(b * 10000, (b + 1) * 10000) for b in (3, 2, 1, 4, 0)])
d = {"x": _ds.data[_order], "y": np.asarray(_ds.targets, dtype=np.int64)[_order]}
X, Y = torch.tensor(d["x"]).permute(0, 3, 1, 2).contiguous(), torch.tensor(d["y"])
perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(12345))      # (fixed split for every arm)
tr, va = perm[:45000], perm[45000:]
mean = X[tr].float().mean((0, 2, 3), keepdim=True) / 255
std = X[tr].float().std((0, 2, 3), keepdim=True) / 255
Xtr, Ytr = ((X[tr].float() / 255 - mean) / std).to(dev), Y[tr].to(dev)
Xva, Yva = ((X[va].float() / 255 - mean) / std).to(dev), Y[va].to(dev)


def augment(x):
    flip = torch.rand(x.shape[0], device=dev) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    p = F.pad(x, (4, 4, 4, 4), mode="reflect")
    i, j = np.random.randint(0, 9, 2)
    return p[:, :, i:i + 32, j:j + 32]


# ------------------------------------------------------------------ model
class FlatConv(nn.Module):
    """3x3 (or kxk) convolution whose weight is a 2-D parameter (out, in*k*k): matrix optimizers see a matrix."""

    def __init__(self, cin, cout, k=3):
        super().__init__()
        self.cin, self.cout, self.k = cin, cout, k
        self.weight = nn.Parameter(torch.randn(cout, cin * k * k) * (2.0 / (cin * k * k)) ** 0.5)

    def forward(self, x):
        return F.conv2d(x, self.weight.view(self.cout, self.cin, self.k, self.k).to(x.dtype), padding=self.k // 2)


def block(cin, cout):
    return nn.Sequential(FlatConv(cin, cout), nn.MaxPool2d(2), nn.BatchNorm2d(cout), nn.GELU(),
                         FlatConv(cout, cout), nn.BatchNorm2d(cout), nn.GELU())


w = a.width
model = nn.Sequential(FlatConv(3, w), nn.BatchNorm2d(w), nn.GELU(),
                      block(w, 2 * w), block(2 * w, 4 * w), block(4 * w, 4 * w),
                      nn.AdaptiveMaxPool2d(1), nn.Flatten(), nn.Linear(4 * w, 10)).to(dev).to(memory_format=torch.channels_last)
mats = [m.weight for m in model.modules() if isinstance(m, FlatConv) and m.cin > 3]      # (stem conv + head: rest)
mat_ids = {id(p) for p in mats}
rest = [p for p in model.parameters() if id(p) not in mat_ids]
n_params = sum(p.numel() for p in model.parameters())

# ------------------------------------------------------------------ optimizer
from streamstate import Muon, StreamMuonOptimizer, optimizer_state_bytes   # noqa: E402

wd = a.wd if a.wd is not None else {"sgd": 5e-4, "adamw": 0.05, "muon": 0.05, "sgdw": 5e-3, "streamsgd": 5e-3}[a.opt]
opts = []
MICRO = 2 if a.opt in ("sgdw", "streamsgd") else 1
decoupled = []
if a.opt in ("sgdw", "streamsgd"):
    rest_opt = torch.optim.SGD(rest, lr=a.lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
    if a.opt == "sgdw":
        opts = [torch.optim.SGD(mats, lr=a.lr, momentum=0.9, nesterov=True, weight_decay=0.0), rest_opt]
        decoupled = mats                      # W <- W (1 - lr wd) after each step
    else:
        for p in mats:
            p.data = p.data.bfloat16()
        # torch SGD-Nesterov step = lr_t (g + mu buf); ours = lr (beta M + (1 - beta) G) with M = (1 - beta) buf,
        # so lr = lr_t / (1 - beta); decoupled decay lr * wd_ours = lr_t * wd  ->  wd_ours = wd (1 - beta)
        o = StreamMuonOptimizer(mats, lr=a.lr / (1 - 0.9), weight_decay=wd * (1 - 0.9), beta1=0.9, mode="gbar",
                                rule="sgd", accum_steps=MICRO, clip=None)
        opts = [o, rest_opt]
elif a.opt == "sgd":
    opts = [torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9, nesterov=True, weight_decay=wd)]
elif a.opt == "adamw":
    opts = [torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=wd, fused=True)]
elif a.opt == "muon":
    opts = [Muon(mats, lr=a.lr, weight_decay=wd), torch.optim.AdamW(rest, lr=a.lr, weight_decay=0.0, fused=True)]
base_lrs = [[g["lr"] for g in o.param_groups] for o in opts]

steps_per_epoch = len(Xtr) // a.batch
total = steps_per_epoch * a.epochs
warm = max(1, total // 20)


def set_lr(s):
    f = (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1)))
    for o, bl in zip(opts, base_lrs):
        for g, l in zip(o.param_groups, bl):
            g["lr"] = l * f


@torch.no_grad()
def evaluate():
    model.eval()
    correct = 0
    for i in range(0, len(Xva), 1000):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            correct += int((model(Xva[i:i + 1000].contiguous(memory_format=torch.channels_last)).argmax(1)
                            == Yva[i:i + 1000]).sum())
    model.train()
    return correct / len(Xva)


torch.cuda.reset_peak_memory_stats()
curve, step, opt_ms = [], 0, []
t0 = time.time()
for ep in range(a.epochs):
    order = torch.randperm(len(Xtr), device=dev)
    for b in range(steps_per_epoch):
        idx = order[b * a.batch:(b + 1) * a.batch]
        x = augment(Xtr[idx]).contiguous(memory_format=torch.channels_last)
        set_lr(step)
        for xm, ym in zip(x.chunk(MICRO), Ytr[idx].chunk(MICRO)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(xm), ym, label_smoothing=0.2) / MICRO
            loss.backward()
        timed = step % 25 == 10
        if timed:
            torch.cuda.synchronize()
            ts = time.perf_counter()
        for o in opts:
            o.step()
        if decoupled:
            with torch.no_grad():
                f = opts[0].param_groups[0]["lr"] * wd
                for p in decoupled:
                    p.mul_(1 - f)
        if timed:
            torch.cuda.synchronize()
            opt_ms.append((time.perf_counter() - ts) * 1e3)
        for o in opts:
            o.zero_grad(set_to_none=True)
        step += 1
    curve.append(round(evaluate(), 4))
wall = time.time() - t0
state = {"state_bytes": 0, "weight_bytes": sum(p.numel() * p.element_size() for p in model.parameters())}
for o in opts:
    state["state_bytes"] += optimizer_state_bytes(o, model)["state_bytes"]
print("RESULT " + json.dumps({
    "opt": a.opt, "lr": a.lr, "wd": wd, "width": w, "epochs": a.epochs, "seed": a.seed,
    "params_M": round(n_params / 1e6, 2), "val_acc": curve[-1], "best_val_acc": max(curve), "curve": curve,
    "wall_s": round(wall, 1), "ms_per_step": round(wall / total * 1e3, 1),
    "opt_step_ms": round(float(np.median(opt_ms)), 2) if opt_ms else None,
    "peak_gb": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
    "opt_state_MB": round(state["state_bytes"] / 2 ** 20, 1), "weights_MB": round(state["weight_bytes"] / 2 ** 20, 1),
    "state_B_per_param": round(state["state_bytes"] / n_params, 2)}), flush=True)
