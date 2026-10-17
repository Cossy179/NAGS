#!/usr/bin/env python3
"""Train an experimental version 4 NNUE (a hidden stack after the accumulators).

Version 4 keeps everything a version 3 network has up to the accumulators:
the same 768 features per perspective, king buckets with left-right
mirroring, the feature transformer and its quantization (QA = 255), so the
engine's incremental accumulator updates are the same code. Instead of one
linear output per bucket it has, per output bucket, two small hidden layers:

    inputs (2H: clipped accumulators, side to move first; or their squares
            with --activation screlu; or H with --activation pairwise: the
            product of the two halves of each clipped accumulator)
    -> L1 (--l1, default 16) -> clipped ReLU
    -> L2 (--l2, default 32) -> clipped ReLU
    -> 1, x SCALE = centipawns for the side to move.

Training data, target, loss, optimizer and schedule are those of train.py
(this file imports its data code, which it does not change); train.py and
the version 1-3 formats stay as they are, and nothing uses a version 4
network unless it is loaded with EvalFile.

Quantization (exact integer arithmetic, mirrored by quantized_eval below):
inputs x in [0, QA] (clamp(a), clamp(a)^2 / QA, or clamp(a) * clamp(b) / QA,
truncated); h1 = clamp((sum(x * w1) + b1) / QH, 0, QA) with w1 x QH and b1 x
QA x QH; h2 likewise from h1; output = (sum(h2 * w3) + b3) * SCALE / (QA x
QH), truncated towards zero. QH = 64; weights are clipped to +-1.98 so that
they fit in int16 products.

File (little endian):

    char[8] "NAGSNNUE"  uint32 4  uint32 H  uint32 B  uint32 K
    uint32 flags (bit 0 SCReLU, bit 1 mirroring, bit 2 pairwise)
    uint8 layout[64]  uint32 L1  uint32 L2
    int16 ft_weights[K][768][H]  int16 ft_bias[H]           (as version 3)
    per output bucket: int16 w1[L1][I]  int32 b1[L1]  int16 w2[L2][L1]
                       int32 b2[L2]  int16 w3[L2]  int32 b3

Usage (as train.py, plus --l1/--l2; compare with a version 3 network trained
on the same data with the same --epochs before using it, see docs/NNUE.md):

    python tools/nnue/train_v4.py --data "data/*.bin" --epochs 30 --buckets 8 \\
        --king-buckets 4 --activation pairwise --out nets/v4.nnue
"""

import argparse
import os
import struct
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402  (data, features and king buckets; not modified)

VERSION = 4
QA, SCALE = train.QA, train.SCALE
QH = 64
MAX_LAYER = 32
HEADER = 100


class NnueV4(train.Nnue):
    """train.Nnue's feature transformer followed by a bucketed hidden stack."""

    def __init__(self, hidden=256, buckets=1, king_buckets="none", activation="crelu", l1=16, l2=32):
        super().__init__(hidden, buckets, king_buckets, activation == "screlu")
        del self.out
        self.activation = activation
        self.l1_size, self.l2_size = l1, l2
        inputs = hidden if activation == "pairwise" else 2 * hidden
        self.l1 = torch.nn.Linear(inputs, buckets * l1)
        self.l2 = torch.nn.Linear(l1, buckets * l2)
        self.l3 = torch.nn.Linear(l2, buckets)

    def _inputs(self, a):
        a = torch.clamp(a, 0.0, 1.0)
        if self.activation == "screlu":
            return a * a
        if self.activation == "pairwise":
            h = self.hidden // 2
            return a[:, :h] * a[:, h:]
        return a

    def forward(self, white, black, stm, bucket):
        aw = self._inputs(self.ft(self.king(white)) + self.ft_bias)
        ab = self._inputs(self.ft(self.king(black)) + self.ft_bias)
        s = stm.unsqueeze(1).bool()
        x = torch.cat([torch.where(s, ab, aw), torch.where(s, aw, ab)], dim=1)
        n = len(bucket)
        pick = lambda y, size: y.view(n, self.buckets, size)[torch.arange(n, device=y.device), bucket]
        h1 = torch.clamp(pick(self.l1(x), self.l1_size), 0.0, 1.0)
        # Layers 2 and 3 of every bucket are applied to h1 (small), then the bucket's are kept.
        h2 = torch.clamp(pick(self.l2(h1), self.l2_size), 0.0, 1.0)
        out = (self.l3.weight.view(self.buckets, self.l2_size)[bucket] * h2).sum(dim=1) + self.l3.bias[bucket]
        return out

    def clip(self):
        with torch.no_grad():
            self.ft.weight.clamp_(-train.WEIGHT_CLIP, train.WEIGHT_CLIP)
            self.ft.weight[self.pad].zero_()
            for layer in (self.l1, self.l2, self.l3):
                layer.weight.clamp_(-train.WEIGHT_CLIP, train.WEIGHT_CLIP)


def export(model, path):
    k = model.king
    flags = (1 if model.activation == "screlu" else 0) | (2 if k.mirror else 0) | (
        4 if model.activation == "pairwise" else 0)
    q16 = lambda a, s: np.clip(np.round(a * s), -32767, 32767).astype("<i2")
    q32 = lambda a, s: np.round(a * s).astype("<i4")
    b, l1, l2 = model.buckets, model.l1_size, model.l2_size
    w1 = model.l1.weight.detach().cpu().numpy().reshape(b, l1, -1)
    b1 = model.l1.bias.detach().cpu().numpy().reshape(b, l1)
    w2 = model.l2.weight.detach().cpu().numpy().reshape(b, l2, l1)
    b2 = model.l2.bias.detach().cpu().numpy().reshape(b, l2)
    w3 = model.l3.weight.detach().cpu().numpy().reshape(b, l2)
    b3 = model.l3.bias.detach().cpu().numpy().reshape(b)
    with open(path + ".tmp", "wb") as fh:
        fh.write(train.MAGIC)
        fh.write(struct.pack("<IIIII", VERSION, model.hidden, b, k.count, flags))
        fh.write(k.layout.numpy().astype(np.uint8).tobytes())
        fh.write(struct.pack("<II", l1, l2))
        fh.write(q16(model.feature_weights().cpu().numpy(), QA).tobytes())
        fh.write(q16(model.ft_bias.detach().cpu().numpy(), QA).tobytes())
        for i in range(b):
            fh.write(q16(w1[i], QH).tobytes())
            fh.write(q32(b1[i], QA * QH).tobytes())
            fh.write(q16(w2[i], QH).tobytes())
            fh.write(q32(b2[i], QA * QH).tobytes())
            fh.write(q16(w3[i], QH).tobytes())
            fh.write(q32(b3[i : i + 1], QA * QH).tobytes())
    os.replace(path + ".tmp", path)


def read_net(path):
    """A version 4 file as integer numpy arrays: dict with ft_w [K * 768, H],
    ft_b, w1 [B, L1, I], b1 [B, L1], w2, b2, w3 [B, L2], b3 [B], and info."""
    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:8] == train.MAGIC, "not a NAGS NNUE file"
    version, hidden, buckets, kb, flags = struct.unpack_from("<IIIII", data, 8)
    assert version == VERSION, f"version {version}, not {VERSION}"
    layout = np.frombuffer(data, np.uint8, 64, 28).astype(np.int64)
    l1, l2 = struct.unpack_from("<II", data, 92)
    inputs = hidden if flags & 4 else 2 * hidden
    off = HEADER

    def take(dtype, count, shape=None):
        nonlocal off
        a = np.frombuffer(data, dtype, count, off).astype(np.int64)
        off += np.dtype(dtype).itemsize * count
        return a.reshape(shape) if shape else a

    net = {"ft_w": take("<i2", kb * train.FEATURES * hidden, (kb * train.FEATURES, hidden)),
           "ft_b": take("<i2", hidden)}
    parts = {key: [] for key in ("w1", "b1", "w2", "b2", "w3", "b3")}
    for _ in range(buckets):
        parts["w1"].append(take("<i2", l1 * inputs, (l1, inputs)))
        parts["b1"].append(take("<i4", l1))
        parts["w2"].append(take("<i2", l2 * l1, (l2, l1)))
        parts["b2"].append(take("<i4", l2))
        parts["w3"].append(take("<i2", l2))
        parts["b3"].append(take("<i4", 1)[0])
    assert off == len(data), "network file has the wrong size"
    net.update({key: np.array(v) for key, v in parts.items()})
    net["info"] = {"hidden": hidden, "buckets": buckets, "king_buckets": kb, "layout": layout,
                   "screlu": bool(flags & 1), "mirror": bool(flags & 2), "pairwise": bool(flags & 4),
                   "l1": l1, "l2": l2}
    return net


def _div(a, b):
    """Integer division truncating towards zero (C++), elementwise."""
    return np.sign(a) * (np.abs(a) // b)


def quantized_eval(net, fen):
    """Integer evaluation of `fen` (centipawns, side to move), as the engine computes it."""
    info = net["info"]
    fields = fen.split()
    wf = np.array(train.white_features(fields[0]), dtype=np.int64)
    k = train.bucket_of(len(wf), info["buckets"])

    def rows(f):
        ksq = int(f[(f >= 320) & (f < 384)][0]) - 320
        x = f ^ 7 if info["mirror"] and (ksq & 7) >= 4 else f
        return int(info["layout"][ksq]) * train.FEATURES + x

    acc_w = net["ft_b"] + net["ft_w"][rows(wf)].sum(axis=0)
    acc_b = net["ft_b"] + net["ft_w"][rows(train.mirror(wf))].sum(axis=0)
    # The accumulators are int16 in the engine; real networks stay far inside that range.
    assert np.abs(np.concatenate([acc_w, acc_b])).max() < 32768
    us, them = (acc_w, acc_b) if fields[1] == "w" else (acc_b, acc_w)

    def inputs(a):
        c = np.clip(a, 0, QA)
        if info["pairwise"]:
            h = len(c) // 2
            return c[:h] * c[h:] // QA
        return c * c // QA if info["screlu"] else c

    x = np.concatenate([inputs(us), inputs(them)])
    h1 = np.clip(_div(net["w1"][k] @ x + net["b1"][k], QH), 0, QA)
    h2 = np.clip(_div(net["w2"][k] @ h1 + net["b2"][k], QH), 0, QA)
    out = int(net["w3"][k] @ h2 + net["b3"][k])
    return int(_div(out * SCALE, QA * QH))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="as train.py: text, .npz or .bin files")
    ap.add_argument("--cache", help="parsed-data cache for text input (.npz), created if missing")
    ap.add_argument("--out", required=True, help="exported network (version 4)")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--buckets", type=int, default=1, help=f"output buckets by piece count (1-{train.MAX_BUCKETS})")
    ap.add_argument("--king-buckets", default="none", help="as train.py")
    ap.add_argument("--activation", choices=["crelu", "screlu", "pairwise"], default="crelu")
    ap.add_argument("--l1", type=int, default=16, help=f"first hidden layer (1-{MAX_LAYER})")
    ap.add_argument("--l2", type=int, default=32, help=f"second hidden layer (1-{MAX_LAYER})")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.75, help="weight of the search score in the target")
    ap.add_argument("--val", type=float, default=0.01, help="fraction held out for validation")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true", help="continue from <out>.ckpt with the same data and settings")
    ap.add_argument("--stop-after", type=int, default=0)
    args = ap.parse_args(argv)

    if not 1 <= args.l1 <= MAX_LAYER or not 1 <= args.l2 <= MAX_LAYER:
        sys.exit(f"--l1 and --l2 must be 1-{MAX_LAYER}")
    if args.activation == "pairwise" and args.hidden % 2:
        sys.exit("--activation pairwise needs an even --hidden")
    if not 1 <= args.buckets <= train.MAX_BUCKETS:
        sys.exit(f"--buckets must be 1-{train.MAX_BUCKETS}")
    try:
        train.king_config(args.king_buckets)
    except ValueError as e:
        sys.exit(f"--king-buckets: {e}")
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    paths = train.expand_paths(args.data)
    binary = [p.endswith(".bin") for p in paths]
    if any(binary) and not all(binary):
        sys.exit("--data: .bin files cannot be mixed with other inputs")
    t0 = time.time()
    source = (train.BinData(paths, args.val, device) if binary and all(binary)
              else train.ArrayData(train.load(paths, args.cache), args.val, rng, device))
    n = source.n
    print(f"{n} positions loaded in {time.time() - t0:.0f}s", flush=True)
    if n == 0:
        sys.exit("no positions")

    model = NnueV4(args.hidden, args.buckets, args.king_buckets, args.activation, args.l1, args.l2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = args.epochs * source.steps(args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=args.lr * 0.01)
    ckpt_path = args.out + ".ckpt"
    design = {"format": VERSION, "positions": n, "hidden": args.hidden, "epochs": args.epochs, "buckets": args.buckets,
              "king_buckets": args.king_buckets, "activation": args.activation, "l1": args.l1, "l2": args.l2,
              "batch": args.batch, "lr": args.lr, "lam": args.lam, "val": args.val, "seed": args.seed}
    first_epoch = 1
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        changed = [key for key, v in design.items() if ckpt.get(key) != v]
        if changed:
            sys.exit(f"{ckpt_path} was made with different data or settings: {', '.join(changed)}")
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        rng.bit_generator.state = ckpt["rng"]
        first_epoch = ckpt["epoch"] + 1
        print(f"resuming after epoch {ckpt['epoch']}", flush=True)
    last_epoch = args.epochs if not args.stop_after else min(args.epochs, first_epoch + args.stop_after - 1)
    for epoch in range(first_epoch, last_epoch + 1):
        t = time.time()
        total, count = 0.0, 0
        for w, b, s, k, target in source.train_batches(rng, args.batch, args.lam, args.buckets):
            loss = ((torch.sigmoid(model(w, b, s, k)) - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            model.clip()
            total += loss.item() * len(target)
            count += len(target)
        msg = f"epoch {epoch:3d}  train {total / count:.6f}"
        if source.n_val:
            msg += f"  val {source.val_loss(model, args.lam):.6f}"
        print(f"{msg}  ({time.time() - t:.0f}s)", flush=True)
        export(model, args.out)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.bit_generator.state, "epoch": epoch, **design}, ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
    print(f"wrote {args.out}" + (f" (stopped after epoch {last_epoch} of {args.epochs})" if last_epoch < args.epochs else ""))


if __name__ == "__main__":
    main()
