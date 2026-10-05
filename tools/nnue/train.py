#!/usr/bin/env python3
"""Train the NNUE evaluation from nags_datagen output and export it.

Network ("768 -> 2x256 -> 1", perspective):

* Input: 768 binary features per perspective (colour relative to the
  perspective x 6 piece types x 64 squares, squares mirrored vertically for
  Black's perspective).
* Feature transformer: one 768 x 256 layer shared by both perspectives,
  giving two accumulators (side to move, other side). The C++ engine keeps
  them up to date incrementally as pieces move.
* Clipped ReLU on each accumulator, concatenated side to move first, and a
  linear output. With output buckets (--buckets N) there are N output layers
  and the one used depends on the number of pieces on the board:
  bucket = min(N - 1, (pieces - 1) * N // 32), so the opening, middlegame and
  endgame get their own final weights at no extra cost. Output x SCALE =
  centipawns for the side to move.

Training target: lambda * sigmoid(score / SCALE) + (1 - lambda) * result,
both for the side to move, with a squared error on sigmoid(output).

Export (little endian), read by src/Nnue.cpp:

    char[8]  "NAGSNNUE"     uint32 version     uint32 hidden size H
    (version 2 only:)       uint32 buckets B (1..8; version 1 means B = 1)
    int16    ft_weights[768][H]  (x QA)
    int16    ft_bias[H]          (x QA)
    int16    out_weights[B][2H]  (x QB)
    int32    out_bias[B]         (x QA x QB)

Networks with one bucket are written as version 1.

Usage:

    python tools/nnue/train.py --data data/selfplay.txt --epochs 30 --out nets/nags.nnue
"""

import argparse
import math
import multiprocessing as mp
import os
import struct
import sys
import time

import numpy as np
import torch

FEATURES = 768
PAD = FEATURES  # index of the all-zero padding row
MAX_PIECES = 32
QA, QB, SCALE = 255, 64, 400
WEIGHT_CLIP = 1.98  # keeps the quantized sums inside int16 / int32
MAGIC = b"NAGSNNUE"
MAX_BUCKETS = 8


def bucket_of(pieces, buckets):
    """Output bucket for a position with `pieces` pieces (numpy or torch)."""
    b = (pieces - 1) * buckets // 32
    return b.clip(0, buckets - 1) if hasattr(b, "clip") else min(max(b, 0), buckets - 1)

PIECE_INDEX = {c: i for i, c in enumerate("PNBRQKpnbrqk")}


def white_features(placement):
    """Feature indices from White's perspective for a FEN placement field."""
    out = []
    rank, file = 7, 0
    for ch in placement:
        if ch == "/":
            rank -= 1
            file = 0
        elif ch.isdigit():
            file += int(ch)
        else:
            p = PIECE_INDEX[ch]
            colour, ptype = (0, p) if p < 6 else (1, p - 6)
            out.append(colour * 384 + ptype * 64 + rank * 8 + file)
            file += 1
    return out


def mirror(features):
    """White-perspective indices -> Black-perspective indices (numpy or torch)."""
    colour = features // 384
    rest = features % 384
    ptype, square = rest // 64, rest % 64
    flipped = (1 - colour) * 384 + ptype * 64 + (square ^ 56)
    return flipped


def parse_chunk(lines):
    n = len(lines)
    feats = np.full((n, MAX_PIECES), PAD, dtype=np.int16)
    stm = np.zeros(n, dtype=np.int8)
    score = np.zeros(n, dtype=np.int16)
    result = np.zeros(n, dtype=np.float32)
    kept = 0
    for line in lines:
        parts = line.split("|")
        if len(parts) != 3:
            continue
        fen = parts[0].split()
        f = white_features(fen[0])
        if len(f) > MAX_PIECES:
            continue
        feats[kept, : len(f)] = f
        stm[kept] = 0 if fen[1] == "w" else 1
        score[kept] = max(-32000, min(32000, int(parts[1])))
        result[kept] = float(parts[2])
        kept += 1
    return feats[:kept], stm[:kept], score[:kept], result[:kept]


def parse_files(paths, workers=None):
    """Parses nags_datagen text files."""
    lines = []
    for p in paths:
        with open(p) as fh:
            lines.extend(fh.readlines())
    chunk = 200000
    chunks = [lines[i : i + chunk] for i in range(0, len(lines), chunk)]
    del lines
    if len(chunks) > 1:
        with mp.Pool(workers or os.cpu_count()) as pool:
            return pool.map(parse_chunk, chunks)
    return [parse_chunk(c) for c in chunks]


def load(paths, cache=None, workers=None):
    """Loads the data: text files are parsed, .npz files (caches written by
    an earlier run) are read as they are. With `cache`, the combined arrays
    are saved there, or loaded from it if it already exists."""
    if cache and os.path.exists(cache):
        d = np.load(cache)
        return d["feats"], d["stm"], d["score"], d["result"]
    parts = []
    text = []
    for p in list(paths) + [None]:
        if p is not None and not p.endswith(".npz"):
            text.append(p)
            continue
        if text:  # keep the files' order
            parts.extend(parse_files(text, workers))
            text = []
        if p is not None:
            d = np.load(p)
            parts.append((d["feats"], d["stm"], d["score"], d["result"]))
    feats = np.concatenate([p[0] for p in parts]) if parts else np.zeros((0, MAX_PIECES), np.int16)
    stm = np.concatenate([p[1] for p in parts]) if parts else np.zeros(0, np.int8)
    score = np.concatenate([p[2] for p in parts]) if parts else np.zeros(0, np.int16)
    result = np.concatenate([p[3] for p in parts]) if parts else np.zeros(0, np.float32)
    if cache:
        np.savez(cache, feats=feats, stm=stm, score=score, result=result)
    return feats, stm, score, result


class Nnue(torch.nn.Module):
    def __init__(self, hidden=256, buckets=1):
        super().__init__()
        self.hidden = hidden
        self.buckets = buckets
        self.ft = torch.nn.EmbeddingBag(FEATURES + 1, hidden, mode="sum", padding_idx=PAD)
        self.ft_bias = torch.nn.Parameter(torch.zeros(hidden))
        self.out = torch.nn.Linear(2 * hidden, buckets)
        with torch.no_grad():
            self.ft.weight.uniform_(-0.1, 0.1)
            self.ft.weight[PAD].zero_()
            self.out.weight.uniform_(-0.05, 0.05)
            self.out.bias.zero_()

    def forward(self, white, black, stm, bucket):
        """white/black: [N, 32] feature indices per perspective; stm: [N] 0/1;
        bucket: [N] output bucket."""
        aw = torch.clamp(self.ft(white) + self.ft_bias, 0.0, 1.0)
        ab = torch.clamp(self.ft(black) + self.ft_bias, 0.0, 1.0)
        s = stm.unsqueeze(1).bool()
        us = torch.where(s, ab, aw)
        them = torch.where(s, aw, ab)
        out = self.out(torch.cat([us, them], dim=1))
        return out.gather(1, bucket.unsqueeze(1)).squeeze(1)

    def clip(self):
        with torch.no_grad():
            self.ft.weight.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
            self.ft.weight[PAD].zero_()
            self.out.weight.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)


def batches(n, size, shuffle, rng):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, size):
        yield order[i : i + size]


def tensors(feats, stm, score, result, idx, lam, buckets=1):
    """Batch tensors: white/black features, side to move, output bucket, target."""
    white = torch.from_numpy(feats[idx].astype(np.int64))
    bucket = bucket_of((white != PAD).sum(dim=1), buckets)
    black = mirror(white)
    black[white == PAD] = PAD
    s = torch.from_numpy(stm[idx].astype(np.int64))
    sign = 1.0 - 2.0 * s.float()  # +1 White to move, -1 Black
    sc = torch.from_numpy(score[idx].astype(np.float32)) * sign
    res = torch.from_numpy(result[idx]).float()
    res = torch.where(s.bool(), 1.0 - res, res)
    target = lam * torch.sigmoid(sc / SCALE) + (1.0 - lam) * res
    return white, black, s, bucket, target


def evaluate_loss(model, data, idx, lam, size=65536):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(idx), size):
            part = idx[start : start + size]
            w, b, s, k, t = tensors(*data, part, lam, model.buckets)
            p = torch.sigmoid(model(w, b, s, k))
            total += float(((p - t) ** 2).sum())
            count += len(part)
    model.train()
    return total / max(1, count)


def export(model, path):
    ft_w = model.ft.weight.detach()[:FEATURES].numpy()  # [768, H]
    ft_b = model.ft_bias.detach().numpy()
    out_w = model.out.weight.detach().numpy()  # [B, 2H]
    out_b = model.out.bias.detach().numpy()  # [B]
    q = lambda a, s: np.clip(np.round(a * s), -32767, 32767).astype("<i2")
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        if model.buckets == 1:
            fh.write(struct.pack("<II", 1, model.hidden))
        else:
            fh.write(struct.pack("<III", 2, model.hidden, model.buckets))
        fh.write(q(ft_w, QA).tobytes())
        fh.write(q(ft_b, QA).tobytes())
        fh.write(q(out_w, QB).tobytes())
        fh.write(np.round(out_b * QA * QB).astype("<i4").tobytes())


def read_net(path):
    """Loads an exported net as integer numpy arrays (for checks)."""
    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:8] == MAGIC, "not a NAGS NNUE file"
    version, hidden = struct.unpack_from("<II", data, 8)
    assert version in (1, 2)
    off = 16
    buckets = 1
    if version == 2:
        (buckets,) = struct.unpack_from("<I", data, off)
        off += 4
    ft_w = np.frombuffer(data, "<i2", FEATURES * hidden, off).reshape(FEATURES, hidden).astype(np.int32)
    off += 2 * FEATURES * hidden
    ft_b = np.frombuffer(data, "<i2", hidden, off).astype(np.int32)
    off += 2 * hidden
    out_w = np.frombuffer(data, "<i2", buckets * 2 * hidden, off).reshape(buckets, 2 * hidden).astype(np.int32)
    off += 2 * buckets * 2 * hidden
    out_b = np.frombuffer(data, "<i4", buckets, off).astype(np.int64)
    return ft_w, ft_b, out_w, out_b


def quantized_eval(net, fen):
    """Integer evaluation of `fen` (centipawns, side to move), as the engine computes it."""
    ft_w, ft_b, out_w_all, out_b_all = net
    fields = fen.split()
    wf = np.array(white_features(fields[0]), dtype=np.int64)
    k = bucket_of(len(wf), len(out_b_all))
    out_w, out_b = out_w_all[k], int(out_b_all[k])
    bf = mirror(wf)
    acc_w = ft_b + ft_w[wf].sum(axis=0)
    acc_b = ft_b + ft_w[bf].sum(axis=0)
    us, them = (acc_w, acc_b) if fields[1] == "w" else (acc_b, acc_w)
    h = len(ft_b)
    total = int((np.clip(us, 0, QA) * out_w[:h]).sum() + (np.clip(them, 0, QA) * out_w[h:]).sum()) + out_b
    q = abs(total * SCALE) // (QA * QB)  # truncates towards zero, like C++
    return q if total >= 0 else -q


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="nags_datagen output files (or .npz caches)")
    ap.add_argument("--cache", help="parsed-data cache (.npz), created if missing")
    ap.add_argument("--out", required=True, help="exported network")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--buckets", type=int, default=1, help=f"output buckets by piece count (1-{MAX_BUCKETS})")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.75, help="weight of the search score in the target")
    ap.add_argument("--val", type=float, default=0.01, help="fraction held out for validation")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>.ckpt (saved after every epoch) with the same data and settings")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="stop after this many epochs in this run (continue later with --resume)")
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    data = load(args.data, args.cache)
    n = len(data[0])
    print(f"{n} positions loaded in {time.time() - t0:.0f}s", flush=True)
    if n == 0:
        sys.exit("no positions")
    perm = rng.permutation(n)
    n_val = max(1, int(n * args.val)) if n > 100 else 0
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    if not 1 <= args.buckets <= MAX_BUCKETS:
        sys.exit(f"--buckets must be 1-{MAX_BUCKETS}")
    model = Nnue(args.hidden, args.buckets)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = args.epochs * math.ceil(len(train_idx) / args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=args.lr * 0.01)
    ckpt_path = args.out + ".ckpt"
    first_epoch = 1
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        if (ckpt["positions"] != n or ckpt["hidden"] != args.hidden or ckpt["epochs"] != args.epochs
                or ckpt.get("buckets", 1) != args.buckets):
            sys.exit(f"{ckpt_path} was made with different data, network size, buckets or --epochs")
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
        for idx in batches(len(train_idx), args.batch, True, rng):
            w, b, s, k, target = tensors(*data, train_idx[idx], args.lam, args.buckets)
            loss = ((torch.sigmoid(model(w, b, s, k)) - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            model.clip()
            total += loss.item() * len(idx)
            count += len(idx)
        msg = f"epoch {epoch:3d}  train {total / count:.6f}"
        if n_val:
            msg += f"  val {evaluate_loss(model, data, val_idx, args.lam):.6f}"
        print(f"{msg}  ({time.time() - t:.0f}s)", flush=True)
        export(model, args.out)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.bit_generator.state, "epoch": epoch, "positions": n, "hidden": args.hidden,
                    "epochs": args.epochs, "buckets": args.buckets},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
    print(f"wrote {args.out}" + (f" (stopped after epoch {last_epoch} of {args.epochs})" if last_epoch < args.epochs else ""))


if __name__ == "__main__":
    main()
