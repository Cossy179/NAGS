#!/usr/bin/env python3
"""Train the NNUE evaluation from nags_datagen output and export it.

Network (perspective, "768 -> 2xH -> 1" by default):

* Input: 768 binary features per perspective (colour relative to the
  perspective x 6 piece types x 64 squares, squares mirrored vertically for
  Black's perspective). With king buckets (--king-buckets) each perspective
  uses one of several 768-feature sets, chosen by its own king's square, and
  its squares are mirrored left-right when that king is on files e-h. While
  training, a shared "factor" set is added to every bucket so that what is
  common to all of them is learned from all the data; it is folded into
  each bucket on export.
* Feature transformer: one layer shared by both perspectives, giving two
  accumulators (side to move, other side). The C++ engine keeps them up to
  date incrementally as pieces move.
* Clipped ReLU (or its square, SCReLU: --activation screlu) on each
  accumulator, concatenated side to move first, and a linear output. With
  output buckets (--buckets N) there are N output layers and the one used
  depends on the number of pieces on the board:
  bucket = min(N - 1, (pieces - 1) * N // 32), so the opening, middlegame and
  endgame get their own final weights at no extra cost. Output x SCALE =
  centipawns for the side to move.

Training target: lambda * sigmoid(score / SCALE) + (1 - lambda) * result,
both for the side to move, with a squared error on sigmoid(output).

Export (little endian), read by src/Nnue.cpp:

    char[8]  "NAGSNNUE"     uint32 version     uint32 hidden size H
    version 2:              uint32 output buckets B (1..8; version 1 means B = 1)
    version 3:              uint32 B, uint32 king buckets K (1..32),
                            uint32 flags (bit 0 SCReLU, bit 1 mirroring),
                            uint8  layout[64] (king square -> bucket, from
                                   the perspective, Black's flipped vertically)
    int16    ft_weights[K][768][H]  (x QA; K = 1 before version 3)
    int16    ft_bias[H]             (x QA)
    int16    out_weights[B][2H]     (x QB)
    int32    out_bias[B]            (x QA x QB)

Networks without king buckets, mirroring or SCReLU are written as version 1
(one output bucket) or 2.

Training data: nags_datagen text files (parsed, optionally cached as .npz)
or its 32-byte binary records (.bin; see src/datagen_main.cpp, and
tools/nnue/pack.py to convert text files). Binary data is held in GPU memory
when it fits and otherwise streamed from disk in shuffled chunks, so data
sets much larger than memory train at full speed.

Usage:

    python tools/nnue/train.py --data data/selfplay.txt --epochs 30 --out nets/nags.nnue
    python tools/nnue/train.py --data "data/*.bin" --epochs 30 --out nets/nags.nnue
"""

import argparse
import concurrent.futures
import glob
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
MAX_KING_BUCKETS = 32

# King bucket layouts for the 32 squares a1-d8 (rank by rank, files a-d) as
# seen by the perspective after mirroring; the king on files e-h uses the
# mirrored square.
KING_LAYOUTS = {
    "none": None,
    "mirror": [0] * 32,
    "4": [0, 0, 1, 1] + [2] * 4 + [3] * 24,
    "8": [0, 1, 2, 3, 4, 4, 5, 5] + [6] * 8 + [7] * 16,
    "16": [0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 9, 9, 10, 10, 11, 11] + [12, 12, 13, 13] * 2 + [14, 14, 15, 15] * 2,
}


def king_config(name):
    """(buckets, mirror, layout[64]) for a --king-buckets value: a name from
    KING_LAYOUTS or 32 comma-separated bucket numbers."""
    if name in KING_LAYOUTS:
        half = KING_LAYOUTS[name]
    else:
        half = [int(x) for x in name.split(",")]
        if len(half) != 32 or min(half) < 0:
            raise ValueError("a king bucket layout needs 32 non-negative numbers")
    if half is None:
        return 1, False, np.zeros(64, np.int64)
    layout = np.zeros(64, np.int64)
    for sq in range(64):
        f, r = sq % 8, sq // 8
        layout[sq] = half[r * 4 + (f if f < 4 else 7 - f)]
    kb = int(layout.max()) + 1
    if kb > MAX_KING_BUCKETS:
        raise ValueError(f"at most {MAX_KING_BUCKETS} king buckets")
    return kb, True, layout


class KingBuckets:
    """Maps plain perspective features to rows of the bucketed feature
    transformer (plus the factor rows while training)."""

    def __init__(self, name="none", factor=True):
        self.name = name
        self.count, self.mirror, layout = king_config(name)
        self.layout = torch.from_numpy(layout)
        self.factor = factor and self.count > 1
        self.rows = self.count * FEATURES + (FEATURES if self.factor else 0)  # the padding row comes after

    @property
    def plain(self):
        return self.count == 1 and not self.mirror

    def __call__(self, f):
        """[N, 32] perspective features (PAD = 768) -> [N, 32] (or [N, 64] with
        the factor rows) row indices; the padding row is self.rows."""
        if self.plain:
            return f
        pad = f == PAD
        own_king = (f >= 320) & (f < 384)
        ksq = ((f - 320) * own_king).sum(dim=1)
        flip = ((ksq & 7) >= 4) & self.mirror
        x = torch.where(flip.unsqueeze(1), f ^ 7, f)
        bucket = self.layout.to(f.device)[ksq]
        out = torch.where(pad, self.rows, bucket.unsqueeze(1) * FEATURES + x)
        if self.factor:
            out = torch.cat([out, torch.where(pad, self.rows, self.count * FEATURES + x)], dim=1)
        return out


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


RECORD_BYTES = 32


def encode(feats, stm, score, result):
    """Positions as parsed (White-perspective features with PAD, side to move,
    White's score and result) -> binary records, uint8 [N, 32]."""
    n = len(feats)
    f = feats.astype(np.int64)
    valid = f != PAD
    sq = np.where(valid, f % 64, 64)
    order = np.argsort(sq, axis=1, kind="stable")
    sq = np.take_along_axis(sq, order, axis=1)
    f = np.take_along_axis(f, order, axis=1)
    valid = sq < 64
    codes = np.where(valid, (f // 384) * 6 + (f % 384) // 64, 0).astype(np.uint8)
    occ = np.zeros(n, np.uint64)
    for j in range(MAX_PIECES):
        occ |= np.where(valid[:, j], np.left_shift(np.uint64(1), np.minimum(sq[:, j], 63).astype(np.uint64)),
                        np.uint64(0))
    rec = np.zeros((n, RECORD_BYTES), np.uint8)
    rec[:, :8] = occ.astype("<u8").view(np.uint8).reshape(n, 8)
    rec[:, 8:24] = codes[:, 0::2] | (codes[:, 1::2] << 4)
    rec[:, 24:26] = np.asarray(score, "<i2").view(np.uint8).reshape(n, 2)
    rec[:, 26] = np.round(np.asarray(result) * 2).astype(np.uint8)
    rec[:, 27] = np.asarray(stm, np.uint8)
    return rec


def decode(rec):
    """Binary records (uint8 tensor [N, 32] on any device) -> White-perspective
    features [N, 32] (PAD-filled), side to move, White's score and result."""
    n, dev = rec.shape[0], rec.device
    rec = rec.long()
    bits = ((rec[:, :8].unsqueeze(2) >> torch.arange(8, device=dev)) & 1).reshape(n, 64)
    occupied = bits.bool()
    order = bits.cumsum(1) - 1  # piece number of each occupied square
    nib = torch.stack([rec[:, 8:24] & 15, rec[:, 8:24] >> 4], dim=2).reshape(n, 32)
    code = nib.gather(1, order.clamp(0, MAX_PIECES - 1))
    f = (code // 6) * 384 + (code % 6) * 64 + torch.arange(64, device=dev)
    feats = torch.full((n, MAX_PIECES + 1), PAD, dtype=torch.long, device=dev)
    feats.scatter_(1, torch.where(occupied, order, MAX_PIECES), torch.where(occupied, f, PAD))
    score = rec[:, 24] | (rec[:, 25] << 8)
    score = torch.where(score >= 32768, score - 65536, score)
    return feats[:, :MAX_PIECES], rec[:, 27], score, rec[:, 26].float() / 2


def expand_paths(paths):
    """Expands wildcards (Windows shells pass them through unexpanded)."""
    out = []
    for p in paths:
        if any(c in p for c in "*?["):
            matches = sorted(glob.glob(p))
            if not matches:
                sys.exit(f"no files match {p}")
            out.extend(matches)
        else:
            out.append(p)
    return out


class ArrayData:
    """Parsed positions held in memory (text files and .npz caches)."""

    def __init__(self, data, val, rng, device):
        self.n = len(data[0])
        self.device = device
        # On a GPU the whole data set stays in GPU memory when it fits
        # (batches are then gathered on the GPU); otherwise batches are built
        # on the CPU.
        data_device = torch.device("cpu")
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            if sum(x.nbytes for x in data) < free // 2:
                data_device = device
            print(f"training on {torch.cuda.get_device_name(device)}, data in "
                  f"{'GPU' if data_device.type == 'cuda' else 'CPU'} memory", flush=True)
        self.data = tuple(torch.from_numpy(x).to(data_device) for x in data)
        perm = rng.permutation(self.n)
        n_val = max(1, int(self.n * val)) if self.n > 100 else 0
        self.val_idx, self.train_idx = perm[:n_val], perm[n_val:]
        self.n_val = n_val

    def steps(self, batch):
        return math.ceil(len(self.train_idx) / batch)

    def train_batches(self, rng, batch, lam, buckets):
        for idx in batches(len(self.train_idx), batch, True, rng):
            yield tensors(*self.data, self.train_idx[idx], lam, buckets, self.device)

    def val_loss(self, model, lam):
        return evaluate_loss(model, self.data, self.val_idx, lam, device=self.device)


class BinData:
    """Binary record files. The last records (up to 1M) are held out for
    validation. The rest is kept in GPU memory when it fits, and otherwise
    read from disk in groups of chunks (in a new random order every epoch,
    shuffled within each group, the next group loading while the current one
    trains)."""

    CHUNK = 1 << 20  # records (32 MB)
    GROUP = 8        # chunks shuffled together

    def __init__(self, paths, val, device):
        self.device = device
        self.maps = []
        for p in paths:
            size = os.path.getsize(p)
            if size % RECORD_BYTES:
                sys.exit(f"{p}: the size is not a multiple of {RECORD_BYTES} bytes")
            if size:
                self.maps.append(np.memmap(p, np.uint8, "r", shape=(size // RECORD_BYTES, RECORD_BYTES)))
        self.n = sum(len(m) for m in self.maps)
        self.n_val = min(max(1, int(self.n * val)), 1 << 20) if self.n > 100 else 0
        self.n_train = self.n - self.n_val
        # Training and validation ranges as (file, start, end) pieces.
        self.chunks, self.val_parts = [], []
        offset = 0
        for i, m in enumerate(self.maps):
            lo, hi = offset, offset + len(m)
            for s in range(lo, min(hi, self.n_train), self.CHUNK):
                self.chunks.append((i, s - lo, min(hi, self.n_train, s + self.CHUNK) - lo))
            if hi > self.n_train:
                self.val_parts.append((i, max(lo, self.n_train) - lo, len(m)))
            offset = hi
        self.resident = None
        if device.type == "cuda":
            free, _ = torch.cuda.mem_get_info(device)
            if self.n * RECORD_BYTES < free - (2 << 30):
                self.resident = torch.empty((self.n_train, RECORD_BYTES), dtype=torch.uint8, device=device)
                pos = 0
                for i, s, e in self.chunks:
                    self.resident[pos : pos + e - s] = torch.from_numpy(np.array(self.maps[i][s:e])).to(device)
                    pos += e - s
            print(f"training on {torch.cuda.get_device_name(device)}, data "
                  f"{'in GPU memory' if self.resident is not None else 'streamed from disk'}", flush=True)

    def steps(self, batch):
        return math.ceil(self.n_train / batch)

    def _read(self, chunks):
        return np.concatenate([self.maps[i][s:e] for i, s, e in chunks])

    def _records(self, rng, batch):
        """Batches of records covering the training range once."""
        if self.resident is not None:
            order = torch.from_numpy(rng.permutation(self.n_train)).to(self.device)
            for i in range(0, self.n_train, batch):
                yield self.resident[order[i : i + batch]]
            return
        chunk_order = rng.permutation(len(self.chunks))
        groups = [[self.chunks[c] for c in chunk_order[g : g + self.GROUP]]
                  for g in range(0, len(chunk_order), self.GROUP)]
        perms = [rng.permutation(sum(e - s for _, s, e in grp)) for grp in groups]
        leftover = None
        with concurrent.futures.ThreadPoolExecutor(1) as pool:
            pending = pool.submit(self._read, groups[0]) if groups else None
            for g in range(len(groups)):
                block = pending.result()
                pending = pool.submit(self._read, groups[g + 1]) if g + 1 < len(groups) else None
                block = torch.from_numpy(block).to(self.device)[torch.from_numpy(perms[g]).to(self.device)]
                if leftover is not None:
                    block = torch.cat([leftover, block])
                full = len(block) // batch * batch
                for i in range(0, full, batch):
                    yield block[i : i + batch]
                leftover = block[full:] if full < len(block) else None
        if leftover is not None:
            yield leftover

    def train_batches(self, rng, batch, lam, buckets):
        for rec in self._records(rng, batch):
            feats, stm, score, result = decode(rec)
            yield tensors(feats, stm, score, result, torch.arange(len(rec), device=rec.device), lam, buckets,
                          self.device)

    def val_loss(self, model, lam, size=65536):
        model.eval()
        total = 0.0
        with torch.no_grad():
            for i, s, e in self.val_parts:
                for start in range(s, e, size):
                    rec = torch.from_numpy(np.array(self.maps[i][start : min(e, start + size)])).to(self.device)
                    feats, stm, score, result = decode(rec)
                    w, b, st, k, t = tensors(feats, stm, score, result, torch.arange(len(rec), device=rec.device), lam,
                                             model.buckets, self.device)
                    total += float(((torch.sigmoid(model(w, b, st, k)) - t) ** 2).sum())
        model.train()
        return total / max(1, self.n_val)


class Nnue(torch.nn.Module):
    def __init__(self, hidden=256, buckets=1, king_buckets="none", screlu=False):
        super().__init__()
        self.hidden = hidden
        self.buckets = buckets
        self.screlu = screlu
        self.king = KingBuckets(king_buckets)
        self.pad = self.king.rows
        self.ft = torch.nn.EmbeddingBag(self.pad + 1, hidden, mode="sum", padding_idx=self.pad)
        self.ft_bias = torch.nn.Parameter(torch.zeros(hidden))
        self.out = torch.nn.Linear(2 * hidden, buckets)
        with torch.no_grad():
            self.ft.weight.uniform_(-0.1, 0.1)
            if self.king.factor:  # the bucket rows start at zero; the factor rows carry the start
                self.ft.weight[: self.king.count * FEATURES].zero_()
            self.ft.weight[self.pad].zero_()
            self.out.weight.uniform_(-0.05, 0.05)
            self.out.bias.zero_()

    def forward(self, white, black, stm, bucket):
        """white/black: [N, 32] plain feature indices per perspective; stm: [N]
        0/1; bucket: [N] output bucket."""
        aw = torch.clamp(self.ft(self.king(white)) + self.ft_bias, 0.0, 1.0)
        ab = torch.clamp(self.ft(self.king(black)) + self.ft_bias, 0.0, 1.0)
        if self.screlu:
            aw, ab = aw * aw, ab * ab
        s = stm.unsqueeze(1).bool()
        us = torch.where(s, ab, aw)
        them = torch.where(s, aw, ab)
        out = self.out(torch.cat([us, them], dim=1))
        return out.gather(1, bucket.unsqueeze(1)).squeeze(1)

    def clip(self):
        with torch.no_grad():
            self.ft.weight.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)
            self.ft.weight[self.pad].zero_()
            self.out.weight.clamp_(-WEIGHT_CLIP, WEIGHT_CLIP)

    def feature_weights(self):
        """[K * 768, H] feature-transformer weights with the factor folded in."""
        w = self.ft.weight.detach()
        k = self.king
        rows = w[: k.count * FEATURES]
        if k.factor:
            rows = rows + w[k.count * FEATURES : (k.count + 1) * FEATURES].repeat(k.count, 1)
        return rows


def batches(n, size, shuffle, rng):
    order = rng.permutation(n) if shuffle else np.arange(n)
    for i in range(0, n, size):
        yield order[i : i + size]


def tensors(feats, stm, score, result, idx, lam, buckets=1, device="cpu"):
    """Batch tensors on `device`: white/black features, side to move, output
    bucket, target. The data may be numpy arrays or torch tensors (on the CPU
    or the GPU); idx selects the positions."""
    feats, stm, score, result = (torch.as_tensor(x) for x in (feats, stm, score, result))
    idx = torch.as_tensor(idx, device=feats.device)
    white = feats[idx].to(device=device, dtype=torch.int64)
    bucket = bucket_of((white != PAD).sum(dim=1), buckets)
    black = mirror(white)
    black[white == PAD] = PAD
    s = stm[idx].to(device=device, dtype=torch.int64)
    sign = 1.0 - 2.0 * s.float()  # +1 White to move, -1 Black
    sc = score[idx].to(device=device, dtype=torch.float32) * sign
    res = result[idx].to(device=device, dtype=torch.float32)
    res = torch.where(s.bool(), 1.0 - res, res)
    target = lam * torch.sigmoid(sc / SCALE) + (1.0 - lam) * res
    return white, black, s, bucket, target


def evaluate_loss(model, data, idx, lam, size=65536, device="cpu"):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(idx), size):
            part = idx[start : start + size]
            w, b, s, k, t = tensors(*data, part, lam, model.buckets, device)
            p = torch.sigmoid(model(w, b, s, k))
            total += float(((p - t) ** 2).sum())
            count += len(part)
    model.train()
    return total / max(1, count)


def export(model, path):
    ft_w = model.feature_weights().cpu().numpy()  # [K * 768, H]
    ft_b = model.ft_bias.detach().cpu().numpy()
    out_w = model.out.weight.detach().cpu().numpy()  # [B, 2H]
    out_b = model.out.bias.detach().cpu().numpy()  # [B]
    q = lambda a, s: np.clip(np.round(a * s), -32767, 32767).astype("<i2")
    with open(path, "wb") as fh:
        fh.write(MAGIC)
        k = model.king
        if not k.plain or model.screlu:
            flags = (1 if model.screlu else 0) | (2 if k.mirror else 0)
            fh.write(struct.pack("<IIIII", 3, model.hidden, model.buckets, k.count, flags))
            fh.write(k.layout.numpy().astype(np.uint8).tobytes())
        elif model.buckets == 1:
            fh.write(struct.pack("<II", 1, model.hidden))
        else:
            fh.write(struct.pack("<III", 2, model.hidden, model.buckets))
        fh.write(q(ft_w, QA).tobytes())
        fh.write(q(ft_b, QA).tobytes())
        fh.write(q(out_w, QB).tobytes())
        fh.write(np.round(out_b * QA * QB).astype("<i4").tobytes())


def read_net(path):
    """Loads an exported net as integer numpy arrays (for checks):
    (ft_w [K * 768, H], ft_b, out_w [B, 2H], out_b [B], info) where info has
    king_buckets, mirror, layout and screlu."""
    with open(path, "rb") as fh:
        data = fh.read()
    assert data[:8] == MAGIC, "not a NAGS NNUE file"
    version, hidden = struct.unpack_from("<II", data, 8)
    assert version in (1, 2, 3)
    off = 16
    buckets, kb, flags, layout = 1, 1, 0, np.zeros(64, np.int64)
    if version == 2:
        (buckets,) = struct.unpack_from("<I", data, off)
        off += 4
    elif version == 3:
        buckets, kb, flags = struct.unpack_from("<III", data, off)
        off += 12
        layout = np.frombuffer(data, np.uint8, 64, off).astype(np.int64)
        off += 64
    rows = kb * FEATURES
    ft_w = np.frombuffer(data, "<i2", rows * hidden, off).reshape(rows, hidden).astype(np.int32)
    off += 2 * rows * hidden
    ft_b = np.frombuffer(data, "<i2", hidden, off).astype(np.int32)
    off += 2 * hidden
    out_w = np.frombuffer(data, "<i2", buckets * 2 * hidden, off).reshape(buckets, 2 * hidden).astype(np.int32)
    off += 2 * buckets * 2 * hidden
    out_b = np.frombuffer(data, "<i4", buckets, off).astype(np.int64)
    info = {"king_buckets": kb, "mirror": bool(flags & 2), "layout": layout, "screlu": bool(flags & 1)}
    return ft_w, ft_b, out_w, out_b, info


def _trunc_div(a, b):
    """Integer division truncating towards zero, like C++."""
    q = abs(a) // b
    return q if a >= 0 else -q


def quantized_eval(net, fen):
    """Integer evaluation of `fen` (centipawns, side to move), as the engine computes it."""
    ft_w, ft_b, out_w_all, out_b_all = net[:4]
    info = net[4] if len(net) > 4 else {"king_buckets": 1, "mirror": False, "layout": np.zeros(64, np.int64),
                                        "screlu": False}
    fields = fen.split()
    wf = np.array(white_features(fields[0]), dtype=np.int64)
    k = bucket_of(len(wf), len(out_b_all))
    out_w, out_b = out_w_all[k], int(out_b_all[k])

    def rows(f):
        ksq = int(f[(f >= 320) & (f < 384)][0]) - 320
        x = f ^ 7 if info["mirror"] and (ksq & 7) >= 4 else f
        return int(info["layout"][ksq]) * FEATURES + x

    acc_w = ft_b + ft_w[rows(wf)].sum(axis=0)
    acc_b = ft_b + ft_w[rows(mirror(wf))].sum(axis=0)
    us, them = (acc_w, acc_b) if fields[1] == "w" else (acc_b, acc_w)
    h = len(ft_b)
    a, b = np.clip(us, 0, QA).astype(np.int64), np.clip(them, 0, QA).astype(np.int64)
    if info["screlu"]:
        total = _trunc_div(int((a * a * out_w[:h]).sum() + (b * b * out_w[h:]).sum()), QA) + out_b
    else:
        total = int((a * out_w[:h]).sum() + (b * out_w[h:]).sum()) + out_b
    return _trunc_div(total * SCALE, QA * QB)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True,
                    help="nags_datagen output: text files (or .npz caches) or .bin files; wildcards are expanded")
    ap.add_argument("--cache", help="parsed-data cache for text input (.npz), created if missing")
    ap.add_argument("--out", required=True, help="exported network")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--buckets", type=int, default=1, help=f"output buckets by piece count (1-{MAX_BUCKETS})")
    ap.add_argument("--king-buckets", default="none",
                    help=f"input buckets by king square: {', '.join(KING_LAYOUTS)} or 32 comma-separated numbers")
    ap.add_argument("--activation", choices=["crelu", "screlu"], default="crelu")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lambda", dest="lam", type=float, default=0.75, help="weight of the search score in the target")
    ap.add_argument("--val", type=float, default=0.01, help="fraction held out for validation")
    ap.add_argument("--threads", type=int, default=0, help="CPU threads (CPU training)")
    ap.add_argument("--device", default="auto", help="auto (CUDA if available), cpu or cuda")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>.ckpt (saved after every epoch) with the same data and settings")
    ap.add_argument("--stop-after", type=int, default=0,
                    help="stop after this many epochs in this run (continue later with --resume)")
    args = ap.parse_args(argv)

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        sys.exit("CUDA is not available: install the CUDA build of PyTorch (see docs/NNUE.md) or use --device cpu")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    paths = expand_paths(args.data)
    binary = [p.endswith(".bin") for p in paths]
    if any(binary) and not all(binary):
        sys.exit("--data: .bin files cannot be mixed with other inputs (convert those with tools/nnue/pack.py)")
    t0 = time.time()
    if binary and all(binary):
        source = BinData(paths, args.val, device)
    else:
        source = ArrayData(load(paths, args.cache), args.val, rng, device)
    n = source.n
    print(f"{n} positions loaded in {time.time() - t0:.0f}s", flush=True)
    if n == 0:
        sys.exit("no positions")

    if not 1 <= args.buckets <= MAX_BUCKETS:
        sys.exit(f"--buckets must be 1-{MAX_BUCKETS}")
    try:
        king_config(args.king_buckets)
    except ValueError as e:
        sys.exit(f"--king-buckets: {e}")
    model = Nnue(args.hidden, args.buckets, args.king_buckets, args.activation == "screlu").to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = args.epochs * source.steps(args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=args.lr * 0.01)
    ckpt_path = args.out + ".ckpt"
    first_epoch = 1
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        if (ckpt["positions"] != n or ckpt["hidden"] != args.hidden or ckpt["epochs"] != args.epochs
                or ckpt.get("buckets", 1) != args.buckets or ckpt.get("king_buckets", "none") != args.king_buckets
                or ckpt.get("activation", "crelu") != args.activation):
            sys.exit(f"{ckpt_path} was made with different data, network design or --epochs")
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
                    "rng": rng.bit_generator.state, "epoch": epoch, "positions": n, "hidden": args.hidden,
                    "epochs": args.epochs, "buckets": args.buckets, "king_buckets": args.king_buckets,
                    "activation": args.activation},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
    print(f"wrote {args.out}" + (f" (stopped after epoch {last_epoch} of {args.epochs})" if last_epoch < args.epochs else ""))


if __name__ == "__main__":
    main()
