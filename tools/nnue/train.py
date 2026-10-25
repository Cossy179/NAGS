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

Provenance: the training data is fingerprinted (SHA-256 of each file's content, cached next to the file
as <file>.sha256 by size and modification time), the fingerprint is kept in the checkpoint, and --resume
refuses data with the same number of positions but other content. After every epoch <out>.json records
what the network is: its architecture, the settings, the data fingerprint, the loss history, the weights
it started from and the code that trained it.

Two ways to start from an existing network, kept apart:

    --resume            continue this run exactly (the same data, settings, optimiser and schedule state)
    --init-from NET     a new run that starts from NET's weights (an exported .nnue such as nets/nags.nnue,
                        or a training .ckpt) with a fresh optimiser and learning-rate schedule; NET must have
                        this run's architecture (hidden size, output buckets, king buckets, activation)
"""

import argparse
import concurrent.futures
import datetime
import glob
import hashlib
import json
import math
import multiprocessing as mp
import os
import struct
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shards  # noqa: E402

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
        self.n_train = self.n - n_val
        self.split = "random positions"  # text data has no game index: a game's positions fall on both sides

    def steps(self, batch):
        return math.ceil(len(self.train_idx) / batch)

    def train_batches(self, rng, batch, lam, buckets):
        for idx in batches(len(self.train_idx), batch, True, rng):
            yield tensors(*self.data, self.train_idx[idx], lam, buckets, self.device)

    def val_loss(self, model, lam):
        return evaluate_loss(model, self.data, self.val_idx, lam, device=self.device)


def game_boundary(paths, counts, target):
    """(n_train, split): the validation boundary moved from record `target` to the nearest start of a game in
    the file it falls in, using that file's nags_datagen shard index (<file>.games), so that no game has
    positions on both sides ("games"). Without a usable index the boundary stays and one game may be split
    ("records")."""
    offset = 0
    for path, count in zip(paths, counts):
        if offset + count <= target:
            offset += count
            continue
        local = target - offset
        if local == 0:
            return target, "games"  # at a file boundary: games never span files
        try:
            _, games = shards.read_index(path)
        except (OSError, ValueError):
            games = None
        if not games or sum(g.bytes for g in games) != count * RECORD_BYTES:
            print(f"validation: {os.path.basename(path)} has no complete game index, so the boundary may cut one "
                  "game (nags_datagen --first-game writes one)", flush=True)
            return target, "records"
        starts = [g.offset // RECORD_BYTES for g in games] + [count]
        best = min(starts, key=lambda s: (abs(s - local), s))
        if offset + best >= sum(counts):  # keep some validation
            best = max(s for s in starts if offset + s < sum(counts))
        return offset + best, "games"
    return target, "games"


def record_hashes(recs):
    """64-bit hashes of 32-byte records (uint8 [N, 32]), to find records two files share."""
    w = np.ascontiguousarray(recs).view("<u8").reshape(-1, 4)
    with np.errstate(over="ignore"):
        h = w[:, 0] * np.uint64(0x9E3779B97F4A7C15)
        for k, c in ((1, 0xC2B2AE3D27D4EB4F), (2, 0x165667B19E3779F9), (3, 0xD6E8FEB86659FD93)):
            h = (h ^ (h >> np.uint64(29))) * np.uint64(0xBF58476D1CE4E5B9) + w[:, k] * np.uint64(c)
    return h


def eval_overlap(eval_path, paths, data_sha, cache=True):
    """How an --eval-set file overlaps the training data: the same file, games of the same seed and number in
    both (from their nags_datagen indexes), and records that occur in both (one pass over the data, kept in
    <eval>.overlap.json for this data and eval file)."""
    eval_sha = file_sha256(eval_path, cache=False)
    side = eval_path + ".overlap.json"
    if cache and os.path.exists(side):
        try:
            with open(side) as fh:
                kept = json.load(fh)
            if kept.get("eval_sha256") == eval_sha and kept.get("data_sha256") == data_sha:
                return kept
        except (OSError, ValueError):
            pass
    same_file = [p for p in paths if os.path.realpath(p) == os.path.realpath(eval_path)
                 or file_sha256(p, cache=True) == eval_sha]
    shared_games = 0
    try:
        eh, eg = shards.read_index(eval_path)
        mine = {(shards.header_fields(eh).get("seed"), g.number) for g in eg}
        for p in paths:
            try:
                th, tg = shards.read_index(p)
            except (OSError, ValueError):
                continue
            seed = shards.header_fields(th).get("seed")
            shared_games += sum((seed, g.number) in mine for g in tg)
    except (OSError, ValueError):
        pass
    raw = np.fromfile(eval_path, np.uint8)
    if raw.size % RECORD_BYTES:
        sys.exit(f"--eval-set {eval_path}: the size is not a multiple of {RECORD_BYTES} bytes")
    eval_h = record_hashes(raw.reshape(-1, RECORD_BYTES))
    unique = np.unique(eval_h)
    found = np.zeros(len(unique), bool)
    for p in paths:
        size = os.path.getsize(p)
        if not size or size % RECORD_BYTES:
            continue
        m = np.memmap(p, np.uint8, "r", shape=(size // RECORD_BYTES, RECORD_BYTES))
        for start in range(0, len(m), 1 << 20):
            h = record_hashes(np.array(m[start : start + (1 << 20)]))
            idx = np.minimum(np.searchsorted(unique, h), len(unique) - 1)
            found[idx[unique[idx] == h]] = True
    shared = int(np.isin(eval_h, unique[found]).sum())
    result = {"eval_sha256": eval_sha, "data_sha256": data_sha, "records": int(len(eval_h)),
              "records_in_training_data": shared, "same_file_as": [os.path.basename(p) for p in same_file],
              "games_in_training_data": shared_games}
    if cache:
        try:
            with open(side, "w") as fh:
                json.dump(result, fh, indent=1)
        except OSError:
            pass
    return result


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
        self.maps, self.paths = [], []
        for p in paths:
            size = os.path.getsize(p)
            if size % RECORD_BYTES:
                sys.exit(f"{p}: the size is not a multiple of {RECORD_BYTES} bytes")
            if size:
                self.maps.append(np.memmap(p, np.uint8, "r", shape=(size // RECORD_BYTES, RECORD_BYTES)))
                self.paths.append(p)
        self.n = sum(len(m) for m in self.maps)
        self.n_val = min(max(1, int(self.n * val)), 1 << 20) if self.n > 100 else 0
        self.n_train, self.split = self.n - self.n_val, "records"
        if self.n_val:
            self.n_train, self.split = game_boundary(self.paths, [len(m) for m in self.maps], self.n_train)
            self.n_val = self.n - self.n_train
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


def eval_set_report(model, path, lam, device, size=65536):
    """The loss on a fixed set of 32-byte records (--eval-set): overall, by output bucket (piece count) and
    by the king bucket of the side to move (its king's square in the network's layout), so a change that
    helps the middlegame and hurts the endgame shows as such."""
    raw = np.fromfile(path, np.uint8)
    if raw.size % RECORD_BYTES:
        sys.exit(f"--eval-set {path}: the size is not a multiple of {RECORD_BYTES} bytes")
    recs = raw.reshape(-1, RECORD_BYTES)
    nb, nk = model.buckets, model.king.count
    err_b, cnt_b = torch.zeros(nb, dtype=torch.float64), torch.zeros(nb, dtype=torch.float64)
    err_k, cnt_k = torch.zeros(nk, dtype=torch.float64), torch.zeros(nk, dtype=torch.float64)
    layout = model.king.layout.to(device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(recs), size):
            rec = torch.from_numpy(np.array(recs[start : start + size])).to(device)
            feats, stm, score, result = decode(rec)
            w, b, s, k, t = tensors(feats, stm, score, result, torch.arange(len(rec), device=device), lam,
                                    model.buckets, device)
            err = ((torch.sigmoid(model(w, b, s, k)) - t) ** 2).double()
            us = torch.where(s.bool().unsqueeze(1), b, w)  # the side to move's perspective features
            ksq = ((us - 320) * ((us >= 320) & (us < 384))).sum(dim=1)
            kb = layout[ksq] if nk > 1 else torch.zeros_like(ksq)
            err_b += torch.bincount(k, weights=err, minlength=nb).cpu()
            cnt_b += torch.bincount(k, minlength=nb).double().cpu()
            err_k += torch.bincount(kb, weights=err, minlength=nk).cpu()
            cnt_k += torch.bincount(kb, minlength=nk).double().cpu()
    model.train()
    per = lambda e, c: [float(x / y) if y else None for x, y in zip(e, c)]  # noqa: E731
    n = int(cnt_b.sum())
    return {"positions": n, "loss": float(err_b.sum() / max(1, n)), "by_output_bucket": per(err_b, cnt_b),
            "output_bucket_positions": [int(x) for x in cnt_b], "by_king_bucket": per(err_k, cnt_k),
            "king_bucket_positions": [int(x) for x in cnt_k]}


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


def file_sha256(path, cache=True):
    """SHA-256 of a file's content. With `cache`, kept in <path>.sha256 with the file's size and modification
    time and reused while both are unchanged (hashing 200M positions takes about a minute from a hard disk)."""
    st = os.stat(path)
    side = path + ".sha256"
    if cache:
        try:
            with open(side) as fh:
                size, mtime, digest = fh.read().split()
            if int(size) == st.st_size and int(mtime) == st.st_mtime_ns and len(digest) == 64:
                return digest
        except (OSError, ValueError):
            pass
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(8 << 20), b""):
            h.update(block)
    digest = h.hexdigest()
    if cache:
        try:
            with open(side, "w") as fh:
                fh.write(f"{st.st_size} {st.st_mtime_ns} {digest}\n")
        except OSError:
            pass  # a read-only data directory: hash again next time
    return digest


def data_fingerprint(paths, cache=True):
    """What the training data is, by content: each file's name, size and SHA-256 in training order, and a
    digest of that list (the record count alone cannot tell two data sets apart)."""
    files = [{"name": os.path.basename(p), "size": os.path.getsize(p), "sha256": file_sha256(p, cache)}
             for p in paths]
    combined = hashlib.sha256(json.dumps([[f["size"], f["sha256"]] for f in files]).encode()).hexdigest()
    return {"sha256": combined, "files": files}


def architecture(model):
    """The exported network's design, as the engine reads it."""
    k = model.king
    version = 3 if (not k.plain or model.screlu) else 1 if model.buckets == 1 else 2
    return {"format_version": version, "hidden": model.hidden, "buckets": model.buckets,
            "king_buckets": k.count, "king_layout_name": k.name, "mirror": bool(k.mirror),
            "activation": "screlu" if model.screlu else "crelu",
            "layout": [int(x) for x in k.layout.tolist()]}


def init_from(model, path):
    """Loads `path`'s weights into `model` (an exported .nnue or a training .ckpt of the same design).
    An exported network is read back exactly: its integer weights over the export scales, folded into the
    king-bucket rows with the factor rows at zero, so exporting before any training writes the same file."""
    want = architecture(model)
    if path.endswith(".ckpt"):
        ckpt = torch.load(path, weights_only=False, map_location="cpu")
        have = {"hidden": ckpt.get("hidden"), "buckets": ckpt.get("buckets", 1),
                "king_buckets": ckpt.get("king_buckets", "none"), "activation": ckpt.get("activation", "crelu")}
        need = {"hidden": model.hidden, "buckets": model.buckets, "king_buckets": model.king.name,
                "activation": want["activation"]}
        if have != need:
            sys.exit(f"--init-from {path}: a {have} network, not this run's {need}")
        model.load_state_dict(ckpt["model"])
        return
    try:
        ft_w, ft_b, out_w, out_b, info = read_net(path)
    except (AssertionError, OSError, ValueError, struct.error) as e:
        sys.exit(f"--init-from {path}: not a network train.py can read ({e})")
    have = {"hidden": ft_b.shape[0], "buckets": out_b.shape[0], "king_buckets": info["king_buckets"],
            "mirror": info["mirror"], "activation": "screlu" if info["screlu"] else "crelu",
            "layout": [int(x) for x in info["layout"].tolist()] if info["king_buckets"] > 1 or info["mirror"]
            else want["layout"]}
    need = {k: want[k] for k in have}
    if have != need:
        diff = ", ".join(f"{k} {have[k]} (this run: {need[k]})" for k in have if have[k] != need[k])
        sys.exit(f"--init-from {path}: another architecture: {diff}")
    k = model.king
    with torch.no_grad():
        w = model.ft.weight
        w.zero_()
        w[: k.count * FEATURES] = torch.from_numpy(ft_w.astype(np.float32) / QA).to(w.device)
        model.ft_bias.copy_(torch.from_numpy(ft_b.astype(np.float32) / QA))
        model.out.weight.copy_(torch.from_numpy(out_w.astype(np.float32) / QB))
        model.out.bias.copy_(torch.from_numpy(out_b.astype(np.float64) / (QA * QB)).float())


def code_version():
    """The commit the trainer ran from (None outside a git checkout) and the trainer file's SHA-256."""
    commit = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(os.path.abspath(__file__)),
                                capture_output=True, text=True, timeout=10).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    return {"git_commit": commit, "train_py_sha256": file_sha256(os.path.abspath(__file__), cache=False)}


def profile(model, opt, sched, source, rng, args, device, warmup=5):
    """--profile: where a training batch's time goes, measured (the model is trained for the batches timed,
    and nothing is written)."""
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == "cuda" else (lambda: None)

    def batch_stream():
        while True:
            yield from source.train_batches(rng, args.batch, args.lam, args.buckets)

    stream = batch_stream()
    phases = {"data": [], "forward": [], "backward": [], "step": [], "loss read": []}
    for i in range(warmup + args.profile):
        sync()
        t0 = time.perf_counter()
        w, b, s, k, target = next(stream)
        sync()
        t1 = time.perf_counter()
        loss = ((torch.sigmoid(model(w, b, s, k)) - target) ** 2).mean()
        sync()
        t2 = time.perf_counter()
        opt.zero_grad()
        loss.backward()
        sync()
        t3 = time.perf_counter()
        opt.step()
        sched.step()
        model.clip()
        sync()
        t4 = time.perf_counter()
        loss.item()
        t5 = time.perf_counter()
        if i >= warmup:
            for name, d in zip(phases, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                phases[name].append(d * 1000)
    med = {k: float(np.median(v)) for k, v in phases.items()}
    print(f"profile: {args.profile} batches of {args.batch} on {device}, median ms per batch (each phase "
          "synchronised): " + ", ".join(f"{k} {v:.2f}" for k, v in med.items())
          + f"; total {sum(med.values()):.2f}", flush=True)

    def end_to_end(every):
        sync()
        t0 = time.perf_counter()
        pending = None
        for i in range(args.profile):
            w, b, s, k, target = next(stream)
            loss = ((torch.sigmoid(model(w, b, s, k)) - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            model.clip()
            part = loss.detach() * len(target)
            pending = part if pending is None else pending + part
            if (i + 1) % every == 0:
                pending.item()
                pending = None
        if pending is not None:
            pending.item()
        sync()
        return args.profile * args.batch / (time.perf_counter() - t0)

    every = args.sync_every if args.sync_every > 1 else 100
    per_batch, deferred = end_to_end(1), end_to_end(every)
    print(f"profile: end to end {per_batch:,.0f} positions/s reading the loss every batch, {deferred:,.0f} reading "
          f"it every {every} ({deferred / per_batch:.2f}x)", flush=True)
    return med, per_batch, deferred


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
    ap.add_argument("--init-from", metavar="NET",
                    help="start a new run from NET's weights (.nnue or .ckpt of the same architecture) with a "
                         "fresh optimiser and schedule; not a resume")
    ap.add_argument("--no-data-cache", action="store_true",
                    help="hash the data files again instead of reusing their <file>.sha256 caches")
    ap.add_argument("--eval-set", action="append", default=[], metavar="FILE",
                    help="a fixed set of 32-byte records (not trained on) evaluated after every epoch, overall, by "
                         "output bucket and by king bucket (repeatable; recorded in <out>.json)")
    ap.add_argument("--eval-overlap-limit", type=float, default=0.01, metavar="FRACTION",
                    help="refuse an --eval-set more than this fraction of whose records also occur in the training "
                         "data (default 0.01; independent self-play shares about 0.01%%); the same file, or a game "
                         "(seed and number) in both, is always refused")
    ap.add_argument("--sync-every", type=int, default=1,
                    help="read the training loss back from the GPU every N batches instead of every batch (each "
                         "read waits for the GPU); the network trained is the same, the logged loss differs in "
                         "rounding")
    ap.add_argument("--profile", type=int, default=0, metavar="N",
                    help="time N training batches by phase (data, forward, backward, step, loss read) and end to "
                         "end with and without a loss read per batch, then stop without writing anything")
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
    t0 = time.time()
    fingerprint = data_fingerprint([p for p in paths if not p.endswith(".npz")] or paths,
                                   cache=not args.no_data_cache)
    print(f"data: {len(paths)} files, content sha256 {fingerprint['sha256'][:16]} ({time.time() - t0:.0f}s)",
          flush=True)
    binary = [p.endswith(".bin") for p in paths]
    if any(binary) and not all(binary):
        sys.exit("--data: .bin files cannot be mixed with other inputs (convert those with tools/nnue/pack.py)")
    t0 = time.time()
    if binary and all(binary):
        source = BinData(paths, args.val, device)
    else:
        source = ArrayData(load(paths, args.cache), args.val, rng, device)
    n = source.n
    print(f"{n} positions loaded in {time.time() - t0:.0f}s; validation {source.n_val} (split by {source.split})",
          flush=True)
    if n == 0:
        sys.exit("no positions")
    overlaps = {}
    for path in args.eval_set:  # an evaluation set must be independent of what is trained on
        ov = eval_overlap(path, [p for p in paths if p.endswith(".bin")], fingerprint["sha256"],
                          cache=not args.no_data_cache)
        frac = ov["records_in_training_data"] / max(1, ov["records"])
        if ov["same_file_as"]:
            sys.exit(f"--eval-set {path} is training data ({', '.join(ov['same_file_as'])})")
        if ov["games_in_training_data"]:
            sys.exit(f"--eval-set {path}: {ov['games_in_training_data']} of its games (seed and number) are in "
                     "the training data")
        if frac > args.eval_overlap_limit:
            sys.exit(f"--eval-set {path}: {ov['records_in_training_data']} of its {ov['records']} records "
                     f"({frac:.2%}) also occur in the training data (limit {args.eval_overlap_limit:.2%})")
        print(f"eval set {os.path.basename(path)}: {ov['records']} records, {ov['records_in_training_data']} "
              f"({frac:.3%}) also in the training data", flush=True)
        overlaps[path] = ov

    if not 1 <= args.buckets <= MAX_BUCKETS:
        sys.exit(f"--buckets must be 1-{MAX_BUCKETS}")
    try:
        king_config(args.king_buckets)
    except ValueError as e:
        sys.exit(f"--king-buckets: {e}")
    model = Nnue(args.hidden, args.buckets, args.king_buckets, args.activation == "screlu").to(device)
    started_from = None
    if args.init_from:
        init_from(model, args.init_from)
        started_from = {"path": args.init_from, "sha256": file_sha256(args.init_from, cache=False)}
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    steps = args.epochs * source.steps(args.batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps), eta_min=args.lr * 0.01)
    ckpt_path = args.out + ".ckpt"
    # Settings that must also match on --resume. Checkpoints written before
    # they were recorded lack them and are resumed as before.
    settings = {"batch": args.batch, "lr": args.lr, "lam": args.lam, "val": args.val, "seed": args.seed,
                "king_layout": [int(x) for x in king_config(args.king_buckets)[2]]}
    first_epoch = 1
    history = []
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
        if (ckpt["positions"] != n or ckpt["hidden"] != args.hidden or ckpt["epochs"] != args.epochs
                or ckpt.get("buckets", 1) != args.buckets or ckpt.get("king_buckets", "none") != args.king_buckets
                or ckpt.get("activation", "crelu") != args.activation):
            sys.exit(f"{ckpt_path} was made with different data, network design or --epochs")
        changed = [k for k, v in settings.items() if k in ckpt and ckpt[k] != v]
        if changed:
            sys.exit(f"{ckpt_path} was made with different settings: {', '.join(changed)}")
        # Checkpoints written before the data was fingerprinted resume as before.
        if "data_sha256" in ckpt and ckpt["data_sha256"] != fingerprint["sha256"]:
            sys.exit(f"{ckpt_path} was made with other data: the same number of positions, other content "
                     f"(sha256 {ckpt['data_sha256'][:16]}, now {fingerprint['sha256'][:16]})")
        if "n_train" in ckpt and ckpt["n_train"] != source.n_train:
            sys.exit(f"{ckpt_path} was made with another training/validation split ({ckpt['n_train']} training "
                     f"positions, now {source.n_train})")
        # A resumed run keeps the weights it started from; naming other ones is a contradiction.
        if args.init_from and (ckpt.get("init_from") or {}).get("sha256") != started_from["sha256"]:
            sys.exit(f"{ckpt_path} started from {(ckpt.get('init_from') or {}).get('path')!r}, not from "
                     f"{args.init_from!r}")
        started_from = ckpt.get("init_from", started_from)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        rng.bit_generator.state = ckpt["rng"]
        history = list(ckpt.get("history", []))
        first_epoch = ckpt["epoch"] + 1
        print(f"resuming after epoch {ckpt['epoch']}", flush=True)
    if args.profile:
        profile(model, opt, sched, source, rng, args, device)
        return
    last_epoch = args.epochs if not args.stop_after else min(args.epochs, first_epoch + args.stop_after - 1)
    version = code_version()
    for epoch in range(first_epoch, last_epoch + 1):
        t = time.time()
        total, count = 0.0, 0
        pending, n_pending = None, 0  # loss sums still on the device (--sync-every)
        for w, b, s, k, target in source.train_batches(rng, args.batch, args.lam, args.buckets):
            loss = ((torch.sigmoid(model(w, b, s, k)) - target) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            model.clip()
            if args.sync_every <= 1:
                total += loss.item() * len(target)  # waits for the GPU to finish this batch
            else:
                part = loss.detach() * len(target)
                pending, n_pending = (part if pending is None else pending + part), n_pending + 1
                if n_pending >= args.sync_every:
                    total, pending, n_pending = total + pending.item(), None, 0
            count += len(target)
        if pending is not None:
            total += pending.item()
        msg = f"epoch {epoch:3d}  train {total / count:.6f}"
        val = source.val_loss(model, args.lam) if source.n_val else None
        if val is not None:
            msg += f"  val {val:.6f}"
        print(f"{msg}  ({time.time() - t:.0f}s)", flush=True)
        entry = {"epoch": epoch, "train": total / count, "val": val, "seconds": round(time.time() - t, 1)}
        if args.eval_set:
            entry["eval"] = {}
            for path in args.eval_set:
                rep = eval_set_report(model, path, args.lam, device)
                entry["eval"][os.path.basename(path)] = rep
                print(f"  eval {os.path.basename(path)}: {rep['loss']:.6f} on {rep['positions']} positions; by piece "
                      "count " + " ".join("-" if x is None else f"{x:.5f}" for x in rep["by_output_bucket"]), flush=True)
        history.append(entry)
        export(model, args.out)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.bit_generator.state, "epoch": epoch, "positions": n, "hidden": args.hidden,
                    "epochs": args.epochs, "buckets": args.buckets, "king_buckets": args.king_buckets,
                    "activation": args.activation, **settings,
                    "data_sha256": fingerprint["sha256"], "n_train": source.n_train, "split": source.split,
                    "init_from": started_from, "history": history},
                   ckpt_path + ".tmp")
        os.replace(ckpt_path + ".tmp", ckpt_path)
        manifest = {
            "format": "nags-nnue-training-1", "network": os.path.basename(args.out),
            "network_sha256": file_sha256(args.out, cache=False), "architecture": architecture(model),
            "settings": {**settings, "epochs": args.epochs, "val": args.val, "seed": args.seed},
            "data": {"positions": n, "train": n - source.n_val, "validation": source.n_val, "split": source.split,
                     **fingerprint},
            "init_from": started_from, "epochs_done": epoch, "history": history, "code": version,
            "eval_sets": [{"name": os.path.basename(p), "sha256": file_sha256(p, cache=False),
                           "records": overlaps[p]["records"],
                           "records_in_training_data": overlaps[p]["records_in_training_data"]}
                          for p in args.eval_set],
            "torch": torch.__version__, "device": str(device),
            "written": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }
        with open(args.out + ".json.tmp", "w") as fh:
            json.dump(manifest, fh, indent=1)
        os.replace(args.out + ".json.tmp", args.out + ".json")
    print(f"wrote {args.out}" + (f" (stopped after epoch {last_epoch} of {args.epochs})" if last_epoch < args.epochs else ""))


if __name__ == "__main__":
    main()
