#!/usr/bin/env python3
"""Training-data mixtures: what a data set is made of, and new data sets mixed from several sources.

  report  the composition of 32-byte record files (a random sample of --sample records): pieces on the board,
          material balance and imbalances, pawn endings, queenless positions, |score| and results, and how
          many positions occur more than once.
  build   a mixture of sources in given shares of --positions, by whole games (from nags_datagen shard
          indexes; files without one are taken in blocks of 1024 consecutive records), with at most
          --max-copies records of any position (the board and side to move) kept, so near-identical
          opening positions do not swamp the rest. The games are written in random order with an index
          (one line per game), so train.py's validation split (the file's tail) is a sample of the whole
          mixture and falls between games. <out>.mix.json records the sources, their files, the shares
          asked for and obtained, the duplicates removed, and the composition of the result.

  book    start positions for nags_datagen --book, drawn from record files by kind: 'imbalanced' (material
          two or more pawns apart, or level material with different pieces), 'level-different' (only the
          latter), 'pawn-endings', 'queenless' or 'pieces:LO-HI'; --max-score keeps starts the search does not
          already call decided (nags_datagen skips openings beyond its --max-opening-score). Castling rights
          are inferred (tools/nnue/rescore.py, record_fen).

  python tools/nnue/mix.py report data/selfplay_*.bin
  python tools/nnue/mix.py book data/selfplay_*.bin --select imbalanced --count 5000 --max-score 300 \
      --out books/imbalanced.epd
  python tools/nnue/mix.py build --source "data/r8/selfplay_*.bin=0.4" --source "data/r9/selfplay_*.bin=0.5" \\
      --source "data/imbalanced/*.bin=0.1" --positions 100000000 --max-copies 4 --out data/mix/mix_a.bin

A source's share is of the positions before duplicates are removed; when a source has fewer positions than
its share, all of it is taken and the report says so. Several mixtures (different shares, --max-copies) of
the same sources are what docs/STRENGTH_PLAN.md (item 3) compares: train the same network design on each,
then play the networks.
"""

import argparse
import glob
import hashlib
import json
import os
import sys
from collections import namedtuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shards  # noqa: E402

RECORD_BYTES = 32
BLOCK = 1024  # records taken together from a file without an index
VALUES = np.array([1, 3, 3, 5, 9, 0], np.int64)  # P N B R Q K
Unit = namedtuple("Unit", "path offset records result origin")


def expand(pattern):
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    if not paths:
        sys.exit(f"no files match {pattern}")
    return paths


def units_of(path):
    """The file's games (from its shard index) or, without a usable index, blocks of BLOCK records."""
    size = os.path.getsize(path)
    if size % RECORD_BYTES:
        sys.exit(f"{path}: the size is not a multiple of {RECORD_BYTES} bytes")
    try:
        header, games = shards.read_index(path)
        if sum(g.bytes for g in games) == size:
            seed = shards.header_fields(header).get("seed")
            return [Unit(path, g.offset, g.records, g.result, f"{seed}:{g.number}") for g in games], True
    except (OSError, ValueError):
        pass
    n = size // RECORD_BYTES
    return [Unit(path, s * RECORD_BYTES, min(BLOCK, n - s), None, None) for s in range(0, n, BLOCK)], False


_maps = {}


def read_unit(u):
    m = _maps.get(u.path)
    if m is None:
        m = _maps[u.path] = np.memmap(u.path, np.uint8, "r")
    return np.asarray(m[u.offset : u.offset + u.records * RECORD_BYTES]).reshape(-1, RECORD_BYTES)


def position_keys(recs):
    """64-bit keys of the positions (occupancy, pieces and side to move; not the score or result)."""
    r = np.ascontiguousarray(recs).copy()
    r[:, 24:27] = 0
    r[:, 28:] = 0
    w = r.view("<u8").reshape(-1, 4)
    with np.errstate(over="ignore"):
        h = w[:, 0] * np.uint64(0x9E3779B97F4A7C15)
        for k, c in ((1, 0xC2B2AE3D27D4EB4F), (2, 0x165667B19E3779F9), (3, 0xD6E8FEB86659FD93)):
            h = (h ^ (h >> np.uint64(29))) * np.uint64(0xBF58476D1CE4E5B9) + w[:, k] * np.uint64(c)
    return h


def piece_counts(recs):
    """[N, 12] number of each piece (P N B R Q K p n b r q k) on the board."""
    n = len(recs)
    total = np.unpackbits(recs[:, :8], axis=1).sum(axis=1)
    nib = np.stack([recs[:, 8:24] & 15, recs[:, 8:24] >> 4], axis=2).reshape(n, 32).astype(np.int64)
    valid = np.arange(32)[None, :] < total[:, None]
    codes = np.where(valid, nib, 12)
    return np.stack([(codes == c).sum(axis=1) for c in range(12)], axis=1)


def composition(recs):
    """The report's numbers for a set of records (shares of the positions)."""
    n = len(recs)
    if n == 0:
        return {"positions": 0}
    pc = piece_counts(recs)
    pieces = pc.sum(axis=1)
    white, black = pc[:, :6], pc[:, 6:]
    mat_w, mat_b = white @ VALUES, black @ VALUES
    diff = np.abs(mat_w - mat_b)
    minors_w, minors_b = white[:, 1] + white[:, 2], black[:, 1] + black[:, 2]
    # Material about level but different pieces: the exchange, a queen against rooks or minor pieces, etc.
    different_sets = ((minors_w != minors_b) | (white[:, 3] != black[:, 3]) | (white[:, 4] != black[:, 4])) & (diff <= 1)
    pawn_ending = (pc[:, [1, 2, 3, 4, 7, 8, 9, 10]].sum(axis=1) == 0)
    score = np.abs(recs[:, 24:26].copy().view("<i2").reshape(-1).astype(np.int64))
    result = recs[:, 26]
    keys = position_keys(recs)
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    share = lambda m: float(np.mean(m))  # noqa: E731
    bins = {"2-6": (2, 6), "7-12": (7, 12), "13-20": (13, 20), "21-26": (21, 26), "27-32": (27, 32)}
    return {
        "positions": int(n),
        "pieces": {k: share((pieces >= lo) & (pieces <= hi)) for k, (lo, hi) in bins.items()},
        "material_diff": {"level": share(diff == 0), "1": share(diff == 1), "2": share(diff == 2),
                          "3-4": share((diff >= 3) & (diff <= 4)), "5+": share(diff >= 5)},
        "level_material_different_pieces": share(different_sets),
        "pawn_endings": share(pawn_ending),
        "queenless": share((pc[:, 4] + pc[:, 10]) == 0),
        "abs_score": {"p50": float(np.percentile(score, 50)), "p90": float(np.percentile(score, 90)),
                      "over_1000": share(score > 1000)},
        "results_white": {"win": share(result == 2), "draw": share(result == 1), "loss": share(result == 0)},
        "repeated_positions": share(counts[inverse] > 1),
        "distinct_positions": float(len(counts) / n),
    }


def sample(paths, count, seed):
    maps = [np.memmap(p, np.uint8, "r").reshape(-1, RECORD_BYTES) for p in paths if os.path.getsize(p)]
    total = sum(len(m) for m in maps)
    if total == 0:
        return np.zeros((0, RECORD_BYTES), np.uint8)
    pick = np.sort(np.random.default_rng(seed).choice(total, min(count, total), replace=False))
    out, offset = [], 0
    for m in maps:
        here = pick[(pick >= offset) & (pick < offset + len(m))] - offset
        out.append(np.asarray(m[here]))
        offset += len(m)
    return np.concatenate(out)


def print_composition(name, c):
    if not c.get("positions"):
        print(f"{name}: no positions")
        return
    pct = lambda d: ", ".join(f"{k} {v:.1%}" for k, v in d.items())  # noqa: E731
    print(f"{name}: {c['positions']} positions sampled")
    print(f"  pieces: {pct(c['pieces'])}")
    print(f"  material difference (pawns): {pct(c['material_diff'])}; level material, different pieces "
          f"{c['level_material_different_pieces']:.1%}")
    print(f"  pawn endings {c['pawn_endings']:.1%}, queenless {c['queenless']:.1%}; |score| p50 "
          f"{c['abs_score']['p50']:.0f}, p90 {c['abs_score']['p90']:.0f}, over 1000 {c['abs_score']['over_1000']:.1%}")
    print(f"  results (White): {pct(c['results_white'])}; positions occurring more than once "
          f"{c['repeated_positions']:.1%} (distinct {c['distinct_positions']:.1%})")


def selection(recs, kind):
    """Boolean mask of the records of a kind (see the module docstring, 'book')."""
    pc = piece_counts(recs)
    white, black = pc[:, :6], pc[:, 6:]
    diff = np.abs(white @ VALUES - black @ VALUES)
    minors_w, minors_b = white[:, 1] + white[:, 2], black[:, 1] + black[:, 2]
    different = ((minors_w != minors_b) | (white[:, 3] != black[:, 3]) | (white[:, 4] != black[:, 4])) & (diff <= 1)
    if kind == "imbalanced":
        return different | (diff >= 2)
    if kind == "level-different":
        return different
    if kind == "pawn-endings":
        return pc[:, [1, 2, 3, 4, 7, 8, 9, 10]].sum(axis=1) == 0
    if kind == "queenless":
        return (pc[:, 4] + pc[:, 10]) == 0
    if kind.startswith("pieces:"):
        lo, _, hi = kind[7:].partition("-")
        pieces = pc.sum(axis=1)
        return (pieces >= int(lo)) & (pieces <= int(hi or lo))
    sys.exit(f"--select {kind}: imbalanced, level-different, pawn-endings, queenless or pieces:LO-HI")


def book(args):
    import rescore
    paths = [p for pat in args.data for p in expand(pat)]
    recs = sample(paths, args.sample, args.seed)
    mask = selection(recs, args.select)
    if args.max_score is not None:
        mask &= np.abs(recs[:, 24:26].copy().view("<i2").reshape(-1).astype(np.int64)) <= args.max_score
    picked = recs[mask]
    keys = position_keys(picked)
    _, first = np.unique(keys, return_index=True)  # each position once
    picked = picked[np.sort(first)]
    if len(picked) > args.count:
        picked = picked[np.sort(np.random.default_rng(args.seed).choice(len(picked), args.count, replace=False))]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="\n") as fh:
        for r in picked:
            fh.write(" ".join(rescore.record_fen(r).split()[:4]) + "\n")
    print(f"{int(mask.sum())} of {len(recs)} sampled positions are '{args.select}'"
          + (f" within {args.max_score} cp" if args.max_score is not None else "")
          + f"; wrote {len(picked)} distinct ones to {args.out}")
    return len(picked)


def report(args):
    paths = [p for pat in args.data for p in expand(pat)]
    c = composition(sample(paths, args.sample, args.seed))
    print_composition(f"{len(paths)} file(s)", c)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(c, fh, indent=1)
    return c


def parse_source(text):
    pattern, sep, weight = text.rpartition("=")
    if not sep:
        sys.exit(f"--source {text}: use PATTERN=SHARE")
    try:
        w = float(weight)
    except ValueError:
        sys.exit(f"--source {text}: the share must be a number")
    if w <= 0:
        sys.exit(f"--source {text}: the share must be positive")
    return pattern, w


def build(args):
    sources = [parse_source(s) for s in args.source]
    total_w = sum(w for _, w in sources)
    rng = np.random.default_rng(args.seed)
    chosen, info = [], []
    for k, (pattern, w) in enumerate(sources):
        paths = expand(pattern)
        units, indexed = [], True
        for p in paths:
            u, ok = units_of(p)
            units.extend(u)
            indexed &= ok
        want = round(args.positions * w / total_w)
        order = rng.permutation(len(units))
        got, take = 0, []
        for i in order:
            if got >= want:
                break
            take.append(units[i])
            got += units[i].records
        available = sum(u.records for u in units)
        info.append({"pattern": pattern, "share": w / total_w, "files": [
            {"name": os.path.basename(p), "size": os.path.getsize(p)} for p in paths],
            "indexed": indexed, "available": int(available), "wanted": int(want), "taken": int(got),
            "short": got < want})
        if got < want:
            print(f"source {pattern}: {available} positions, fewer than its share ({want}): all taken", flush=True)
        chosen.extend((k, u) for u in take)
    order = rng.permutation(len(chosen))
    chosen = [chosen[i] for i in order]
    # Pass 1: position keys in output order; keep the first --max-copies of each.
    keys = np.concatenate([position_keys(read_unit(u)) for _, u in chosen]) if chosen else np.zeros(0, np.uint64)
    keep = np.ones(len(keys), bool)
    if args.max_copies and len(keys):
        idx = np.argsort(keys, kind="stable")
        sk = keys[idx]
        start = np.r_[0, np.flatnonzero(sk[1:] != sk[:-1]) + 1]
        run = np.repeat(start, np.diff(np.r_[start, len(sk)]))
        rank = np.arange(len(sk)) - run
        keep[idx[rank >= args.max_copies]] = False
    out = args.out
    if os.path.exists(out) or os.path.exists(shards.index_path(out)):
        sys.exit(f"{out} exists: a mixture is written once (choose another name)")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    digest = hashlib.sha256(json.dumps([info, args.positions, args.max_copies, args.seed]).encode()).hexdigest()[:16]
    header = (shards.HEADER + f"seed=mix-{digest} format=bin mix-sources={len(sources)} "
              f"max-copies={args.max_copies} mix-seed={args.seed}")
    pos, number, removed = 0, 0, [0] * len(sources)
    written = [0] * len(sources)
    with open(out + ".tmp", "wb") as data, open(shards.index_path(out) + ".tmp", "w", newline="\n") as idx:
        idx.write(header + "\n")
        for k, u in chosen:
            recs = read_unit(u)
            mask = keep[pos : pos + len(recs)]
            pos += len(recs)
            recs = recs[mask]
            removed[k] += int((~mask).sum())
            if not len(recs):
                continue
            blob = recs.tobytes()
            data.write(blob)
            result = u.result if u.result is not None else 0.5  # a block is not a game; its result is not used
            res = "0.5" if result == 0.5 else str(int(result))
            idx.write(f"{number} {len(recs)} {len(blob)} {res} {shards.fnv1a(blob):016x}\n")
            written[k] += len(recs)
            number += 1
    os.replace(out + ".tmp", out)
    os.replace(shards.index_path(out) + ".tmp", shards.index_path(out))
    for k, s in enumerate(info):
        s["duplicates_removed"], s["written"] = removed[k], written[k]
    total = sum(written)
    for s in info:
        s["share_written"] = s["written"] / total if total else 0.0
    result = {"out": os.path.basename(out), "positions": int(total), "units": number, "max_copies": args.max_copies,
              "seed": args.seed, "sources": info,
              "composition": composition(sample([out], args.sample, args.seed))}
    with open(out + ".mix.json", "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"wrote {out}: {total} positions in {number} games or blocks")
    for s in info:
        print(f"  {s['pattern']}: share {s['share']:.1%} asked, {s['share_written']:.1%} written ({s['written']} "
              f"positions, {s['duplicates_removed']} duplicates removed{', short' if s['short'] else ''})")
    print_composition(os.path.basename(out), result["composition"])
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report", help="the composition of record files")
    r.add_argument("data", nargs="+")
    r.add_argument("--sample", type=int, default=1_000_000)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--json")
    b = sub.add_parser("build", help="mix sources into a new data file")
    b.add_argument("--source", action="append", required=True, metavar="PATTERN=SHARE")
    b.add_argument("--positions", type=int, required=True, help="positions to take before duplicates are removed")
    b.add_argument("--max-copies", type=int, default=4, help="keep at most this many records of a position (0: all)")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--sample", type=int, default=1_000_000, help="records sampled for the composition report")
    b.add_argument("--out", required=True)
    k = sub.add_parser("book", help="start positions of a kind for nags_datagen --book")
    k.add_argument("data", nargs="+")
    k.add_argument("--select", required=True)
    k.add_argument("--count", type=int, default=5000)
    k.add_argument("--max-score", type=int, help="only positions whose |score| is at most this (centipawns)")
    k.add_argument("--sample", type=int, default=5_000_000, help="records drawn to choose from")
    k.add_argument("--seed", type=int, default=1)
    k.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "report":
        report(args)
    elif args.cmd == "book":
        book(args)
    else:
        build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
