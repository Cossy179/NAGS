#!/usr/bin/env python3
"""Convert nags_datagen text files (or .npz caches) to 32-byte binary records.

Binary files are about half the size of text, load much faster, and let
train.py stream data sets larger than memory (see the record layout at the
top of src/datagen_main.cpp). nags_datagen writes this format directly when
its output name ends in ".bin".

    python tools/nnue/pack.py data/selfplay_101.txt data/selfplay_102.txt
    python tools/nnue/pack.py "data/selfplay*.txt" --out-dir data/bin

Each input X.txt (or X.npz) becomes X.bin next to it, or in --out-dir.
Wildcards are expanded (Windows shells pass them through unexpanded).
"""

import argparse
import multiprocessing as mp
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402

LINES_PER_BLOCK = 500000


def _encode_lines(lines):
    return train.encode(*train.parse_chunk(lines)).tobytes()


def _blocks(path):
    with open(path) as fh:
        block = []
        for line in fh:
            block.append(line)
            if len(block) == LINES_PER_BLOCK:
                yield block
                block = []
        if block:
            yield block


def pack(path, out, workers=None):
    """Writes `out` from one input file; returns the number of positions."""
    tmp = out + ".tmp"
    count = 0
    with open(tmp, "wb") as fh:
        if path.endswith(".npz"):
            d = np.load(path)
            rec = train.encode(d["feats"], d["stm"], d["score"], d["result"])
            fh.write(rec.tobytes())
            count = len(rec)
        else:
            with mp.Pool(workers or os.cpu_count()) as pool:
                for data in pool.imap(_encode_lines, _blocks(path)):
                    fh.write(data)
                    count += len(data) // train.RECORD_BYTES
    os.replace(tmp, out)
    return count


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="nags_datagen text files or .npz caches")
    ap.add_argument("--out-dir", help="directory for the .bin files (default: next to each input)")
    ap.add_argument("--workers", type=int, default=0, help="parsing processes (default: all CPUs)")
    args = ap.parse_args(argv)
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    for path in train.expand_paths(args.inputs):
        if path.endswith(".bin"):
            continue
        base = os.path.splitext(os.path.basename(path))[0] + ".bin"
        out = os.path.join(args.out_dir or os.path.dirname(path), base)
        t = time.time()
        n = pack(path, out, args.workers or None)
        print(f"{path} -> {out}: {n} positions ({time.time() - t:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
