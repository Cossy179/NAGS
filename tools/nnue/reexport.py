"""Export a training checkpoint again, with the output biases corrected for the rounding (no training).

    python tools/nnue/reexport.py --ckpt nets/run8/net_kb4.nnue.ckpt --data data/selfplay_*.bin \\
        --out nets/net8_corrected.nnue --check nets/nags.nnue

--check compares the plain export of the checkpoint with a network file (for example the one the engine
embeds), so it is certain that the checkpoint is that network's. The corrected file is then measured on
positions it was not corrected on: the float network against the plain and the corrected export, as the
engine computes them. Like any network, the corrected one replaces nets/nags.nnue only after an SPRT.
"""

import argparse
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import quant_error  # noqa: E402
import train  # noqa: E402


def tensors_of(recs, buckets):
    """(white, black, stm, bucket) of 32-byte records."""
    rec = torch.from_numpy(np.ascontiguousarray(recs))
    feats, stm, score, result = train.decode(rec)
    return train.tensors(feats, stm, score, result, torch.arange(len(rec)), 1.0, buckets)[:4]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="a train.py checkpoint (<net>.nnue.ckpt)")
    ap.add_argument("--data", nargs="+", required=True, help="32-byte record files to draw positions from")
    ap.add_argument("--positions", type=int, default=20000, help="positions the correction is measured on")
    ap.add_argument("--out", required=True, help="the corrected network")
    ap.add_argument("--check", metavar="NNUE", help="refuse unless the plain export equals this network")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)
    if os.path.exists(args.out):
        sys.exit(f"{args.out} already exists: choose another --out (an existing network is never overwritten)")

    model = quant_error.load_float(args.ckpt)
    paths = train.expand_paths(args.data)
    with tempfile.TemporaryDirectory() as tmp:
        plain = os.path.join(tmp, "plain.nnue")
        train.export(model, plain)
        if args.check:
            with open(plain, "rb") as a, open(args.check, "rb") as b:
                if a.read() != b.read():
                    sys.exit(f"the plain export of {args.ckpt} is not {args.check}: not that network's checkpoint")
            print(f"the plain export of {os.path.basename(args.ckpt)} is {args.check}, byte for byte")
        # 2N positions at random: the correction is measured on the first N, the result on the other N.
        recs = quant_error.positions(paths, 2 * args.positions, args.seed)
        recs = recs[np.random.default_rng(args.seed).permutation(len(recs))]  # both halves from every file
        fit, held_out = recs[: len(recs) // 2], recs[len(recs) // 2 :]
        train.export(model, args.out, tensors_of(fit, model.buckets))
        print(f"wrote {args.out}: output biases corrected on {len(fit)} positions")
        other = os.path.join(tmp, "held_out.bin")
        held_out.tofile(other)
        for name, path in (("plain", plain), ("corrected", args.out)):
            rep = quant_error.measure(args.ckpt, [other], len(held_out), args.seed, exported=path)
            t = rep["total (float - engine)"]["overall"]
            print(f"  {name:9} export on {t['positions']} other positions: float - engine {t['mean']:.2f} cp mean, "
                  f"p50 {t['p50']:.2f}, bias {t['bias']:+.2f}, sign changes {t['sign_changes']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
