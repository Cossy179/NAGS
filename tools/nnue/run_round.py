#!/usr/bin/env python3
"""One NNUE round on your own machine: generate data, train, test.

1. nags_datagen plays games with the current engine (CPU, all threads) and
   appends positions to <data-dir>/selfplay_<seed>.bin (binary records).
2. tools/nnue/train.py trains a new network on every selfplay*.bin in the
   data directory, on the GPU when PyTorch has CUDA. Older selfplay*.txt
   files are converted to .bin once (tools/nnue/pack.py) and then used too.
3. tools/sprt.py plays the new network against the engine's built-in one.
   If it wins (H1), the network is copied to nets/nags.nnue; rebuild the
   engines and commit it.

    python tools/nnue/run_round.py --games 100000 --threads 11 --seed 101

Use a new --seed every round so no games repeat. --skip-datagen trains on
the existing data only; --skip-test stops after training. --king-buckets
and --activation choose the network design (see train.py).
"""

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def find_binary(name):
    for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe", ROOT / "build" / f"{name}.exe"):
        if c.exists():
            return c
    sys.exit(f"{name} not found under build/: build the engines first (see docs/NNUE.md)")


def run(cmd):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd]).returncode


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--games", type=int, default=100000, help="self-play games to generate")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for data generation and testing")
    ap.add_argument("--nodes", type=int, default=5000, help="search nodes per move in self-play")
    ap.add_argument("--seed", type=int, required=True, help="self-play seed (use a new one every round)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--buckets", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=256, help="must match the engine build (NAGS_NNUE_HIDDEN)")
    ap.add_argument("--king-buckets", default="none", help="input buckets by king square (see train.py)")
    ap.add_argument("--activation", default="crelu", choices=["crelu", "screlu"])
    ap.add_argument("--device", default="auto", help="training device: auto, cpu or cuda")
    ap.add_argument("--tc", default="3+0.03", help="time control of the test match")
    ap.add_argument("--skip-datagen", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    net = data_dir / f"net_{stamp}.nnue"

    if not args.skip_datagen:
        out = data_dir / f"selfplay_{args.seed}.bin"
        if run([find_binary("nags_datagen"), "--out", out, "--games", args.games, "--threads", args.threads,
                "--nodes", args.nodes, "--seed", args.seed]) != 0:
            sys.exit("data generation failed")

    sys.path.insert(0, str(ROOT / "tools" / "nnue"))
    import pack  # noqa: E402
    import train  # noqa: E402

    for t in sorted(data_dir.glob("selfplay*.txt")):
        b = t.with_suffix(".bin")
        if not b.exists() or b.stat().st_mtime < t.stat().st_mtime:
            print(f"converting {t.name} to {b.name}", flush=True)
            pack.pack(str(t), str(b))
    files = sorted(data_dir.glob("selfplay*.bin"))
    if not files:
        sys.exit(f"no selfplay*.bin or selfplay*.txt files in {data_dir}")
    train.main(["--data", *map(str, files), "--out", str(net), "--epochs", str(args.epochs),
                "--buckets", str(args.buckets), "--hidden", str(args.hidden), "--device", args.device,
                "--king-buckets", args.king_buckets, "--activation", args.activation])

    if args.skip_test:
        print(f"trained {net}")
        return
    engine = find_binary("nags_enhanced")
    rc = run([sys.executable, ROOT / "tools" / "sprt.py", "--engine", engine, "--engine", engine,
              "--name", "new", "--name", "current", "--option1", f"EvalFile={net}", "--tc", args.tc,
              "--openings", ROOT / "tools" / "openings" / "nags_balanced.epd",
              "--concurrency", max(1, args.threads - 1), "--sprt", "0", "10",
              "--resign-adjudication", "4", "1000", "--games", "4000"])
    if rc == 0:
        shutil.copyfile(net, ROOT / "nets" / "nags.nnue")
        print(f"\nThe new network is stronger: copied {net.name} to nets/nags.nnue.\n"
              "Rebuild the engines (cmake --build build --config Release), check bench, and commit it.")
    elif rc == 1:
        print(f"\nThe new network is not stronger; the built-in one stays. ({net} is kept.)")
    else:
        print("\nThe test match ended without a decision.")


if __name__ == "__main__":
    main()
