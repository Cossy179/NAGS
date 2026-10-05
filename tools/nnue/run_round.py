#!/usr/bin/env python3
"""One NNUE round on your own machine: generate data, train, test.

1. nags_datagen plays games with the current engine (CPU, all threads) and
   appends positions to <data-dir>/selfplay_<seed>.txt.
2. tools/nnue/train.py trains a new network on every selfplay*.txt in the
   data directory (each file is parsed once into a .npz cache next to it),
   on the GPU when PyTorch has CUDA.
3. tools/sprt.py plays the new network against the engine's built-in one.
   If it wins (H1), the network is copied to nets/nags.nnue; rebuild the
   engines and commit it.

    python tools/nnue/run_round.py --games 100000 --threads 11 --seed 101

Use a new --seed every round so no games repeat. --skip-datagen trains on
the existing data only; --skip-test stops after training.
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
        out = data_dir / f"selfplay_{args.seed}.txt"
        if run([find_binary("nags_datagen"), "--out", out, "--games", args.games, "--threads", args.threads,
                "--nodes", args.nodes, "--seed", args.seed]) != 0:
            sys.exit("data generation failed")

    texts = sorted(data_dir.glob("selfplay*.txt"))
    if not texts:
        sys.exit(f"no selfplay*.txt files in {data_dir}")
    sys.path.insert(0, str(ROOT / "tools" / "nnue"))
    import train  # noqa: E402

    caches = []
    for t in texts:
        cache = t.with_suffix(".npz")
        if not cache.exists() or cache.stat().st_mtime < t.stat().st_mtime:
            print(f"parsing {t.name}", flush=True)
            cache.unlink(missing_ok=True)
            train.load([str(t)], cache=str(cache))
        caches.append(str(cache))
    train.main(["--data", *caches, "--out", str(net), "--epochs", str(args.epochs), "--buckets", str(args.buckets),
                "--hidden", str(args.hidden), "--device", args.device])

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
