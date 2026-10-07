#!/usr/bin/env python3
"""NNUE training on your own machine, start to finish, with one command.

    python tools/nnue/run_round.py --target-positions 200000000 --threads 11

1. Data: nags_datagen plays games with the current engine (CPU, --threads)
   into <data-dir>/selfplay_<seed>.bin until the data directory holds
   --target-positions positions (each batch is --games games with a new
   seed). Older selfplay*.txt files are converted to .bin once and count
   too. Without --target-positions one batch is generated (or none with
   --skip-datagen).
2. Training: every design in --designs is trained on all selfplay*.bin
   files (on the GPU when PyTorch has CUDA). Designs: current (that of the
   built-in network: 768 -> 256, 8 output buckets), and kb4 / kb8 / kb16
   (4, 8 or 16 king buckets with SCReLU, which need more data).
3. Test: each new network plays the engine's built-in network (SPRT [0, 10]
   at --tc). If several pass, the two best play each other. The winner is
   copied to nets/nags.nnue; rebuild the engines, check bench and commit it.

Everything can be interrupted: run the same command again and it continues
(data batches already written are kept, training resumes from its
checkpoint, test matches resume from their logs). Results of every step are
written to <data-dir>/round_<positions>.txt.
"""

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
import pack  # noqa: E402
import train  # noqa: E402

# Network designs (train.py arguments). "current" is the design of the
# built-in network; the others are bigger and need more data to pay off.
DESIGNS = {
    "current": ["--buckets", "8"],
    "kb4": ["--buckets", "8", "--king-buckets", "4", "--activation", "screlu"],
    "kb8": ["--buckets", "8", "--king-buckets", "8", "--activation", "screlu"],
    "kb16": ["--buckets", "8", "--king-buckets", "16", "--activation", "screlu"],
}


def find_binary(name):
    for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe", ROOT / "build" / f"{name}.exe"):
        if c.exists():
            return c
    sys.exit(f"{name} not found under build/: build the engines first (see docs/NNUE.md)")


def run(cmd, log=None):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    if log is None:
        return subprocess.run([str(c) for c in cmd]).returncode
    with open(log, "a") as fh:
        proc = subprocess.Popen([str(c) for c in cmd], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            fh.write(line)
            fh.flush()
        return proc.wait()


def convert_text(data_dir):
    for t in sorted(data_dir.glob("selfplay*.txt")):
        b = t.with_suffix(".bin")
        if not b.exists() or b.stat().st_mtime < t.stat().st_mtime:
            print(f"converting {t.name} to {b.name}", flush=True)
            pack.pack(str(t), str(b))


def data_files(data_dir):
    files = sorted(data_dir.glob("selfplay*.bin"))
    for f in files:
        # A batch cut off mid-write (power loss, killed) may end in a partial
        # record: drop it.
        extra = f.stat().st_size % train.RECORD_BYTES
        if extra:
            with open(f, "r+b") as fh:
                fh.truncate(f.stat().st_size - extra)
    return files


def count_positions(files):
    return sum(f.stat().st_size // train.RECORD_BYTES for f in files)


def next_seed(data_dir, start):
    used = [int(m.group(1)) for f in data_dir.glob("selfplay_*.*") if (m := re.match(r"selfplay_(\d+)\.", f.name))]
    return max([start - 1] + used) + 1


def generate(args, data_dir):
    datagen = find_binary("nags_datagen")

    def batch(seed):
        out = data_dir / f"selfplay_{seed}.bin"
        if run([datagen, "--out", out, "--games", args.games, "--threads", args.threads, "--nodes", args.nodes,
                "--seed", seed]) != 0:
            sys.exit("data generation failed")

    if args.target_positions:
        while (have := count_positions(data_files(data_dir))) < args.target_positions:
            print(f"\n{have:,} of {args.target_positions:,} positions", flush=True)
            batch(next_seed(data_dir, args.seed or 1))
    else:
        batch(next_seed(data_dir, args.seed or 1))


def sprt_result(log):
    """(decision, elo, error) from a finished match log, or None."""
    if not log.exists():
        return None
    text = log.read_text()
    decision = "H1" if "H1 accepted" in text else "H0" if "H0 accepted" in text else None
    if decision is None and "inconclusive" not in text:
        return None
    games = re.findall(r"Elo ([+-][\d.]+) ± ([\d.]+)", text)
    elo, err = (float(games[-1][0]), float(games[-1][1])) if games else (0.0, 0.0)
    return decision or "none", elo, err


def match(args, engine, net1, net2, log, bounds):
    """SPRT between two networks in the same engine (net2 None: the built-in
    one); resumes an interrupted match from its log."""
    done = sprt_result(log)
    if done:
        return done
    cmd = [sys.executable, ROOT / "tools" / "sprt.py", "--engine", engine, "--engine", engine,
           "--name", Path(net1).stem, "--name", Path(net2).stem if net2 else "built-in",
           "--option1", f"EvalFile={net1}", "--tc", args.tc,
           "--openings", ROOT / "tools" / "openings" / "nags_balanced.epd",
           "--concurrency", max(1, args.threads - 1), "--sprt", *bounds,
           "--resign-adjudication", "4", "1000", "--games", "20000"]
    if net2:
        cmd += ["--option2", f"EvalFile={net2}"]
    if log.exists() and "Games " in log.read_text():
        partial = log.with_suffix(".part")
        shutil.copyfile(log, partial)
        cmd += ["--resume", partial]
    run(cmd, log)
    return sprt_result(log) or ("none", 0.0, 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--target-positions", type=int, default=0,
                    help="generate batches until the data directory holds this many positions")
    ap.add_argument("--games", type=int, default=150000, help="self-play games per batch")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads for data generation and testing")
    ap.add_argument("--nodes", type=int, default=5000, help="search nodes per move in self-play")
    ap.add_argument("--seed", type=int, help="first self-play seed (default: after the highest one in use)")
    ap.add_argument("--designs", nargs="+", default=["current", "kb4"], choices=sorted(DESIGNS),
                    help="network designs to train and test (default: current kb4)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden", type=int, default=256, help="must match the engine build (NAGS_NNUE_HIDDEN)")
    ap.add_argument("--device", default="auto", help="training device: auto, cpu or cuda")
    ap.add_argument("--tc", default="3+0.03", help="time control of the test matches")
    ap.add_argument("--skip-datagen", action="store_true", help="train on the data that is there")
    ap.add_argument("--skip-test", action="store_true", help="stop after training")
    args = ap.parse_args(argv)

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    convert_text(data_dir)
    if not args.skip_datagen:
        generate(args, data_dir)
    files = data_files(data_dir)
    if not files:
        sys.exit(f"no selfplay*.bin or selfplay*.txt files in {data_dir}")
    positions = count_positions(files)
    summary = data_dir / f"round_{positions}.txt"
    print(f"\ntraining on {positions:,} positions in {len(files)} files", flush=True)

    # Training: one network per design. Names include the data size and the
    # epochs, so rerunning with the same settings resumes (or skips a
    # finished network), and new data starts afresh.
    nets = {}
    for d in args.designs:
        net = data_dir / f"net_{d}_{positions}_e{args.epochs}.nnue"
        train.main(["--data", *map(str, files), "--out", str(net), "--epochs", str(args.epochs),
                    "--hidden", str(args.hidden), "--device", args.device, "--resume", *DESIGNS[d]])
        if train.torch.cuda.is_available():
            train.torch.cuda.empty_cache()  # so the next design can keep its data in GPU memory too
        nets[d] = net
        with open(summary, "a") as fh:
            fh.write(f"trained {d}: {net}\n")

    if args.skip_test:
        print("trained " + ", ".join(str(n) for n in nets.values()))
        return
    engine = find_binary("nags_enhanced")
    results = {}
    for d, net in nets.items():
        decision, elo, err = match(args, engine, net, None, data_dir / f"sprt_{d}_{positions}.log", ["0", "10"])
        results[d] = (decision, elo, err)
        with open(summary, "a") as fh:
            fh.write(f"{d} vs built-in: {decision}, {elo:+.1f} +- {err:.1f} Elo\n")

    winners = sorted((d for d, r in results.items() if r[0] == "H1"), key=lambda d: -results[d][1])
    if len(winners) >= 2:
        a, b = winners[:2]
        decision, elo, err = match(args, engine, nets[a], nets[b], data_dir / f"sprt_{a}_vs_{b}_{positions}.log",
                                   ["-5", "5"])
        with open(summary, "a") as fh:
            fh.write(f"{a} vs {b}: {decision}, {elo:+.1f} +- {err:.1f} Elo\n")
        winners = [b if decision == "H0" else a]  # undecided: the one that did better against the built-in

    print("\nResults:")
    for d, (decision, elo, err) in results.items():
        print(f"  {d:8s} vs built-in: {decision}, {elo:+.1f} +- {err:.1f} Elo")
    if winners:
        best = nets[winners[0]]
        shutil.copyfile(best, ROOT / "nets" / "nags.nnue")
        msg = (f"\nThe {winners[0]} network is stronger than the built-in one: copied {best.name} to "
               "nets/nags.nnue.\nRebuild the engines (cmake --build build --config Release), check bench, record "
               "the result in docs/TESTING.md and docs/NNUE.md, and commit nets/nags.nnue.")
    else:
        msg = "\nNo new network beat the built-in one; it stays. Generate more data and run again."
    print(msg)
    with open(summary, "a") as fh:
        fh.write(msg.strip() + "\n")


if __name__ == "__main__":
    main()
