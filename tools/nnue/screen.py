"""Staged screening of NNUE training candidates (designs or settings): many cheap runs, a few full ones.

    python tools/nnue/screen.py --candidates runs/candidates.json --out runs/screen1 \\
        --stage0-data data/selfplay_1.bin --stage0-epochs 4 \\
        --stage1-data data/selfplay_*.bin --stage1-epochs 20 --keep 2 --eval-set data/evalset_9999.bin

candidates.json maps a name to its train.py arguments, for example
{"kb4": ["--buckets", "8", "--king-buckets", "4", "--activation", "screlu"],
 "kb4-lr2e-3": ["--buckets", "8", "--king-buckets", "4", "--activation", "screlu", "--lr", "2e-3"]}.

Stage 0 trains every candidate on --stage0-data for --stage0-epochs and ranks them by the evaluation set's
loss at their best epoch (the same positions for every candidate, independent of the training data). Stage 1
trains the best --keep on --stage1-data with --keep-best. Every run resumes where it stopped, so the command
can be run again after an interruption. results.json and the printed table hold both stages' losses.

A lower loss is not more Elo (a 512-wide network with a lower loss lost 48 Elo, being slower): the last
stage is a match at equal time, for which the commands are printed. Whether stage 0 ranks candidates as
stage 1 does is itself a measurement: the table shows both, so it can be checked as results accumulate.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train  # noqa: E402


def _run(name, targs, data, epochs, eval_set, out_dir, extra):
    out = out_dir / f"{name}.nnue"
    argv = ["--data", *data, "--out", str(out), "--epochs", str(epochs), "--eval-set", eval_set,
            "--select-by", f"eval:{os.path.basename(eval_set)}", "--resume", *extra, *targs]
    train.main(argv)
    m = json.loads(Path(str(out) + ".json").read_text())
    return {"network": str(out), "best_epoch": m["best"]["epoch"], "eval_loss": m["best"]["loss"],
            "epochs_done": m["epochs_done"], "val": m["history"][-1]["val"]}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", required=True, help="JSON file: name -> list of train.py arguments")
    ap.add_argument("--out", required=True, help="directory for the runs and results.json")
    ap.add_argument("--eval-set", required=True, help="the fixed evaluation set every candidate is ranked on")
    ap.add_argument("--stage0-data", nargs="+", required=True)
    ap.add_argument("--stage0-epochs", type=int, default=4)
    ap.add_argument("--stage1-data", nargs="+", required=True)
    ap.add_argument("--stage1-epochs", type=int, default=20)
    ap.add_argument("--keep", type=int, default=2, help="candidates taken to stage 1")
    ap.add_argument("--early-stop", type=int, default=0, help="passed to train.py in both stages")
    ap.add_argument("--train-arg", action="append", default=[], help="an argument for every train.py run "
                    "(repeatable, e.g. --train-arg=--device --train-arg=cuda)")
    args = ap.parse_args(argv)

    candidates = json.loads(Path(args.candidates).read_text())
    if not candidates or not all(isinstance(v, list) for v in candidates.values()):
        sys.exit(f"{args.candidates}: a JSON object of name -> list of train.py arguments")
    out = Path(args.out)
    (out / "stage0").mkdir(parents=True, exist_ok=True)
    (out / "stage1").mkdir(parents=True, exist_ok=True)
    extra = list(args.train_arg) + (["--early-stop", str(args.early_stop)] if args.early_stop else [])
    s0 = {n: _run(n, a, train.expand_paths(args.stage0_data), args.stage0_epochs, args.eval_set, out / "stage0",
                  extra) for n, a in candidates.items()}
    ranked = sorted(s0, key=lambda n: s0[n]["eval_loss"])
    chosen = ranked[: args.keep]
    s1 = {n: _run(n, candidates[n], train.expand_paths(args.stage1_data), args.stage1_epochs, args.eval_set,
                  out / "stage1", extra + ["--keep-best"]) for n in chosen}
    results = {"eval_set": args.eval_set, "stage0": s0, "stage1": s1, "ranking_stage0": ranked,
               "ranking_stage1": sorted(s1, key=lambda n: s1[n]["eval_loss"])}
    (out / "results.json").write_text(json.dumps(results, indent=1) + "\n")
    print(f"\n{'candidate':24} {'stage 0 eval loss':>18} {'stage 1 eval loss':>18}")
    for n in ranked:
        print(f"{n:24} {s0[n]['eval_loss']:18.6f} " + (f"{s1[n]['eval_loss']:18.6f}" if n in s1 else f"{'-':>18}"))
    print("\nThe networks still need a match at equal time against the current one, for example:")
    for n in results["ranking_stage1"]:
        best = Path(s1[n]["network"]).with_suffix("").as_posix() + ".best.nnue"
        print(f"  python tools/sprt.py --engine build/nags_enhanced --engine build/nags_enhanced --name {n} "
              f"--name current --option1 EvalFile={best} --tc 3+0.03 --sprt 0 10 --games 8000 --concurrency 3 "
              "--openings tools/openings/nags_balanced.epd --resign-adjudication 4 1000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
