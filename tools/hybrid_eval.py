#!/usr/bin/env python3
"""Does the GNN make NAGS stronger? Measurements for the hybrid (docs/TESTING.md, "The NAGS hybrid").

  accuracy   the network alone, on held-out labelled positions (rows of the 'label' step):
             top-1 / top-3 agreement of its prior with the search's best move, the
             negative log-likelihood of that move, and its value against the search's,
             next to a uniform prior; plus the time per position.
  decisions  the hybrid's per-move decisions on sample positions: how often MCTS proposes a
             move other than the alpha-beta arm's, how often verification lets it through,
             and, from a much deeper reference search of both moves, whether the moves it
             substituted were better (the cost of the network side comes with it: simulations,
             round trips).
  match      a match (tools/sprt.py) between two arms, with one rpc_server.py serving both:
               ab          nags_enhanced: pure alpha-beta
               plain       nags without the network (must equal ab)
               hybrid      nags with the network, default hybrid settings
               hybrid-opt  nags with the network and the hybrid search options (--opt-option)
               nn-ab       nags with the network ordering the alpha-beta root moves only (no MCTS)

  python tools/hybrid_eval.py accuracy --model models/gnn_v3_distill_X.pth --data heldout.jsonl
  python tools/hybrid_eval.py decisions --model models/gnn_v3_distill_X.pth --positions 200 --movetime 2000
  python tools/hybrid_eval.py match --model models/gnn_v3_distill_X.pth --arms hybrid ab --tc 10+0.1 \\
      --concurrency 2 --sprt 0 10 -- --pgnout hybrid.pgn

Everything after `--` goes to tools/sprt.py as it is.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "tools"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import chess  # noqa: E402
import chess.engine  # noqa: E402


def _exe(name: str) -> str:
    for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe", ROOT / "build" / f"{name}.exe"):
        if c.exists():
            return str(c)
    raise SystemExit(f"{name} not built (cmake --build build --config Release)")


# ----------------------------------------------------------------------------
# accuracy

def read_rows(path: str, limit: int) -> List[Dict[str, Any]]:
    from distillation import valid_row
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if valid_row(r):
                    rows.append(r)
            if limit and len(rows) >= limit:
                break
    return rows


def accuracy(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    from distillation import row_targets
    from gnn_v3 import load_any_checkpoint
    from training_pipeline import DISTILL_DEFAULTS

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    adapter = load_any_checkpoint(args.model, device=torch.device(args.device))
    backend = None
    if adapter.format == "nags-gnn-v3":
        from gnn_infer import TorchBackendV3
        backend = TorchBackendV3(adapter.model)
    rows = read_rows(args.data, args.limit)
    if not rows:
        raise SystemExit(f"no usable rows in {args.data}")
    top1 = top3 = 0
    nll: List[float] = []
    uniform_nll: List[float] = []
    uniform_top1 = 0.0
    sq: List[float] = []
    pairs = []
    start = time.perf_counter()
    for i in range(0, len(rows), args.batch):
        chunk = rows[i:i + args.batch]
        fens = [r["fen"] for r in chunk]
        if backend is not None:
            from relational_fast import build_batch
            probs, values, _ = backend.run(build_batch(fens), mc_samples=1)
        else:
            probs, values, _ = adapter.evaluate(adapter.build_batch(fens), mc_samples=1)
        for j, r in enumerate(chunk):
            legal = r["legal"]
            priors = adapter.move_priors(probs[j], legal)
            best = r["scored"][0]["move"]
            p = dict(zip(legal, priors))
            ranked = sorted(legal, key=lambda m: -p[m])
            top1 += ranked[0] == best
            top3 += best in ranked[:3]
            nll.append(-math.log(max(p.get(best, 0.0), 1e-12)))
            uniform_nll.append(math.log(len(legal)))
            uniform_top1 += 1.0 / len(legal)
            target = row_targets(r, DISTILL_DEFAULTS)["search_value"]
            v = float(values[j])
            sq.append((v - target) ** 2)
            pairs.append((v, target))
    elapsed = time.perf_counter() - start
    n = len(rows)
    vs, ts = zip(*pairs)
    corr = statistics.correlation(vs, ts) if n > 2 and len(set(vs)) > 1 and hasattr(statistics, "correlation") else None
    out = {"model": args.model, "format": adapter.format, "positions": n, "top1": top1 / n, "top3": top3 / n,
           "nll": statistics.mean(nll), "uniform_top1": uniform_top1 / n, "uniform_nll": statistics.mean(uniform_nll),
           "value_mse": statistics.mean(sq), "value_mse_zero": statistics.mean(t * t for t in ts),
           "value_corr": corr, "ms_per_position": 1000 * elapsed / n}
    print(f"{adapter.format}, {n} held-out positions ({out['ms_per_position']:.2f} ms each, batch {args.batch})")
    print(f"  prior   top-1 {out['top1']:.1%}  top-3 {out['top3']:.1%}  NLL {out['nll']:.3f}   "
          f"(uniform: top-1 {out['uniform_top1']:.1%}, NLL {out['uniform_nll']:.3f})")
    print(f"  value   MSE vs search {out['value_mse']:.4f} (predicting 0: {out['value_mse_zero']:.4f})"
          + (f"  correlation {corr:.3f}" if corr is not None else ""))
    return out


# ----------------------------------------------------------------------------
# decisions

class Engine:
    """A UCI engine driven line by line, to read the hybrid's own info strings."""

    def __init__(self, cmd: Sequence[str], options: Dict[str, str]):
        self.proc = subprocess.Popen(list(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.send("uci")
        self.wait("uciok")
        for k, v in options.items():
            self.send(f"setoption name {k} value {v}")
        self.send("isready")
        self.wait("readyok")

    def send(self, line: str) -> None:
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def wait(self, token: str) -> List[str]:
        lines = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError("engine exited")
            lines.append(line.rstrip("\n"))
            if line.startswith(token):
                return lines

    def go(self, fen: str, go: str) -> List[str]:
        self.send("ucinewgame")
        self.send(f"position fen {fen}")
        self.send(go)
        return self.wait("bestmove")

    def close(self) -> None:
        try:
            self.send("quit")
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()


def parse_summary(lines: Sequence[str]) -> Dict[str, str]:
    """The fields of the hybrid's 'info string nags ...' line (name -> value), plus 'reason'."""
    line = next((l for l in reversed(lines) if l.startswith("info string nags ")), None)
    if line is None:
        return {}
    fields = line.split()
    out = {fields[i]: fields[i + 1] for i in range(len(fields) - 1)}
    if " chose " in line and "(" in line.split(" chose ", 1)[1]:
        out["reason"] = line.split(" chose ", 1)[1].split("(", 1)[1].split(") mcts_ms")[0]
    return out


def category(f: Dict[str, str]) -> str:
    reason = f.get("reason", "")
    if f.get("mcts_sims", "0") == "0" or "MCTS off" in reason:
        return "mcts off"
    if reason.startswith("mcts, verified"):
        return "override (verified)"
    if "rejected" in reason:
        return "proposal rejected"
    if "no time to verify" in reason:
        return "no time to verify"
    if "mate score" in reason:
        return "mate score"
    if f.get("mcts_top") == f.get("alpha_beta_move"):
        return "agree"
    return "differs, not proposed"  # MCTS's top move had too few visits to be proposed


def decisions(args: argparse.Namespace) -> Dict[str, Any]:
    from search_benefit import sample_positions, value_of

    from training_pipeline import ServiceProcess

    fens = sample_positions(args.openings, args.positions, args.plies, args.seed)
    rows: List[Dict[str, Any]] = []
    server_args = ["--model", args.model, "--device", args.device, "--mc-samples", str(args.mc_samples),
                   *args.server_arg]
    log = Path(args.log)
    with ServiceProcess("rpc_server.py", server_args, log) as rpc:
        options = {"UseNN": "true", "NNPort": str(rpc.port), "UseMetaLearner": "false", "MctsMinTime": "0"}
        for o in args.option:
            k, v = o.split("=", 1)
            options[k] = v
        hybrid = Engine([_exe("nags")], options)
        ref = chess.engine.SimpleEngine.popen_uci([_exe("nags_enhanced")])
        ref.configure({"Threads": 1})
        try:
            for i, fen in enumerate(fens):
                f = parse_summary(hybrid.go(fen, f"go movetime {args.movetime}"))
                row: Dict[str, Any] = {"fen": fen, "category": category(f), **{k: f.get(k) for k in (
                    "alpha_beta_move", "alpha_beta_score", "chose", "mcts_top", "visits", "mcts_sims",
                    "mcts_sims_per_s", "rpc_mean_ms", "rpc_requests", "cache_hits", "reason")}}
                ab, top = f.get("alpha_beta_move"), f.get("mcts_top")
                board = chess.Board(fen)
                # (With no simulation the summary still names the first root move: not a proposal.)
                if top and ab and top != ab and row["category"] != "mcts off" and int(f.get("visits", 0)) > 0:
                    nodes = args.reference_nodes
                    row["value_ab"] = value_of(ref, fen, chess.Move.from_uci(ab), nodes)
                    row["value_mcts"] = value_of(ref, fen, chess.Move.from_uci(top), nodes)
                    row["gain"] = row["value_mcts"] - row["value_ab"]  # > 0: the MCTS move is better
                ref_best = ref.analyse(board, chess.engine.Limit(nodes=args.reference_nodes))["pv"][0].uci()
                row["reference_best"] = ref_best
                rows.append(row)
                print(f"{i + 1}/{len(fens)} {row['category']}", file=sys.stderr, end="\r", flush=True)
        finally:
            hybrid.close()
            ref.quit()
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    return summarise_decisions(rows, args)


def summarise_decisions(rows: List[Dict[str, Any]], args: Any) -> Dict[str, Any]:
    n = len(rows)
    cats: Dict[str, int] = {}
    for r in rows:
        cats[r["category"]] = cats.get(r["category"], 0) + 1

    def clip(x: float) -> float:
        return max(-1000.0, min(1000.0, x))

    def gains(cat: str) -> List[float]:
        return [clip(r["gain"]) for r in rows if r["category"] == cat and "gain" in r]

    out: Dict[str, Any] = {"positions": n, "categories": cats}
    print(f"\n{n} positions, movetime {getattr(args, 'movetime', '?')} ms, reference "
          f"{getattr(args, 'reference_nodes', '?')} nodes")
    for c, k in sorted(cats.items(), key=lambda x: -x[1]):
        g = gains(c)
        extra = (f"  MCTS move vs alpha-beta move: mean {statistics.mean(g):+.1f} cp, better {sum(x > 10 for x in g)}, "
                 f"worse {sum(x < -10 for x in g)}" if g else "")
        print(f"  {c:<24} {k:>5} ({k / n:.0%}){extra}")
        out[f"gain_{c}"] = statistics.mean(g) if g else None
    for name, key in (("alpha-beta move", "alpha_beta_move"), ("MCTS top move", "mcts_top"), ("chosen move", "chose")):
        have = [r for r in rows if r.get(key)]
        if have:
            agree = sum(r[key] == r["reference_best"] for r in have) / len(have)
            out[f"{key}_agrees_with_reference"] = agree
            print(f"  {name:<16} = reference best move: {agree:.1%}")
    sims = [float(r["mcts_sims"]) for r in rows if r.get("mcts_sims") not in (None, "0")]
    if sims:
        rates = [float(r["mcts_sims_per_s"]) for r in rows if r.get("mcts_sims") not in (None, "0")]
        rtt = [float(r["rpc_mean_ms"]) for r in rows if r.get("rpc_mean_ms") not in (None, "0.0")]
        out.update(mean_sims=statistics.mean(sims), sims_per_s=statistics.mean(rates),
                   rpc_mean_ms=statistics.mean(rtt) if rtt else None)
        print(f"  cost: {statistics.mean(sims):.0f} simulations a move, {statistics.mean(rates):.0f}/s"
              + (f", {statistics.mean(rtt):.1f} ms a round trip" if rtt else ""))
    return out


# ----------------------------------------------------------------------------
# match

def arm(name: str, port: int, opt_options: Sequence[str]) -> Dict[str, Any]:
    nn = [f"UseNN=true", f"NNPort={port}", "UseMetaLearner=false"]
    arms = {
        "ab": (_exe("nags_enhanced"), []),
        "plain": (_exe("nags"), []),
        "hybrid": (_exe("nags"), nn),
        "hybrid-opt": (_exe("nags"), nn + list(opt_options)),
        "nn-ab": (_exe("nags"), nn + ["NNRootOrder=true", "MctsMinTime=600000"]),
    }
    if name not in arms:
        raise SystemExit(f"unknown arm {name!r} (one of {', '.join(arms)})")
    exe, options = arms[name]
    return {"engine": exe, "options": options, "name": name}


DEFAULT_OPT = ["MctsReuse=true", "MctsSmartStop=true", "MctsTTValue=6", "NNBatch=8", "NNRootOrder=true"]


def match(args: argparse.Namespace, sprt_args: Sequence[str]) -> int:
    from training_pipeline import ServiceProcess

    server_args = ["--model", args.model, "--device", args.device, "--mc-samples", str(args.mc_samples),
                   *args.server_arg]
    if args.concurrency > 1 and not any(a.startswith("--batch-wait-ms") for a in args.server_arg):
        server_args += ["--batch-wait-ms", "1"]  # several games share the server
    needs_server = any(a in ("hybrid", "hybrid-opt", "nn-ab") for a in args.arms)
    log = Path(args.log)

    def run(port: int) -> int:
        a, b = (arm(n, port, args.opt_option or DEFAULT_OPT) for n in args.arms)
        cmd = [sys.executable, str(ROOT / "tools" / "sprt.py"), "--engine", a["engine"], "--engine", b["engine"],
               "--name", a["name"], "--name", b["name"], "--concurrency", str(args.concurrency)]
        for o in a["options"]:
            cmd += ["--option1", o]
        for o in b["options"]:
            cmd += ["--option2", o]
        if args.tc:
            cmd += ["--tc", args.tc]
        if args.sprt:
            cmd += ["--sprt", str(args.sprt[0]), str(args.sprt[1])]
        if args.games:
            cmd += ["--games", str(args.games)]
        cmd += ["--openings", args.openings, *sprt_args]
        print(" ".join(cmd), flush=True)
        return subprocess.call(cmd)

    if not needs_server:
        return run(0)
    with ServiceProcess("rpc_server.py", server_args, log) as rpc:
        return run(rpc.port)


# ----------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    sprt_args: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, sprt_args = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("accuracy", help="the network's prior and value on held-out labelled positions")
    a.add_argument("--model", required=True)
    a.add_argument("--data", required=True, help="label rows (jsonl) not used in training")
    a.add_argument("--limit", type=int, default=0)
    a.add_argument("--batch", type=int, default=64)
    a.add_argument("--device", default="cpu")
    a.add_argument("--threads", type=int, default=1)
    a.add_argument("--json", help="write the summary here")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True, help="checkpoint served by rpc_server.py")
    common.add_argument("--device", default="cpu")
    common.add_argument("--mc-samples", type=int, default=1)
    common.add_argument("--server-arg", action="append", default=[], help="extra rpc_server.py argument (repeatable)")
    common.add_argument("--openings", default=str(ROOT / "tools" / "openings" / "nags_balanced.epd"))
    common.add_argument("--log", default="hybrid_eval_rpc.log", help="rpc_server.py's log")

    d = sub.add_parser("decisions", parents=[common], help="per-move decisions of the hybrid, scored by a deep search")
    d.add_argument("--positions", type=int, default=100)
    d.add_argument("--plies", type=int, default=20)
    d.add_argument("--seed", type=int, default=1)
    d.add_argument("--movetime", type=int, default=2000, help="ms per move for the hybrid")
    d.add_argument("--reference-nodes", type=int, default=2000000)
    d.add_argument("--option", action="append", default=[], help="nags UCI option NAME=VALUE (repeatable)")
    d.add_argument("--out", help="per-position rows (jsonl)")
    d.add_argument("--json", help="write the summary here")

    m = sub.add_parser("match", parents=[common], help="tools/sprt.py between two arms")
    m.add_argument("--arms", nargs=2, required=True, metavar=("ARM1", "ARM2"))
    m.add_argument("--opt-option", action="append", default=[],
                   help=f"options of the hybrid-opt arm (default: {' '.join(DEFAULT_OPT)})")
    m.add_argument("--tc")
    m.add_argument("--sprt", nargs=2, type=float)
    m.add_argument("--games", type=int)
    m.add_argument("--concurrency", type=int, default=1)

    args = ap.parse_args(argv)
    if args.cmd == "match":
        return match(args, sprt_args)
    out = accuracy(args) if args.cmd == "accuracy" else decisions(args)
    if getattr(args, "json", None):
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
