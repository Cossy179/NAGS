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
               ab-2t       nags_enhanced with two threads: the same cores as a hybrid's two arms
               plain       nags without the network (must equal ab)
               hybrid      nags with the network, default hybrid settings
               hybrid-opt  nags with the network and the hybrid search options (--opt-option)
               nn-ab       nags with the network ordering the alpha-beta root moves only (no MCTS)

  python tools/hybrid_eval.py accuracy --model models/gnn_v3_distill_X.pth --data heldout.jsonl
  python tools/hybrid_eval.py decisions --model models/gnn_v3_distill_X.pth --positions 200 --movetime 2000
  python tools/hybrid_eval.py match --model models/gnn_v3_distill_X.pth --arms hybrid ab --tc 10+0.1 \\
      --concurrency 2 --sprt 0 10 -- --pgnout hybrid.pgn

  gauntlet   every arm against one baseline (default ab) with one rpc_server.py for all:
             the same openings in the same order (sprt.py --seed), each played with both
             colours, the same time control and hash for every engine. Results per arm with
             95% intervals; an interrupted gauntlet continues where it stopped.

  python tools/hybrid_eval.py gauntlet --model models/gnn_v3.pth --device cuda \
      --arms hybrid hybrid-opt nn-ab ab-2t --tc 60+0.6 --games 400 --concurrency 2 --out-dir gauntlet

Everything after `--` goes to tools/sprt.py as it is.
"""

from __future__ import annotations

import argparse
import json
import math
import re
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


def warm_server(port: int, fens: Sequence[str], batches: Sequence[int] = (1,), timeout: float = 900.0) -> None:
    """Evaluate a batch of each size once, with a long timeout, before an engine connects. A server's
    first inference (model load, an ONNX export, a TensorRT engine build) can take much longer than a
    search's request timeout (a tenth of its hard limit), and after a request that times out the engine
    leaves the server alone for a minute: the searches in that minute would run without MCTS."""
    import socket

    boards = [chess.Board(f) for f in fens] or [chess.Board()]
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as s, s.makefile("rwb") as io:
        for n in sorted(set(batches)):
            sample = [boards[i % len(boards)] for i in range(max(1, n))]
            io.write((json.dumps({"fens": [b.fen() for b in sample],
                                  "moves": [[m.uci() for m in b.legal_moves] for b in sample]}) + "\n").encode())
            io.flush()
            reply = json.loads(io.readline() or "{}")
            if "results" not in reply:
                raise RuntimeError(f"server warm-up failed: {reply.get('error', 'no reply')}")


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
        warm_server(rpc.port, fens)
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
        "ab-2t": (_exe("nags_enhanced"), ["Threads=2"]),
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
        warm_server(rpc.port, [chess.STARTING_FEN], (1, 8, 32))
        return run(rpc.port)


# ----------------------------------------------------------------------------
# gauntlet

MCTS_ARMS = ("hybrid", "hybrid-opt")


def soft_ms(tc: str) -> float:
    """Roughly the soft limit a move gets at the start of a game (time / 40 + increment)."""
    base, _, inc = tc.partition("+")
    return (float(base) / 40 + float(inc or 0)) * 1000


def read_status(log: Path) -> Optional[Dict[str, Any]]:
    """The last status line of a sprt.py log as numbers, and whether the match finished."""
    if not log.exists():
        return None
    text = log.read_text()
    lines = [l for l in text.splitlines() if l.startswith("Games ")]
    if not lines:
        return None
    m = re.search(r"Games (\d+): \+(\d+) =(\d+) -(\d+) \| penta \[([\d, ]+)\] \| Elo ([+-][\d.]+|nan) ± ([\d.]+|inf)",
                  lines[-1])
    if not m:
        return None
    return {"games": int(m.group(1)), "wdl": [int(m.group(2)), int(m.group(3)), int(m.group(4))],
            "penta": [int(x) for x in m.group(5).split(",")], "elo": float(m.group(6)), "error": float(m.group(7)),
            "finished": "\nEndings: " in "\n" + text or "accepted" in text or "inconclusive" in text}


def gauntlet(args: argparse.Namespace, sprt_args: Sequence[str]) -> int:
    from training_pipeline import ServiceProcess

    if args.baseline in args.arms:
        raise SystemExit("the baseline is not also an arm")
    for name in args.arms:
        arm(name, 0, [])  # unknown names fail before anything starts
    if args.tc and any(a in MCTS_ARMS for a in args.arms):
        limit = args.mcts_min_time if args.mcts_min_time is not None else 1000
        if soft_ms(args.tc) < limit:
            print(f"WARNING: at {args.tc} a move gets about {soft_ms(args.tc):.0f} ms, under MctsMinTime {limit}: "
                  "the MCTS arms would mostly play pure alpha-beta (give a longer --tc or --mcts-min-time)",
                  file=sys.stderr, flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    server_args = ["--model", args.model, "--device", args.device, "--mc-samples", str(args.mc_samples),
                   *args.server_arg]
    if args.concurrency > 1 and not any(a.startswith("--batch-wait-ms") for a in args.server_arg):
        server_args += ["--batch-wait-ms", "1"]
    needs_server = any(a in ("hybrid", "hybrid-opt", "nn-ab") for a in [args.baseline, *args.arms])

    def play(port: int) -> List[Dict[str, Any]]:
        rows = []
        base = arm(args.baseline, port, args.opt_option or DEFAULT_OPT)
        for name in args.arms:
            a = arm(name, port, args.opt_option or DEFAULT_OPT)
            if name in MCTS_ARMS and args.mcts_min_time is not None:
                a["options"] = a["options"] + [f"MctsMinTime={args.mcts_min_time}"]
            log = out_dir / f"{name}_vs_{args.baseline}.log"
            status = read_status(log)
            if not (status and status["finished"]):
                cmd = [sys.executable, str(ROOT / "tools" / "sprt.py"), "--engine", a["engine"], "--engine",
                       base["engine"], "--name", name, "--name", args.baseline, "--concurrency", str(args.concurrency),
                       "--seed", str(args.seed), "--openings", args.openings, "--option", f"Hash={args.hash}",
                       "--pgnout", str(out_dir / f"{name}_vs_{args.baseline}.pgn")]
                for o in a["options"]:
                    cmd += ["--option1", o]
                for o in base["options"]:
                    cmd += ["--option2", o]
                if args.tc:
                    cmd += ["--tc", args.tc]
                if args.sprt:
                    cmd += ["--sprt", str(args.sprt[0]), str(args.sprt[1])]
                if args.games:
                    cmd += ["--games", str(args.games)]
                if status or (log.exists() and "\nFingerprint: " in "\n" + log.read_text(errors="replace")):
                    cmd += ["--resume", str(log)]  # (no status line yet: resumed at zero games, its PGN kept)
                cmd += list(sprt_args)
                print(f"== {name} vs {args.baseline}" + (f" (resuming after {status['games']} games)" if status else ""),
                      flush=True)
                with open(log, "a") as f:
                    rc = subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT)
                status = read_status(log)
                if rc == 3 or status is None:
                    raise SystemExit(f"{name} vs {args.baseline} failed: see {log}")
            rows.append({"arm": name, "baseline": args.baseline, "options": a["options"], **status})
            print(f"   {name}: {status['elo']:+.1f} ± {status['error']:.1f} Elo over {status['games']} games", flush=True)
        return rows

    if needs_server:
        with ServiceProcess("rpc_server.py", server_args, out_dir / "rpc_server.log") as rpc:
            warm_server(rpc.port, [chess.STARTING_FEN], (1, 8, 32))
            rows = play(rpc.port)
    else:
        rows = play(0)
    print(f"\n{'arm':12} {'games':>6} {'W':>5} {'D':>5} {'L':>5} {'Elo vs ' + args.baseline:>18}")
    for r in rows:
        print(f"{r['arm']:12} {r['games']:6d} {r['wdl'][0]:5d} {r['wdl'][1]:5d} {r['wdl'][2]:5d} "
              f"{r['elo']:+10.1f} ± {r['error']:5.1f}")
    (out_dir / "results.json").write_text(json.dumps({"baseline": args.baseline, "tc": args.tc, "seed": args.seed,
                                                     "results": rows}, indent=2))
    return 0


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

    g = sub.add_parser("gauntlet", parents=[common], help="every arm against one baseline, same openings")
    g.add_argument("--arms", nargs="+", required=True)
    g.add_argument("--baseline", default="ab")
    g.add_argument("--opt-option", action="append", default=[],
                   help=f"options of the hybrid-opt arm (default: {' '.join(DEFAULT_OPT)})")
    g.add_argument("--mcts-min-time", type=int, help="MctsMinTime (ms) for the MCTS arms (engine default: 1000)")
    g.add_argument("--tc")
    g.add_argument("--sprt", nargs=2, type=float)
    g.add_argument("--games", type=int)
    g.add_argument("--concurrency", type=int, default=1)
    g.add_argument("--seed", type=int, default=1, help="opening order, the same for every arm")
    g.add_argument("--hash", type=int, default=64, help="MB for every engine")
    g.add_argument("--out-dir", default="gauntlet", help="logs, PGNs and results.json")

    args = ap.parse_args(argv)
    if args.cmd == "match":
        return match(args, sprt_args)
    if args.cmd == "gauntlet":
        return gauntlet(args, sprt_args)
    out = accuracy(args) if args.cmd == "accuracy" else decisions(args)
    if getattr(args, "json", None):
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
