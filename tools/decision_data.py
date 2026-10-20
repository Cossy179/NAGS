#!/usr/bin/env python3
"""Per-decision measurements for time management (docs/TESTING.md, "Decisions").

The meta-learner was trained on game results credited to every move of a game, a signal that
says almost nothing about one decision (docs/ARCHITECTURE.md). This measures decisions directly:

  collect  for sample positions, the engine's search at node budgets N, 2N, 4N, ..., with what
           the search knows after its last iteration at each budget (best-move changes and
           stability, score and its swings, depth, legal moves, check, material), and a deep
           reference value of every move chosen: the value each budget's move loses.
  fit      a logistic model of P(twice the budget changes the move | the features at a budget),
           with held-out log-loss and AUC against the base rate; writes its coefficients.
  policy   offline evaluation of time allocation. Every budget was played in every position, so
           the outcome of any allocation is known (a full-information contextual bandit): a rule
           spends twice the base budget on average, giving 4N where it expects more search to
           matter and N elsewhere, and is scored by the value its moves lose, next to spending
           2N everywhere and to the best allocation in hindsight.

  python tools/decision_data.py collect --engine build/nags_enhanced --positions 300 --nodes 20000 \\
      --budgets 1 2 4 8 --reference 64 --out decisions.jsonl
  python tools/decision_data.py fit --data decisions.jsonl --json tm_model.json
  python tools/decision_data.py policy --data decisions.jsonl --model tm_model.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import chess
import chess.engine

sys.path.insert(0, str(Path(__file__).resolve().parent))
from search_benefit import cp, sample_positions, value_of  # noqa: E402

FEATURES = ("log_budget", "depth", "best_changes", "stable_iterations", "score_abs", "score_drop", "score_swing",
            "legal_moves", "in_check", "pieces")
CAP = 1000  # cp: a move's loss is counted up to this (mate-sized losses would dominate any mean)
SCORE_CAP = 2000  # cp: search scores, mates included, as features


# ----------------------------------------------------------------------------
# collect

def search_features(board: chess.Board, iterations: List[Dict[str, Any]], budget_factor: float) -> Dict[str, float]:
    """What the search knows at its end: per-iteration best moves and scores (last one = its decision)."""
    moves = [it["move"] for it in iterations]
    # Mate scores (about 30000) are capped like score_abs: a swing from a mate score to a normal one
    # would otherwise be some 800 standard deviations of the usual swings, and the model certain.
    scores = [max(-SCORE_CAP, min(SCORE_CAP, it["score"])) for it in iterations]
    changes = sum(1 for a, b in zip(moves, moves[1:]) if a != b)
    stable = 0
    for a, b in zip(reversed(moves[:-1]), reversed(moves[1:])):
        if a != b:
            break
        stable += 1
    last3 = scores[-3:] or [0]
    return {"log_budget": math.log2(budget_factor), "depth": iterations[-1]["depth"] if iterations else 0,
            "best_changes": changes, "stable_iterations": stable, "score_abs": abs(scores[-1]) if scores else 0,
            "score_drop": max(0, scores[-2] - scores[-1]) if len(scores) > 1 else 0,
            "score_swing": max(last3) - min(last3), "legal_moves": board.legal_moves.count(),
            "in_check": float(board.is_check()), "pieces": len(board.piece_map())}


def run_search(engine, fen: str, nodes: int) -> Tuple[Optional[chess.Move], List[Dict[str, Any]], int]:
    board = chess.Board(fen)
    if "Clear Hash" in engine.options:
        engine.configure({"Clear Hash": None})
    iterations: Dict[int, Dict[str, Any]] = {}
    used = 0
    with engine.analysis(board, chess.engine.Limit(nodes=nodes)) as analysis:
        for info in analysis:
            if "pv" in info and info["pv"] and "score" in info and "depth" in info:
                iterations[info["depth"]] = {"depth": info["depth"], "move": info["pv"][0].uci(),
                                             "score": cp(info["score"])}
            used = max(used, info.get("nodes", 0))
        best = analysis.wait().move
    its = [iterations[d] for d in sorted(iterations)]
    return best, its, used


def collect(args: argparse.Namespace) -> int:
    fens = sample_positions(args.openings, args.positions, args.plies, args.seed)
    engine = chess.engine.SimpleEngine.popen_uci([args.engine])
    engine.configure({k: v for k, v in (("Threads", 1), ("Hash", args.hash)) if k in engine.options})
    ref_nodes = args.nodes * args.reference
    with open(args.out, "w") as out:
        try:
            for i, fen in enumerate(fens):
                board = chess.Board(fen)
                values: Dict[str, int] = {}
                ref = engine.analyse(board, chess.engine.Limit(nodes=ref_nodes))
                ref_move = ref["pv"][0]
                values[ref_move.uci()] = value_of(engine, fen, ref_move, ref_nodes)
                row: Dict[str, Any] = {"fen": fen, "reference_move": ref_move.uci(), "budgets": []}
                for k in args.budgets:
                    move, its, used = run_search(engine, fen, args.nodes * k)
                    if move is None or not its:
                        break
                    if move.uci() not in values:
                        values[move.uci()] = value_of(engine, fen, move, ref_nodes)
                    loss = max(0, values[ref_move.uci()] - values[move.uci()])
                    row["budgets"].append({"budget": k, "nodes": used, "move": move.uci(), "loss": min(loss, CAP),
                                           "features": search_features(board, its, k)})
                if len(row["budgets"]) == len(args.budgets):
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                print(f"{i + 1}/{len(fens)}", file=sys.stderr, end="\r", flush=True)
        finally:
            engine.quit()
    return 0


# ----------------------------------------------------------------------------
# fit

def read(path: str) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def examples(rows: Sequence[Dict[str, Any]]) -> Tuple[List[List[float]], List[int], List[int]]:
    """(features at budget b_j, did b_{j+1} choose another move, row index) for each consecutive pair."""
    X, y, idx = [], [], []
    for r_i, r in enumerate(rows):
        bs = r["budgets"]
        for a, b in zip(bs, bs[1:]):
            X.append([float(a["features"][f]) for f in FEATURES])
            y.append(int(a["move"] != b["move"]))
            idx.append(r_i)
    return X, y, idx


def standardise(X, mean=None, std=None):
    n, d = len(X), len(X[0])
    if mean is None:
        mean = [sum(x[j] for x in X) / n for j in range(d)]
        std = [math.sqrt(sum((x[j] - mean[j]) ** 2 for x in X) / n) or 1.0 for j in range(d)]
    return [[(x[j] - mean[j]) / std[j] for j in range(d)] for x in X], mean, std


def logistic_fit(X: List[List[float]], y: List[int], l2: float = 1.0, iters: int = 50) -> List[float]:
    """Newton's method on the L2-regularised log-likelihood; w[0] is the intercept."""
    import numpy as np

    A = np.hstack([np.ones((len(X), 1)), np.array(X)])
    t = np.array(y, dtype=float)
    w = np.zeros(A.shape[1])
    reg = np.eye(A.shape[1]) * l2
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-A @ w))
        g = A.T @ (p - t) + reg @ w
        H = (A * (p * (1 - p))[:, None]).T @ A + reg
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w.tolist()


def predict(w: Sequence[float], x: Sequence[float]) -> float:
    z = w[0] + sum(a * b for a, b in zip(w[1:], x))
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, z))))


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def logloss(ps: Sequence[float], ys: Sequence[int]) -> float:
    return -sum(math.log(max(1e-12, p if y else 1 - p)) for p, y in zip(ps, ys)) / len(ys)


def fit(args: argparse.Namespace) -> Dict[str, Any]:
    rows = read(args.data)
    rng = random.Random(args.seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    test_rows = set(order[:max(1, int(len(rows) * args.holdout))])
    X, y, idx = examples(rows)
    train = [i for i, r in enumerate(idx) if r not in test_rows]
    test = [i for i, r in enumerate(idx) if r in test_rows]
    Xs, mean, std = standardise([X[i] for i in train])
    w = logistic_fit(Xs, [y[i] for i in train], l2=args.l2)
    Xt, _, _ = standardise([X[i] for i in test], mean, std)
    yt = [y[i] for i in test]
    pt = [predict(w, x) for x in Xt]
    base = sum(y[i] for i in train) / len(train)
    out = {"features": list(FEATURES), "mean": mean, "std": std, "weights": w, "train_examples": len(train),
           "test_examples": len(test), "base_rate": base, "test_logloss": logloss(pt, yt),
           "test_logloss_base": logloss([base] * len(yt), yt), "test_auc": auc(pt, yt)}
    print(f"{len(rows)} positions, {len(train)} training and {len(test)} held-out decisions "
          f"(moves change with twice the budget in {base:.1%})")
    print(f"  held-out log-loss {out['test_logloss']:.4f} (base rate alone {out['test_logloss_base']:.4f}), "
          f"AUC {out['test_auc']:.3f}")
    print("  weights (per standard deviation): " + ", ".join(f"{f} {c:+.2f}" for f, c in zip(FEATURES, w[1:])))
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return out


# ----------------------------------------------------------------------------
# policy

def policy(args: argparse.Namespace) -> Dict[str, Any]:
    """Each position gets N or 4N (or 2N for 'uniform'); the same nodes on average as 2N everywhere."""
    rows = read(args.data)
    by = lambda r, k: next(b for b in r["budgets"] if b["budget"] == k)  # noqa: E731
    need = {1, 2, 4}
    rows = [r for r in rows if need <= {b["budget"] for b in r["budgets"]}]
    if not rows:
        raise SystemExit("needs budgets 1, 2 and 4 in every row")
    model = json.loads(Path(args.model).read_text()) if args.model else None

    def benefit(r) -> float:
        f = by(r, 1)["features"]
        x = [float(f[k]) for k in FEATURES]
        if model:
            xs = [(v - m) / s for v, m, s in zip(x, model["mean"], model["std"])]
            return predict(model["weights"], xs)
        # Without a model: the stability rule (fewer stable iterations, more search).
        return -f["stable_iterations"] + 0.01 * f["score_swing"]

    def spend(scores: Sequence[float]) -> float:
        # A third of the positions at 4N and two thirds at N cost 2N on average.
        ranked = sorted(range(len(rows)), key=lambda i: -scores[i])
        extra = set(ranked[:len(rows) // 3])
        return sum(by(rows[i], 4 if i in extra else 1)["loss"] for i in range(len(rows))) / len(rows)

    uniform = sum(by(r, 2)["loss"] for r in rows) / len(rows)
    rule = spend([benefit(r) for r in rows])
    oracle = spend([by(r, 1)["loss"] - by(r, 4)["loss"] for r in rows])
    random_rule = statistics_mean([spend([random.Random(s).random() for _ in rows]) for s in range(20)])
    out = {"positions": len(rows), "uniform_2N": uniform, "rule": rule, "random": random_rule, "oracle": oracle,
           "rule_name": "model" if model else "stability"}
    print(f"{len(rows)} positions, mean loss in cp at the same total nodes (2N on average):")
    print(f"  2N everywhere            {uniform:7.1f}")
    print(f"  N or 4N, chosen at random {random_rule:6.1f}")
    print(f"  N or 4N by the {out['rule_name']:<9} {rule:7.1f}")
    print(f"  N or 4N in hindsight      {oracle:6.1f} (the best any rule could do)")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return out


def statistics_mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--engine", required=True)
    c.add_argument("--openings", default=str(Path(__file__).with_name("openings") / "nags_balanced.epd"))
    c.add_argument("--positions", type=int, default=200)
    c.add_argument("--plies", type=int, default=20)
    c.add_argument("--nodes", type=int, default=20000)
    c.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8])
    c.add_argument("--reference", type=int, default=64, help="reference search as a multiple of --nodes")
    c.add_argument("--hash", type=int, default=64)
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--out", required=True)
    f = sub.add_parser("fit")
    f.add_argument("--data", required=True)
    f.add_argument("--holdout", type=float, default=0.25)
    f.add_argument("--l2", type=float, default=1.0)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--json")
    p = sub.add_parser("policy")
    p.add_argument("--data", required=True)
    p.add_argument("--model", help="a 'fit' --json (default: the stability rule)")
    p.add_argument("--json")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        return collect(args)
    (fit if args.cmd == "fit" else policy)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
