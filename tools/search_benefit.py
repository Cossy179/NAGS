#!/usr/bin/env python3
"""What does more search buy? A per-move measurement for time management.

For each sample position the engine searches with several node budgets
(N, 2N, 4N, ...). The move each budget chooses is then scored by a much
deeper reference search of the position after it, so every budget gets:

  * how often its move differs from the reference search's best move,
  * the value it loses against that best move (centipawns, reference scores),
  * and, between consecutive budgets, how often the move changed at all.

Unlike rewarding search settings with game results (one result shared by
every move of a game), this measures the decision itself, per position and
per unit of search. Per-position rows (--out) also record each budget's
score and depth, the number of legal moves and whether the side to move is
in check, so one can look for positions where extra search rarely changes
the decision.

    python tools/search_benefit.py --engine build/nags_enhanced \\
        --openings tools/openings/nags_balanced.epd --positions 200 --nodes 20000 \\
        --budgets 1 2 4 8 --reference 64 --out benefit.jsonl
"""

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

import chess
import chess.engine


def sample_positions(path, count, plies, seed):
    """Positions a few random moves after the openings in an EPD/FEN file."""
    rng = random.Random(seed)
    starts = [l.split(";")[0].strip() for l in Path(path).read_text().splitlines() if l.strip()]
    out = []
    while len(out) < count and starts:
        board = chess.Board(rng.choice(starts))
        for _ in range(rng.randint(0, plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over() and len(list(board.legal_moves)) > 1:
            out.append(board.fen())
    return out


def cp(score):
    """Centipawns from the side to move's view; mates as large values."""
    return score.pov(score.turn).score(mate_score=30000)


def search(engine, fen, nodes):
    board = chess.Board(fen)
    engine.configure({"Clear Hash": None})
    info = engine.analyse(board, chess.engine.Limit(nodes=nodes), info=chess.engine.INFO_ALL)
    return info


def value_of(engine, fen, move, nodes):
    """Reference value of playing `move`: minus the reference score after it."""
    board = chess.Board(fen)
    board.push(move)
    if board.is_game_over():
        result = board.result(claim_draw=True)
        return 0 if result == "1/2-1/2" else 30000
    info = search(engine, board.fen(), nodes)
    return -cp(info["score"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--openings", required=True, help="EPD/FEN file to sample positions from")
    ap.add_argument("--positions", type=int, default=100)
    ap.add_argument("--plies", type=int, default=20, help="random plies played after an opening (0..this)")
    ap.add_argument("--nodes", type=int, default=20000, help="smallest node budget N")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8], help="budgets as multiples of N")
    ap.add_argument("--reference", type=int, default=64, help="reference search as a multiple of N")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", help="per-position rows (jsonl)")
    args = ap.parse_args(argv)

    fens = sample_positions(args.openings, args.positions, args.plies, args.seed)
    engine = chess.engine.SimpleEngine.popen_uci([args.engine])
    rows = []
    try:
        for i, fen in enumerate(fens):
            ref = search(engine, fen, args.nodes * args.reference)
            ref_move = ref["pv"][0]
            values = {}  # move -> reference value, computed once per move
            row = {"fen": fen, "reference_move": ref_move.uci(), "legal": len(list(chess.Board(fen).legal_moves)),
                   "in_check": chess.Board(fen).is_check(), "budgets": []}
            values[ref_move] = value_of(engine, fen, ref_move, args.nodes * args.reference)
            for k in args.budgets:
                info = search(engine, fen, args.nodes * k)
                move = info["pv"][0]
                if move not in values:
                    values[move] = value_of(engine, fen, move, args.nodes * args.reference)
                row["budgets"].append({"budget": k, "move": move.uci(), "score": cp(info["score"]),
                                       "depth": info.get("depth"), "loss": max(0, values[ref_move] - values[move])})
            rows.append(row)
            print(f"{i + 1}/{len(fens)}", file=sys.stderr, end="\r", flush=True)
    finally:
        engine.quit()
    if args.out:
        with open(args.out, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

    print(f"\n{len(rows)} positions, N = {args.nodes} nodes, reference {args.reference}N")
    print(f"{'budget':>8} {'differs from ref':>17} {'mean loss cp':>13} {'median loss':>12} {'changed vs prev':>16}")
    for j, k in enumerate(args.budgets):
        differs = sum(r["budgets"][j]["move"] != r["reference_move"] for r in rows)
        losses = [min(r["budgets"][j]["loss"], 1000) for r in rows]  # cap mate-sized losses for the mean
        changed = (sum(r["budgets"][j]["move"] != r["budgets"][j - 1]["move"] for r in rows) if j else None)
        print(f"{str(k) + 'N':>8} {differs / len(rows):17.1%} {statistics.mean(losses):13.1f} "
              f"{statistics.median(losses):12.1f} {('-' if changed is None else f'{changed / len(rows):.1%}'):>16}")


if __name__ == "__main__":
    main()
