#!/usr/bin/env python3
"""Search mistakes, collected and classified by cause.

  collect  NAGS chooses a move in each sample position with N nodes; a deeper reference (another engine, or
           NAGS itself with many more nodes) scores that move and its own best move by searching each one
           alone (`searchmoves`). A position is a mistake when NAGS's move loses at least --threshold
           centipawns by the reference. Each mistake is then classified by what fixes it:

             time        NAGS with more nodes (--budgets, e.g. 4N and 16N) chooses a move within half the
                         threshold of the reference's best: the search was right, only short.
             selectivity NAGS, made to search only the reference's move with the same N nodes, scores it
                         above the move it chose: it would have preferred the move had it searched it as
                         deeply, so move ordering, pruning or reductions kept it from doing so.
             evaluation  even searched alone with N nodes the reference's move scores below NAGS's choice:
                         NAGS misjudges where it leads (the evaluation, or a horizon beyond N nodes).

           The rows (--out, one JSON line a position) also hold NAGS's shallow (depth 1, quiescence)
           scores of both moves next to the reference's, so an evaluation mistake shows how much of it the
           static evaluation already makes, and whether the reference move is a capture, check or quiet.
  summary  counts and mean loss per class, by phase (pieces) and by the reference move's kind, and writes
           the mistakes as an EPD suite (bm = the reference's move) that tools/suite.py run can replay to
           check whether a change fixes them.

  python tools/mistakes.py collect --engine build/nags_enhanced --reference /usr/games/stockfish \\
      --source games.pgn --positions 400 --nodes 40000 --ref-nodes 2000000 --workers 3 --out mistakes.jsonl
  python tools/mistakes.py summary mistakes.jsonl --epd mistakes.epd

Positions come from a PGN file's games (real decisions; the default for this measurement), a jsonl file's
'fen' fields, or an EPD file after random plies (tools/suite.py's sampling). Every search starts from a new
game (ucinewgame), so the results do not depend on the order. With another engine as the reference its
centipawns are on its own scale; --ref-scale multiplies them (tools/nnue/rescore.py compare measures the
factor; Stockfish 16 about 1.6 to NAGS's scale) before the threshold and the comparisons with NAGS's scores.
"""

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import threading
from pathlib import Path

import chess
import chess.engine

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from suite import source_positions  # noqa: E402

MATE = 30000
CLASSES = ("time", "selectivity", "evaluation")


def cp(score):
    """Centipawns from the side to move's view; mates as +-MATE."""
    return score.relative.score(mate_score=MATE)


def command(text):
    return text.split() if " " in text and not os.path.exists(text) else [text]


class Engines:
    """NAGS and the reference, one process each, owned by one worker thread."""

    def __init__(self, args):
        self.nags = chess.engine.SimpleEngine.popen_uci(command(args.engine))
        self.ref = (chess.engine.SimpleEngine.popen_uci(command(args.reference)) if args.reference else self.nags)
        for eng, opts in ((self.nags, args.option), (self.ref, args.ref_option if args.reference else None)):
            if opts:
                eng.configure(dict(o.split("=", 1) for o in opts))

    def search(self, eng, board, limit, moves=None):
        return eng.analyse(board, limit, root_moves=moves, game=object())  # a new game: no table carried over

    def close(self):
        self.nags.quit()
        if self.ref is not self.nags:
            self.ref.quit()


def examine(eng, fen, args):
    """One position's row."""
    board = chess.Board(fen)
    n = chess.engine.Limit(nodes=args.nodes)
    ref_limit = chess.engine.Limit(nodes=args.ref_nodes)
    scale = args.ref_scale if args.reference else 1.0
    mine = eng.search(eng.nags, board, n)
    best = eng.search(eng.ref, board, ref_limit)
    if not mine.get("pv") or not best.get("pv"):
        return {"fen": fen, "error": "no principal variation", "mistake": False, "loss": 0.0,
                "pieces": len(board.piece_map()), "ref_move_kind": "none"}
    move, ref_move = mine["pv"][0], best["pv"][0]
    ref_best = cp(best["score"]) * scale
    ref_mine = ref_best if move == ref_move else cp(eng.search(eng.ref, board, ref_limit, [move])["score"]) * scale
    loss = max(0.0, ref_best - ref_mine)
    pieces = len(board.piece_map())
    kind = ("capture" if board.is_capture(ref_move) else "promotion" if ref_move.promotion else
            "check" if board.gives_check(ref_move) else "quiet")
    row = {"fen": fen, "pieces": pieces, "nodes": args.nodes, "move": move.uci(), "score": cp(mine["score"]),
           "ref_move": ref_move.uci(), "ref_best": ref_best, "ref_of_move": ref_mine, "loss": loss,
           "ref_move_kind": kind, "mistake": loss >= args.threshold}
    if not row["mistake"]:
        return row
    # Time: does NAGS find a good move with more nodes?
    row["budgets"] = {}
    for k in args.budgets:
        info = eng.search(eng.nags, board, chess.engine.Limit(nodes=args.nodes * k))
        m = info["pv"][0]
        value = ref_best if m == ref_move else ref_mine if m == move else \
            cp(eng.search(eng.ref, board, ref_limit, [m])["score"]) * scale
        row["budgets"][str(k)] = {"move": m.uci(), "loss": max(0.0, ref_best - value)}
    # Selectivity: NAGS's score of the reference's move, searched alone with the same nodes.
    forced = eng.search(eng.nags, board, n, [ref_move])
    row["forced_ref_score"] = cp(forced["score"])
    # Shallow scores of both moves (depth 1: the static evaluation and a quiescence search).
    row["shallow_move"] = cp(eng.search(eng.nags, board, chess.engine.Limit(depth=1), [move])["score"])
    row["shallow_ref"] = cp(eng.search(eng.nags, board, chess.engine.Limit(depth=1), [ref_move])["score"])
    fixed = [k for k in args.budgets if row["budgets"][str(k)]["loss"] < args.threshold / 2]
    row["fixed_by_budget"] = min(fixed) if fixed else None
    if fixed:
        row["class"] = "time"
    elif row["forced_ref_score"] > row["score"]:
        row["class"] = "selectivity"
    else:
        row["class"] = "evaluation"
    return row


def collect(args):
    fens = list(source_positions(args.source, args.positions, args.plies, args.seed))
    done = set()
    if os.path.exists(args.out):  # continue a stopped run
        done = {json.loads(l)["fen"] for l in Path(args.out).read_text().splitlines() if l.strip()}
    todo = [f for f in fens if f not in done]
    print(f"{len(fens)} positions, {len(done)} already examined, {len(todo)} to go", flush=True)
    pool_engines, lock = [], threading.Lock()
    local = threading.local()

    def work(fen):
        if not hasattr(local, "eng"):
            local.eng = Engines(args)
            with lock:
                pool_engines.append(local.eng)
        return examine(local.eng, fen, args)

    try:
        with open(args.out, "a") as out, concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
            for i, fut in enumerate(concurrent.futures.as_completed([pool.submit(work, f) for f in todo]), 1):
                row = fut.result()
                out.write(json.dumps(row) + "\n")
                out.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"  {i}/{len(todo)}", flush=True)
    finally:
        for e in pool_engines:
            e.close()
    return summary(argparse.Namespace(rows=[args.out], epd=None, json=None, top=0))


def summary(args):
    rows = [json.loads(l) for p in args.rows for l in Path(p).read_text().splitlines() if l.strip()]
    if not rows:
        sys.exit("no rows")
    mistakes = [r for r in rows if r.get("mistake")]
    print(f"{len(rows)} positions, {len(mistakes)} mistakes ({len(mistakes) / len(rows):.1%}); mean loss over all "
          f"positions {statistics.mean(r['loss'] for r in rows):.1f} cp")
    report = {"positions": len(rows), "mistakes": len(mistakes), "classes": {}}
    for c in CLASSES:
        part = [r for r in mistakes if r.get("class") == c]
        entry = {"count": len(part), "share": len(part) / len(mistakes) if mistakes else 0.0,
                 "mean_loss": statistics.mean(r["loss"] for r in part) if part else None}
        if c == "time" and part:
            entry["fixed_by"] = {str(k): sum(r["fixed_by_budget"] == k for r in part)
                                 for k in sorted({r["fixed_by_budget"] for r in part})}
        if c == "evaluation" and part:
            # How much the shallow scores already prefer NAGS's move over the reference's.
            entry["shallow_preference_cp"] = statistics.median(r["shallow_move"] - r["shallow_ref"] for r in part)
        report["classes"][c] = entry
        print(f"  {c:<12} {len(part):4d} ({entry['share']:.0%})"
              + (f"  mean loss {entry['mean_loss']:.0f} cp" if part else "")
              + (f"  fixed by {entry['fixed_by']} x N" if entry.get("fixed_by") else "")
              + (f"  shallow search prefers NAGS's move by {entry['shallow_preference_cp']:.0f} cp (median)"
                 if entry.get("shallow_preference_cp") is not None else ""))
    for name, key in (("pieces", lambda r: "<=10" if r["pieces"] <= 10 else "11-20" if r["pieces"] <= 20
                       else "21-32"), ("reference move", lambda r: r["ref_move_kind"])):
        groups = {}
        for r in rows:
            groups.setdefault(key(r), []).append(r)
        cells = []
        for g, part in sorted(groups.items()):
            m = [r for r in part if r.get("mistake")]
            by = ", ".join(f"{c} {sum(r.get('class') == c for r in m)}" for c in CLASSES)
            cells.append(f"{g}: {len(m)}/{len(part)} mistakes ({by})")
            report.setdefault("by_" + name.replace(" ", "_"), {})[g] = {
                "positions": len(part), "mistakes": len(m), **{c: sum(r.get("class") == c for r in m) for c in CLASSES}}
        print(f"  by {name}: " + "; ".join(cells))
    if args.epd:
        with open(args.epd, "w") as fh:
            for r in sorted(mistakes, key=lambda r: -r["loss"]):
                board = chess.Board(r["fen"])
                bm = board.san(chess.Move.from_uci(r["ref_move"]))
                fh.write(f"{board.epd()} bm {bm}; c0 \"{r['class']}\"; ce {int(r['ref_best'])}; "
                         f"id \"loss {int(r['loss'])} played {r['move']}\";\n")
        print(f"wrote {len(mistakes)} positions to {args.epd}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=1)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect", help="examine positions and classify NAGS's mistakes")
    c.add_argument("--engine", required=True, help="NAGS (a UCI command)")
    c.add_argument("--reference", help="the reference engine (default: --engine itself, with --ref-nodes)")
    c.add_argument("--source", required=True, help="PGN, jsonl ('fen' fields) or EPD positions")
    c.add_argument("--positions", type=int, default=300)
    c.add_argument("--plies", type=int, default=0, help="random plies after EPD positions")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--nodes", type=int, default=40000, help="NAGS's nodes a decision (N)")
    c.add_argument("--budgets", type=int, nargs="+", default=[4, 16], help="multiples of N tried for 'time'")
    c.add_argument("--ref-nodes", type=int, default=2_000_000)
    c.add_argument("--ref-scale", type=float, default=1.0, help="multiply the reference's centipawns by this")
    c.add_argument("--threshold", type=float, default=50.0, help="a loss of at least this (cp) is a mistake")
    c.add_argument("--option", action="append", metavar="NAME=VALUE", help="UCI option for NAGS")
    c.add_argument("--ref-option", action="append", metavar="NAME=VALUE", help="UCI option for the reference")
    c.add_argument("--workers", type=int, default=1, help="positions examined at once (two engines each)")
    c.add_argument("--out", required=True, help="rows (jsonl); an existing file is continued")
    s = sub.add_parser("summary", help="summarise collected rows")
    s.add_argument("rows", nargs="+")
    s.add_argument("--epd", help="write the mistakes as an EPD suite")
    s.add_argument("--json", help="also write the summary here")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        collect(args)
    else:
        summary(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
