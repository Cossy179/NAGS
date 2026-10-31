#!/usr/bin/env python3
"""Search mistakes, collected and classified by cause.

  collect  NAGS chooses a move in each sample position with N nodes; a deeper reference (another engine, or
           NAGS itself with many more nodes) scores that move and its own best move, each by searching the
           position after it (minus that search's score; NAGS ignores `searchmoves`, so no engine is asked
           to restrict its root moves). A position is a mistake when NAGS's move loses at least --threshold
           centipawns by the reference. Each mistake is then classified by what fixes it:

             time        NAGS with more nodes (--budgets, e.g. 4N and 16N) chooses a move within half the
                         threshold of the reference's best: the search was right, only short.
             selectivity NAGS, searching the position after the reference's move with N nodes and the
                         position after its own move with N nodes, values the reference's move higher: it
                         would have preferred it had it searched it as deeply, so move ordering, pruning or
                         reductions at the root kept it from doing so.
             evaluation  searched that way the reference's move still scores below NAGS's choice: NAGS
                         misjudges where it leads (the evaluation, or a horizon beyond N nodes).

           The rows (--out, one JSON line a position) also hold NAGS's shallow (depth 1, quiescence)
           scores of both moves next to the reference's, so an evaluation mistake shows how much of it the
           static evaluation already makes, and whether the reference move is a capture, check or quiet.
  reclassify  recomputes NAGS's side of each mistake (selectivity or evaluation, shallow scores) with the
           current method and --engine, keeping the reference's values: for rows collected before the method
           changed.
  summary  counts and loss per class (mean with each loss capped at 1000 cp, so mates do not dominate, and
           median), by phase (pieces) and by the reference move's kind, and writes
           the mistakes as an EPD suite (bm = the reference's move) that tools/suite.py run can replay to
           check whether a change fixes them.

  python tools/mistakes.py collect --engine build/nags_enhanced --reference /usr/games/stockfish \\
      --source games.pgn --positions 400 --nodes 40000 --ref-nodes 2000000 --workers 3 --out mistakes.jsonl
  python tools/mistakes.py summary mistakes.jsonl --epd mistakes.epd

When the mistakes go into training data (a book for nags_datagen --book), collect them with NAGS itself as the
reference (no --reference, deeper --ref-nodes): nothing NAGS trains on may come from another engine
(docs/STRENGTH_PLAN.md). Another engine as the reference is for diagnosis only.

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

    def value(self, eng, board, move, limit):
        """The value of playing `move` for the side to move: minus a search of the position after it."""
        child = board.copy()
        child.push(move)
        if child.is_checkmate():
            return MATE
        if child.is_game_over(claim_draw=True):
            return 0
        return -cp(self.search(eng, child, limit)["score"])

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
    values = {}  # the reference's value of each move asked about

    def ref_value(m):
        if m not in values:
            values[m] = eng.value(eng.ref, board, m, ref_limit) * scale
        return values[m]

    ref_best = ref_value(ref_move)
    ref_mine = ref_value(move)
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
        row["budgets"][str(k)] = {"move": m.uci(), "loss": max(0.0, ref_best - ref_value(m))}
    nags_side(eng, board, move, ref_move, row, args)
    return row


def nags_side(eng, board, move, ref_move, row, args):
    """Selectivity or evaluation, and the shallow scores: NAGS's values of both moves, each from a search of
    the position after it with N nodes (and with depth 1: the evaluation and a quiescence search)."""
    n = chess.engine.Limit(nodes=args.nodes)
    row["nags_value_move"] = eng.value(eng.nags, board, move, n)
    row["nags_value_ref"] = eng.value(eng.nags, board, ref_move, n)
    row["shallow_move"] = eng.value(eng.nags, board, move, chess.engine.Limit(depth=1))
    row["shallow_ref"] = eng.value(eng.nags, board, ref_move, chess.engine.Limit(depth=1))
    row.pop("forced_ref_score", None)  # the earlier method's (searchmoves, which NAGS ignores)
    fixed = [k for k in args.budgets if row["budgets"][str(k)]["loss"] < args.threshold / 2]
    row["fixed_by_budget"] = min(fixed) if fixed else None
    if fixed:
        row["class"] = "time"
    elif row["nags_value_ref"] > row["nags_value_move"]:
        row["class"] = "selectivity"
    else:
        row["class"] = "evaluation"


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


def reclassify(args):
    rows = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    eng = Engines(argparse.Namespace(engine=args.engine, reference=None, option=args.option, ref_option=None))
    try:
        for r in rows:
            if r.get("mistake"):
                budgets = argparse.Namespace(nodes=r["nodes"], budgets=[int(k) for k in r["budgets"]],
                                             threshold=args.threshold)
                board = chess.Board(r["fen"])
                nags_side(eng, board, chess.Move.from_uci(r["move"]), chess.Move.from_uci(r["ref_move"]), r, budgets)
    finally:
        eng.close()
    with open(args.out, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return summary(argparse.Namespace(rows=[args.out], epd=None, json=None))


CAP = 1000  # cp: a loss counted in means at most this (mate scores would dominate them)


def summary(args):
    rows = [json.loads(l) for p in args.rows for l in Path(p).read_text().splitlines() if l.strip()]
    if not rows:
        sys.exit("no rows")
    mistakes = [r for r in rows if r.get("mistake")]
    print(f"{len(rows)} positions, {len(mistakes)} mistakes ({len(mistakes) / len(rows):.1%}); mean loss over all "
          f"positions {statistics.mean(min(CAP, r['loss']) for r in rows):.1f} cp (each capped at {CAP})")
    report = {"positions": len(rows), "mistakes": len(mistakes), "classes": {}}
    for c in CLASSES:
        part = [r for r in mistakes if r.get("class") == c]
        entry = {"count": len(part), "share": len(part) / len(mistakes) if mistakes else 0.0,
                 "mean_loss": statistics.mean(min(CAP, r["loss"]) for r in part) if part else None,
                 "median_loss": statistics.median(r["loss"] for r in part) if part else None}
        if c == "time" and part:
            entry["fixed_by"] = {str(k): sum(r["fixed_by_budget"] == k for r in part)
                                 for k in sorted({r["fixed_by_budget"] for r in part})}
        if c == "evaluation" and part:
            # How much the shallow scores already prefer NAGS's move over the reference's.
            entry["shallow_preference_cp"] = statistics.median(r["shallow_move"] - r["shallow_ref"] for r in part)
        report["classes"][c] = entry
        print(f"  {c:<12} {len(part):4d} ({entry['share']:.0%})"
              + (f"  loss mean {entry['mean_loss']:.0f} cp (capped), median {entry['median_loss']:.0f}" if part else "")
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
    rc = sub.add_parser("reclassify", help="recompute NAGS's side of collected mistakes with the current method")
    rc.add_argument("rows")
    rc.add_argument("--engine", required=True)
    rc.add_argument("--option", action="append", metavar="NAME=VALUE")
    rc.add_argument("--threshold", type=float, default=50.0, help="as collect's (for the time class)")
    rc.add_argument("--out", required=True)
    c.add_argument("--out", required=True, help="rows (jsonl); an existing file is continued")
    s = sub.add_parser("summary", help="summarise collected rows")
    s.add_argument("rows", nargs="+")
    s.add_argument("--epd", help="write the mistakes as an EPD suite")
    s.add_argument("--json", help="also write the summary here")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        collect(args)
    elif args.cmd == "reclassify":
        reclassify(args)
    else:
        summary(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
