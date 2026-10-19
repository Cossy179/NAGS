#!/usr/bin/env python3
"""Test suites: positions with one clearly best move, by category (docs/TESTING.md, "Suites").

  make  analyses positions (an EPD file after random plies, a PGN file's games or a jsonl file's
        'fen' fields) with a strong reference engine (by default Stockfish 16, MultiPV 2)
        and keeps those whose best move beats every other by at least --min-gap centipawns:
        'tactical' (the best move captures, promotes or checks, or mates), 'endgame' (at most
        --endgame-pieces pieces besides the kings) or 'positional' (the rest). Writes EPD lines
        with bm (SAN), c0 (category), ce (the reference score) and id.
  run   lets an engine choose a move in each position at fixed nodes or move time, and reports
        the share solved per category; with --baseline (an earlier run's --json) it lists the
        positions solved then and not now.

  python tools/suite.py make --source tools/openings/nags_balanced.epd --plies 30 --count 300 \\
      --nodes 2000000 --out tools/suites/nags_sf16.epd
  python tools/suite.py run --suite tools/suites/nags_sf16.epd --engine build/nags_enhanced --nodes 200000 \\
      --json suite_new.json --baseline suite_main.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import chess
import chess.engine
import chess.pgn

CATEGORIES = ("tactical", "positional", "endgame")


def source_positions(path: str, limit: int, plies: int, seed: int) -> Iterator[str]:
    """Up to `limit` FENs, lazily: the 'fen' fields of a jsonl file, or the positions of a PGN file's
    games, in random order; or an EPD/FEN file's positions after 0..plies random moves."""
    rng = random.Random(seed)
    if path.endswith(".jsonl") or path.endswith(".pgn"):
        if path.endswith(".pgn"):
            fens = []
            with open(path) as f:
                while (game := chess.pgn.read_game(f)) is not None:
                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                        if not board.is_game_over() and board.legal_moves.count() > 1:
                            fens.append(board.fen())
        else:
            fens = [json.loads(l)["fen"] for l in Path(path).read_text().splitlines() if l.strip()]
        rng.shuffle(fens)
        yield from fens[:limit]
        return
    starts = [" ".join(l.split(";")[0].split()[:4]) + " 0 1" for l in Path(path).read_text().splitlines() if l.strip()]
    made = 0
    while made < limit and starts:
        board = chess.Board(rng.choice(starts))
        for _ in range(rng.randint(0, plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over() and board.legal_moves.count() > 1:
            made += 1
            yield board.fen()


def category(board: chess.Board, move: chess.Move, score: chess.engine.PovScore, endgame_pieces: int) -> str:
    pieces = len(board.piece_map()) - 2
    if pieces <= endgame_pieces:
        return "endgame"
    if board.is_capture(move) or move.promotion or board.gives_check(move) or score.relative.is_mate():
        return "tactical"
    return "positional"


def make(args: argparse.Namespace) -> int:
    engine = chess.engine.SimpleEngine.popen_uci([args.reference])
    engine.configure({k: v for k, v in (("Threads", args.threads), ("Hash", args.hash)) if k in engine.options})
    limit = chess.engine.Limit(nodes=args.nodes) if args.nodes else chess.engine.Limit(depth=args.depth)
    kept: Dict[str, List[str]] = {c: [] for c in CATEGORIES}
    per_cat = {c: args.count // len(CATEGORIES) for c in CATEGORIES} if args.balanced else None
    seen = set()
    # Kept positions are appended as they are found (an interrupted run keeps them) and the file is
    # rewritten, by category, at the end.
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    partial.write_text("")
    try:
        for fen in source_positions(args.source, args.max_candidates, args.plies, args.seed):
            if sum(map(len, kept.values())) >= args.count:
                break
            board = chess.Board(fen)
            key = board.epd()
            if key in seen:
                continue
            seen.add(key)
            if "Clear Hash" in engine.options:
                engine.configure({"Clear Hash": None})
            infos = engine.analyse(board, limit, multipv=2)
            if len(infos) < 2 or "pv" not in infos[0] or "pv" not in infos[1]:
                continue
            best, second = infos[0]["score"].relative, infos[1]["score"].relative
            b, s = best.score(mate_score=100000), second.score(mate_score=100000)
            if b - s < args.min_gap or abs(b) > args.max_score and not best.is_mate():
                continue
            move = infos[0]["pv"][0]
            cat = category(board, move, infos[0]["score"], args.endgame_pieces)
            if per_cat and len(kept[cat]) >= per_cat[cat]:
                continue
            ce = best.score(mate_score=100000)
            line = f'{key} bm {board.san(move)}; c0 "{cat}"; ce {ce}; id "{args.prefix}.{len(seen)}";'
            kept[cat].append(line)
            with partial.open("a") as f:
                f.write(line + "\n")
            print(f"{sum(map(len, kept.values()))}/{args.count} {cat} ({len(seen)} analysed)", file=sys.stderr,
                  flush=True)
    finally:
        engine.quit()
    lines = [l for c in CATEGORIES for l in kept[c]]
    out.write_text("\n".join(lines) + "\n")
    partial.unlink(missing_ok=True)
    print(f"\n{len(lines)} positions ({', '.join(f'{c} {len(kept[c])}' for c in CATEGORIES)}) from {len(seen)} "
          f"analysed -> {args.out}")
    return 0


def read_suite(path: str) -> List[Dict[str, Any]]:
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        board, ops = chess.Board.from_epd(line)
        bm = ops.get("bm") or []
        out.append({"id": ops.get("id", str(len(out))), "fen": board.fen(), "bm": [m.uci() for m in bm],
                    "category": ops.get("c0", "positional")})
    return out


def run(args: argparse.Namespace) -> int:
    positions = read_suite(args.suite)
    engine = chess.engine.SimpleEngine.popen_uci(args.engine.split())
    options = {k: v for k, v in (o.split("=", 1) for o in args.option)}
    if options:
        engine.configure(options)
    limit = chess.engine.Limit(nodes=args.nodes) if args.nodes else chess.engine.Limit(time=args.movetime)
    results = {}
    try:
        for p in positions:
            if "Clear Hash" in engine.options:
                engine.configure({"Clear Hash": None})
            move = engine.play(chess.Board(p["fen"]), limit).move
            results[p["id"]] = {"category": p["category"], "move": move.uci() if move else None,
                                "solved": bool(move and move.uci() in p["bm"])}
    finally:
        engine.quit()
    summary: Dict[str, Any] = {"suite": args.suite, "engine": args.engine, "options": options,
                               "limit": f"nodes {args.nodes}" if args.nodes else f"movetime {args.movetime}",
                               "positions": results}
    print(f"{args.engine} on {args.suite} ({summary['limit']}):")
    for c in CATEGORIES + ("all",):
        rows = [r for r in results.values() if c == "all" or r["category"] == c]
        if rows:
            solved = sum(r["solved"] for r in rows)
            summary[c] = solved / len(rows)
            print(f"  {c:<10} {solved:>4}/{len(rows):<4} {solved / len(rows):6.1%}")
    if args.baseline:
        base = json.loads(Path(args.baseline).read_text())["positions"]
        lost = sorted(i for i, r in results.items() if not r["solved"] and base.get(i, {}).get("solved"))
        gained = sorted(i for i, r in results.items() if r["solved"] and not base.get(i, {}).get("solved", True))
        summary["regressions"], summary["gains"] = lost, gained
        print(f"  against the baseline: {len(gained)} newly solved, {len(lost)} no longer solved"
              + (f": {', '.join(lost[:20])}" if lost else ""))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--source", required=True, help="EPD/FEN file (positions after random plies) or jsonl with 'fen'")
    m.add_argument("--plies", type=int, default=30, help="random plies after an EPD position (0..this)")
    m.add_argument("--count", type=int, default=300)
    m.add_argument("--max-candidates", type=int, default=20000,
                   help="positions to analyse at most (with a 150 cp gap, a few percent of game positions qualify)")
    m.add_argument("--balanced", action="store_true", help="the same number of positions per category")
    m.add_argument("--reference", default=shutil.which("stockfish") or "/usr/games/stockfish")
    m.add_argument("--nodes", type=int, default=2000000)
    m.add_argument("--depth", type=int, default=22, help="used when --nodes is 0")
    m.add_argument("--threads", type=int, default=1)
    m.add_argument("--hash", type=int, default=128)
    m.add_argument("--min-gap", type=int, default=150, help="cp the best move must beat the second by")
    m.add_argument("--max-score", type=int, default=1500, help="skip positions decided beyond this (except mates)")
    m.add_argument("--endgame-pieces", type=int, default=6, help="at most this many pieces besides the kings")
    m.add_argument("--prefix", default="nags")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--suite", required=True)
    r.add_argument("--engine", required=True)
    r.add_argument("--option", action="append", default=[], help="UCI option NAME=VALUE")
    r.add_argument("--nodes", type=int)
    r.add_argument("--movetime", type=float, default=1.0, help="seconds per position (without --nodes)")
    r.add_argument("--json")
    r.add_argument("--baseline", help="an earlier run's --json")
    args = ap.parse_args(argv)
    return make(args) if args.cmd == "make" else run(args)


if __name__ == "__main__":
    sys.exit(main())
