#!/usr/bin/env python3
"""Builds a balanced opening set (EPD) for engine testing.

Candidates come either from random playouts of a few plies (default) or from
real games in a PGN file (--pgn, e.g. the correspondence database). Each
candidate is searched by a UCI engine and kept only if the evaluation is
within --max-score centipawns of equality, so test games start from playable
positions. Duplicates are removed.

    python tools/make_openings.py --engine build/nags_enhanced --count 500 \\
        --out tools/openings/nags_balanced.epd

Random playouts give varied but sometimes odd openings; for serious testing
prefer positions sampled from strong games (--pgn) or an established book.
"""

from __future__ import annotations

import argparse
import random
import sys
from typing import Iterator, Optional, Sequence

import chess
import chess.engine
import chess.pgn


def random_candidates(rng: random.Random, min_plies: int, max_plies: int) -> Iterator[chess.Board]:
    while True:
        board = chess.Board()
        for _ in range(rng.randint(min_plies, max_plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if not board.is_game_over():
            yield board


def pgn_candidates(path: str, rng: random.Random, min_plies: int, max_plies: int) -> Iterator[chess.Board]:
    with open(path, encoding="utf-8", errors="replace") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                return
            board = game.board()
            target = rng.randint(min_plies, max_plies)
            for ply, move in enumerate(game.mainline_moves()):
                if ply >= target:
                    break
                board.push(move)
            if board.ply() == target and not board.is_game_over():
                yield board


def epd_key(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:4])


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="UCI engine used to evaluate candidates")
    ap.add_argument("--out", required=True, help="output EPD file")
    ap.add_argument("--count", type=int, default=500)
    ap.add_argument("--pgn", help="sample positions from these games instead of random playouts")
    ap.add_argument("--min-plies", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=10)
    ap.add_argument("--depth", type=int, default=7, help="search depth for the balance check")
    ap.add_argument("--max-score", type=int, default=60, help="keep positions with |eval| <= this (cp)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    source = (pgn_candidates(args.pgn, rng, args.min_plies, args.max_plies) if args.pgn
              else random_candidates(rng, args.min_plies, args.max_plies))
    seen = set()
    kept = tried = 0
    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    try:
        with open(args.out, "w") as out:
            for board in source:
                if kept >= args.count:
                    break
                key = epd_key(board)
                if key in seen:
                    continue
                seen.add(key)
                tried += 1
                info = engine.analyse(board, chess.engine.Limit(depth=args.depth), game=object())
                score = info["score"].white()
                if score.is_mate() or abs(score.score()) > args.max_score:
                    continue
                out.write(key + "\n")
                kept += 1
                if kept % 50 == 0:
                    print(f"{kept}/{args.count} positions kept ({tried} evaluated)", flush=True)
    finally:
        engine.quit()
    print(f"Wrote {kept} positions to {args.out} ({tried} evaluated)")
    return 0 if kept > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
