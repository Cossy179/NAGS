#!/usr/bin/env python3
"""Rough absolute rating for an engine, measured against Stockfish's
strength-limited mode (UCI_LimitStrength + UCI_Elo).

Plays a fixed number of games against each Stockfish level and estimates the
engine's rating as level + measured Elo difference; levels are combined
weighted by the inverse variance of each estimate. Stockfish documents its
UCI_Elo scale as calibrated at 60+0.6 and anchored to CCRL 40/4, so ratings
measured at much faster time controls are only approximate.

    python tools/calibrate.py --engine build/nags_enhanced --levels 1500 1800 2100 \\
        --games 100 --tc 10+0.1 --concurrency 3
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import shlex
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

_spec = importlib.util.spec_from_file_location("sprt", Path(__file__).with_name("sprt.py"))
sprt = importlib.util.module_from_spec(_spec)
sys.modules["sprt"] = sprt
_spec.loader.exec_module(sprt)


def combine(estimates: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Inverse-variance weighted mean of (rating, 95% half-width) pairs."""
    usable = [(r, e) for r, e in estimates if math.isfinite(e) and e > 0]
    if not usable:
        return float("nan"), float("inf")
    weights = [1.0 / (e / 1.96) ** 2 for _, e in usable]
    mean = sum(w * r for w, (r, _) in zip(weights, usable)) / sum(weights)
    return mean, 1.96 / math.sqrt(sum(weights))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="engine to rate")
    ap.add_argument("--option", action="append", default=[], help="UCI option NAME=VALUE for the rated engine")
    ap.add_argument("--stockfish", default=shutil.which("stockfish") or "/usr/games/stockfish")
    ap.add_argument("--levels", nargs="+", type=int, default=[1500, 1800, 2100], help="Stockfish UCI_Elo levels")
    ap.add_argument("--games", type=int, default=100, help="games per level")
    ap.add_argument("--tc", default="10+0.1", help="time control base+inc in seconds")
    ap.add_argument("--openings", default=str(Path(__file__).with_name("openings") / "nags_balanced.epd"))
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--pgnout", help="PGN file prefix (one file per level)")
    args = ap.parse_args(argv)

    if not os.path.exists(args.stockfish):
        print(f"ERROR: Stockfish not found at {args.stockfish} (use --stockfish)", file=sys.stderr)
        return 3
    base, inc = sprt.parse_tc(args.tc)
    openings = sprt.load_openings(args.openings)
    name = os.path.basename(shlex.split(args.engine)[0])
    rows: List[Tuple[int, sprt.Match]] = []
    for level in args.levels:
        specs = (sprt.EngineSpec(name, shlex.split(args.engine), sprt.parse_options(args.option)),
                 sprt.EngineSpec(f"stockfish-{level}", [args.stockfish],
                                 {"UCI_LimitStrength": "true", "UCI_Elo": str(level)}))
        print(f"== {name} vs Stockfish UCI_Elo {level}: {args.games} games at {args.tc}", flush=True)
        match = sprt.Match(specs, openings, sprt.Limits(base=base, inc=inc), sprt.Adjudication(),
                           max_pairs=(args.games + 1) // 2, concurrency=args.concurrency, sprt=None, alpha=0.05,
                           beta=0.05, pgnout=f"{args.pgnout}_{level}.pgn" if args.pgnout else None, seed=level,
                           shuffle=True)
        if match.run(report_interval=60) != 0:
            return 3
        rows.append((level, match))

    print("\nLevel   Games   +    =    -    Score   Elo diff        Rating estimate")
    estimates = []
    for level, m in rows:
        diff, err = m.penta.elo()
        games = sum(m.wdl)
        score = (m.wdl[0] + 0.5 * m.wdl[1]) / max(1, games)
        estimates.append((level + diff, err))
        print(f"{level:<7} {games:<7} {m.wdl[0]:<4} {m.wdl[1]:<4} {m.wdl[2]:<4} {score:6.1%}  "
              f"{diff:+7.0f} ± {err:<5.0f}  {level + diff:7.0f} ± {err:.0f}")
    rating, err = combine(estimates)
    print(f"\nCombined estimate: {rating:.0f} ± {err:.0f} (Stockfish UCI_Elo scale, {args.tc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
