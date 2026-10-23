#!/usr/bin/env python3
"""An Elo ladder of unrestricted engines at fixed strength (docs/TESTING.md, "Ladder").

calibrate.py rates against Stockfish's strength-limited levels, whose scale ends at UCI_Elo
3190 (NAGS scores 35% there). Here the opponents play at full strength, held fixed by a node
limit per move: Stockfish at N nodes plays the same moves on any machine, so the ladder's scale
does not move when the hardware or the time control changes. A ladder file (JSON) lists the
rungs; `anchor` measures each rung against the next once and stores their ratings on the
ladder's own scale (rung 0 = its given rating, by default 0; a match already in the file with
at least --games games is not played again, so an interrupted anchor continues); `rate` plays an engine against
the rungs near its strength and combines the results into a rating on that scale.

  python tools/ladder.py anchor --ladder tools/ladder/sf16_nodes.json --games 200 --concurrency 3
  python tools/ladder.py rate --ladder tools/ladder/sf16_nodes.json --engine build/nags_enhanced \\
      --tc 10+0.1 --rungs sf16-64k sf16-128k sf16-256k --games 100 --concurrency 3 --results ladder.jsonl

With --nodes instead of --tc the rated engine is node-limited too, which takes the hardware out
of the measurement entirely (but not how the engine uses its time). Every `rate` run appends
one JSON line to --results: the date, the engine's bench signature, the machine, the limits and
each rung's result with its 95% interval.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.util
import json
import math
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_spec = importlib.util.spec_from_file_location("sprt", Path(__file__).with_name("sprt.py"))
sprt = importlib.util.module_from_spec(_spec)
sys.modules["sprt"] = sprt
_spec.loader.exec_module(sprt)

OPENINGS = str(Path(__file__).with_name("openings") / "nags_balanced.epd")


def combine(estimates: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """Inverse-variance weighted mean of (rating, 95% half-width) pairs."""
    usable = [(r, e) for r, e in estimates if math.isfinite(r) and math.isfinite(e) and e > 0]
    if not usable:
        return float("nan"), float("inf")
    weights = [1.0 / (e / 1.96) ** 2 for _, e in usable]
    return sum(w * r for w, (r, _) in zip(weights, usable)) / sum(weights), 1.96 / math.sqrt(sum(weights))


def rung_spec(rung: Dict[str, Any]) -> "sprt.EngineSpec":
    lim = rung.get("limits") or {}
    limits = sprt.make_limits(lim.get("tc"), lim.get("nodes"), lim.get("depth"), lim.get("movetime"), 0.0)
    if limits is None:
        raise ValueError(f"rung {rung['name']}: no limits")
    return sprt.EngineSpec(rung["name"], shlex.split(rung["engine"]),
                           {k: str(v) for k, v in (rung.get("options") or {}).items()}, limits)


def play(specs, games: int, concurrency: int, openings, seed: int, pgnout: Optional[str]) -> "sprt.Match":
    # The shared limit is never used: both sides bring their own.
    match = sprt.Match(specs, openings, specs[0].limits, sprt.Adjudication(), max_pairs=(games + 1) // 2,
                       concurrency=concurrency, sprt=None, alpha=0.05, beta=0.05, pgnout=pgnout, seed=seed,
                       shuffle=True)
    if match.run(report_interval=60) != 0:
        raise RuntimeError(f"{specs[0].name} vs {specs[1].name}: match failed ({'; '.join(match.errors)})")
    return match


def anchor(args: argparse.Namespace) -> int:
    path = Path(args.ladder)
    ladder = json.loads(path.read_text())
    rungs = ladder["rungs"]
    openings = sprt.load_openings(args.openings)
    if rungs[0].get("rating") is None:  # the ladder files list every rung with "rating": null
        rungs[0]["rating"] = ladder.get("base_rating", 0.0)
    rungs[0]["rating_error"] = 0.0
    for i in range(len(rungs) - 1):
        a, b = rungs[i], rungs[i + 1]
        done = b.get("anchored_against") or {}
        if done.get("rung") == a["name"] and done.get("games", 0) >= args.games:
            # Measured by an earlier run (the file is saved after every match): its result
            # stands, on top of the rung below as it is now.
            b["rating"] = round(a["rating"] + done["elo"], 1)
            b["rating_error"] = round(math.hypot(a["rating_error"], done["error"]), 1)
            print(f"== {a['name']} vs {b['name']}: anchored already ({done['games']} games)", flush=True)
            continue
        print(f"== {a['name']} vs {b['name']}: {args.games} games", flush=True)
        m = play((rung_spec(a), rung_spec(b)), args.games, args.concurrency, openings, seed=i,
                 pgnout=f"{args.pgnout}_{a['name']}_{b['name']}.pgn" if args.pgnout else None)
        diff, err = m.penta.elo()  # a's Elo relative to b
        b["rating"] = round(a["rating"] - diff, 1)
        b["rating_error"] = round(math.hypot(a["rating_error"], err), 1)  # errors add up along the ladder
        b["anchored_against"] = {"rung": a["name"], "games": sum(m.wdl), "wdl": list(m.wdl),
                                 "elo": round(-diff, 1), "error": round(err, 1)}
        print(f"   {b['name']}: {b['rating']:+.0f} ± {b['rating_error']:.0f} ({-diff:+.0f} ± {err:.0f} over "
              f"{a['name']})", flush=True)
        path.write_text(json.dumps(ladder, indent=2) + "\n")  # an interrupted run keeps what it measured
    ladder["anchored"] = dt.date.today().isoformat()
    path.write_text(json.dumps(ladder, indent=2) + "\n")
    return 0


def bench_signature(command: Sequence[str]) -> Optional[str]:
    try:
        out = subprocess.run(list(command) + ["bench"], capture_output=True, text=True, timeout=600).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    line = next((l for l in out.splitlines() if l.startswith("Nodes searched")), None)
    return line.split(":")[1].strip() if line else None


def machine() -> Dict[str, Any]:
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"cpu": cpu, "cores": os.cpu_count(), "system": platform.system()}


def rate(args: argparse.Namespace) -> int:
    ladder = json.loads(Path(args.ladder).read_text())
    by_name = {r["name"]: r for r in ladder["rungs"]}
    chosen = [by_name[n] for n in args.rungs] if args.rungs else ladder["rungs"]
    missing = [r["name"] for r in chosen if r.get("rating") is None]
    if missing:
        print(f"ERROR: rungs without a rating (run 'anchor' first): {', '.join(missing)}", file=sys.stderr)
        return 3
    command = shlex.split(args.engine)
    limits = sprt.make_limits(args.tc, args.nodes, None, None, args.timemargin)
    if limits is None:
        print("ERROR: give --tc or --nodes", file=sys.stderr)
        return 3
    name = args.name or os.path.basename(command[0])
    me = sprt.EngineSpec(name, command, sprt.parse_options(args.option), limits)
    openings = sprt.load_openings(args.openings)
    rows = []
    for i, rung in enumerate(chosen):
        print(f"== {name} ({limits.describe()}) vs {rung['name']}: {args.games} games", flush=True)
        m = play((me, rung_spec(rung)), args.games, args.concurrency, openings, seed=100 + i,
                 pgnout=f"{args.pgnout}_{rung['name']}.pgn" if args.pgnout else None)
        diff, err = m.penta.elo()
        total_err = math.hypot(err, rung.get("rating_error") or 0.0)
        rows.append({"rung": rung["name"], "rung_rating": rung["rating"], "games": sum(m.wdl), "wdl": list(m.wdl),
                     "elo": diff, "error": err, "rating": rung["rating"] + diff, "rating_error": total_err})
    rating, err = combine([(r["rating"], r["rating_error"]) for r in rows])
    print(f"\n{'Rung':<14} {'rating':>8} {'games':>6}   +    =    -   {'Elo diff':>14} {'estimate':>16}")
    for r in rows:
        print(f"{r['rung']:<14} {r['rung_rating']:>8.0f} {r['games']:>6} {r['wdl'][0]:>4} {r['wdl'][1]:>4} "
              f"{r['wdl'][2]:>4}   {r['elo']:+7.0f} ± {r['error']:<4.0f} {r['rating']:>8.0f} ± {r['rating_error']:.0f}")
    print(f"\n{name}: {rating:.0f} ± {err:.0f} on the {ladder.get('name', 'ladder')} scale ({limits.describe()})")
    if args.results:
        record = {"date": dt.datetime.now().isoformat(timespec="seconds"), "engine": name,
                  "command": command, "options": sprt.parse_options(args.option), "bench": bench_signature(command),
                  "limits": limits.describe(), "ladder": ladder.get("name"), "machine": machine(),
                  "rating": rating, "error": err, "rungs": rows}
        with open(args.results, "a") as f:
            f.write(json.dumps(record) + "\n")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("anchor", "rate"):
        p = sub.add_parser(cmd)
        p.add_argument("--ladder", required=True, help="ladder file (JSON)")
        p.add_argument("--games", type=int, default=100, help="games per match")
        p.add_argument("--concurrency", type=int, default=1)
        p.add_argument("--openings", default=OPENINGS)
        p.add_argument("--pgnout", help="PGN file prefix (one file per match)")
        if cmd == "rate":
            p.add_argument("--engine", required=True)
            p.add_argument("--name")
            p.add_argument("--option", action="append", default=[], help="UCI option NAME=VALUE")
            p.add_argument("--tc", help="the rated engine's time control (base+inc seconds)")
            p.add_argument("--nodes", type=int, help="or a fixed node count per move")
            p.add_argument("--timemargin", type=float, default=0.0)
            p.add_argument("--rungs", nargs="+", help="rungs to play (default: all)")
            p.add_argument("--results", help="append the result to this file (JSON lines)")
    args = ap.parse_args(argv)
    return anchor(args) if args.cmd == "anchor" else rate(args)


if __name__ == "__main__":
    sys.exit(main())
