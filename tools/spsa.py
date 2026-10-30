#!/usr/bin/env python3
"""SPSA tuning of search parameters, with a game budget fixed before the first game.

    cmake -B build-tune -DCMAKE_BUILD_TYPE=Release -DNAGS_TUNE=ON
    cmake --build build-tune --config Release --target nags_enhanced
    python tools/spsa.py --engine build-tune/nags_enhanced --params RfpMargin FutilityMargin \\
        --iterations 2000 --pairs 4 --tc 3+0.03 --concurrency 3 --state runs/spsa_rfp_fut.json

The engine must be a NAGS_TUNE build: its parameters (src/Tune.h) are UCI spin options with a default and a
range. Every iteration k perturbs each tuned parameter by +-c_k (an independent random sign each), plays
--pairs game pairs (each opening with both colours) between theta + c_k*delta and theta - c_k*delta, and moves
theta by a_k / c_k * (wins - losses of the + side) * delta, the schedule fishtest uses: c_k = c / (k + 1)^0.101,
a_k = a / (A + k + 1)^0.602 with A = iterations / 10, c = c_end * iterations^0.101 and a = r_end * c_end^2 *
(A + iterations)^0.602. c_end defaults to a twentieth of the parameter's range and r_end to 0.002.

The state file holds the settings (engine SHA-256, parameters, budget, time control, openings) and theta after
every iteration; the same command continues a stopped run and refuses other settings. At the end it prints
the tuned values and the SPRT that must decide them: the tuned values against the defaults, in the same tuning
build (equal speed), before any default in src/Tune.h changes.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import sprt  # noqa: E402

DEFAULT_OPENINGS = ROOT / "tools" / "openings" / "nags_balanced.epd"


def engine_params(command: List[str]) -> Dict[str, Dict[str, int]]:
    """{name: {default, min, max}} of the engine's spin options (a tuning build's parameters among them)."""
    import subprocess

    out = subprocess.run(command, input="uci\nquit\n", capture_output=True, text=True, timeout=60).stdout
    found = {}
    for line in out.splitlines():
        t = line.split()
        if t[:2] == ["option", "name"] and "type" in t and t[t.index("type") + 1] == "spin":
            name = " ".join(t[2 : t.index("type")])
            get = lambda key: int(t[t.index(key) + 1])  # noqa: E731
            found[name] = {"default": get("default"), "min": get("min"), "max": get("max")}
    return found


def schedule(k: int, n: int, c_end: float, r_end: float):
    """(c_k, a_k) for iteration k of n."""
    big_a = n / 10
    c = c_end * n ** 0.101
    a = r_end * c_end ** 2 * (big_a + n) ** 0.602
    return c / (k + 1) ** 0.101, a / (big_a + k + 1) ** 0.602


def clamp_round(x: float, lo: int, hi: int) -> int:
    return int(min(hi, max(lo, round(x))))


def play_pairs(state: Dict[str, Any], plus: Dict[str, int], minus: Dict[str, int], seed: int,
               concurrency: int) -> int:
    """Wins minus losses of `plus` over the state's number of game pairs (each opening with both colours)."""
    command = sprt.engine_command(state["engine"]["text"])
    specs = (sprt.EngineSpec("plus", command, {k: str(v) for k, v in plus.items()}),
             sprt.EngineSpec("minus", command, {k: str(v) for k, v in minus.items()}))
    tc = state["tc"]  # a time control, or depth=N / nodes=N
    limits = (sprt.make_limits(None, None, int(tc[6:]), None, 0.0) if tc.startswith("depth=") else
              sprt.make_limits(None, int(tc[6:]), None, None, 0.0) if tc.startswith("nodes=") else
              sprt.make_limits(tc, None, None, None, 0.0))
    adj = sprt.Adjudication(max_plies=400)
    adj.resign_count, adj.resign_score = 4, 1000
    match = sprt.Match(specs, sprt.load_openings(state["openings"]["path"], 0), limits, adj,
                       max_pairs=state["pairs"], concurrency=concurrency, sprt=None, alpha=0.05, beta=0.05,
                       pgnout=None, seed=seed, shuffle=True)
    with contextlib.redirect_stdout(io.StringIO()):
        code = match.run(report_interval=1e9)
    if code != 0:
        raise RuntimeError(f"iteration match failed (exit {code}): {'; '.join(match.errors) or 'interrupted'}")
    return match.wdl[0] - match.wdl[2]


def iterate(state: Dict[str, Any], play: Callable[..., int], concurrency: int, save: Callable[[], None],
            log=print) -> None:
    """Runs the remaining iterations, saving the state after each."""
    params, n = state["params"], state["iterations"]
    while state["done"] < n:
        k = state["done"]
        rng = random.Random(state["seed"] * 1_000_003 + k)
        delta = {name: rng.choice((-1, 1)) for name in params}
        plus, minus, steps = {}, {}, {}
        for name, p in params.items():
            c_k, a_k = schedule(k, n, p["c_end"], p["r_end"])
            plus[name] = clamp_round(p["theta"] + c_k * delta[name], p["min"], p["max"])
            minus[name] = clamp_round(p["theta"] - c_k * delta[name], p["min"], p["max"])
            steps[name] = (c_k, a_k)
        result = play(state, plus, minus, seed=state["seed"] + k, concurrency=concurrency)
        for name, p in params.items():
            c_k, a_k = steps[name]
            p["theta"] = min(p["max"], max(p["min"], p["theta"] + a_k / c_k * result * delta[name]))
        state["done"] = k + 1
        state["games"] = state.get("games", 0) + 2 * state["pairs"]
        state.setdefault("trace", []).append({"k": k + 1, "result": result,
                                              "theta": {m: round(q["theta"], 3) for m, q in params.items()}})
        save()
        if (k + 1) % max(1, n // 20) == 0 or k + 1 == n:
            log(f"iteration {k + 1}/{n} ({state['games']} games): "
                + ", ".join(f"{m} {q['theta']:.2f}" for m, q in params.items()), flush=True)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True, help="a NAGS_TUNE build")
    ap.add_argument("--params", nargs="+", required=True, help="parameters to tune (UCI option names)")
    ap.add_argument("--iterations", type=int, required=True, help="the budget: iterations of --pairs pairs each")
    ap.add_argument("--pairs", type=int, default=4, help="game pairs per iteration")
    ap.add_argument("--tc", default="3+0.03", help="time control, or depth=N / nodes=N")
    ap.add_argument("--openings", default=str(DEFAULT_OPENINGS))
    ap.add_argument("--concurrency", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--c-end", action="append", default=[], metavar="NAME=C", help="final perturbation of a "
                    "parameter (default: a twentieth of its range)")
    ap.add_argument("--r-end", type=float, default=0.002, help="final learning rate (fishtest's default)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--state", required=True, help="JSON state file (created, or continued)")
    args = ap.parse_args(argv)

    command = sprt.engine_command(args.engine)
    available = engine_params(command)
    missing = [p for p in args.params if p not in available]
    if missing:
        print(f"Refused: {', '.join(missing)} not among the engine's spin options (a NAGS_TUNE build has them)",
              file=sys.stderr)
        return 3
    c_end = dict(item.split("=", 1) for item in args.c_end)
    params = {}
    for name in args.params:
        o = available[name]
        params[name] = {"theta": float(o["default"]), "default": o["default"], "min": o["min"], "max": o["max"],
                        "c_end": float(c_end.get(name, max(1.0, (o["max"] - o["min"]) / 20))), "r_end": args.r_end}
    settings = {"format": "nags-spsa-1",
                "engine": {"text": args.engine, "sha256": sprt.command_sha256(command)},
                "iterations": args.iterations, "pairs": args.pairs, "tc": args.tc, "seed": args.seed,
                "openings": {"path": str(Path(args.openings).resolve()), "sha256": sprt.file_sha256(args.openings)}}
    path = Path(args.state)
    if path.exists():
        state = json.loads(path.read_text())
        mine = {k: state.get(k) for k in settings}
        mine["engine"] = {"text": args.engine, "sha256": (state.get("engine") or {}).get("sha256")}
        fixed = {name: {k: v for k, v in p.items() if k != "theta"} for name, p in state.get("params", {}).items()}
        want = {name: {k: v for k, v in p.items() if k != "theta"} for name, p in params.items()}
        if mine != settings or fixed != want:
            print("Refused: the state file was started with other settings (engine, parameters, budget, time "
                  "control, openings or seed); use another --state", file=sys.stderr)
            return 3
        print(f"Continuing after iteration {state['done']} of {state['iterations']}", flush=True)
    else:
        state = {**settings, "params": params, "done": 0, "games": 0}
        path.parent.mkdir(parents=True, exist_ok=True)

    def save():
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(state, indent=1) + "\n")
        os.replace(tmp, path)

    save()
    try:
        iterate(state, play_pairs, args.concurrency, save)
    except KeyboardInterrupt:
        print(f"Interrupted after iteration {state['done']}: run the same command again to continue")
        return 4
    except RuntimeError as e:
        print(f"Stopped: {e}", file=sys.stderr)
        return 3
    tuned = {name: clamp_round(p["theta"], p["min"], p["max"]) for name, p in state["params"].items()}
    print(f"\nTuned after {state['games']} games: " + ", ".join(f"{k}={v} (default {state['params'][k]['default']})"
                                                             for k, v in tuned.items()))
    opts = " ".join(f"--option1 {k}={v}" for k, v in tuned.items())
    print("Decide them by SPRT in the same build, against the defaults, before changing src/Tune.h:\n"
          f"  python tools/sprt.py --engine {args.engine} --engine {args.engine} --name tuned --name defaults "
          f"{opts} --tc {args.tc} --sprt 0 10 --games 8000 --concurrency 3 --openings {args.openings} "
          "--resign-adjudication 4 1000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
