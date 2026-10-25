#!/usr/bin/env python3
"""An experiment funnel: one candidate engine against its base, through stages fixed before the first game.

    python tools/funnel.py --name capture-history --candidate cand/build/nags_enhanced \\
        --base main/build/nags_enhanced --dir results/capture-history --concurrency 3

Stages (--kind feature, the default): an SPRT [0, 10] at 3+0.03 with an 8000-game cap and, only after H1
there, a non-regression SPRT [-5, 0] at 10+0.1 with a 6000-game cap. --kind fix: one non-regression SPRT
[-5, 0] at 3+0.03 with a 6000-game cap, for corrections that should not cost strength. --stage
NAME,LIMIT,ELO0,ELO1,GAMES (repeatable) replaces them; LIMIT is a time control (3+0.03) or depth=N or nodes=N.

Before the first game the plan is written to <dir>/plan.json: the stages, both engines (command, SHA-256,
bench node count), the openings (SHA-256) and the seed. Running the same command again continues that plan:
a finished stage is reused and a stopped one resumed from its log. Another plan in the same directory (other
stages, binaries, openings or seed) is refused, so the stages and caps cannot be changed after the results
start coming in. The concurrency may change.

Both benches are recorded. A candidate whose bench equals the base's is still played: bench searches to a
fixed depth, so time management changes and fixes for positions bench does not reach leave it unchanged (a
pure speed change needs no games: measure it with bench). The same binary as the base is refused. An
invalid match (an engine that cannot start) stops the funnel with no result; a match that stops without a
decision (Ctrl-C, a crash, a kill) is continued from its log by the next run. The losses on time, crashes
and illegal moves of each stage (the runner's "Forfeits:" line) are kept in result.json.

Exit status: 0 = passed every stage, 1 = rejected (H0) at a stage, 2 = undecided at a stage's game cap,
3 = invalid or refused, 4 = stopped before a decision (run the same command again to continue). The last
lines are one row per stage for the strength log in docs/TESTING.md.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import sprt  # noqa: E402

KINDS = {
    "feature": ["stc,3+0.03,0,10,8000", "ltc,10+0.1,-5,0,6000"],
    "fix": ["nonreg,3+0.03,-5,0,6000"],
}
DEFAULT_OPENINGS = ROOT / "tools" / "openings" / "nags_balanced.epd"
STATUS = re.compile(r"Games (\d+): \+(\d+) =(\d+) -(\d+) \| penta \[[\d, ]+\] \| Elo ([+-]?[\d.]+) ± ([\d.]+)")


def parse_stage(text: str) -> Dict[str, Any]:
    parts = text.split(",")
    if len(parts) != 5:
        raise ValueError(f"stage {text!r}: give NAME,LIMIT,ELO0,ELO1,GAMES")
    name, limit, elo0, elo1, games = parts
    if not re.fullmatch(r"[\w.-]+", name):
        raise ValueError(f"stage {text!r}: the name is used for file names (letters, digits, - _ .)")
    if not (re.fullmatch(r"(depth|nodes)=\d+", limit) or re.fullmatch(r"[\d.]+\+[\d.]+", limit)):
        raise ValueError(f"stage {text!r}: the limit is a time control such as 3+0.03, depth=N or nodes=N")
    stage = {"name": name, "limit": limit, "elo0": float(elo0), "elo1": float(elo1), "games": int(games)}
    if stage["elo0"] >= stage["elo1"] or stage["games"] < 2:
        raise ValueError(f"stage {text!r}: needs ELO0 < ELO1 and at least 2 games")
    return stage


def limit_args(limit: str) -> List[str]:
    if limit.startswith(("depth=", "nodes=")):
        key, value = limit.split("=")
        return [f"--{key}", value]
    return ["--tc", limit]


def bench_nodes(command: List[str]) -> Optional[str]:
    """The node count the engine's bench command reports (its search signature), or None."""
    try:
        out = subprocess.run(command + ["bench"], capture_output=True, text=True, timeout=900).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    line = next((l for l in out.splitlines() if l.startswith("Nodes searched")), None)
    return line.split(":")[1].strip() if line else None


def stage_status(log: Path) -> Optional[Dict[str, Any]]:
    """What a stage's log says: games, W/D/L, Elo and its error from the last status line, and the decision
    ("H1", "H0", "cap") or None while the stage is unfinished. None without a status line."""
    if not log.exists():
        return None
    text = log.read_text(errors="replace")
    found = STATUS.findall(text)
    if not found:
        return None
    games, w, d, l, elo, err = found[-1]
    decision = ("H1" if "H1 accepted" in text else "H0" if "H0 accepted" in text
                else "cap" if "SPRT inconclusive" in text else None)
    forfeits = [line[len("Forfeits: "):] for line in text.splitlines() if line.startswith("Forfeits: ")]
    return {"games": int(games), "wdl": [int(w), int(d), int(l)], "elo": float(elo), "error": float(err),
            "decision": decision, "forfeits": forfeits[-1] if forfeits else None}


def engine_record(text: str) -> Dict[str, Any]:
    command = sprt.engine_command(text)
    return {"text": text, "command": command, "sha256": sprt.command_sha256(command)}


def plan_differences(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    if old.get("format") != new["format"]:
        return [f"plan format {old.get('format')}"]
    out = []
    if old.get("stages") != new["stages"]:
        out.append(f"stages {old.get('stages')} -> {new['stages']}")
    for side in ("candidate", "base"):
        if (old.get(side) or {}).get("sha256") != new[side]["sha256"]:
            out.append(f"{side} binary {(old.get(side) or {}).get('sha256')} -> {new[side]['sha256']}")
    if (old.get("openings") or {}).get("sha256") != new["openings"]["sha256"]:
        out.append("openings")
    if old.get("seed") != new["seed"]:
        out.append(f"seed {old.get('seed')} -> {new['seed']}")
    return out


def run_logged(cmd: List[str], log: Path) -> int:
    print("+ " + " ".join(cmd), flush=True)
    with open(log, "a") as fh:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        while True:
            try:
                for line in proc.stdout:
                    print(line, end="", flush=True)
                    fh.write(line)
                    fh.flush()
                break
            except KeyboardInterrupt:
                continue  # the match finishes its games in progress, reports them and exits 4
        return proc.wait()


def run_stage(stage: Dict[str, Any], plan: Dict[str, Any], args: argparse.Namespace, out: Path) -> Dict[str, Any]:
    log = out / f"{stage['name']}.log"
    done = stage_status(log)
    if done and done["decision"]:
        print(f"{stage['name']}: finished earlier ({done['decision']} after {done['games']} games)", flush=True)
        return done
    cmd = [sys.executable, str(ROOT / "tools" / "sprt.py"),
           "--engine", plan["candidate"]["text"], "--engine", plan["base"]["text"],
           "--name", plan["name"], "--name", "base", *limit_args(stage["limit"]),
           "--openings", plan["openings"]["path"], "--seed", str(plan["seed"]),
           "--concurrency", str(args.concurrency), "--sprt", f"{stage['elo0']:g}", f"{stage['elo1']:g}",
           "--games", str(stage["games"]), "--resign-adjudication", "4", "1000",
           "--pgnout", str(out / f"{stage['name']}.pgn"), "--report", str(args.report)]
    if done:  # stopped part-way: continue from its last status line
        part = log.with_suffix(".part")
        shutil.copyfile(log, part)
        cmd += ["--resume", str(part)]
    code = run_logged(cmd, log)
    st = stage_status(log) or {}
    if code == 3:  # an engine that cannot start: the games played (if any) are not a result
        return {**st, "decision": "invalid", "exit_code": code}
    if st.get("decision") in ("H1", "H0", "cap"):  # decided by the log, not the exit status
        return st
    # No decision: interrupted (exit 4), or the runner ended otherwise (a crash, a kill). Its log is resumed.
    return {**st, "decision": "interrupted" if code == 4 else "stopped", "exit_code": code}


def strength_row(name: str, stage: Dict[str, Any], st: Dict[str, Any], plan: Dict[str, Any], concurrency: int) -> str:
    kind = "non-regression, " if stage["elo1"] <= 0 else ""
    outcome = {"H1": "H1", "H0": "H0", "cap": "no decision"}[st["decision"]]
    w, d, l = st["wdl"]
    after = f"after {st['games']} games" if st["decision"] != "cap" else f"at the {st['games']}-game cap"
    return (f"| {name} | SPRT [{stage['elo0']:g}, {stage['elo1']:g}] ({kind}{stage['name']}), {stage['limit']}, "
            f"{concurrency} at a time | {outcome} {after} (+{w} ={d} -{l}) | {st['elo']:+.1f} ± {st['error']:.1f} | "
            f"{plan['candidate']['bench']} |")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="the candidate's name in logs and the result rows")
    ap.add_argument("--candidate", required=True, help="the candidate engine (path or command)")
    ap.add_argument("--base", required=True, help="the engine it is tested against")
    ap.add_argument("--dir", required=True, help="directory for the plan, logs, games and result")
    ap.add_argument("--kind", choices=sorted(KINDS), default="feature")
    ap.add_argument("--stage", action="append", metavar="NAME,LIMIT,ELO0,ELO1,GAMES",
                    help="replace the kind's stages (repeatable; run in order, each after H1 in the one before)")
    ap.add_argument("--openings", default=str(DEFAULT_OPENINGS))
    ap.add_argument("--seed", type=int, default=1, help="seed for the opening order")
    ap.add_argument("--concurrency", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--report", type=float, default=60.0, help="seconds between status lines")
    args = ap.parse_args(argv)

    try:
        stages = [parse_stage(s) for s in (args.stage or KINDS[args.kind])]
    except ValueError as e:
        ap.error(str(e))
    out = Path(args.dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = {"format": "nags-funnel-1", "name": args.name, "kind": None if args.stage else args.kind,
            "stages": stages, "candidate": engine_record(args.candidate), "base": engine_record(args.base),
            "openings": {"path": str(Path(args.openings).resolve()), "sha256": sprt.file_sha256(args.openings)},
            "seed": args.seed}
    if plan["candidate"]["sha256"] is None or plan["base"]["sha256"] is None:
        print("Refused: the candidate or the base is not a file (or a command naming one)", file=sys.stderr)
        return 3
    if plan["candidate"]["sha256"] == plan["base"]["sha256"]:
        print("Refused: the candidate is the same binary as the base", file=sys.stderr)
        return 3
    if plan["openings"]["sha256"] is None:
        print(f"Refused: no openings file {args.openings}", file=sys.stderr)
        return 3
    plan_file = out / "plan.json"
    if plan_file.exists():
        old = json.loads(plan_file.read_text())
        differences = plan_differences(old, plan)
        if differences:
            print("Refused: this directory holds another plan (" + "; ".join(differences) + "); its stages are "
                  "fixed once games are played. Use another --dir.", file=sys.stderr)
            return 3
        plan = old
        print(f"Continuing the plan of {old['created']}", flush=True)
    else:
        for side in ("candidate", "base"):
            plan[side]["bench"] = bench_nodes(plan[side]["command"])
            if plan[side]["bench"] is None:
                print(f"Refused: the {side} did not run its bench", file=sys.stderr)
                return 3
        plan["same_bench"] = plan["candidate"]["bench"] == plan["base"]["bench"]
        if plan["same_bench"]:
            print(f"Note: the candidate's bench equals the base's ({plan['base']['bench']} nodes): its change does "
                  "not show at bench's fixed depth (time management, rare positions); played as a functional change",
                  flush=True)
        plan["created"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        plan_file.write_text(json.dumps(plan, indent=1) + "\n")
    print(f"{plan['name']}: bench {plan['candidate']['bench']} against {plan['base']['bench']}; stages "
          + ", ".join(f"{s['name']} [{s['elo0']:g}, {s['elo1']:g}] {s['limit']} cap {s['games']}" for s in stages),
          flush=True)

    results, code = [], 0
    for stage in stages:
        st = run_stage(stage, plan, args, out)
        results.append({"stage": stage["name"], **st})
        if st["decision"] == "H1":
            continue
        code = {"H0": 1, "cap": 2, "invalid": 3, "interrupted": 4, "stopped": 4}[st["decision"]]
        break
    last = results[-1] if results else {}
    verdict = {0: "passed every stage", 1: f"rejected at {last.get('stage')}",
               2: f"undecided at the cap of {last.get('stage')} (not adopted)",
               3: f"invalid match at {last.get('stage')}: no result",
               4: f"stopped at {last.get('stage')} before a decision ("
                  + ("interrupted" if last.get("decision") == "interrupted"
                     else f"the runner ended with exit status {last.get('exit_code')}; see its log")
                  + "): run the same command again to continue"}[code]
    (out / "result.json").write_text(json.dumps({"name": plan["name"], "verdict": verdict, "stages": results},
                                                indent=1) + "\n")
    print(f"\n{plan['name']}: {verdict}")
    rows = [strength_row(plan["name"], s, r, plan, args.concurrency) for s, r in zip(stages, results)
            if r["decision"] in ("H1", "H0", "cap")]
    if rows:
        print("Rows for the strength log (docs/TESTING.md):")
        print("\n".join(rows))
    return code


if __name__ == "__main__":
    sys.exit(main())
