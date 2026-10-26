"""tools/funnel.py: stages fixed before the first game, run in order, reused and resumed, and refused when
the plan changes or the search does not."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import funnel  # noqa: E402


def _engine(name):
    for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe"):
        if c.exists():
            return str(c)
    return None


needs_engines = pytest.mark.skipif(_engine("nags_basic") is None or _engine("nags_fast") is None,
                                   reason="engines not built")
# Wide bounds so the (far stronger) NNUE engine decides each stage in a few dozen depth-1 games.
WIDE = ["--stage", "first,depth=1,0,200,200", "--stage", "second,depth=1,-200,0,200"]


def _run(tmp_path, candidate, base, *extra):
    return funnel.main(["--name", "cand", "--candidate", candidate, "--base", base, "--dir", str(tmp_path / "f"),
                        "--concurrency", "1", "--report", "1000", *extra])


def test_stage_parsing():
    assert funnel.parse_stage("stc,3+0.03,0,10,8000") == {"name": "stc", "limit": "3+0.03", "elo0": 0.0,
                                                          "elo1": 10.0, "games": 8000}
    assert funnel.limit_args("depth=3") == ["--depth", "3"] and funnel.limit_args("10+0.1") == ["--tc", "10+0.1"]
    for bad in ("stc,3+0.03,0,10", "s/x,3+0.03,0,10,8", "stc,fast,0,10,8", "stc,3+0.03,10,0,8"):
        with pytest.raises(ValueError):
            funnel.parse_stage(bad)
    assert [funnel.parse_stage(s)["name"] for s in funnel.KINDS["feature"]] == ["stc", "ltc"]


@needs_engines
def test_a_candidate_through_every_stage_then_reused(tmp_path, capsys):
    assert _run(tmp_path, _engine("nags_fast"), _engine("nags_basic"), *WIDE) == 0
    out = capsys.readouterr().out
    plan = json.loads((tmp_path / "f" / "plan.json").read_text())
    assert [s["name"] for s in plan["stages"]] == ["first", "second"]
    assert plan["candidate"]["bench"] != plan["base"]["bench"] and plan["candidate"]["sha256"]
    result = json.loads((tmp_path / "f" / "result.json").read_text())
    assert result["verdict"] == "passed every stage" and [r["decision"] for r in result["stages"]] == ["H1", "H1"]
    rows = [l for l in out.splitlines() if l.startswith("| cand |")]
    assert len(rows) == 2 and "SPRT [0, 200] (first), depth=1, 1 at a time | H1 after" in rows[0]
    assert "non-regression, second" in rows[1] and rows[0].endswith(f"| {plan['candidate']['bench']} |")
    # Run again: both stages are taken from their logs, nothing is played.
    assert _run(tmp_path, _engine("nags_fast"), _engine("nags_basic"), *WIDE) == 0
    again = capsys.readouterr().out
    assert "first: finished earlier (H1" in again and "second: finished earlier (H1" in again
    assert "+ " not in again.replace("| +", "")  # no match was started


@needs_engines
def test_a_stopped_stage_is_resumed(tmp_path, capsys):
    assert _run(tmp_path, _engine("nags_fast"), _engine("nags_basic"), *WIDE[:2]) == 0
    capsys.readouterr()
    log = tmp_path / "f" / "first.log"
    text = log.read_text()
    log.write_text(text[: text.index("H1 accepted")])  # as if it had stopped just before deciding
    assert funnel.stage_status(log)["decision"] is None
    assert _run(tmp_path, _engine("nags_fast"), _engine("nags_basic"), *WIDE[:2]) == 0
    assert "Resuming after" in capsys.readouterr().out and funnel.stage_status(log)["decision"] == "H1"


@needs_engines
def test_a_stage_stopped_before_its_first_status_line_is_resumed_with_its_games(tmp_path, capsys):
    assert _run(tmp_path, _engine("nags_fast"), _engine("nags_basic"), *WIDE[:2]) == 0
    capsys.readouterr()
    log, pgn = tmp_path / "f" / "first.log", tmp_path / "f" / "first.pgn"
    text = log.read_text()
    log.write_text(text[: text.index("\nGames ") + 1])  # the fingerprint, no status line; the PGN holds games
    games = pgn.read_text()
    assert funnel.stage_status(log) is None and games
    assert _run(tmp_path, _engine("nags_fast"), _engine("nags_basic"), *WIDE[:2]) == 0
    assert "Resuming after 0 games" in capsys.readouterr().out and funnel.stage_status(log)["decision"] == "H1"
    assert pgn.read_text().startswith(games)  # kept, not overwritten


@needs_engines
def test_a_rejected_candidate_stops_at_that_stage(tmp_path, capsys):
    assert _run(tmp_path, _engine("nags_basic"), _engine("nags_fast"), *WIDE) == 1
    assert not (tmp_path / "f" / "second.log").exists()
    result = json.loads((tmp_path / "f" / "result.json").read_text())
    assert result["verdict"] == "rejected at first" and "| H0 after" in capsys.readouterr().out
    # The plan is fixed: other stages or another binary in the same directory are refused.
    assert _run(tmp_path, _engine("nags_basic"), _engine("nags_fast"), "--stage", "first,depth=1,0,100,200") == 3
    assert "another plan" in capsys.readouterr().err
    assert _run(tmp_path, _engine("nags_enhanced") or _engine("nags_fast"), _engine("nags_fast"), *WIDE) == 3


@needs_engines
def test_the_cap_is_no_decision(tmp_path, capsys):
    assert _run(tmp_path, _engine("nags_fast"), _engine("nags_basic"), "--stage", "only,depth=1,0,5,2") == 2
    out = capsys.readouterr().out
    assert "undecided at the cap of only (not adopted)" in out and "| no decision at the 2-game cap (" in out


@needs_engines
def test_the_same_bench_is_played_the_same_binary_and_a_missing_engine_refused(tmp_path, capsys):
    # Time management changes and rare-position fixes leave bench unchanged: they are played.
    copy = tmp_path / "copy_of_fast"
    copy.write_bytes(Path(_engine("nags_fast")).read_bytes() + b"\0")  # another binary, the same search
    copy.chmod(0o755)
    assert _run(tmp_path, str(copy), _engine("nags_fast"), "--stage", "only,depth=1,0,5,2") == 2
    out = capsys.readouterr().out
    assert "bench equals the base's" in out and "played as a functional change" in out
    assert json.loads((tmp_path / "f" / "plan.json").read_text())["same_bench"] is True
    assert _run(tmp_path / "g", _engine("nags_fast"), _engine("nags_fast"), *WIDE) == 3
    assert "same binary" in capsys.readouterr().err and not (tmp_path / "g" / "f" / "plan.json").exists()
    assert _run(tmp_path / "h", str(tmp_path / "missing"), _engine("nags_fast"), *WIDE) == 3


def _plan(tmp_path):
    return {"name": "cand", "candidate": {"text": "c"}, "base": {"text": "b"}, "openings": {"path": "o"}, "seed": 1}


def _fake_runner(lines, code):
    def run_logged(cmd, log):
        with open(log, "a") as fh:
            fh.write("".join(l + "\n" for l in lines))
        return code
    return run_logged


STATUS_LINE = "Games 4: +2 =1 -1 | penta [0, 1, 0, 1, 0] | Elo +0.0 ± 50.0 | LLR +0.10 [-2.94, 2.94] for [0, 10]"


@pytest.mark.parametrize("code,lines,decision", [
    (4, [STATUS_LINE, "Match interrupted after 4 games, before its end: no result yet"], "interrupted"),
    (1, [STATUS_LINE, "Traceback (most recent call last):"], "stopped"),  # a crash: exit 1 is not H0
    (-9, [STATUS_LINE[:30]], "stopped"),  # killed while writing its first status line
    (3, ["Match invalid: an engine could not be started or set up; no decision"], "invalid"),
    (1, [STATUS_LINE, "H0 accepted: cand is not stronger by 10 Elo"], "H0"),
    (2, [STATUS_LINE, "SPRT inconclusive: maximum number of games reached"], "cap"),
])
def test_how_a_stage_ended_is_read_from_its_log(tmp_path, monkeypatch, code, lines, decision):
    # The decision comes from the log, never from the exit status alone; a stage that stopped without one
    # (interrupted, crashed, killed) is resumable. Exit 4 used to raise a KeyError here.
    monkeypatch.setattr(funnel, "run_logged", _fake_runner(lines, code))
    stage = funnel.parse_stage("s,depth=1,0,10,100")
    args = argparse.Namespace(concurrency=1, report=1000)
    st = funnel.run_stage(stage, _plan(tmp_path), args, tmp_path)
    assert st["decision"] == decision
    if decision in ("interrupted", "stopped"):
        assert st["exit_code"] == code


@needs_engines
@pytest.mark.skipif(os.name == "nt", reason="sends SIGINT to a POSIX process group (Ctrl-C); the resume logic is "
                                            "tested on every platform by the fake-log test above")
def test_an_interrupted_funnel_resumes_where_it_stopped(tmp_path):
    cmd = [sys.executable, str(ROOT / "tools" / "funnel.py"), "--name", "cand", "--candidate", _engine("nags_fast"),
           "--base", _engine("nags_basic"), "--dir", str(tmp_path / "f"), "--concurrency", "1", "--report", "0.5",
           "--stage", "only,depth=2,0,5,60"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    log = tmp_path / "f" / "only.log"
    deadline = time.time() + 120
    while time.time() < deadline and not (log.exists() and funnel.STATUS.search(log.read_text())):
        time.sleep(0.2)
    os.killpg(proc.pid, signal.SIGINT)  # Ctrl-C reaches the funnel and its match
    out = proc.communicate(timeout=300)[0]
    assert proc.returncode == 4, out
    assert "stopped at only before a decision (interrupted)" in out
    result = json.loads((tmp_path / "f" / "result.json").read_text())
    assert result["stages"][0]["decision"] == "interrupted" and result["stages"][0]["games"] > 0
    done = result["stages"][0]["games"]
    again = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert again.returncode in (0, 1, 2), again.stdout + again.stderr
    assert f"Resuming after {done} games" in again.stdout
    final = funnel.stage_status(log)
    assert final["decision"] in ("H1", "H0", "cap") and final["games"] >= done
