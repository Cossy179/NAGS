"""tools/funnel.py: stages fixed before the first game, run in order, reused and resumed, and refused when
the plan changes or the search does not."""

import json
import sys
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
def test_an_unchanged_search_or_a_missing_engine_is_refused(tmp_path, capsys):
    assert _run(tmp_path, _engine("nags_fast"), _engine("nags_fast"), *WIDE) == 3
    assert "bench equals the base's" in capsys.readouterr().err and not (tmp_path / "f" / "plan.json").exists()
    assert _run(tmp_path, str(tmp_path / "missing"), _engine("nags_fast"), *WIDE) == 3
