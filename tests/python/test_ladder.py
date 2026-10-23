"""tools/ladder.py: anchoring a ladder and rating an engine on it (small engines, tiny node counts)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("ladder", ROOT / "tools" / "ladder.py")
ladder = importlib.util.module_from_spec(_spec)
sys.modules["ladder"] = ladder
_spec.loader.exec_module(ladder)


def _engine(name):
    for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe"):
        if c.exists():
            return str(c)
    return None


@pytest.mark.skipif(_engine("nags_basic") is None or _engine("nags_fast") is None, reason="engines not built")
def test_anchor_and_rate(tmp_path, capsys):
    path = tmp_path / "ladder.json"
    rungs = [{"name": f"basic-{n}", "engine": _engine("nags_basic"), "options": {}, "limits": {"nodes": n}}
             for n in (100, 400)]
    path.write_text(json.dumps({"name": "test", "base_rating": 1000.0, "rungs": rungs}))
    assert ladder.main(["anchor", "--ladder", str(path), "--games", "4"]) == 0
    anchored = json.loads(path.read_text())
    r0, r1 = anchored["rungs"]
    assert r0["rating"] == 1000.0 and r0["rating_error"] == 0.0
    assert r1["rating"] is not None and r1["anchored_against"]["games"] == 4 and "anchored" in anchored

    results = tmp_path / "results.jsonl"
    assert ladder.main(["rate", "--ladder", str(path), "--engine", _engine("nags_fast"), "--nodes", "200",
                        "--games", "2", "--results", str(results)]) == 0
    rec = json.loads(results.read_text().splitlines()[0])
    assert rec["limits"] == "nodes 200" and len(rec["rungs"]) == 2 and rec["bench"].isdigit()
    assert rec["machine"]["cores"] >= 1 and "on the test scale" in capsys.readouterr().out

    # Rungs must be anchored before an engine is rated on them.
    path.write_text(json.dumps({"name": "test", "rungs": rungs}))
    assert ladder.main(["rate", "--ladder", str(path), "--engine", _engine("nags_fast"), "--nodes", "200"]) == 3


@pytest.mark.skipif(_engine("nags_basic") is None, reason="engines not built")
def test_anchor_from_a_shipped_file_and_resume(tmp_path, capsys):
    # As in tools/ladder/*.json: every rung starts with "rating": null.
    path = tmp_path / "ladder.json"
    rungs = [{"name": f"basic-{n}", "engine": _engine("nags_basic"), "options": {}, "limits": {"nodes": n},
              "rating": None} for n in (100, 200, 400)]
    path.write_text(json.dumps({"name": "test", "base_rating": 0.0, "rungs": rungs}))
    assert ladder.main(["anchor", "--ladder", str(path), "--games", "2"]) == 0
    first = json.loads(path.read_text())["rungs"]
    assert first[0]["rating"] == 0.0 and all(r["rating"] is not None for r in first)
    # A second run plays nothing that is already measured, and gives the same ratings.
    capsys.readouterr()
    assert ladder.main(["anchor", "--ladder", str(path), "--games", "2"]) == 0
    assert capsys.readouterr().out.count("anchored already") == 2
    assert [r["rating"] for r in json.loads(path.read_text())["rungs"]] == [r["rating"] for r in first]


def test_combine():
    rating, err = ladder.combine([(100.0, 20.0), (130.0, 20.0)])
    assert abs(rating - 115.0) < 1e-9 and abs(err - 20.0 / 2 ** 0.5) < 1e-9
    assert ladder.combine([(1.0, float("inf"))])[0] != ladder.combine([(1.0, float("inf"))])[0]  # nan: nothing usable
