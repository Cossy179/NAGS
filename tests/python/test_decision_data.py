"""tools/decision_data.py: collecting decisions, the logistic model and offline policy evaluation."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("decision_data", ROOT / "tools" / "decision_data.py")
dd = importlib.util.module_from_spec(_spec)
sys.modules["decision_data"] = dd
_spec.loader.exec_module(dd)


def _engine(name):
    for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe"):
        if c.exists():
            return str(c)
    return None


@pytest.mark.skipif(_engine("nags_enhanced") is None, reason="engines not built")
def test_collect(tmp_path):
    out = tmp_path / "d.jsonl"
    assert dd.main(["collect", "--engine", _engine("nags_enhanced"), "--positions", "4", "--nodes", "300",
                    "--budgets", "1", "2", "4", "--reference", "4", "--out", str(out)]) == 0
    rows = dd.read(str(out))
    assert rows and all(len(r["budgets"]) == 3 for r in rows)
    for r in rows:
        for b in r["budgets"]:
            assert set(b["features"]) == set(dd.FEATURES) and 0 <= b["loss"] <= dd.CAP and b["nodes"] > 0


def synthetic(n, seed=0):
    """Positions where the move changes with more search unless the search was stable."""
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        stable = rng.randint(0, 6)
        unstable = stable < 2
        budgets = []
        for k in (1, 2, 4, 8):
            feats = {f: rng.random() for f in dd.FEATURES}
            feats.update(stable_iterations=stable + (k > 1), log_budget=float(k.bit_length() - 1))
            budgets.append({"budget": k, "nodes": 1000 * k, "features": feats,
                            "move": f"m{k}" if unstable or rng.random() < 0.05 else "m",
                            "loss": (80 if k < 4 else 0) if unstable else 0})
        rows.append({"fen": str(i), "reference_move": "m", "budgets": budgets})
    return rows


def test_fit_and_policy(tmp_path):
    data = tmp_path / "s.jsonl"
    data.write_text("\n".join(json.dumps(r) for r in synthetic(600)) + "\n")
    model = tmp_path / "m.json"
    assert dd.main(["fit", "--data", str(data), "--json", str(model)]) == 0
    m = json.loads(model.read_text())
    assert m["test_auc"] > 0.8 and m["test_logloss"] < m["test_logloss_base"]
    assert m["weights"][1 + dd.FEATURES.index("stable_iterations")] < 0  # stable searches change less
    out = tmp_path / "p.json"
    assert dd.main(["policy", "--data", str(data), "--model", str(model), "--json", str(out)]) == 0
    p = json.loads(out.read_text())
    assert p["oracle"] <= p["rule"] < p["random"] and p["rule"] <= p["uniform_2N"]
    assert dd.auc([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0 and dd.auc([0.1, 0.9], [1, 0]) == 0.0


def test_mate_scores_are_capped_in_the_features():
    board = chess.Board()
    its = [{"move": "e2e4", "score": 30, "depth": 1}, {"move": "e2e4", "score": 29990, "depth": 2},
           {"move": "e2e4", "score": 40, "depth": 3}]
    f = dd.search_features(board, its, 1.0)
    assert f["score_abs"] == 40 and f["score_drop"] == dd.SCORE_CAP - 40 and f["score_swing"] == dd.SCORE_CAP - 30
    f = dd.search_features(board, [{"move": "e2e4", "score": -29990, "depth": 1}], 1.0)
    assert f["score_abs"] == dd.SCORE_CAP and f["score_swing"] == 0
