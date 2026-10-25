"""tools/gnn_compare.py end to end at a tiny scale: every arm trains, the
held-out metrics come out in range, and the run directory is all it writes."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
ENGINE = next((c for c in (ROOT / "build" / "nags_enhanced", ROOT / "build" / "Release" / "nags_enhanced.exe")
               if c.exists()), ROOT / "build" / "nags_enhanced")
sys.path.insert(0, str(ROOT / "tools"))


@pytest.mark.skipif(not ENGINE.exists(), reason="nags_enhanced not built")
def test_gnn_compare_smoke(tmp_path):
    import gnn_compare

    out = tmp_path / "run"
    gnn_compare.main(["--out", str(out), "--engine", str(ENGINE), "--games", "4", "--play-nodes", "300",
                      "--label-positions", "60", "--multipv", "3", "--label-nodes", "1500", "--epochs", "1"])
    results = json.loads((out / "results.json").read_text())
    arms = [k for k in results if k[0] in "ABC"]
    assert len(arms) == 4
    for name in arms:
        r = results[name]
        assert 0 <= r["top1_engine"] <= 1 and 0 <= r["top1_played"] <= 1 and r["nll_engine_best"] > 0
        assert 0 <= r["mass_on_engine_topk"] <= 1 + 1e-6 and r["value_mse_vs_result"] >= 0
    assert {p.name for p in tmp_path.iterdir()} == {"run"}
