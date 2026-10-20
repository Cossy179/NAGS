"""tools/e2e_bench.py: configurations, the numbers it reports, and a short real run (tiny network)."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from gnn_v3 import RelationalGNNEvaluator, save_checkpoint_v3

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("e2e_bench", ROOT / "tools" / "e2e_bench.py")
e2e = importlib.util.module_from_spec(_spec)
sys.modules["e2e_bench"] = e2e
_spec.loader.exec_module(e2e)


def _engine():
    for c in (ROOT / "build" / "nags", ROOT / "build" / "Release" / "nags.exe"):
        if c.exists():
            return str(c)
    return None


def test_presets_and_custom_configs():
    for name, configs in e2e.PRESETS.items():
        assert len({c.name for c in configs}) == len(configs), name
        assert configs[0].options["NNBatch"] == "1"  # the reference row: one request per simulation
    c = e2e.parse_config("mine;--backend onnx --providers cuda,cpu;NNBatch=16,NNGraphs=true")
    assert c == e2e.Config("mine", ["--backend", "onnx", "--providers", "cuda,cpu"],
                           {"NNBatch": "16", "NNGraphs": "true"})
    assert e2e.parse_config("plain;;") == e2e.Config("plain", [], {})
    for bad in ("", "x", "a;b", ";--backend torch;"):
        with pytest.raises(ValueError):
            e2e.parse_config(bad)


def test_summary_numbers():
    lines = ["info depth 9 seldepth 12 score cp 10 nodes 9000 nps 900000 time 10 pv e2e4",
             "info depth 10 seldepth 14 score cp 12 nodes 20000 nps 1000000 time 20 pv e2e4",
             "info string nags alpha_beta_depth 10 mcts_sims 120"]
    assert e2e.last_info(lines) == {"depth": 10, "nps": 1000000}
    runs = [dict(mcts_sims=100, mcts_ms=500.0, rpc_requests=10, rpc_positions=80, rpc_wait_ms=50.0, cache_hits=20,
                 rpc_failures=0, ab_depth=10, ab_nps=1_000_000),
            dict(mcts_sims=300, mcts_ms=1500.0, rpc_requests=30, rpc_positions=240, rpc_wait_ms=150.0, cache_hits=60,
                 rpc_failures=0, ab_depth=12, ab_nps=2_000_000)]
    s = e2e.summarise("x", runs)
    assert s["sims_per_s"] == pytest.approx(200.0) and s["server_positions_per_s"] == pytest.approx(160.0)
    assert s["rtt_ms"] == pytest.approx(5.0) and s["positions_per_request"] == pytest.approx(8.0)
    assert s["ab_depth"] == 11 and s["ab_knps"] == pytest.approx(1500.0) and s["cache_hits"] == 80
    table = e2e.format_table([s, dict(s, config="y", sims_per_s=400.0), {"config": "z", "error": "no provider"}])
    assert "(2.0x)" in table and "z" in table and "failed: no provider" in table


@pytest.mark.skipif(_engine() is None, reason="nags not built")
def test_short_run(tmp_path):
    torch.manual_seed(0)
    model = RelationalGNNEvaluator(hidden_dim=16, layers=2, num_bases=2, device=torch.device("cpu")).eval()
    path = tmp_path / "tiny.pth"
    save_checkpoint_v3(model, str(path))
    out = tmp_path / "e2e.json"
    rc = e2e.main(["--model", str(path), "--positions", "2", "--movetime", "300", "--log-dir", str(tmp_path),
                   "--config", "b1;--backend torch;NNBatch=1", "--config", "b4g;--backend torch;NNBatch=4,NNGraphs=true",
                   "--json", str(out)])
    assert rc == 0
    rows = json.loads(out.read_text())["results"]
    assert [r["config"] for r in rows] == ["b1", "b4g"]
    for r in rows:
        assert r["searches"] == 2 and r["sims_per_s"] > 0 and r["rpc_failures"] == 0 and r["ab_depth"] > 0
    assert rows[0]["positions_per_request"] == pytest.approx(1.0) and rows[1]["positions_per_request"] > 1.0
