"""Smoke test of tools/gnn_bench.py with tiny models."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import gnn_bench  # noqa: E402


def test_bench_reports_both_formats(tmp_path, capsys):
    out = tmp_path / "bench.json"
    rows = gnn_bench.run(["--tiny", "--batch-sizes", "1,3", "--repeats", "1", "--warmup", "0", "--device", "cpu",
                          "--mc-samples", "2", "--json", str(out)])
    assert [(r["model"], r["batch"]) for r in rows] == [("nags-gnn-v2", 1), ("nags-gnn-v2", 3),
                                                        ("nags-gnn-v3", 1), ("nags-gnn-v3", 3)]
    for r in rows:
        assert r["build_ms"] > 0 and r["model_ms"] > 0 and r["total_pos_s"] > 0 and r["params"] > 0
    assert len(json.loads(out.read_text())["rows"]) == 4
    assert "build pos/s" in capsys.readouterr().out


def test_positions_are_distinct_and_playable():
    import chess

    fens = gnn_bench.positions(40, seed=1)
    assert len(set(fens)) == 40 and all(any(chess.Board(f).legal_moves) for f in fens)
