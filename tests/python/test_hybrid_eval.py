"""tools/hybrid_eval.py: parsing the hybrid's summary line, and short accuracy / decisions runs."""

import argparse
import json
import math
import sys
from pathlib import Path

import chess
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import hybrid_eval  # noqa: E402

LINE = ("info string nags alpha_beta_depth 9 mcts_sims 1200 mcts_top g1f3 visits 700 q 0.120 (cp 48) evaluator network "
        "chose g1f3 (mcts, verified >= cp 3 (alpha-beta cp 28)) mcts_ms 980.0 mcts_sims_per_s 1224.5 rpc_requests 150 "
        "rpc_failures 0 rpc_wait_ms 800.0 rpc_mean_ms 5.3 rpc_max_ms 9.0 rpc_positions 1200 rpc_batch 8.0 cache_hits 40 "
        "mcts_collisions 3 mcts_reused 0 tt_values 0 smart_stop 0 root_order 0 alpha_beta_move e2e4 alpha_beta_score 28")


def test_parse_summary_and_categories():
    f = hybrid_eval.parse_summary(["info depth 9", LINE, "info string nags_meta time_left 1"])
    assert f["chose"] == "g1f3" and f["alpha_beta_move"] == "e2e4" and f["mcts_sims"] == "1200"
    assert f["reason"] == "mcts, verified >= cp 3 (alpha-beta cp 28)"
    assert hybrid_eval.category(f) == "override (verified)"
    cases = {
        "alpha-beta (MCTS off)": "mcts off",
        "alpha-beta, mcts proposal rejected (< cp 3)": "proposal rejected",
        "alpha-beta, no time to verify the mcts proposal": "no time to verify",
        "alpha-beta (mate score: no override)": "mate score",
    }
    for reason, cat in cases.items():
        assert hybrid_eval.category({**f, "reason": reason}) == cat
    assert hybrid_eval.category({**f, "reason": "alpha-beta", "mcts_top": "e2e4"}) == "agree"
    assert hybrid_eval.category({**f, "reason": "alpha-beta"}) == "differs, not proposed"
    assert hybrid_eval.category({**f, "mcts_sims": "0", "reason": "alpha-beta"}) == "mcts off"
    assert hybrid_eval.parse_summary(["bestmove e2e4"]) == {}


def _tiny_checkpoint(path):
    from gnn_v3 import RelationalGNNEvaluator, save_checkpoint_v3

    torch.manual_seed(0)
    save_checkpoint_v3(RelationalGNNEvaluator(hidden_dim=16, layers=1, num_bases=2, device=torch.device("cpu")),
                       str(path))


def test_accuracy(tmp_path, capsys):
    model = tmp_path / "v3.pth"
    _tiny_checkpoint(model)
    rows = tmp_path / "rows.jsonl"
    boards = [chess.Board(), chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")]
    with open(rows, "w") as f:
        for b in boards:
            legal = [m.uci() for m in b.legal_moves]
            f.write(json.dumps({"fen": b.fen(), "legal": legal, "scored": [{"move": legal[0], "cp": 30},
                                                                          {"move": legal[1], "cp": 10}]}) + "\n")
    out = hybrid_eval.accuracy(argparse.Namespace(model=str(model), data=str(rows), limit=0, batch=8, device="cpu",
                                                  threads=1))
    assert out["positions"] == 2 and 0 <= out["top1"] <= out["top3"] <= 1 and out["nll"] > 0
    assert abs(out["uniform_nll"] - sum(math.log(len(list(b.legal_moves))) for b in boards) / 2) < 1e-9
    assert "held-out positions" in capsys.readouterr().out


@pytest.mark.skipif(not (ROOT / "build" / "nags").exists(), reason="nags not built")
def test_decisions(tmp_path):
    model = tmp_path / "v3.pth"
    _tiny_checkpoint(model)
    out_rows = tmp_path / "rows.jsonl"
    rc = hybrid_eval.main(["decisions", "--model", str(model), "--positions", "2", "--movetime", "400",
                           "--reference-nodes", "20000", "--log", str(tmp_path / "rpc.log"), "--option", "NNBatch=4",
                           "--out", str(out_rows), "--json", str(tmp_path / "summary.json")])
    assert rc == 0
    rows = [json.loads(l) for l in out_rows.read_text().splitlines()]
    assert len(rows) == 2 and all(r["alpha_beta_move"] and r["reference_best"] for r in rows)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["positions"] == 2 and sum(summary["categories"].values()) == 2


def test_gauntlet_status_and_tc(tmp_path):
    log = tmp_path / "x.log"
    assert hybrid_eval.read_status(log) is None
    log.write_text("a vs b: nodes 2000\nGames 2: +1 =1 -0 | penta [0, 0, 1, 1, 0] | Elo +191.0 ± 900.1\n")
    s = hybrid_eval.read_status(log)
    assert s["games"] == 2 and s["wdl"] == [1, 1, 0] and s["penta"] == [0, 0, 1, 1, 0] and not s["finished"]
    log.write_text(log.read_text() + "Games 4: +2 =1 -1 | penta [0, 1, 0, 1, 0] | Elo +88.0 ± 410.0\n"
                                     "Endings: checkmate 3, threefold_repetition 1\n")
    s = hybrid_eval.read_status(log)
    assert s["games"] == 4 and s["elo"] == 88.0 and s["error"] == 410.0 and s["finished"]
    assert hybrid_eval.soft_ms("10+0.1") == pytest.approx(350.0) and hybrid_eval.soft_ms("60") == pytest.approx(1500.0)


@pytest.mark.skipif(not (ROOT / "build" / "nags").exists(), reason="nags not built")
def test_gauntlet(tmp_path, capsys):
    model = tmp_path / "v3.pth"
    _tiny_checkpoint(model)
    out = tmp_path / "g"
    argv = ["gauntlet", "--model", str(model), "--arms", "hybrid", "ab-2t", "--mcts-min-time", "0", "--games", "2",
            "--out-dir", str(out), "--", "--nodes", "3000"]
    assert hybrid_eval.main(argv) == 0
    res = json.loads((out / "results.json").read_text())
    assert res["baseline"] == "ab" and [r["arm"] for r in res["results"]] == ["hybrid", "ab-2t"]
    assert all(r["games"] == 2 and r["finished"] for r in res["results"])
    assert "MctsMinTime=0" in res["results"][0]["options"] and "Threads=2" in res["results"][1]["options"]
    # Run again: both matches are finished, so nothing is played and the results stand.
    capsys.readouterr()
    assert hybrid_eval.main(argv) == 0
    assert "==" not in capsys.readouterr().out
    with pytest.raises(SystemExit):
        hybrid_eval.main(["gauntlet", "--model", str(model), "--arms", "ab", "--out-dir", str(out)])
    # An unknown arm fails before anything is played.
    with pytest.raises(SystemExit):
        hybrid_eval.main(["gauntlet", "--model", str(model), "--arms", "nonsense", "--out-dir", str(out)])
