"""tools/nnue/tb_data.py: tablebase positions with results under the fifty-move rule, written as training
records, and the endgame measurement (with the 3-piece tables in tests/data/syzygy)."""

import json
import random
import sys
from pathlib import Path

import chess
import chess.syzygy
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
import rescore  # noqa: E402
import tb_data  # noqa: E402
import train  # noqa: E402

TB = ROOT / "tests" / "data" / "syzygy"


def _engine():
    for c in (ROOT / "build" / "nags_enhanced", ROOT / "build" / "Release" / "nags_enhanced.exe"):
        if c.exists():
            return c
    return None


def test_random_positions_are_legal_quiet_and_of_the_material():
    rng = random.Random(3)
    for material in ("KPvK", "KvKR", "KRPvKR", "KBNvK"):
        w, b = tb_data.parse_material(material)
        for _ in range(50):
            board = tb_data.random_position(material, rng)
            assert board.is_valid() and not board.is_check()
            assert not any(board.is_capture(m) or m.promotion for m in board.legal_moves)
            have = sorted(p.symbol() for p in board.piece_map().values())
            assert have == sorted(w.upper() + b.lower())
    with pytest.raises(SystemExit):
        tb_data.parse_material("RKvK")


def test_cursed_wins_are_draws():
    class Fake:
        def __init__(self, wdl):
            self.wdl = wdl

        def probe_wdl(self, board):
            return self.wdl
    assert [tb_data.tb_result(Fake(w), None) for w in (2, 1, 0, -1, -2)] == ["win", "draw", "draw", "draw", "loss"]


@pytest.mark.skipif(_engine() is None, reason="nags_enhanced not built")
def test_generate_writes_records_with_the_tablebase_result(tmp_path):
    out = tmp_path / "tb.bin"
    tb_data.main(["generate", "--syzygy", str(TB), "--material", "KPvK", "KvKB", "--count", "40", "--engine",
                  str(_engine()), "--nodes", "500", "--out", str(out), "--workers", "2"])
    recs = np.fromfile(out, np.uint8).reshape(-1, 32)
    counts = json.loads(Path(str(out) + ".tb.json").read_text())["materials"]
    assert len(recs) == sum(c["positions"] for c in counts.values()) and counts["KvKB"]["draw"] == counts["KvKB"]["positions"]
    with chess.syzygy.open_tablebase(str(TB)) as tb:
        for r in recs:
            board = chess.Board(rescore.record_fen(r))
            board.castling_rights = 0
            res = {"win": 1.0, "draw": 0.5, "loss": 0.0}[tb_data.tb_result(tb, board)]
            white = res if board.turn == chess.WHITE else 1.0 - res
            assert r[26] / 2 == white
    feats, stm, score, result = train.decode(__import__("torch").from_numpy(recs))
    assert len(feats) == len(recs)


@pytest.mark.skipif(_engine() is None, reason="nags_enhanced not built")
def test_measure_reports_by_material(tmp_path):
    rep = tb_data.main(["measure", "--syzygy", str(TB), "--material", "KvKB", "KRvK", "--count", "20", "--engine",
                        str(_engine()), "--nodes", "2000", "--json", str(tmp_path / "m.json")])
    assert rep == 0
    m = json.loads((tmp_path / "m.json").read_text())["materials"]
    assert m["KvKB"]["positions"] == 20 and m["KvKB"]["right"] == 20  # insufficient material: the engine says 0
    assert m["KRvK"]["right"] >= 18
