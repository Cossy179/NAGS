"""tools/suite.py: making a suite with a reference engine and scoring an engine on it."""

import importlib.util
import json
import sys
from pathlib import Path

import chess
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("suite", ROOT / "tools" / "suite.py")
suite = importlib.util.module_from_spec(_spec)
sys.modules["suite"] = suite
_spec.loader.exec_module(suite)


def _engine(name):
    for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe"):
        if c.exists():
            return str(c)
    return None


def test_category():
    b = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")
    assert suite.category(b, chess.Move.from_uci("a1a8"), chess.engine.PovScore(chess.engine.Mate(1), chess.WHITE),
                          6) == "endgame"
    b = chess.Board()
    assert suite.category(b, chess.Move.from_uci("e2e4"), chess.engine.PovScore(chess.engine.Cp(30), chess.WHITE),
                          6) == "positional"


@pytest.mark.skipif(_engine("nags_enhanced") is None or _engine("nags_basic") is None, reason="engines not built")
def test_make_and_run(tmp_path):
    src = tmp_path / "src.epd"
    src.write_text("\n".join([
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq -",  # Qxf7#
        "4k3/8/8/3q4/8/8/8/3QK3 w - -",                                   # Qxd5
        "6k1/5ppp/8/8/8/8/5PPP/R5K1 w - -",                               # Ra8#
    ]) + "\n")
    out = tmp_path / "suite.epd"
    assert suite.main(["make", "--source", str(src), "--plies", "0", "--count", "3", "--reference",
                       _engine("nags_enhanced"), "--nodes", "20000", "--min-gap", "100", "--out", str(out)]) == 0
    rows = suite.read_suite(str(out))
    assert {r["bm"][0] for r in rows} == {"f3f7", "d1d5", "a1a8"}
    assert {r["category"] for r in rows} <= set(suite.CATEGORIES)
    base = tmp_path / "base.json"
    assert suite.main(["run", "--suite", str(out), "--engine", _engine("nags_enhanced"), "--nodes", "20000",
                       "--json", str(base)]) == 0
    assert json.loads(base.read_text())["all"] == 1.0
    new = tmp_path / "new.json"
    assert suite.main(["run", "--suite", str(out), "--engine", _engine("nags_basic"), "--nodes", "50",
                       "--json", str(new), "--baseline", str(base)]) == 0
    summary = json.loads(new.read_text())
    assert set(summary["regressions"]) == {i for i, r in summary["positions"].items() if not r["solved"]}


def test_pgn_source(tmp_path):
    pgn = tmp_path / "games.pgn"
    pgn.write_text('[Event "t"]\n[Result "1-0"]\n\n1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7# 1-0\n\n'
                   '[Event "t"]\n[Result "*"]\n\n1. d4 d5 *\n')
    fens = list(suite.source_positions(str(pgn), 100, 0, seed=1))
    # Every position after a move, except the final mate (game over).
    assert len(fens) == 6 + 2 and len(set(fens)) == 8
    assert all(not chess.Board(f).is_game_over() for f in fens)
    assert len(list(suite.source_positions(str(pgn), 3, 0, seed=1))) == 3


@pytest.mark.skipif(_engine("nags_enhanced") is None, reason="engines not built")
def test_make_with_too_few_qualifying_positions(tmp_path):
    src = tmp_path / "src.epd"
    src.write_text("4k3/8/8/3q4/8/8/8/3QK3 w - -\n" + chess.STARTING_FEN.rsplit(" ", 2)[0] + "\n")
    out = tmp_path / "suite.epd"
    # Only Qxd5 qualifies; the run stops after --max-candidates and writes what it found.
    assert suite.main(["make", "--source", str(src), "--plies", "0", "--count", "5", "--max-candidates", "50",
                       "--reference", _engine("nags_enhanced"), "--nodes", "20000", "--min-gap", "100",
                       "--out", str(out)]) == 0
    assert [r["bm"] for r in suite.read_suite(str(out))] == [["d1d5"]]
    assert not (tmp_path / "suite.epd.partial").exists()
