"""tools/mistakes.py: a mistake is classified by what fixes it (more nodes, searching the reference's move,
neither), rows are resumable, and the summary writes a replayable EPD suite."""

import argparse
import json
import sys
from pathlib import Path

import chess
import chess.engine
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import mistakes  # noqa: E402

FEN = "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"
GOOD, BAD, OTHER = chess.Move.from_uci("f1b5"), chess.Move.from_uci("h2h3"), chess.Move.from_uci("b1c3")


class Scripted:
    """A stand-in engine: analyse() answers from a function of (limit, root moves)."""

    def __init__(self, answer):
        self.answer = answer

    def analyse(self, board, limit, root_moves=None, game=None):
        move, cp = self.answer(limit, root_moves)
        return {"pv": [move], "score": chess.engine.PovScore(chess.engine.Cp(cp), board.turn)}


class Pair:
    def __init__(self, nags, ref):
        self.nags, self.ref = nags, ref

    def search(self, eng, board, limit, moves=None):
        return eng.analyse(board, limit, root_moves=moves)


def _args(**kw):
    base = dict(nodes=1000, ref_nodes=10**6, budgets=[4, 16], threshold=50.0, reference="ref", ref_scale=1.0)
    base.update(kw)
    return argparse.Namespace(**base)


def _reference(limit, moves):
    if moves == [BAD]:
        return BAD, -40
    if moves == [OTHER]:
        return OTHER, 20
    return GOOD, 60


@pytest.mark.parametrize("cls", ["time", "selectivity", "evaluation"])
def test_classes(cls):
    def nags(limit, moves):
        if moves == [GOOD]:  # the reference's move searched alone with N nodes
            return GOOD, 30 if cls == "selectivity" else -10
        if limit.depth == 1:
            return (moves[0], 5)
        if limit.nodes and limit.nodes >= 4000 and cls == "time":
            return GOOD, 50
        return BAD, 10

    row = mistakes.examine(Pair(Scripted(nags), Scripted(_reference)), FEN, _args())
    assert row["mistake"] and row["loss"] == 100 and row["move"] == "h2h3" and row["ref_move"] == "f1b5"
    assert row["class"] == cls
    assert row["fixed_by_budget"] == (4 if cls == "time" else None)
    assert row["budgets"]["16"]["loss"] == (0 if cls == "time" else 100)


def test_small_losses_are_not_mistakes_and_scale_applies():
    nags = lambda limit, moves: (OTHER, 0)  # noqa: E731
    row = mistakes.examine(Pair(Scripted(nags), Scripted(_reference)), FEN, _args())
    assert not row["mistake"] and row["loss"] == 40 and "class" not in row
    row = mistakes.examine(Pair(Scripted(nags), Scripted(_reference)), FEN, _args(ref_scale=2.0))
    assert row["mistake"] and row["loss"] == 80


def _engine():
    for c in (ROOT / "build" / "nags_enhanced", ROOT / "build" / "Release" / "nags_enhanced.exe"):
        if c.exists():
            return c
    return None


@pytest.mark.skipif(_engine() is None, reason="nags_enhanced not built")
def test_collect_resume_and_epd(tmp_path):
    src = tmp_path / "p.epd"
    src.write_text("\n".join(l for l in (ROOT / "tools" / "openings" / "nags_balanced.epd").read_text()
                             .splitlines()[:12]) + "\n")
    out = tmp_path / "rows.jsonl"
    argv = ["collect", "--engine", str(_engine()), "--source", str(src), "--positions", "6", "--nodes", "300",
            "--ref-nodes", "20000", "--threshold", "20", "--out", str(out), "--workers", "2"]
    mistakes.main(argv)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    assert len(rows) == 6 and all("loss" in r for r in rows)
    mistakes.main(argv)  # nothing left to do: no row added
    assert len(out.read_text().splitlines()) == 6
    epd = tmp_path / "m.epd"
    rep = mistakes.summary(argparse.Namespace(rows=[str(out)], epd=str(epd), json=None))
    lines = epd.read_text().splitlines()
    assert len(lines) == rep["mistakes"] == sum(r["mistake"] for r in rows)
    for line in lines:
        board, ops = chess.Board.from_epd(line)
        assert ops["bm"][0] in board.legal_moves
