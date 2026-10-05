"""Tests for tools/sprt.py: statistics, parsing, adjudication and real matches."""

import importlib.util
import math
import random
import sys
import textwrap
from pathlib import Path

import chess
import chess.pgn
import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("sprt", ROOT / "tools" / "sprt.py")
sprt = importlib.util.module_from_spec(_spec)
sys.modules["sprt"] = sprt
_spec.loader.exec_module(sprt)


def test_elo_score_round_trip():
    for elo in (-400, -50, 0, 5, 100, 400):
        assert sprt.score_to_elo(sprt.elo_to_score(elo)) == pytest.approx(elo, abs=1e-6)
    assert sprt.elo_to_score(0) == 0.5


def test_bounds():
    lo, hi = sprt.sprt_bounds(0.05, 0.05)
    assert lo == pytest.approx(-2.944, abs=1e-3) and hi == pytest.approx(2.944, abs=1e-3)


def test_llr_sign_and_zero_point():
    p = sprt.Pentanomial([10, 40, 100, 40, 10])  # symmetric: mean score exactly 0.5
    assert p.llr(-5, 5) == pytest.approx(0.0, abs=1e-9)
    assert p.llr(0, 5) < 0  # observed 0 Elo favours H0 (elo0 = 0)
    better = sprt.Pentanomial([5, 30, 100, 50, 15])
    assert better.llr(0, 5) > 0
    elo, err = better.elo()
    assert elo > 0 and err > 0
    # More data at the same proportions: larger |LLR|, tighter interval.
    more = sprt.Pentanomial([c * 4 for c in better.counts])
    assert more.llr(0, 5) == pytest.approx(4 * better.llr(0, 5), rel=1e-6)
    assert more.elo()[1] < err


def _pair_probabilities(elo, draw_rate=0.4):
    s = sprt.elo_to_score(elo)
    w, d = s - draw_rate / 2, draw_rate
    l = 1 - w - d
    game = [l, d, w]  # 0, 0.5, 1 point
    probs = [0.0] * 5
    for i, a in enumerate(game):
        for j, b in enumerate(game):
            probs[i + j] += a * b
    return probs


def _simulate(true_elo, elo0, elo1, runs, rng, max_pairs=20000):
    lo, hi = sprt.sprt_bounds(0.05, 0.05)
    probs = _pair_probabilities(true_elo)
    accepted_h1 = 0
    for _ in range(runs):
        p = sprt.Pentanomial()
        for _ in range(max_pairs):
            p.add(rng.choices(range(5), probs)[0] / 2)
            llr = p.llr(elo0, elo1)
            if llr >= hi:
                accepted_h1 += 1
                break
            if llr <= lo:
                break
    return accepted_h1 / runs


def test_sprt_error_rates_monte_carlo():
    """With alpha = beta = 0.05 the test should accept H1 for an equal engine
    about 5% of the time and for an elo1-stronger engine about 95% of the time."""
    rng = random.Random(1234)
    false_positive = _simulate(0.0, 0.0, 20.0, runs=200, rng=rng)
    power = _simulate(20.0, 0.0, 20.0, runs=200, rng=rng)
    assert false_positive <= 0.10, false_positive
    assert power >= 0.88, power


def test_parse_tc():
    assert sprt.parse_tc("8+0.08") == (8.0, 0.08)
    assert sprt.parse_tc("60") == (60.0, 0.0)
    with pytest.raises(ValueError):
        sprt.parse_tc("0+1")


def test_load_openings(tmp_path):
    epd = tmp_path / "o.epd"
    epd.write_text(textwrap.dedent("""\
        # comment
        rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq -
        rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2
        rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - bm Nc6;
        """))
    ops = sprt.load_openings(str(epd))
    assert len(ops) == 3 and ops[0].board().turn == chess.BLACK

    pgn = tmp_path / "o.pgn"
    pgn.write_text('[Event "a"]\n\n1. e4 e5 2. Nf3 Nc6 3. Bb5 *\n\n[Event "b"]\n\n1. d4 d5 *\n')
    ops = sprt.load_openings(str(pgn), plies=3)
    assert [len(o.moves) for o in ops] == [3, 2]
    assert ops[0].board().fen().startswith("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2")  # e4 e5 Nf3

    shipped = ROOT / "tools" / "openings" / "nags_balanced.epd"
    assert len(sprt.load_openings(str(shipped))) >= 100


def test_adjudication():
    adj = sprt.Adjudication(draw_movenumber=40, draw_count=4, draw_score=10, resign_count=3, resign_score=500)
    assert sprt.adjudicate([600] * 6, 30, adj) == ("1-0", "adjudication: black resigns")
    assert sprt.adjudicate([-700] * 6, 30, adj) == ("0-1", "adjudication: white resigns")
    assert sprt.adjudicate([600] * 5 + [100], 30, adj) is None
    assert sprt.adjudicate([3, -5] * 4, 45, adj) == ("1/2-1/2", "adjudication: draw")
    assert sprt.adjudicate([3, -5] * 4, 39, adj) is None  # too early
    assert sprt.adjudicate([3, None] * 4, 45, adj) is None  # missing scores never adjudicate


def _engine(name):
    for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe"):
        if c.exists():
            return str(c)
    return None


needs_engines = pytest.mark.skipif(_engine("nags_basic") is None or _engine("nags_fast") is None,
                                   reason="engines not built")


@needs_engines
def test_fixed_match_end_to_end(tmp_path):
    pgn = tmp_path / "games.pgn"
    code = sprt.main(["--engine", _engine("nags_fast"), "--engine", _engine("nags_basic"), "--depth", "2",
                      "--games", "6", "--concurrency", "2", "--openings", str(ROOT / "tools/openings/nags_balanced.epd"),
                      "--pgnout", str(pgn), "--report", "1000"])
    assert code == 0
    games = []
    with open(pgn) as f:
        while (g := chess.pgn.read_game(f)) is not None:
            games.append(g)
    assert len(games) == 6
    for g in games:
        assert g.headers["Result"] in ("1-0", "0-1", "1/2-1/2")
        board = g.board()
        for m in g.mainline_moves():
            assert m in board.legal_moves
            board.push(m)
    # Each opening is played with both colours.
    assert games[0].board().fen() == games[1].board().fen()
    assert games[0].headers["White"] == games[1].headers["Black"]


@needs_engines
def test_broken_engine_loses(tmp_path):
    fake = tmp_path / "broken_engine.py"
    fake.write_text(textwrap.dedent("""\
        import sys
        for line in sys.stdin:
            cmd = line.split()
            if not cmd:
                continue
            if cmd[0] == "uci":
                print("id name Broken"); print("uciok", flush=True)
            elif cmd[0] == "isready":
                print("readyok", flush=True)
            elif cmd[0] == "go":
                print("bestmove a1a1", flush=True)
            elif cmd[0] == "quit":
                break
        """))
    match = sprt.Match((sprt.EngineSpec("nags", [_engine("nags_basic")]), sprt.EngineSpec("broken", [sys.executable, str(fake)])),
                       [sprt.Opening(chess.STARTING_FEN, [])], sprt.Limits(depth=1), sprt.Adjudication(), max_pairs=2,
                       concurrency=1, sprt=None, alpha=0.05, beta=0.05, pgnout=None, seed=1, shuffle=False)
    assert match.run(report_interval=1000) == 0
    assert match.wdl == [4, 0, 0]
    assert any("broken" in reason for reason in match.reasons)


def test_calibration_combine():
    spec = importlib.util.spec_from_file_location("calibrate", ROOT / "tools" / "calibrate.py")
    calibrate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(calibrate)
    rating, err = calibrate.combine([(1800, 100), (1800, 100)])
    assert rating == pytest.approx(1800) and err == pytest.approx(100 / math.sqrt(2))
    # The more precise estimate dominates.
    rating, _ = calibrate.combine([(1700, 50), (2000, 500)])
    assert 1700 < rating < 1710
    assert math.isnan(calibrate.combine([(1500, float("inf"))])[0])
