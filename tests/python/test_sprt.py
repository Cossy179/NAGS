"""Tests for tools/sprt.py: statistics, parsing, adjudication and real matches."""

import importlib.util
import math
import os
import random
import sys
import textwrap
import time
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
def test_resume_continues_counts_and_openings(tmp_path, capsys):
    common = ["--engine", _engine("nags_fast"), "--engine", _engine("nags_basic"), "--depth", "2", "--concurrency", "1",
              "--openings", str(ROOT / "tools/openings/nags_balanced.epd"), "--report", "1000"]
    full, part = tmp_path / "full.pgn", tmp_path / "part.pgn"
    assert sprt.main(common + ["--games", "8", "--pgnout", str(full)]) == 0
    full_status = [l for l in capsys.readouterr().out.splitlines() if l.startswith("Games ")][-1]
    assert sprt.main(common + ["--games", "4", "--pgnout", str(part)]) == 0
    log = tmp_path / "part.log"
    log.write_text(capsys.readouterr().out)
    assert sprt.main(common + ["--games", "8", "--pgnout", str(part), "--resume", str(log)]) == 0
    out = capsys.readouterr().out
    assert "Resuming after 4 games" in out
    # Deterministic engines at fixed depth: the resumed match equals the uninterrupted one.
    assert [l for l in out.splitlines() if l.startswith("Games ")][-1] == full_status
    assert part.read_text() == full.read_text()
    m = sprt.Match.__new__(sprt.Match)
    m.penta, m.wdl = sprt.Pentanomial(), [0, 0, 0]
    with pytest.raises(ValueError):
        m.resume("Games 3: +1 =1 -1 | penta [0, 1, 0, 0, 0]")  # 3 games cannot be whole pairs


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


def test_engine_paths_stay_whole(tmp_path):
    # run_round passes native paths: on Windows with backslashes, which POSIX
    # command-line splitting dropped (every match then failed to start), and
    # possibly with spaces.
    # A literal backslash in a file name only exists off Windows (on Windows it separates directories, and the
    # native path has backslashes anyway).
    exe = tmp_path / "dir with space" / ("engine.exe" if os.name == "nt" else "eng\\ine.exe")
    exe.parent.mkdir()
    exe.write_text("")
    assert sprt.engine_command(str(exe)) == [str(exe)]
    assert sprt.engine_command("python3 -u engine.py") == ["python3", "-u", "engine.py"]


class _RecordingEngine:
    def __init__(self, log):
        self.log = log

    def play(self, board, limit, info=None, game=None):
        self.log.append((board.turn, limit))
        return type("Play", (), {"move": next(iter(board.legal_moves)), "info": {}})()


class _RecordingPlayer:
    def __init__(self, spec, log):
        self.spec = spec
        self.engine = _RecordingEngine(log)

    def ensure(self):
        return self.engine

    def kill(self):
        pass


def test_per_engine_limits():
    """Time odds and a node-limited opponent: each side gets its own limit."""
    log = []
    timed = sprt.EngineSpec("timed", ["x"], {}, sprt.make_limits("10+0.1", None, None, None, 0.0))
    nodes = sprt.EngineSpec("nodes", ["y"], {}, sprt.make_limits(None, 5000, None, None, 0.0))
    shared = sprt.make_limits(None, None, 3, None, 0.0)
    adj = sprt.Adjudication(max_plies=6)
    opening = sprt.Opening(chess.STARTING_FEN, [])
    game = sprt.play_game(_RecordingPlayer(timed, log), _RecordingPlayer(nodes, log), opening, shared, adj)
    assert len(log) == 6
    for turn, limit in log:
        if turn == chess.WHITE:  # the timed engine: its clock, and its own shown for the clockless opponent
            assert limit.nodes is None and limit.white_inc == 0.1 and limit.black_clock == limit.white_clock
        else:
            assert limit.nodes == 5000 and limit.white_clock is None
    assert game.game.headers["TimeControl"] == "10+0.1 vs nodes 5000"
    # Without its own limits an engine plays with the match's.
    log.clear()
    plain = sprt.EngineSpec("plain", ["z"], {})
    sprt.play_game(_RecordingPlayer(plain, log), _RecordingPlayer(nodes, log), opening, shared, adj)
    assert {l.depth for t, l in log if t == chess.WHITE} == {3} and {l.nodes for t, l in log if t == chess.BLACK} == {5000}
    assert sprt.describe_limits(shared, shared) == "depth 3"


@needs_engines
def test_time_odds_match_end_to_end(tmp_path, capsys):
    code = sprt.main(["--engine", _engine("nags_fast"), "--engine", _engine("nags_basic"), "--tc1", "1+0.01",
                      "--nodes2", "500", "--games", "2", "--openings", str(ROOT / "tools/openings/nags_balanced.epd"),
                      "--report", "1000"])
    assert code == 0
    assert "1+0.01 vs nodes 500" in capsys.readouterr().out
    with pytest.raises(SystemExit):  # no limit at all
        sprt.main(["--engine", "a", "--engine", "b", "--games", "2"])
    with pytest.raises(SystemExit):  # one engine without a limit
        sprt.main(["--engine", "a", "--engine", "b", "--nodes1", "10", "--games", "2"])


# Infrastructure failures make a match invalid; genuine forfeits stay losses and are reported.

_FAKE = """\
import sys
moves = 0
for line in sys.stdin:
    cmd = line.split()
    if not cmd:
        continue
    if cmd[0] == "uci":
        print("id name Fake"); print("option name Mode type string default x"); print("uciok", flush=True)
    elif cmd[0] == "setoption" and len(cmd) > 4 and cmd[4] == "complain":
        print("info string Mode: cannot open 'complain'", flush=True)
    elif cmd[0] == "isready":
        print("readyok", flush=True)
    elif cmd[0] == "go":
        moves += 1
        if moves > 1:  # the preflight's search works; the first move of a game crashes
            sys.exit(1)
        print("bestmove e2e4", flush=True)
    elif cmd[0] == "quit":
        break
"""


def _fake(tmp_path):
    path = tmp_path / "fake_engine.py"
    path.write_text(_FAKE)
    return [sys.executable, str(path)]


def _sprt_match(specs, pairs=4):
    return sprt.Match(specs, [sprt.Opening(chess.STARTING_FEN, [])], sprt.Limits(depth=1), sprt.Adjudication(),
                      max_pairs=pairs, concurrency=1, sprt=(0, 10), alpha=0.05, beta=0.05, pgnout=None, seed=1,
                      shuffle=False)


def test_engines_that_cannot_start_give_no_decision(tmp_path, capsys):
    # The Windows failure of 2026-10-23: every launch failed, every game was scored as a crash, and pairs of
    # crashes looked like a tie until H0 was accepted. Now the first failed launch ends the match, undecided.
    missing = [str(tmp_path / "no_such_engine")]
    match = _sprt_match((sprt.EngineSpec("a", missing), sprt.EngineSpec("b", missing)), pairs=500)
    assert match.run(report_interval=1000) == 3
    assert match.decision is None and match.wdl == [0, 0, 0]
    out = capsys.readouterr().out
    assert "Match invalid" in out and "accepted" not in out
    code = sprt.main(["--engine", missing[0], "--engine", missing[0], "--depth", "1", "--sprt", "0", "10",
                      "--report", "1000"])
    out = capsys.readouterr()
    assert code == 3 and "preflight" in out.err and "accepted" not in out.out


def test_preflight_checks_options_and_records_a_fingerprint(tmp_path):
    fake = _fake(tmp_path)
    fp = sprt.preflight(sprt.EngineSpec("fake", fake, {"Mode": "fine"}))
    assert fp["id"] == "Fake" and fp["options"] == {"Mode": "fine"} and fp["sha256"] is not None  # the interpreter
    with pytest.raises(sprt.InfrastructureError, match="did not take its options"):
        sprt.preflight(sprt.EngineSpec("fake", fake, {"Mode": "complain"}))
    with pytest.raises(sprt.InfrastructureError, match="unknown UCI option"):
        sprt.preflight(sprt.EngineSpec("fake", fake, {"Threads": "2"}))


def test_a_crash_during_a_game_is_a_loss_and_reported(tmp_path, capsys):
    if _engine("nags_basic") is None:
        pytest.skip("engines not built")
    match = _sprt_match((sprt.EngineSpec("nags", [_engine("nags_basic")]), sprt.EngineSpec("fake", _fake(tmp_path))),
                        pairs=2)
    assert match.run(report_interval=1000) in (1, 2)  # games were played: a decision or the cap, never invalid
    assert match.wdl == [4, 0, 0] and match.forfeits == {("fake", "crash"): 4}
    assert "Forfeits: fake: 4 crashed or hung" in capsys.readouterr().out


@needs_engines
def test_preflight_rejects_a_network_that_does_not_load(tmp_path, capsys):
    code = sprt.main(["--engine", _engine("nags_fast"), "--engine", _engine("nags_basic"), "--depth", "1",
                      "--option1", f"EvalFile={tmp_path / 'missing.nnue'}", "--games", "2", "--report", "1000"])
    out = capsys.readouterr()
    assert code == 3 and "EvalFile" in out.err and "Games 2" not in out.out
    fp = sprt.preflight(sprt.EngineSpec("fast", [_engine("nags_fast")], {"EvalFile": "<embedded>"}))
    assert fp["sha256"] == sprt.file_sha256(_engine("nags_fast")) and fp["id"]
    assert fp["network"] or any("hand-written" in s or "classical" in s for s in fp["said"])


@needs_engines
def test_resume_refuses_other_engines(tmp_path, capsys):
    common = ["--depth", "1", "--concurrency", "1", "--openings", str(ROOT / "tools/openings/nags_balanced.epd"),
              "--report", "1000"]
    pair = ["--engine", _engine("nags_fast"), "--engine", _engine("nags_basic")]
    assert sprt.main(pair + common + ["--games", "2"]) == 0
    log = tmp_path / "m.log"
    log.write_text(capsys.readouterr().out)
    assert "Fingerprint: " in log.read_text()
    swapped = ["--engine", _engine("nags_basic"), "--engine", _engine("nags_fast")]
    assert sprt.main(swapped + common + ["--games", "4", "--resume", str(log)]) == 3
    assert "other engines or settings" in capsys.readouterr().err
    assert sprt.main(swapped + common + ["--games", "4", "--resume", str(log), "--force-resume"]) == 0
    assert "WARNING: resuming although engine 1 sha256" in capsys.readouterr().out
    assert sprt.main(pair + common + ["--games", "4", "--resume", str(log)]) == 0  # the same engines resume


def test_tools_keep_windows_engine_paths_whole(tmp_path):
    exe = tmp_path / "dir with space" / ("engine.exe" if os.name == "nt" else "eng\\ine.exe")
    exe.parent.mkdir()
    exe.write_text("")
    spec = importlib.util.spec_from_file_location("ladder", ROOT / "tools" / "ladder.py")
    ladder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ladder)
    rung = ladder.rung_spec({"name": "r", "engine": str(exe), "limits": {"nodes": 10}})
    assert rung.command == [str(exe)]


@needs_engines
def test_an_interrupted_match_is_not_reported_as_finished(tmp_path, monkeypatch, capsys):
    # Ctrl-C used to end with "SPRT inconclusive: maximum number of games reached" (exit 2), which run_round
    # reads as a finished match without a decision, so it never resumed it.
    import types

    calls = []

    def sleep(seconds):
        calls.append(seconds)
        if len(calls) == 1:
            raise KeyboardInterrupt
        time.sleep(seconds)

    monkeypatch.setattr(sprt, "time", types.SimpleNamespace(monotonic=time.monotonic, sleep=sleep,
                                                            perf_counter=time.perf_counter, time=time.time))
    match = _sprt_match((sprt.EngineSpec("a", [_engine("nags_fast")]), sprt.EngineSpec("b", [_engine("nags_basic")])),
                        pairs=500)
    assert match.run(report_interval=1000) == 4
    out = capsys.readouterr().out
    assert "Match interrupted after" in out and "inconclusive" not in out and "accepted" not in out
    sys.path.insert(0, str(ROOT / "tools" / "nnue"))
    import run_round

    log = tmp_path / "m.log"
    log.write_text(out)
    assert run_round.sprt_result(log) is None  # to be resumed, not taken as a result


@needs_engines
def test_resume_skips_a_status_line_cut_off_by_a_kill(tmp_path, capsys):
    common = ["--engine", _engine("nags_fast"), "--engine", _engine("nags_basic"), "--depth", "1", "--concurrency",
              "1", "--openings", str(ROOT / "tools/openings/nags_balanced.epd"), "--report", "1000"]
    assert sprt.main(common + ["--games", "2"]) == 0
    log = tmp_path / "m.log"
    log.write_text(capsys.readouterr().out + "Games 4: +2 =1 -1 | penta [0, 1")  # killed while writing
    assert sprt.main(common + ["--games", "4", "--resume", str(log)]) == 0
    assert "Resuming after 2 games" in capsys.readouterr().out
