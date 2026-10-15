"""tools/nnue/run_round.py: data generation to a target, training several
designs, resuming, and reading match results."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
import run_round  # noqa: E402
import train  # noqa: E402


def _built(name):
    return any(c.exists() for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe"))


@pytest.mark.skipif(not _built("nags_datagen"), reason="nags_datagen not built")
def test_round_generates_trains_and_resumes(tmp_path, capsys):
    args = ["--data-dir", str(tmp_path), "--target-positions", "1500", "--games", "8", "--nodes", "300",
            "--threads", "2", "--designs", "current", "kb4", "--epochs", "1", "--device", "cpu", "--skip-test"]
    run_round.main(args)
    files = run_round.data_files(tmp_path)
    positions = run_round.count_positions(files)
    assert positions >= 1500 and len(files) >= 2
    nets = {d: tmp_path / f"net_{d}_{positions}_e1.nnue" for d in ("current", "kb4")}
    assert all(n.exists() for n in nets.values())
    assert train.read_net(nets["kb4"])[4]["king_buckets"] == 4
    capsys.readouterr()
    # Running again: the target is met and both networks are finished.
    stamps = {d: n.stat().st_mtime for d, n in nets.items()}
    run_round.main(args)
    out = capsys.readouterr().out
    assert "--out" not in out and "  train " not in out  # no data generation command, no training epoch
    assert run_round.count_positions(run_round.data_files(tmp_path)) == positions
    assert {d: n.stat().st_mtime for d, n in nets.items()} == stamps


def test_next_seed_and_text_conversion(tmp_path):
    (tmp_path / "selfplay_7.bin").write_bytes(b"\0" * 64)
    (tmp_path / "selfplay_nnue6.txt").write_text(
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 | 10 | 0.5\n")
    assert run_round.next_seed(tmp_path, 1) == 8
    assert run_round.next_seed(tmp_path, 101) == 101
    run_round.convert_text(tmp_path)
    assert run_round.count_positions(run_round.data_files(tmp_path)) == 3
    with open(tmp_path / "selfplay_7.bin", "ab") as fh:  # a batch cut off mid-record
        fh.write(b"\0" * 5)
    assert run_round.count_positions(run_round.data_files(tmp_path)) == 3
    assert (tmp_path / "selfplay_7.bin").stat().st_size == 64


def test_match_results_are_read_from_logs(tmp_path):
    log = tmp_path / "m.log"
    assert run_round.sprt_result(log) is None
    log.write_text("Games 10: +5 =3 -2 | penta [0, 1, 1, 2, 1] | Elo +105.2 ± 150.0 | LLR +0.4\n")
    assert run_round.sprt_result(log) is None  # still running or interrupted
    log.write_text(log.read_text() + "Games 300: +120 =118 -62 | penta [3, 24, 51, 44, 28] | Elo +68.0 ± 30.1 | "
                   "LLR +2.96\nH1 accepted: x is stronger (Elo bounds [0, 10])\n")
    assert run_round.sprt_result(log) == ("H1", 68.0, 30.1)


def _engine():
    for c in (ROOT / "build" / "nags_enhanced", ROOT / "build" / "Release" / "nags_enhanced.exe"):
        if c.exists():
            return c
    return None


@pytest.mark.skipif(_engine() is None, reason="nags_enhanced not built")
def test_networks_that_do_not_load_are_not_tested(tmp_path):
    # A network the engine rejects would otherwise play as the built-in one.
    engine = _engine()
    assert run_round.load_error(engine, ROOT / "nets" / "nags.nnue") is None
    bad = tmp_path / "bad.nnue"
    bad.write_bytes(b"not a network")
    assert "not a NAGS NNUE file" in run_round.load_error(engine, bad)

    class Args:
        tc, threads = "1+0.01", 2
    log = tmp_path / "match.log"
    assert run_round.match(Args, engine, bad, None, log, ["0", "10"]) == ("error", 0.0, 0.0)
    assert not log.exists()  # no match was played


def test_logs_of_other_networks_are_not_reused(tmp_path):
    # A finished log is reused only for the networks, engine and settings it
    # was made for (recorded in <log>.key); a log without a key is reused as
    # before.
    engine, net, other = tmp_path / "engine", tmp_path / "a.nnue", tmp_path / "b.nnue"
    engine.write_bytes(b"engine")
    net.write_bytes(b"net a")
    other.write_bytes(b"net b")

    class Args:
        tc, threads = "3+0.03", 2
    log = tmp_path / "m.log"
    log.write_text("Games 300: +150 =100 -50 | penta [0, 0, 0, 0, 0] | Elo +100.0 ± 20.0 | LLR 3.0\nH1 accepted\n")
    assert run_round.match(Args, engine, net, None, log, ["0", "10"]) == ("H1", 100.0, 20.0)  # no key: reused
    log.with_suffix(".key").write_text(json.dumps(run_round.match_key(Args, engine, net, None, ["0", "10"])))
    assert run_round.match(Args, engine, net, None, log, ["0", "10"]) == ("H1", 100.0, 20.0)  # same key: reused
    # Another network under the same log name: set aside, and (as the network
    # cannot be loaded by this fake engine) not tested.
    assert run_round.match(Args, engine, other, None, log, ["0", "10"])[0] == "error"
    assert not log.exists() and len(list(tmp_path.glob("m.stale*.log"))) == 1
