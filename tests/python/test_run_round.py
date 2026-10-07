"""tools/nnue/run_round.py: data generation to a target, training several
designs, resuming, and reading match results."""

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
