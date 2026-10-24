"""nags_datagen shards (--first-game): numbered games with their own random streams, independent of the
thread count, resumable after a stop, and safe to split across machines."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _datagen():
    for c in (ROOT / "build" / "nags_datagen", ROOT / "build" / "Release" / "nags_datagen.exe"):
        if c.exists():
            return c
    return None


pytestmark = pytest.mark.skipif(_datagen() is None, reason="nags_datagen not built")
ARGS = ["--nodes", "300", "--seed", "11"]


def _run(out, *args, check=True):
    return subprocess.run([str(_datagen()), "--out", str(out), *ARGS, *map(str, args)], capture_output=True,
                          text=True, timeout=600, check=check)


def _games(out):
    """{game: its records' bytes} from a shard's output and index."""
    lines = Path(str(out) + ".games").read_text().splitlines()
    assert lines[0].startswith("# nags_datagen games seed=11 nodes=300")
    data, pos, games = Path(out).read_bytes(), 0, {}
    for line in lines[1:]:
        game, records, size, _ = line.split()
        assert int(size) == 32 * int(records)
        games[int(game)] = data[pos : pos + int(size)]
        pos += int(size)
    assert pos == len(data)
    return games


def test_games_do_not_depend_on_the_threads(tmp_path):
    _run(tmp_path / "one.bin", "--first-game", 0, "--games", 8, "--threads", 1)
    _run(tmp_path / "three.bin", "--first-game", 0, "--games", 8, "--threads", 3)
    one, three = _games(tmp_path / "one.bin"), _games(tmp_path / "three.bin")
    assert sorted(one) == list(range(8)) and one == three
    assert list(one) == list(range(8))  # one thread writes them in order


def test_a_shard_resumes_and_extends(tmp_path):
    full = tmp_path / "full.bin"
    _run(full, "--first-game", 0, "--games", 10, "--threads", 2)
    part = tmp_path / "part.bin"
    _run(part, "--first-game", 0, "--games", 4, "--threads", 1)
    with open(part, "ab") as fh:  # a game cut off before its index line
        fh.write(b"\x01" * 70)
    out = _run(part, "--first-game", 0, "--games", 10, "--threads", 2).stderr
    assert "4 already written, 6 to play" in out
    assert _games(part) == _games(full)
    again = _run(part, "--first-game", 0, "--games", 10, "--threads", 2).stderr
    assert "10 already written, 0 to play" in again and _games(part) == _games(full)


def test_disjoint_ranges_add_up_and_other_settings_are_refused(tmp_path):
    _run(tmp_path / "w.bin", "--first-game", 100, "--games", 6, "--threads", 1)
    whole = _games(tmp_path / "w.bin")
    _run(tmp_path / "a.bin", "--first-game", 100, "--games", 3, "--threads", 1)
    _run(tmp_path / "b.bin", "--first-game", 103, "--games", 3, "--threads", 1)
    assert {**_games(tmp_path / "a.bin"), **_games(tmp_path / "b.bin")} == whole
    bad = subprocess.run([str(_datagen()), "--out", str(tmp_path / "a.bin"), "--nodes", "400", "--seed", "11",
                          "--first-game", "100", "--games", "3"], capture_output=True, text=True, timeout=600)
    assert bad.returncode == 2 and "other settings" in bad.stderr


def test_without_first_game_nothing_changes(tmp_path):
    _run(tmp_path / "plain.bin", "--games", 3, "--threads", 1)
    assert not Path(str(tmp_path / "plain.bin") + ".games").exists()


def test_run_round_finishes_a_stopped_batch_under_its_own_seed(tmp_path):
    import argparse
    import sys

    sys.path.insert(0, str(ROOT / "tools" / "nnue"))
    import run_round

    args = argparse.Namespace(games=4, threads=1, nodes=300, seed=5, target_positions=0, resumable_batches=True)
    run_round.generate(args, tmp_path)
    assert len((tmp_path / "selfplay_5.bin.games").read_text().splitlines()) == 1 + 4
    # A batch stopped after two of its four games.
    subprocess.run([str(_datagen()), "--out", str(tmp_path / "selfplay_6.bin"), "--games", "2", "--threads", "1",
                    "--nodes", "300", "--seed", "6", "--first-game", "0"], check=True, capture_output=True)
    run_round.generate(args, tmp_path)
    assert len((tmp_path / "selfplay_6.bin.games").read_text().splitlines()) == 1 + 4
    assert not (tmp_path / "selfplay_7.bin").exists()
    run_round.generate(args, tmp_path)  # nothing unfinished: a new seed
    assert (tmp_path / "selfplay_7.bin.games").exists()
