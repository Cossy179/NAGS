"""nags_datagen shards (--first-game): numbered games with their own random streams, independent of the
thread count, resumable after a stop, safe to split across machines, and never destructive to data they did
not write."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
import shards  # noqa: E402


def _bin(name):
    for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe", ROOT / "build" / f"{name}.exe"):
        if c.exists():
            return c
    return None


def _datagen():
    return _bin("nags_datagen")


pytestmark = pytest.mark.skipif(_datagen() is None, reason="nags_datagen not built")
ARGS = ["--nodes", "300", "--seed", "11"]


def _run(out, *args, check=True):
    return subprocess.run([str(_datagen()), "--out", str(out), *ARGS, *map(str, args)], capture_output=True,
                          text=True, timeout=600, check=check)


def _games(out):
    """{game: its records' bytes} from a shard's output and index, after checking both."""
    assert shards.check([str(out)], verify="all") == []
    header, games = shards.read_index(out)
    assert header.startswith("# nags_datagen games v2 seed=11 nodes=300")
    data = Path(out).read_bytes()
    return {g.number: data[g.offset : g.offset + g.bytes] for g in games}


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


def test_data_without_an_index_is_never_touched(tmp_path):
    legacy = tmp_path / "legacy.bin"
    _run(legacy, "--games", 2)  # an ordinary (unnumbered) run: data, no index
    before = legacy.read_bytes()
    bad = _run(legacy, "--first-game", 0, "--games", 2, check=False)
    assert bad.returncode == 2 and "has no index" in bad.stderr
    assert legacy.read_bytes() == before and not Path(str(legacy) + ".games").exists()
    # A shard whose index was deleted is not cut back to nothing either.
    shard = tmp_path / "shard.bin"
    _run(shard, "--first-game", 0, "--games", 3)
    Path(str(shard) + ".games").unlink()
    kept = shard.read_bytes()
    assert _run(shard, "--first-game", 0, "--games", 3, check=False).returncode == 2 and shard.read_bytes() == kept


def test_an_unfinished_index_line_is_dropped_and_its_game_replayed(tmp_path):
    full = tmp_path / "full.bin"
    _run(full, "--first-game", 0, "--games", 5, "--threads", 1)
    part = tmp_path / "part.bin"
    _run(part, "--first-game", 0, "--games", 3, "--threads", 1)
    index = Path(str(part) + ".games")
    game3 = _games(full)[3]
    with open(part, "ab") as fh:  # game 3's bytes were written, its index line only in part
        fh.write(game3)
    with open(index, "ab") as fh:
        fh.write(f"3 {len(game3) // 32} {len(game3)}".encode())
    assert shards.read_index(part)[1][-1].number == 2  # the reader leaves the unfinished line out too
    out = _run(part, "--first-game", 0, "--games", 5, "--threads", 1).stderr
    assert "3 already written, 2 to play" in out
    assert _games(part) == _games(full) and index.read_bytes().endswith(b"\n")


def test_a_damaged_or_short_tail_is_replayed(tmp_path):
    full = tmp_path / "full.bin"
    _run(full, "--first-game", 0, "--games", 6, "--threads", 1)
    expected = _games(full)
    for damage in ("flip", "short"):
        part = tmp_path / f"{damage}.bin"
        _run(part, "--first-game", 0, "--games", 6, "--threads", 1)
        data = bytearray(part.read_bytes())
        last = shards.read_index(part)[1][-1]
        if damage == "flip":  # the last game's bytes changed (what a crash before the data reached the disk leaves)
            data[last.offset + 5] ^= 0xFF
        else:  # the data shorter than its index says
            data = data[: last.offset + 10]
        part.write_bytes(bytes(data))
        assert shards.check([str(part)], verify="all")  # the checker sees it
        out = _run(part, "--first-game", 0, "--games", 6, "--threads", 1).stderr
        assert "1 indexed game(s)" in out and "5 already written, 1 to play" in out
        assert _games(part) == expected


def test_a_damaged_index_or_other_settings_are_refused_untouched(tmp_path):
    part = tmp_path / "part.bin"
    _run(part, "--first-game", 0, "--games", 4, "--threads", 1)
    index = Path(str(part) + ".games")
    data, lines = part.read_bytes(), index.read_text().splitlines(keepends=True)
    for broken in (lines[:2] + ["garbage\n"] + lines[3:], lines[:3] + [lines[1]] + lines[3:]):  # damaged, duplicate
        index.write_text("".join(broken))
        bad = _run(part, "--first-game", 0, "--games", 4, check=False)
        assert bad.returncode == 2 and "damaged line" in bad.stderr
        assert part.read_bytes() == data and index.read_text() == "".join(broken)
    index.write_text("".join(lines))
    bad = subprocess.run([str(_datagen()), "--out", str(part), "--nodes", "400", "--seed", "11", "--first-game", "0",
                          "--games", "4"], capture_output=True, text=True, timeout=600)
    assert bad.returncode == 2 and "other settings" in bad.stderr and part.read_bytes() == data


def test_the_header_names_the_network_and_the_search(tmp_path):
    _run(tmp_path / "s.bin", "--first-game", 0, "--games", 1)
    fields = shards.header_fields(shards.read_index(tmp_path / "s.bin")[0])
    net = ROOT / "nets" / "nags.nnue"
    if fields["network"] == "embedded" and net.exists():
        assert fields["network-fnv"] == f"{shards.fnv1a(net.read_bytes()):016x}"
    engine = _bin("nags_enhanced")
    if engine is not None:  # the search fingerprint is the engine's bench node count
        out = subprocess.run([str(engine), "bench"], capture_output=True, text=True, timeout=600).stdout
        assert f"Nodes searched  : {fields['search']}" in out


def test_shards_from_several_machines_are_checked_together(tmp_path):
    _run(tmp_path / "w.bin", "--first-game", 100, "--games", 6, "--threads", 1)
    whole = _games(tmp_path / "w.bin")
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    _run(a, "--first-game", 100, "--games", 3, "--threads", 2)
    _run(b, "--first-game", 103, "--games", 3, "--threads", 1)
    assert {**_games(a), **_games(b)} == whole and shards.check([str(a), str(b)]) == []
    # The same games twice (an overlapping range) and a shard of other settings are reported.
    c = tmp_path / "c.bin"
    _run(c, "--first-game", 102, "--games", 2, "--threads", 1)
    assert any("game 102 of seed 11 is in both" in p for p in shards.check([str(a), str(c)]))
    d = tmp_path / "d.bin"
    subprocess.run([str(_datagen()), "--out", str(d), "--nodes", "400", "--seed", "11", "--first-game", "200",
                    "--games", "1"], check=True, capture_output=True)
    assert any("different settings" in p for p in shards.check([str(a), str(d)]))
    assert shards.main(["check", str(a), str(b)]) == 0


def test_without_first_game_nothing_changes(tmp_path):
    _run(tmp_path / "plain.bin", "--games", 3, "--threads", 1)
    assert not Path(str(tmp_path / "plain.bin") + ".games").exists()


def test_run_round_finishes_a_stopped_batch_under_its_own_seed(tmp_path):
    import argparse

    import run_round

    args = argparse.Namespace(games=4, threads=1, nodes=300, seed=5, target_positions=0, resumable_batches=True)
    run_round.generate(args, tmp_path)
    assert len((tmp_path / "selfplay_5.bin.games").read_text().splitlines()) == 1 + 4
    # A batch stopped after two of its four games, with a third index line cut off as it was written.
    subprocess.run([str(_datagen()), "--out", str(tmp_path / "selfplay_6.bin"), "--games", "2", "--threads", "1",
                    "--nodes", "300", "--seed", "6", "--first-game", "0"], check=True, capture_output=True)
    with open(tmp_path / "selfplay_6.bin.games", "ab") as fh:
        fh.write(b"2 70 22")
    assert run_round.unfinished_batch(tmp_path, 3) == 6  # the cut-off line is not a game
    run_round.generate(args, tmp_path)
    assert len((tmp_path / "selfplay_6.bin.games").read_text().splitlines()) == 1 + 4
    assert not (tmp_path / "selfplay_7.bin").exists()
    run_round.generate(args, tmp_path)  # nothing unfinished: a new seed
    assert (tmp_path / "selfplay_7.bin.games").exists()
