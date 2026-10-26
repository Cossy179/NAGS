"""nags_datagen --relabel-nodes/--relabel-select: chosen positions get a deeper search's score as their label,
while the games (positions, order, results) stay those of the plain run."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
import shards  # noqa: E402


def _datagen():
    for c in (ROOT / "build" / "nags_datagen", ROOT / "build" / "Release" / "nags_datagen.exe"):
        if c.exists():
            return c
    return None


pytestmark = pytest.mark.skipif(_datagen() is None, reason="nags_datagen not built")


def _run(out, *args):
    return subprocess.run([str(_datagen()), "--out", str(out), "--nodes", "500", "--seed", "21", "--games", "6",
                           "--threads", "1", *map(str, args)], capture_output=True, text=True, timeout=600,
                          check=True).stderr


def _records(path):
    return np.fromfile(path, np.uint8).reshape(-1, 32)


def _is_ordered_subset(sub, full):
    j = 0
    for row in sub:
        while j < len(full) and not np.array_equal(full[j], row):
            j += 1
        if j == len(full):
            return False
        j += 1
    return True


@pytest.mark.parametrize("select", ["disagree:60", "random:0.3"])
def test_labels_change_the_games_do_not(tmp_path, select):
    plain, deeper = tmp_path / "plain.bin", tmp_path / "deeper.bin"
    _run(plain)
    err = _run(deeper, "--relabel-nodes", 4000, "--relabel-select", select)
    assert "relabelled" in err
    p, d = _records(plain), _records(deeper)
    # Without the score bytes the deeper file is the plain one, but for chosen positions whose deeper search
    # found a mate (those are not kept).
    without_score = lambda r: np.delete(r, [24, 25], axis=1)  # noqa: E731
    assert _is_ordered_subset(without_score(d), without_score(p)) and len(p) - len(d) < 0.05 * len(p)
    changed = int((d[:, 24:26] != p[: len(d), 24:26]).any(axis=1).sum()) if len(d) == len(p) else None
    assert changed is None or changed > 0


def test_random_choice_is_the_same_whatever_the_threads(tmp_path):
    one, two = tmp_path / "one.bin", tmp_path / "two.bin"
    args = ["--first-game", 0, "--relabel-nodes", 3000, "--relabel-select", "random:0.25"]
    _run(one, *args)
    subprocess.run([str(_datagen()), "--out", str(two), "--nodes", "500", "--seed", "21", "--games", "6", "--threads",
                    "2", *map(str, args)], check=True, capture_output=True, timeout=600)
    games = lambda p: {g.number: p.read_bytes()[g.offset : g.offset + g.bytes] for g in shards.read_index(p)[1]}  # noqa: E731
    assert games(one) == games(two)
    fields = shards.header_fields(shards.read_index(one)[0])
    assert fields["relabel"].startswith("3000@random:0.25")  # relabelled shards are other data


def test_bad_selections_are_refused(tmp_path):
    for bad in ("random:0", "random:1.5", "deep:3", "disagree:-1", "disagree:x"):
        r = subprocess.run([str(_datagen()), "--out", str(tmp_path / "x.bin"), "--games", "1", "--relabel-nodes", "100",
                            "--relabel-select", bad], capture_output=True, text=True, timeout=60)
        assert r.returncode == 2 and not (tmp_path / "x.bin").exists(), bad


def test_every_run_reports_what_its_searches_cost(tmp_path):
    import re
    plain = _run(tmp_path / "plain.bin")
    deeper = _run(tmp_path / "deeper.bin", "--relabel-nodes", 4000, "--relabel-select", "disagree:60")
    game = re.search(r"searches in this run: game moves (\d+) nodes, (\d+) thread-seconds", plain)
    assert game and int(game.group(1)) > 0 and "relabelling" not in plain
    m = re.search(r"game moves (\d+) nodes, \d+ thread-seconds; relabelling (\d+) nodes \(([\d.]+)% of all\)", deeper)
    assert m and m.group(1) == game.group(1)  # the same games, so the same game searches
    chosen = re.search(r"relabelled (\d+) of \d+ positions written .*; (\d+) chosen positions dropped", deeper)
    searches = int(chosen.group(1)) + int(chosen.group(2))
    assert searches > 0 and int(m.group(2)) >= searches * 4000  # every deeper search ran to its node budget
    share = 100 * int(m.group(2)) / (int(m.group(1)) + int(m.group(2)))
    assert abs(share - float(m.group(3))) < 0.1
