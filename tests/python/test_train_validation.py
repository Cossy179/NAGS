"""tools/nnue/train.py: validation by complete games (from nags_datagen shard indexes) and evaluation sets
that must not overlap the training data."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
import shards  # noqa: E402
import train  # noqa: E402

COMMON = ["--hidden", "8", "--batch", "256", "--threads", "1", "--device", "cpu", "--epochs", "1", "--buckets", "8"]


def _datagen():
    for c in (ROOT / "build" / "nags_datagen", ROOT / "build" / "Release" / "nags_datagen.exe"):
        if c.exists():
            return c
    return None


needs_datagen = pytest.mark.skipif(_datagen() is None, reason="nags_datagen not built")


def _shard(path, seed, first, games):
    subprocess.run([str(_datagen()), "--out", str(path), "--nodes", "300", "--seed", str(seed), "--first-game",
                    str(first), "--games", str(games)], check=True, capture_output=True, timeout=600)
    return str(path)


def test_the_boundary_moves_to_the_nearest_game_start(tmp_path, capsys):
    # Two files of 10 and 30 records; the second is a shard of games of 7, 8, 9 and 6 records.
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(bytes(32 * 10))
    b.write_bytes(bytes(32 * 30))
    lines = ["# nags_datagen games v2 seed=1 nodes=1 format=bin search=1"]
    for g, recs in enumerate((7, 8, 9, 6)):
        lines.append(f"{g} {recs} {32 * recs} 0.5 {shards.fnv1a(bytes(32 * recs)):016x}")
    Path(str(b) + ".games").write_bytes(("\n".join(lines) + "\n").encode())  # as nags_datagen writes it: LF
    counts = [10, 30]
    assert train.game_boundary([str(a), str(b)], counts, 10 + 13) == (10 + 15, "games")  # 13 -> start 15
    assert train.game_boundary([str(a), str(b)], counts, 10 + 26) == (10 + 24, "games")  # 26 -> start 24
    assert train.game_boundary([str(a), str(b)], counts, 10) == (10, "games")  # a file boundary
    assert train.game_boundary([str(a), str(b)], counts, 10 + 29) == (10 + 24, "games")  # keeps a validation part
    # The boundary inside a file without an index stays where it is, and says so.
    assert train.game_boundary([str(a), str(b)], counts, 5) == (5, "records")
    assert "no complete game index" in capsys.readouterr().out


@needs_datagen
def test_training_splits_shards_by_games_and_records_it(tmp_path):
    data = _shard(tmp_path / "train.bin", 11, 0, 40)
    _, games = shards.read_index(data)
    starts = {g.offset // 32 for g in games}
    out = tmp_path / "n.nnue"
    train.main(["--data", data, "--out", str(out), "--val", "0.1"] + COMMON)
    m = json.loads(Path(str(out) + ".json").read_text())
    assert m["data"]["split"] == "games" and m["data"]["train"] in starts  # validation starts at a game
    assert abs(m["data"]["validation"] - 0.1 * m["data"]["positions"]) < max(g.records for g in games)
    # A resume must find the same split.
    ckpt = __import__("torch").load(str(out) + ".ckpt", weights_only=False)
    assert ckpt["n_train"] == m["data"]["train"] and ckpt["split"] == "games"


@needs_datagen
def test_overlapping_evaluation_sets_are_refused(tmp_path, capsys):
    data = _shard(tmp_path / "train.bin", 11, 0, 30)
    out = str(tmp_path / "n.nnue")
    # The training file itself, or a copy of it.
    with pytest.raises(SystemExit, match="is training data"):
        train.main(["--data", data, "--out", out, "--eval-set", data] + COMMON)
    copy = tmp_path / "copy.bin"
    shutil.copyfile(data, copy)
    with pytest.raises(SystemExit, match="is training data"):
        train.main(["--data", data, "--out", out, "--eval-set", str(copy)] + COMMON)
    # Games of the same seed and number: a shard of an overlapping range.
    overlapping = _shard(tmp_path / "overlap.bin", 11, 25, 10)
    with pytest.raises(SystemExit, match="of its games"):
        train.main(["--data", data, "--out", out, "--eval-set", overlapping] + COMMON)
    # Training records pasted into an evaluation file without an index.
    pasted = tmp_path / "pasted.bin"
    independent = Path(_shard(tmp_path / "indep.bin", 99, 0, 10)).read_bytes()
    pasted.write_bytes(independent + Path(data).read_bytes()[: 32 * 200])
    with pytest.raises(SystemExit, match="also occur in the training data"):
        train.main(["--data", data, "--out", out, "--eval-set", str(pasted)] + COMMON)
    # Independent self-play (another seed) is accepted, and the overlap it has is recorded.
    capsys.readouterr()
    train.main(["--data", data, "--out", out, "--eval-set", str(tmp_path / "indep.bin")] + COMMON)
    assert "also in the training data" in capsys.readouterr().out
    m = json.loads(Path(out + ".json").read_text())
    assert m["eval_sets"][0]["records"] == len(independent) // 32
    assert m["eval_sets"][0]["records_in_training_data"] <= 0.01 * m["eval_sets"][0]["records"]
    assert json.loads(Path(str(tmp_path / "indep.bin") + ".overlap.json").read_text())["data_sha256"]
