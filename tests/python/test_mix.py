"""tools/nnue/mix.py: mixtures by whole games in the shares asked for, duplicates capped, a consistent index,
and the composition report."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
import mix  # noqa: E402
import shards  # noqa: E402
import train  # noqa: E402
from test_datagen_relabel import _datagen  # noqa: E402


def _records(fens):
    feats, stm, score, result = train.parse_chunk([f"{f} | {s} | {r}" for f, s, r in fens])
    return train.encode(feats, stm, score, result)


def test_composition_counts_what_it_says():
    recs = _records([
        ("4k3/pp6/8/8/8/8/PPP5/4K3 w - - 0 1", 150, 1.0),            # pawn ending, White a pawn up
        ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 2500, 1.0),               # a rook up, |score| > 1000
        ("r3k3/8/8/8/8/8/8/2B1KN2 w - - 0 1", 20, 0.5),              # the exchange (minor pieces vs a rook)
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 30, 0.5),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 25, 0.0),
    ])
    c = mix.composition(recs)
    assert c["positions"] == 5
    assert c["pawn_endings"] == pytest.approx(0.2)
    assert c["material_diff"]["level"] == pytest.approx(0.4) and c["material_diff"]["5+"] == pytest.approx(0.2)
    assert c["material_diff"]["1"] == pytest.approx(0.4)  # the pawn and the exchange (6 against 5)
    assert c["level_material_different_pieces"] == pytest.approx(0.2)
    assert c["queenless"] == pytest.approx(0.6)
    assert c["abs_score"]["over_1000"] == pytest.approx(0.2)
    assert c["repeated_positions"] == pytest.approx(0.4)  # the start position twice (scores differ)
    assert c["results_white"]["win"] == pytest.approx(0.4)
    assert mix.piece_counts(recs[:1]).tolist() == [[3, 0, 0, 0, 0, 1, 2, 0, 0, 0, 0, 1]]


needs_datagen = pytest.mark.skipif(_datagen() is None, reason="nags_datagen not built")


def _shard(path, seed, games):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(_datagen()), "--out", str(path), "--nodes", "300", "--seed", str(seed), "--first-game", "0",
                    "--games", str(games), "--threads", "2"], capture_output=True, check=True, timeout=600)
    return path


def _source_games(path):
    _, games = shards.read_index(str(path))
    raw = np.fromfile(path, np.uint8).reshape(-1, 32)
    return [raw[g.offset // 32 : (g.offset + g.bytes) // 32] for g in games]


@needs_datagen
def test_build_takes_whole_games_in_the_shares(tmp_path):
    a = _shard(tmp_path / "a" / "s.bin", 31, 40)
    b = _shard(tmp_path / "b" / "s.bin", 32, 40)
    total = sum(len(g) for g in _source_games(a)) + sum(len(g) for g in _source_games(b))
    out = tmp_path / "m" / "mix.bin"
    mix.main(["build", "--source", f"{a}=3", "--source", f"{b}=1", "--positions", str(total // 2),
              "--max-copies", "0", "--out", str(out)])
    assert shards.check([str(out)], verify="all") == []
    info = json.loads(Path(str(out) + ".mix.json").read_text())
    sa, sb = info["sources"]
    assert sa["written"] >= sa["wanted"] and sb["written"] >= sb["wanted"]
    assert abs(sa["share_written"] - 0.75) < 0.1
    # Every game written is a whole game of a source.
    whole = {g.tobytes() for g in _source_games(a) + _source_games(b)}
    assert all(g.tobytes() in whole for g in _source_games(out))
    assert len(_source_games(out)) == info["units"]
    # The same command, the same mixture; another name is needed.
    again = tmp_path / "m" / "again.bin"
    mix.main(["build", "--source", f"{a}=3", "--source", f"{b}=1", "--positions", str(total // 2),
              "--max-copies", "0", "--out", str(again)])
    assert again.read_bytes() == out.read_bytes()
    with pytest.raises(SystemExit, match="exists"):
        mix.main(["build", "--source", f"{a}=1", "--positions", "10", "--out", str(out)])


@needs_datagen
def test_duplicates_are_capped(tmp_path):
    a = _shard(tmp_path / "a" / "s.bin", 33, 60)
    raw = np.fromfile(a, np.uint8).reshape(-1, 32)
    keys = mix.position_keys(raw)
    n_distinct = len(np.unique(keys))
    out = tmp_path / "one.bin"
    mix.main(["build", "--source", f"{a}=1", "--positions", str(len(raw)), "--max-copies", "1", "--out", str(out)])
    kept = np.fromfile(out, np.uint8).reshape(-1, 32)
    assert len(kept) == n_distinct and len(np.unique(mix.position_keys(kept))) == n_distinct
    info = json.loads(Path(str(out) + ".mix.json").read_text())
    assert info["sources"][0]["duplicates_removed"] == len(raw) - n_distinct


@needs_datagen
def test_a_file_without_an_index_is_taken_in_blocks(tmp_path):
    a = _shard(tmp_path / "a" / "s.bin", 34, 30)
    plain = tmp_path / "plain.bin"
    plain.write_bytes(a.read_bytes())  # no index next to it
    units, indexed = mix.units_of(str(plain))
    assert not indexed and sum(u.records for u in units) == plain.stat().st_size // 32
    out = tmp_path / "m.bin"
    mix.main(["build", "--source", f"{plain}=1", "--positions", "100000", "--max-copies", "0", "--out", str(out)])
    assert sorted(map(bytes, np.fromfile(out, np.uint8).reshape(-1, 32))) == \
        sorted(map(bytes, np.fromfile(plain, np.uint8).reshape(-1, 32)))
    info = json.loads(Path(str(out) + ".mix.json").read_text())
    assert info["sources"][0]["indexed"] is False and info["sources"][0]["short"] is True
