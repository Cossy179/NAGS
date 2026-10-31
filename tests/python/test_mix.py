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
    units, header = mix.units_of(str(plain))
    assert header is None and {u.kind for u in units} == {"b"}
    assert sum(u.records for u in units) == plain.stat().st_size // 32
    out = tmp_path / "m.bin"
    mix.main(["build", "--source", f"{plain}=1", "--positions", "100000", "--max-copies", "0", "--out", str(out)])
    assert sorted(map(bytes, np.fromfile(out, np.uint8).reshape(-1, 32))) == \
        sorted(map(bytes, np.fromfile(plain, np.uint8).reshape(-1, 32)))
    info = json.loads(Path(str(out) + ".mix.json").read_text())
    assert info["sources"][0]["indexed"] is False and info["sources"][0]["short"] is True
    header, units = shards.read_index(str(out))
    assert header.startswith(shards.HEADER3) and {u.kind for u in units} == {"b"}
    assert all(u.origin.startswith("@plain.bin:") for u in units)
    # Blocks only: no safe game boundary, so the trainer splits by records and says so.
    n = out.stat().st_size // 32
    assert train.game_boundary([str(out)], [n], n - n // 10) == (n - n // 10, "records")


@needs_datagen
def test_blocks_never_reach_the_validation_side(tmp_path):
    """The reported problem: blocks of a file without an index were written as if they were games, so the
    trainer could split validation between two blocks of one game. Now blocks come first, and the boundary
    moves only to a unit start after the last block."""
    games = _shard(tmp_path / "g" / "s.bin", 35, 40)
    plain = tmp_path / "plain.bin"
    plain.write_bytes(_shard(tmp_path / "p" / "s.bin", 36, 40).read_bytes())  # its index left behind
    out = tmp_path / "m.bin"
    mix.main(["build", "--source", f"{games}=1", "--source", f"{plain}=1", "--positions", "100000",
              "--max-copies", "0", "--out", str(out)])
    header, units = shards.read_index(str(out))
    kinds = [u.kind for u in units]
    last_block = max(i for i, k in enumerate(kinds) if k == "b")
    assert set(kinds[: last_block + 1]) == {"b"} and set(kinds[last_block + 1 :]) == {"g"}
    n = out.stat().st_size // 32
    first_game = units[last_block + 1].offset // 32
    for target in (first_game - 5, first_game + 3, n - 50, n // 3):
        boundary, split = train.game_boundary([str(out)], [n], target)
        assert split == "games" and boundary >= first_game
        assert boundary in {u.offset // 32 for u in units[last_block + 1 :]}
    # The games keep their identity: the same (seed, game) as in the shard they came from.
    assert shards.game_ids(units) == shards.game_ids(shards.read_index(str(games))[1])
    assert all(u.origin.endswith("@s.bin") for u in units if u.kind == "g")
    assert shards.check([str(out)], verify="all") == []


@needs_datagen
def test_an_evaluation_set_is_recognised_inside_a_mixture(tmp_path):
    a = _shard(tmp_path / "a" / "s.bin", 37, 20)
    out = tmp_path / "m.bin"
    mix.main(["build", "--source", f"{a}=1", "--positions", "100000", "--max-copies", "0", "--out", str(out)])
    ov = train.eval_overlap(str(a), [str(out)], "x", cache=False)
    assert ov["games_in_training_data"] == 20  # by origin, although the mixture's own seed is mix-...


@needs_datagen
def test_damaged_or_partial_indexes_are_refused(tmp_path):
    a = _shard(tmp_path / "a" / "s.bin", 38, 10)
    lines = Path(shards.index_path(str(a))).read_text().splitlines()
    Path(shards.index_path(str(a))).write_text("\n".join(lines[:3] + ["3 x y"] + lines[4:]) + "\n")
    with pytest.raises(SystemExit, match="damaged"):
        mix.main(["build", "--source", f"{a}=1", "--positions", "10", "--out", str(tmp_path / "m1.bin")])
    b = _shard(tmp_path / "b" / "s.bin", 39, 10)
    with open(b, "ab") as fh:
        fh.write(b"\0" * 32)  # bytes the index does not list (a game cut off while written)
    with pytest.raises(SystemExit, match="unfinished"):
        mix.main(["build", "--source", f"{b}=1", "--positions", "10", "--out", str(tmp_path / "m2.bin")])


def test_independent_positions_split_anywhere(tmp_path):
    recs = _records([("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 100 + i, 1.0) for i in range(3000)])
    data = tmp_path / "tb.bin"
    recs.tofile(data)
    chunks = [recs[i : i + 1024].tobytes() for i in range(0, len(recs), 1024)]
    with open(shards.index_path(str(data)), "w") as fh:
        fh.write(shards.HEADER3 + "seed=tb-1 format=bin kind=positions\n")
        for n, (blob, c) in enumerate(zip(chunks, shards.fnv1a_many(chunks))):
            fh.write(f"{n} {len(blob) // 32} {len(blob)} 0.5 {c:016x} p tb-1@tb.bin:{n * 1024}\n")
    assert train.game_boundary([str(data)], [3000], 2900) == (2900, "games")
    out = tmp_path / "m.bin"
    mix.main(["build", "--source", f"{data}=1", "--positions", "1500", "--max-copies", "0", "--out", str(out)])
    _, units = shards.read_index(str(out))
    assert {u.kind for u in units} == {"p"} and len(units) == 2  # taken in chunks, still positions


def test_fnv1a_many_is_fnv1a():
    import os as _os
    blobs = [_os.urandom(k) for k in (0, 1, 31, 32, 900, 32, 4000)]
    assert shards.fnv1a_many(blobs) == [shards.fnv1a(b) for b in blobs]


def test_book_selects_kinds(tmp_path):
    recs = _records([
        ("4k3/pp6/8/8/8/8/PPP5/4K3 w - - 0 1", 150, 1.0),            # pawn ending
        ("4k3/8/8/8/8/8/8/R3K3 w - - 0 1", 2500, 1.0),               # a rook up
        ("r3k3/8/8/8/8/8/8/2B1KN2 w - - 0 1", 20, 0.5),              # the exchange
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 30, 0.5),
    ])
    data = tmp_path / "d.bin"
    recs.tofile(data)
    out = tmp_path / "b.epd"
    assert mix.main(["book", str(data), "--select", "imbalanced", "--out", str(out)]) == 0
    lines = out.read_text().splitlines()
    assert sorted(l.split()[0] for l in lines) == sorted(["4k3/8/8/8/8/8/8/R3K3", "r3k3/8/8/8/8/8/8/2B1KN2"])
    mix.main(["book", str(data), "--select", "imbalanced", "--max-score", "300", "--out", str(out)])
    assert [l.split()[0] for l in out.read_text().splitlines()] == ["r3k3/8/8/8/8/8/8/2B1KN2"]
    mix.main(["book", str(data), "--select", "pieces:20-32", "--out", str(out)])
    assert out.read_text().splitlines() == ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - -"]  # no recorded state: no rights
    mix.main(["book", str(data), "--select", "pawn-endings", "--out", str(out)])
    assert out.read_text().splitlines() == ["4k3/pp6/8/8/8/8/PPP5/4K3 w - -"]
