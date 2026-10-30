"""tools/nnue/rescore.py: records back to FENs, teacher labels written as a consistent shard (resumable, the
games and positions unchanged but for the scores), and the comparison of label sets."""

import json
import random
import subprocess
import sys
from pathlib import Path

import chess
import chess.engine
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
import rescore  # noqa: E402
import shards  # noqa: E402
import train  # noqa: E402


def _build(name):
    for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe"):
        if c.exists():
            return c
    return None


needs_engines = pytest.mark.skipif(_build("nags_datagen") is None or _build("nags_enhanced") is None,
                                   reason="nags_datagen / nags_enhanced not built")


def _shard(tmp_path, games=4):
    out = tmp_path / "base.bin"
    subprocess.run([str(_build("nags_datagen")), "--out", str(out), "--nodes", "500", "--seed", "21",
                    "--first-game", "0", "--games", str(games), "--threads", "1"], capture_output=True, check=True,
                   timeout=600)
    return out


def _label(data, out_dir, *extra):
    return rescore.main(["label", "--data", str(data), "--engine", str(_build("nags_enhanced")), "--nodes", "1500",
                         "--out-dir", str(out_dir), *map(str, extra)])


def test_record_fen_round_trip():
    rng = random.Random(5)
    for _ in range(300):
        board = chess.Board()
        for _ in range(rng.randint(0, 60)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        feats, stm, score, result = train.parse_chunk([f"{board.fen()} | -123 | 0.5"])
        rec = train.encode(feats, stm, score, result)[0]
        back = chess.Board(rescore.record_fen(rec))
        assert back.board_fen() == board.board_fen() and back.turn == board.turn
        assert rescore.record_score(rec) == -123
        # Castling rights wherever king and rook stand at home: a superset of the real ones.
        assert board.castling_rights & ~back.castling_rights == 0
        assert rescore.record_score(rescore.with_score(rec, 777)) == 777


def test_fit_scale_finds_the_scale():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 300, 20000)
    res = (rng.random(20000) < 1 / (1 + np.exp(-s / 250))).astype(float)
    k, _ = rescore.fit_scale(s, res)
    assert 230 < k < 270
    k2, _ = rescore.fit_scale(2 * s, res)
    assert abs(k2 / k - 2) < 0.02


@needs_engines
def test_label_is_a_consistent_shard_of_the_same_positions(tmp_path):
    base = _shard(tmp_path)
    _label(base, tmp_path / "t", "--max-games", 3, "--workers", 2)
    out = tmp_path / "t" / "base.bin"
    assert shards.check([str(out)], verify="all") == []
    header, games = shards.read_index(str(out))
    assert "teacher=NAGS" in header and "teacher-limit=nodes:1500" in header and len(games) == 3
    _, src_games = shards.read_index(str(base))
    src = {g.number: g for g in src_games}
    raw, new = np.fromfile(base, np.uint8), np.fromfile(out, np.uint8)
    changed = 0
    for g in games:
        s = src[g.number]
        a = raw[s.offset : s.offset + s.bytes].reshape(-1, 32)
        b = new[g.offset : g.offset + g.bytes].reshape(-1, 32)
        assert g.result == s.result
        if len(a) == len(b):  # no mate dropped: the same records but for the score
            assert np.array_equal(np.delete(a, [24, 25], 1), np.delete(b, [24, 25], 1))
            changed += int((a[:, 24:26] != b[:, 24:26]).any(axis=1).sum())
        # The first position of a game is searched right after ucinewgame: a fresh engine gives its label.
        with chess.engine.SimpleEngine.popen_uci([str(_build("nags_enhanced"))]) as eng:
            info = eng.analyse(chess.Board(rescore.record_fen(b[0])), chess.engine.Limit(nodes=1500))
        assert rescore.record_score(b[0]) == info["score"].white().score()
    assert changed > 0
    stats = json.loads((tmp_path / "t" / "base.bin.teacher.json").read_text())
    assert stats["games"] == 3 and stats["engine_seconds_per_position"] > 0


@needs_engines
def test_label_resumes_to_the_same_bytes(tmp_path):
    base = _shard(tmp_path)
    _label(base, tmp_path / "full", "--max-games", 3)
    _label(base, tmp_path / "cut", "--max-games", 3)
    cut = tmp_path / "cut" / "base.bin"
    _, games = shards.read_index(str(cut))
    # A stop while the last game was written: its index line unfinished, half of its bytes there.
    idx = Path(shards.index_path(str(cut))).read_bytes()
    Path(shards.index_path(str(cut))).write_bytes(idx[: idx.rstrip(b"\n").rfind(b"\n") + 1] + b"2 1")
    with open(cut, "r+b") as fh:
        fh.truncate(games[-1].offset + games[-1].bytes // 2)
    _label(base, tmp_path / "cut", "--max-games", 3)
    assert cut.read_bytes() == (tmp_path / "full" / "base.bin").read_bytes()
    assert Path(shards.index_path(str(cut))).read_bytes() == Path(
        shards.index_path(str(tmp_path / "full" / "base.bin"))).read_bytes()
    with pytest.raises(SystemExit, match="other settings"):
        _label(base, tmp_path / "cut", "--max-games", 3, "--scale", 1.5)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "base.bin").write_bytes(b"\0" * 32)
    with pytest.raises(SystemExit, match="no index"):
        _label(base, tmp_path / "raw", "--max-games", 1)
    plain = tmp_path / "plain.bin"
    plain.write_bytes(base.read_bytes())
    with pytest.raises(SystemExit, match="shard index"):
        _label(plain, tmp_path / "p", "--max-games", 1)


@needs_engines
def test_compare_scales_and_aligns(tmp_path):
    base = _shard(tmp_path, games=3)
    # A "teacher" that says twice the base's score: compare must find the factor 1/2 and no difference after it.
    _, games = shards.read_index(str(base))
    raw = np.fromfile(base, np.uint8).reshape(-1, 32).copy()
    s = raw[:, 24:26].copy().view("<i2").reshape(-1)
    raw[:, 24:26] = np.clip(s.astype(np.int64) * 2, -32000, 32000).astype("<i2").view(np.uint8).reshape(-1, 2)
    (tmp_path / "x2").mkdir()
    doubled = tmp_path / "x2" / "base.bin"
    raw.tofile(doubled)
    lines = Path(shards.index_path(str(base))).read_text().splitlines()
    fixed = [lines[0]]
    for g, line in zip(games, lines[1:]):
        blob = raw[g.offset // 32 : (g.offset + g.bytes) // 32].tobytes()
        parts = line.split()
        fixed.append(" ".join(parts[:4] + [f"{shards.fnv1a(blob):016x}"]))
    Path(shards.index_path(str(doubled))).write_text("\n".join(fixed) + "\n")
    rep = rescore.main(["compare", "--base", str(base), "--labels", str(doubled), "--json", str(tmp_path / "c.json")])
    assert rep == 0
    report = json.loads((tmp_path / "c.json").read_text())
    x2 = report["sets"][1]
    assert abs(x2["scale_to_base"] - 0.5) < 0.01 and x2["mean_abs_diff_to_base_cp"] < 2
    assert report["positions"] == len(raw)
    # A label set with a position dropped (a mate) is aligned by content.
    _label(base, tmp_path / "t", "--max-games", 2)
    rescore.main(["compare", "--base", str(base), "--labels", str(tmp_path / "t" / "base.bin"), str(doubled),
                  "--json", str(tmp_path / "c2.json")])
    assert json.loads((tmp_path / "c2.json").read_text())["games"] == 2
