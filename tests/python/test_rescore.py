"""tools/nnue/rescore.py: exact positions only (records with nags_datagen --record-state, nothing guessed),
labels written as a consistent shard of the same positions, accounting that survives a stop, fingerprints that
refuse a changed source or evaluation file, and the comparison of label sets."""

import json
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


def _datagen(out, *args):
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(_build("nags_datagen")), "--out", str(out), "--nodes", "500", "--seed", "21",
                    "--threads", "1", *map(str, args)], capture_output=True, check=True, timeout=600)
    return out


def _shard(tmp_path, games=4, state=True):
    return _datagen(tmp_path / ("base.bin" if state else "plain.bin"), "--first-game", 0, "--games", games,
                    *(["--record-state", "on"] if state else []))


def _label(data, out_dir, *extra, engine=None):
    return rescore.main(["label", "--data", str(data), "--engine", str(engine or _build("nags_enhanced")),
                         "--nodes", "1500", "--out-dir", str(out_dir), *map(str, extra)])


def test_state_bytes_and_no_guessing():
    feats, stm, score, result = train.parse_chunk(["r3k2r/8/8/3pP3/8/8/8/R3K2R w KQq d6 3 17 | 10 | 0.5"])
    plain = train.encode(feats, stm, score, result)[0]
    with pytest.raises(ValueError, match="record-state"):
        rescore.record_fen(plain)  # castling and en passant unknown: refused, not inferred
    assert rescore.record_board_fen(plain) == "r3k2r/8/8/3pP3/8/8/8/R3K2R w"
    rec = rescore.with_state(plain, "KQq", ep_file=3, halfmove=3, fullmove=17)
    assert rescore.record_fen(rec) == "r3k2r/8/8/3pP3/8/8/8/R3K2R w KQq d6 3 17"
    black = bytearray(rec)
    black[27] = 1
    assert rescore.record_fen(bytes(black)).split()[3] == "d3"  # the en-passant rank follows the side to move
    assert rescore.record_score(rescore.with_score(rec, 777)) == 777


@needs_engines
def test_datagen_records_the_exact_position(tmp_path):
    """--record-state: the bin records' state is the text FEN's, game for game; without it, the old bytes."""
    text = _datagen(tmp_path / "t.txt", "--games", 30)
    state = _datagen(tmp_path / "s.bin", "--games", 30, "--record-state", "on")
    plain = _datagen(tmp_path / "p.bin", "--games", 30)
    fens = [l.split("|")[0].strip() for l in text.read_text().splitlines()]
    s = np.fromfile(state, np.uint8).reshape(-1, 32)
    p = np.fromfile(plain, np.uint8).reshape(-1, 32)
    assert len(fens) == len(s) == len(p)
    assert np.array_equal(s[:, :28], p[:, :28]) and not p[:, 28:].any()
    assert [rescore.record_fen(r) for r in s] == fens
    assert any(f.split()[2] != "-" for f in fens)  # castling rights are exercised
    head = shards.read_index(str(_datagen(tmp_path / "i.bin", "--first-game", 0, "--games", 2,
                                          "--record-state", "on")))[0]
    assert head.endswith(" state=1")


def test_fit_scale_finds_the_scale():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 300, 20000)
    res = (rng.random(20000) < 1 / (1 + np.exp(-s / 250))).astype(float)
    k, _ = rescore.fit_scale(s, res)
    assert 230 < k < 270
    k2, _ = rescore.fit_scale(2 * s, res)
    assert abs(k2 / k - 2) < 0.02


@needs_engines
def test_a_shard_without_state_is_refused(tmp_path):
    plain = _shard(tmp_path, state=False)
    with pytest.raises(SystemExit, match="record-state"):
        _label(plain, tmp_path / "t", "--max-games", 1)
    unindexed = tmp_path / "u.bin"
    unindexed.write_bytes(_shard(tmp_path).read_bytes())
    with pytest.raises(SystemExit, match="shard index"):
        _label(unindexed, tmp_path / "u", "--max-games", 1)


@needs_engines
def test_label_is_a_consistent_shard_of_the_same_positions(tmp_path):
    base = _shard(tmp_path)
    _label(base, tmp_path / "t", "--max-games", 3, "--workers", 2)
    out = tmp_path / "t" / "base.bin"
    assert shards.check([str(out)], verify="all") == []
    header, games = shards.read_index(str(out))
    assert "teacher=NAGS" in header and "teacher-limit=nodes:1500" in header and " source-sha=" in header
    assert len(games) == 3
    _, src_games = shards.read_index(str(base))
    src = {g.number: g for g in src_games}
    raw, new = np.fromfile(base, np.uint8), np.fromfile(out, np.uint8)
    changed = 0
    for g in games:
        s = src[g.number]
        a = raw[s.offset : s.offset + s.bytes].reshape(-1, 32)
        b = new[g.offset : g.offset + g.bytes].reshape(-1, 32)
        assert g.result == s.result
        if len(a) == len(b):  # no mate dropped: the same records (state included) but for the score
            assert np.array_equal(np.delete(a, [24, 25], 1), np.delete(b, [24, 25], 1))
            changed += int((a[:, 24:26] != b[:, 24:26]).any(axis=1).sum())
        # The first position of a game is searched right after ucinewgame: a fresh engine gives its label.
        with chess.engine.SimpleEngine.popen_uci([str(_build("nags_enhanced"))]) as eng:
            info = eng.analyse(chess.Board(rescore.record_fen(b[0])), chess.engine.Limit(nodes=1500))
        assert rescore.record_score(b[0]) == info["score"].white().score()
    assert changed > 0
    stats = json.loads((tmp_path / "t" / "base.bin.teacher.json").read_text())
    assert stats["games"] == 3 and stats["engine_seconds_per_position"] > 0 and stats["nodes"] > 0
    assert stats["positions"] == sum(g.records for g in games) and stats["games_without_accounting"] == 0


@needs_engines
def test_an_interrupted_run_counts_all_its_work(tmp_path):
    """A stop after one finished game (index and accounting written, the summary not), then a resume: the
    output holds both games and the summary counts both, as an uninterrupted run does."""
    base = _shard(tmp_path)
    _label(base, tmp_path / "full", "--max-games", 2)
    _label(base, tmp_path / "cut", "--max-games", 2)
    cut = tmp_path / "cut" / "base.bin"
    _, games = shards.read_index(str(cut))
    idx = Path(shards.index_path(str(cut))).read_text().splitlines(keepends=True)
    Path(shards.index_path(str(cut))).write_text("".join(idx[:2]))  # the header and the first game
    acc = Path(rescore.accounting_path(str(cut))).read_text().splitlines(keepends=True)
    Path(rescore.accounting_path(str(cut))).write_text("".join(acc) + "99 5 0 0.1")  # a line cut mid-write
    with open(cut, "r+b") as fh:
        fh.truncate(games[0].bytes + games[1].bytes // 2)
    Path(str(cut) + ".teacher.json").unlink()  # killed before the summary was ever written
    _label(base, tmp_path / "cut", "--max-games", 2)
    assert cut.read_bytes() == (tmp_path / "full" / "base.bin").read_bytes()
    full = json.loads(Path(str(tmp_path / "full" / "base.bin") + ".teacher.json").read_text())
    resumed = json.loads(Path(str(cut) + ".teacher.json").read_text())
    for key in ("games", "positions", "dropped", "nodes"):
        assert resumed[key] == full[key], key
    assert resumed["games"] == 2 and len(resumed["sittings"]) == 1
    assert sorted(int(l.split()[0]) for l in Path(rescore.accounting_path(str(cut))).read_text().splitlines()) == \
        sorted(g.number for g in games)


@needs_engines
def test_fingerprints_refuse_a_changed_source_or_evaluation_file(tmp_path):
    base = _shard(tmp_path)
    net = tmp_path / "net.nnue"
    net.write_bytes((ROOT / "nets" / "nags.nnue").read_bytes())
    _label(base, tmp_path / "t", "--max-games", 1, "--option", f"EvalFile={net}")
    header = shards.read_index(str(tmp_path / "t" / "base.bin"))[0]
    assert " teacher-file=EvalFile:" in header
    stats = json.loads((tmp_path / "t" / "base.bin.teacher.json").read_text())
    assert stats["option_files_sha256"]["EvalFile"] == rescore.file_sha256(str(net))
    data = bytearray(net.read_bytes())
    data[-1] ^= 1  # the same path, another network
    net.write_bytes(bytes(data))
    with pytest.raises(SystemExit, match="other settings"):
        _label(base, tmp_path / "t", "--max-games", 1, "--option", f"EvalFile={net}")
    net.write_bytes((ROOT / "nets" / "nags.nnue").read_bytes())
    with open(base, "r+b") as fh:  # the same source path, other content (a score byte of the last game)
        fh.seek(-8, 2)
        b = fh.read(1)
        fh.seek(-8, 2)
        fh.write(bytes([b[0] ^ 1]))
    with pytest.raises(SystemExit, match="other settings"):
        _label(base, tmp_path / "t", "--max-games", 1, "--option", f"EvalFile={net}")


@needs_engines
def test_resume_refuses_other_settings(tmp_path):
    base = _shard(tmp_path)
    _label(base, tmp_path / "t", "--max-games", 1)
    with pytest.raises(SystemExit, match="other settings"):
        _label(base, tmp_path / "t", "--max-games", 1, "--scale", 1.5)
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "base.bin").write_bytes(b"\0" * 32)
    with pytest.raises(SystemExit, match="no index"):
        _label(base, tmp_path / "raw", "--max-games", 1)


@needs_engines
def test_compare_scales_and_aligns(tmp_path):
    base = _shard(tmp_path, games=3)
    # A label set that says twice the base's score: compare must find the factor 1/2 and no difference after it.
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
