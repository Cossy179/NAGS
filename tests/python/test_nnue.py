"""NNUE trainer tests: features, training/export, and (when nags_enhanced is
built) exact agreement between the engine's evaluation and the reference."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
import train  # noqa: E402

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/8/1k6/8/8/8/6K1/7Q b - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]


def _dataset(path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    with open(path, "w") as fh:
        for i in range(n):
            fen = FENS[i % len(FENS)]
            fh.write(f"{fen} | {int(rng.integers(-300, 300))} | {rng.choice(['1.0', '0.5', '0.0'])}\n")


def test_features():
    f = train.white_features(FENS[0].split()[0])
    assert len(f) == 32 and len(set(f)) == 32
    assert 0 * 384 + 0 * 64 + 8 in f  # white pawn a2
    assert 1 * 384 + 5 * 64 + 60 in f  # black king e8
    # From Black's side the black king on e8 is "our king on e1".
    black = train.mirror(np.array(f))
    assert 0 * 384 + 5 * 64 + 4 in black
    assert sorted(black.tolist()) == sorted(f)  # the start position is symmetric


DESIGNS = [  # (output buckets, king buckets, activation)
    (1, "none", "crelu"),
    (8, "none", "crelu"),
    (1, "mirror", "crelu"),
    (8, "4", "screlu"),
    (4, "16", "screlu"),
]


@pytest.mark.parametrize("buckets,king,activation", DESIGNS)
def test_train_and_export(tmp_path, buckets, king, activation):
    data = tmp_path / "d.txt"
    _dataset(data)
    out = tmp_path / "n.nnue"
    train.main(["--data", str(data), "--out", str(out), "--epochs", "2", "--hidden", "16", "--batch", "64",
                "--threads", "1", "--buckets", str(buckets), "--king-buckets", king, "--activation", activation])
    kb = train.king_config(king)[0]
    plain = king == "none" and activation == "crelu"
    header = (16 if buckets == 1 else 20) if plain else 92
    assert out.stat().st_size == header + 2 * (kb * 768 * 16 + 16 + buckets * 32) + 4 * buckets
    net = train.read_net(out)
    assert net[2].shape == (buckets, 32) and net[0].shape == (kb * 768, 16)
    assert net[4]["screlu"] == (activation == "screlu") and net[4]["king_buckets"] == kb
    # The quantized evaluation tracks the float network closely.
    model = train.Nnue(16, buckets, king, activation == "screlu")
    with torch.no_grad():
        model.ft.weight.zero_()
        model.ft.weight[: kb * 768] = torch.from_numpy(net[0] / train.QA).float()
        model.ft_bias[:] = torch.from_numpy(net[1] / train.QA).float()
        model.out.weight[:] = torch.from_numpy(net[2] / train.QB).float()
        model.out.bias[:] = torch.from_numpy(net[3] / (train.QA * train.QB)).float()
    for fen in FENS:
        feats, stm, _, _ = train.parse_chunk([f"{fen} | 0 | 0.5"])
        w, b, s, k, _ = train.tensors(feats, stm, np.zeros(1, np.int16), np.zeros(1, np.float32), np.array([0]), 1.0,
                                      buckets)
        with torch.no_grad():
            cp = float(model(w, b, s, k)) * train.SCALE
        assert abs(train.quantized_eval(net, fen) - cp) <= 2


def test_king_buckets():
    kb = train.KingBuckets("4")
    assert (kb.count, kb.mirror, kb.factor) == (4, True, True)
    feats, _, _, _ = train.parse_chunk([f"{FENS[2]} | 0 | 0.5"])  # White king g1, Black king g8
    white = torch.from_numpy(feats).long()
    rows = kb(white)
    assert rows.shape == (1, 64)
    # g1 -> mirrored to b1 -> bucket 0; the white pawn on a3 becomes h3.
    assert 0 * 768 + 0 * 384 + 0 * 64 + 23 in rows[0, :32].tolist()
    # Factor rows: the same mirrored features after the 4 bucket sets.
    assert 4 * 768 + 23 in rows[0, 32:].tolist()
    # The Black perspective sees its king on g1 as well (flipped vertically).
    black = train.mirror(white)
    black[white == train.PAD] = train.PAD
    assert 5 * 64 + 1 in kb(black)[0, :32].tolist()
    assert train.KingBuckets("none")(white) is white
    with pytest.raises(ValueError):
        train.king_config("1,2,3")


def test_buckets_by_piece_count():
    assert [train.bucket_of(p, 8) for p in (2, 5, 8, 9, 16, 17, 32)] == [0, 1, 1, 2, 3, 4, 7]
    assert train.bucket_of(32, 1) == 0


def test_npz_inputs_equal_text(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    _dataset(a, n=50, seed=1)
    _dataset(b, n=30, seed=2)
    train.load([str(a)], cache=str(tmp_path / "a.npz"))
    direct = train.load([str(a), str(b)])
    mixed = train.load([str(tmp_path / "a.npz"), str(b)])
    for x, y in zip(direct, mixed):
        assert np.array_equal(x, y)


def test_resume_reproduces_a_full_run(tmp_path):
    data = tmp_path / "d.txt"
    _dataset(data)
    common = ["--data", str(data), "--epochs", "4", "--hidden", "8", "--batch", "64", "--threads", "1"]
    train.main(common + ["--out", str(tmp_path / "full.nnue")])
    train.main(common + ["--out", str(tmp_path / "split.nnue"), "--stop-after", "2"])
    train.main(common + ["--out", str(tmp_path / "split.nnue"), "--resume"])
    assert (tmp_path / "full.nnue").read_bytes() == (tmp_path / "split.nnue").read_bytes()
    with pytest.raises(SystemExit):  # a different schedule must not be resumed silently
        train.main([a if a != "4" else "5" for a in common] + ["--out", str(tmp_path / "split.nnue"), "--resume"])


def _engine():
    for c in (ROOT / "build" / "nags_enhanced", ROOT / "build" / "Release" / "nags_enhanced.exe"):
        if c.exists():
            return c
    return None


@pytest.mark.skipif(_engine() is None, reason="nags_enhanced not built")
@pytest.mark.parametrize("buckets,king,activation", DESIGNS)
def test_engine_matches_reference(tmp_path, buckets, king, activation):
    data = tmp_path / "d.txt"
    _dataset(data)
    out = tmp_path / "n.nnue"
    train.main(["--data", str(data), "--out", str(out), "--epochs", "2", "--hidden", "256", "--batch", "64",
                "--threads", "1", "--lr", "0.01", "--buckets", str(buckets), "--king-buckets", king,
                "--activation", activation])
    net = train.read_net(out)
    cmds = f"setoption name EvalFile value {out}\n" + "".join(f"position fen {f}\neval\n" for f in FENS)
    lines = subprocess.run([str(_engine())], input=cmds, capture_output=True, text=True, timeout=60).stdout.splitlines()
    evals = [int(l.split()[3]) for l in lines if l.startswith("info string eval")]
    assert any("NNUE evaluation" in l for l in lines)
    assert evals == [train.quantized_eval(net, f) for f in FENS]
