"""Experimental version 4 NNUE (tools/nnue/train_v4.py): export, the integer
reference, and exact agreement with the engine; versions 1-3 unaffected."""

import struct
import subprocess
import sys
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import compat_nets  # noqa: E402
import train  # noqa: E402
import train_v4  # noqa: E402
from test_compat import GOLDEN_EVALS  # noqa: E402

FENS = [
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/8/1k6/8/8/8/6K1/7Q b - - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
]
DESIGNS = [  # (output buckets, king buckets, activation, l1, l2)
    (1, "none", "crelu", 16, 32),
    (8, "4", "screlu", 8, 16),
    (4, "mirror", "pairwise", 16, 8),
]


def _dataset(path, n=400, seed=0):
    rng = np.random.default_rng(seed)
    with open(path, "w") as fh:
        for i in range(n):
            fh.write(f"{FENS[i % len(FENS)]} | {int(rng.integers(-300, 300))} | "
                     f"{rng.choice(['1.0', '0.5', '0.0'])}\n")


def _train(tmp_path, hidden, buckets, king, activation, l1, l2, lr="0.001"):
    data = tmp_path / "d.txt"
    _dataset(data)
    out = tmp_path / "n.nnue"
    train_v4.main(["--data", str(data), "--out", str(out), "--epochs", "2", "--hidden", str(hidden), "--batch", "64",
                   "--threads", "1", "--lr", lr, "--buckets", str(buckets), "--king-buckets", king,
                   "--activation", activation, "--l1", str(l1), "--l2", str(l2)])
    return out


def _float_eval(model, fen):
    feats, stm, _, _ = train.parse_chunk([f"{fen} | 0 | 0.5"])
    w, b, s, k, _ = train.tensors(feats, stm, np.zeros(1, np.int16), np.zeros(1, np.float32), np.array([0]), 1.0,
                                  model.buckets)
    with torch.no_grad():
        return float(model(w, b, s, k)[0]) * train.SCALE


@pytest.mark.parametrize("buckets,king,activation,l1,l2", DESIGNS)
def test_train_export_and_reference(tmp_path, buckets, king, activation, l1, l2):
    out = _train(tmp_path, 16, buckets, king, activation, l1, l2)
    kb = train.king_config(king)[0]
    inputs = 16 if activation == "pairwise" else 32
    per_bucket = 2 * (l1 * inputs + l2 * l1 + l2) + 4 * (l1 + l2 + 1)
    assert out.stat().st_size == train_v4.HEADER + 2 * (kb * 768 * 16 + 16) + buckets * per_bucket
    net = train_v4.read_net(str(out))
    info = net["info"]
    assert (info["buckets"], info["king_buckets"], info["l1"], info["l2"]) == (buckets, kb, l1, l2)
    assert info["screlu"] == (activation == "screlu") and info["pairwise"] == (activation == "pairwise")
    assert net["w1"].shape == (buckets, l1, inputs) and net["w3"].shape == (buckets, l2)
    # The checkpoint's float network and the quantized file agree closely.
    ckpt = torch.load(str(out) + ".ckpt", weights_only=False)
    model = train_v4.NnueV4(16, buckets, king, activation, l1, l2)
    model.load_state_dict(ckpt["model"])
    for fen in FENS:
        assert abs(_float_eval(model, fen) - train_v4.quantized_eval(net, fen)) < 10, fen


def test_resume_refuses_another_design(tmp_path):
    out = _train(tmp_path, 16, 1, "none", "crelu", 16, 32)
    with pytest.raises(SystemExit, match="l1"):
        train_v4.main(["--data", str(tmp_path / "d.txt"), "--out", str(out), "--epochs", "2", "--hidden", "16",
                       "--batch", "64", "--threads", "1", "--l1", "8", "--resume"])


def _engine():
    for c in (ROOT / "build" / "nags_enhanced", ROOT / "build" / "Release" / "nags_enhanced.exe"):
        if c.exists():
            return c
    return None


def _engine_evals(commands):
    out = subprocess.run([str(_engine())], input=commands, capture_output=True, text=True, timeout=120).stdout
    return out, [int(l.split()[3]) for l in out.splitlines() if l.startswith("info string eval")]


GAMES = [  # positions reached by moves, so the engine's accumulators are updated incrementally
    (chess.STARTING_FEN, "e2e4 d7d5 e4d5 g8f6 f1b5 c7c6 d5c6 d8d2 b1d2 b7c6 g1f3 c6b5 e1g1 e7e5 f1e1".split()),
    ("r3k2r/pPpp1ppp/8/8/8/8/PpPP1PPP/R3K2R w KQkq - 0 1", "b7a8n b2a1n e1g1 e8g8 a8c7 a1c2".split()),
    ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6 e8d7 e1e2 d7d6".split()),
]


@pytest.mark.skipif(_engine() is None, reason="nags_enhanced not built")
@pytest.mark.parametrize("buckets,king,activation,l1,l2", DESIGNS)
def test_engine_matches_reference(tmp_path, buckets, king, activation, l1, l2):
    out = _train(tmp_path, 256, buckets, king, activation, l1, l2, lr="0.01")
    net = train_v4.read_net(str(out))
    cmds = f"setoption name EvalFile value {out}\n"
    fens = list(FENS)
    for fen in FENS:
        cmds += f"position fen {fen}\neval\n"
    for fen, moves in GAMES:
        board = chess.Board(fen)
        for i, m in enumerate(moves):
            board.push_uci(m)
            cmds += f"position fen {fen} moves {' '.join(moves[: i + 1])}\neval\n"
            fens.append(board.fen())
    text, evals = _engine_evals(cmds)
    if "expects hidden size" in text:
        pytest.skip("engine built with another NAGS_NNUE_HIDDEN")
    assert any("NNUE evaluation" in l for l in text.splitlines()), text
    expected = [train_v4.quantized_eval(net, f) for f in fens]
    assert evals == expected
    assert len(set(expected)) > 3  # a network that evaluates, not a constant


@pytest.mark.skipif(_engine() is None, reason="nags_enhanced not built")
def test_engine_rejects_bad_files_and_keeps_older_formats(tmp_path):
    out = _train(tmp_path, 256, 2, "none", "pairwise", 8, 8, lr="0.01")
    data = out.read_bytes()
    short = tmp_path / "short.nnue"
    short.write_bytes(data[:-1])
    both = tmp_path / "both.nnue"  # pairwise and SCReLU together: not a valid design
    both.write_bytes(data[:24] + struct.pack("<I", 5) + data[28:])
    big = tmp_path / "big.nnue"  # l1 above the limit
    big.write_bytes(data[:92] + struct.pack("<I", 33) + data[96:])
    for bad in (short, both, big):
        text, _ = _engine_evals(f"setoption name EvalFile value {bad}\n")
        assert "EvalFile:" in text and "NNUE evaluation" not in text, (bad.name, text)
    # A version 3 network loaded after a version 4 one evaluates as it always did.
    v3 = tmp_path / "v3.nnue"
    v3.write_bytes(compat_nets.network_bytes("v3_kb4_screlu_b8"))
    cmds = f"setoption name EvalFile value {out}\nposition startpos\neval\nsetoption name EvalFile value {v3}\n"
    for fen, moves in compat_nets.POSITIONS:
        cmds += f"position fen {fen}" + (" moves " + " ".join(moves) if moves else "") + "\neval\n"
    text, evals = _engine_evals(cmds)
    if "expects hidden size" in text:
        pytest.skip("engine built with another NAGS_NNUE_HIDDEN")
    assert evals[0] == train_v4.quantized_eval(train_v4.read_net(str(out)), chess.STARTING_FEN)
    assert evals[1:] == GOLDEN_EVALS["v3_kb4_screlu_b8"]


def test_train_py_cannot_read_version_4(tmp_path):
    """Version 4 is new: the version 1-3 reader refuses it rather than misreading it."""
    out = _train(tmp_path, 16, 1, "none", "crelu", 4, 4)
    with pytest.raises(AssertionError):
        train.read_net(str(out))
