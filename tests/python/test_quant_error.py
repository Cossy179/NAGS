"""tools/nnue/quant_error.py: the batched integer evaluation is the engine's, and the error splits into
weight rounding and integer arithmetic."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import pack  # noqa: E402
import quant_error  # noqa: E402
import train  # noqa: E402
from test_nnue import FENS, _dataset  # noqa: E402

DESIGNS = [
    (["--buckets", "1"], "none", False),
    (["--buckets", "8"], "none", False),
    (["--buckets", "8", "--king-buckets", "4", "--activation", "screlu"], "4", True),
    (["--buckets", "4", "--king-buckets", "mirror"], "mirror", False),
    (["--buckets", "8", "--king-buckets", "16", "--activation", "screlu"], "16", True),
]


def _trained(tmp_path, design):
    text = tmp_path / "d.txt"
    _dataset(text)
    pack.main([str(text)])
    out = tmp_path / "n.nnue"
    train.main(["--data", str(tmp_path / "d.bin"), "--out", str(out), "--epochs", "2", "--hidden", "8",
                "--batch", "64", "--threads", "1", "--device", "cpu"] + design)
    return out, tmp_path / "d.bin"


@pytest.mark.parametrize("design,king,screlu", DESIGNS)
def test_batched_integer_eval_is_the_engines(tmp_path, design, king, screlu):
    out, _ = _trained(tmp_path, design)
    net = quant_error.IntegerNet(str(out))
    ref = train.read_net(str(out))
    fens = FENS + ["8/8/1k6/8/8/8/6K1/7Q w - - 0 1", "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1",
                   "r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1"]
    feats, stm, score, result = train.parse_chunk([f"{f} | 0 | 0.5" for f in fens])
    w, b, s, k, _ = train.tensors(feats, stm, score, result, np.arange(len(fens)), 1.0, net.buckets)
    engine, rounded = net.evaluate(w, b, s, k)
    assert engine.tolist() == [float(train.quantized_eval(ref, f)) for f in fens]
    assert (engine - rounded).abs().max() < 2  # truncation costs at most about a centipawn


@pytest.mark.parametrize("design,king,screlu", DESIGNS[1:3])
def test_the_error_splits_into_weights_and_arithmetic(tmp_path, design, king, screlu):
    out, data = _trained(tmp_path, design)
    rep = quant_error.measure(str(out) + ".ckpt", [str(data)], count=300)
    total, weights, arith = (rep[k]["overall"] for k in ("total (float - engine)", "weights (float - rounded)",
                                                         "arithmetic (rounded - engine)"))
    assert rep["positions"] == 300 and total["positions"] == 300
    assert total["bias"] == pytest.approx(weights["bias"] + arith["bias"], abs=1e-6)
    assert total["mean"] <= weights["mean"] + arith["mean"] + 1e-9
    assert 0 < arith["mean"] < 1.5 and arith["max"] < 2
    assert sum(b["positions"] for b in rep["total (float - engine)"]["by_output_bucket"]) == 300
    # The exported network alone: the arithmetic part only, and the same numbers.
    alone = quant_error.measure(str(out), [str(data)], count=300, simulate=2)
    assert set(alone) & {"total (float - engine)", "weights (float - rounded)"} == set()
    assert alone["arithmetic (rounded - engine)"]["overall"] == arith
    # Simulated roundings: one more rounding's worth of noise per layer, reproducible from the seed.
    sim = alone["simulated rounding"]
    assert set(sim) == {"feature transformer", "output layer"}
    assert all(len(t) == 2 and all(x["positions"] == 300 and x["mean"] > 0 for x in t) for t in sim.values())
    again = quant_error.measure(str(out), [str(data)], count=300, simulate=2)["simulated rounding"]
    assert again == sim


def test_positions_are_drawn_from_every_file(tmp_path):
    for i in range(2):
        _dataset(tmp_path / f"d{i}.txt", n=200, seed=i)
        pack.main([str(tmp_path / f"d{i}.txt")])
    files = [str(tmp_path / "d0.bin"), str(tmp_path / "d1.bin")]
    recs = quant_error.positions(files, 10_000, seed=3)
    assert len(recs) == 400  # all of them, once
    whole = np.concatenate([np.fromfile(f, np.uint8).reshape(-1, 32) for f in files])
    assert sorted(map(bytes, recs)) == sorted(map(bytes, whole))
    some = quant_error.positions(files, 50, seed=3)
    assert len(some) == 50 and np.array_equal(some, quant_error.positions(files, 50, seed=3))


@pytest.mark.skipif(not (ROOT / "nets" / "nags.nnue").exists(), reason="no built-in network")
def test_the_built_in_network_matches_the_reference():
    net = quant_error.IntegerNet(str(ROOT / "nets" / "nags.nnue"))
    ref = train.read_net(str(ROOT / "nets" / "nags.nnue"))
    feats, stm, score, result = train.parse_chunk([f"{f} | 0 | 0.5" for f in FENS])
    w, b, s, k, _ = train.tensors(feats, stm, score, result, np.arange(len(FENS)), 1.0, net.buckets)
    engine, _ = net.evaluate(w, b, s, k)
    assert engine.tolist() == [float(train.quantized_eval(ref, f)) for f in FENS]
    assert torch.is_tensor(engine)
