"""tools/nnue/train.py: the fused feature-transformer path (--ft-impl fused) computes the same network as the
embedding path, and quantization-aware training (--qat) trains exactly what the export writes."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "nnue"))
sys.path.insert(0, str(ROOT / "tests" / "python"))
import quant_error  # noqa: E402
import train  # noqa: E402
from test_train_provenance import COMMON, _bin_data  # noqa: E402

DESIGNS = [("none", False, 1), ("none", False, 8), ("mirror", False, 4), ("4", True, 8), ("16", True, 8)]


def _batch(path, buckets):
    rec = torch.from_numpy(np.fromfile(path, np.uint8).reshape(-1, train.RECORD_BYTES))
    feats, stm, score, result = train.decode(rec)
    return train.tensors(feats, stm, score, result, torch.arange(len(rec)), 0.75, buckets)


def _random_model(king, screlu, buckets, **kw):
    torch.manual_seed(3)
    m = train.Nnue(32, buckets, king, screlu, **kw)
    with torch.no_grad():  # weights away from the start, factor rows included
        m.ft.weight.uniform_(-0.3, 0.3)
        m.ft.weight[m.pad].zero_()
        m.ft_bias.uniform_(0.0, 0.3)
        m.out.weight.uniform_(-0.5, 0.5)
        m.out.bias.uniform_(-0.1, 0.1)
    return m


@pytest.mark.parametrize("king,screlu,buckets", DESIGNS)
def test_fused_path_is_the_same_function_and_gradient(tmp_path, king, screlu, buckets):
    w, b, s, k, t = _batch(_bin_data(tmp_path), buckets)
    ref = _random_model(king, screlu, buckets)
    fused = _random_model(king, screlu, buckets, ft_impl="fused")
    fused.load_state_dict(ref.state_dict())
    outs = []
    for m in (ref, fused):
        out = m(w, b, s, k)
        ((torch.sigmoid(out) - t) ** 2).mean().backward()
        outs.append(out.detach())
    assert torch.allclose(outs[0], outs[1], atol=1e-5)
    for (name, p), q in zip(ref.named_parameters(), fused.parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-7), name


@pytest.mark.parametrize("king,screlu,buckets", DESIGNS)
def test_qat_forward_is_the_export(tmp_path, king, screlu, buckets):
    w, b, s, k, _ = _batch(_bin_data(tmp_path), buckets)
    m = _random_model(king, screlu, buckets)
    qat = _random_model(king, screlu, buckets, qat=True)
    qat.load_state_dict(m.state_dict())
    path = tmp_path / "n.nnue"
    train.export(m, str(path))
    net = quant_error.IntegerNet(str(path))
    with torch.no_grad():
        engine, rounded = net.evaluate(w, b, s, k)
        mine = qat(w, b, s, k).double() * train.SCALE
        plain = m(w, b, s, k).double() * train.SCALE
    assert (mine - rounded).abs().max() < 1e-3  # the exported weights, in floating point
    assert (mine - engine).abs().max() <= 1.01  # the engine's truncating integer arithmetic
    assert (plain - rounded).abs().mean() > 1.0  # while the float network is elsewhere


def test_qat_gradient_is_straight_through():
    x = torch.tensor([0.1234, -0.5, 1.98, 200.0], requires_grad=True)
    y = train.fake_quant(x, train.QA)
    assert torch.equal(y.detach(), torch.clamp(torch.round(x.detach() * train.QA), -32767, 32767) / train.QA)
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones(4))


def test_qat_run_from_an_exported_network(tmp_path):
    data = _bin_data(tmp_path)
    design = ["--buckets", "8", "--king-buckets", "4", "--activation", "screlu"]
    src = tmp_path / "src.nnue"
    train.main(["--data", str(data), "--out", str(src), "--epochs", "2"] + COMMON + design)
    out = tmp_path / "qat.nnue"
    train.main(["--data", str(data), "--out", str(out), "--epochs", "2", "--lr", "1e-4", "--qat",
                "--init-from", str(src)] + COMMON + design)
    manifest = json.loads((tmp_path / "qat.nnue.json").read_text())
    assert manifest["settings"]["qat"] is True and manifest["settings"]["ft_impl"] == "fused"
    ckpt = torch.load(str(out) + ".ckpt", weights_only=False)
    assert ckpt["qat"] is True
    # The checkpoint's forward pass is the exported file, as the engine evaluates it (to its truncations).
    rep = quant_error.measure(str(out) + ".ckpt", [str(data)], count=300)
    assert rep["weights (float - rounded)"]["overall"]["max"] < 1e-3
    assert rep["total (float - engine)"]["overall"]["max"] <= 1.01
    assert rep["loss"]["engine"] > 0 and rep["loss"]["float"] > 0
    # A QAT run resumes only as a QAT run, and a plain run not as one.
    with pytest.raises(SystemExit, match="qat"):
        train.main(["--data", str(data), "--out", str(out), "--epochs", "2", "--lr", "1e-4", "--resume",
                    "--init-from", str(src)] + COMMON + design)
    plain = tmp_path / "plain.nnue"
    train.main(["--data", str(data), "--out", str(plain), "--epochs", "1"] + COMMON + design)
    with pytest.raises(SystemExit, match="qat"):
        train.main(["--data", str(data), "--out", str(plain), "--epochs", "1", "--resume", "--qat"]
                   + COMMON + design)


def test_ft_impl_can_change_on_resume(tmp_path):
    """The fused path computes the same function with the same parameters, so a run may switch."""
    data = _bin_data(tmp_path)
    design = ["--buckets", "8", "--king-buckets", "4", "--activation", "screlu"]
    out = tmp_path / "n.nnue"
    train.main(["--data", str(data), "--out", str(out), "--epochs", "3", "--stop-after", "1"] + COMMON + design)
    train.main(["--data", str(data), "--out", str(out), "--epochs", "3", "--resume", "--ft-impl", "fused"]
               + COMMON + design)
    assert json.loads((tmp_path / "n.nnue.json").read_text())["epochs_done"] == 3
