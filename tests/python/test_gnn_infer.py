"""gnn_infer backends compute what RelationalGNNEvaluator.evaluate computes:
the torch backend bit for bit (MC-dropout samples included), ONNX Runtime to float precision."""

import chess
import numpy as np
import pytest
import torch

from gnn_infer import OnnxBackendV3, TorchBackendV3, cached_onnx, make_backend
from gnn_v3 import GNNAdapterV3, RelationalGNNEvaluator
from relational_fast import build_batch
from relational_graph import RelationalChessGraph
from test_relational_fast import SPECIAL, random_positions

CPU = torch.device("cpu")


def _onnx_available():
    try:
        import onnxruntime  # noqa: F401
        import onnxscript  # noqa: F401  (torch.onnx.export's exporter)
        return True
    except Exception:
        return False


needs_onnx = pytest.mark.skipif(not _onnx_available(), reason="onnxruntime / onnxscript not installed "
                                                                "(requirements-onnx.txt)")


def model_of(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(hidden_dim=32, layers=2, num_bases=2)
    cfg.update(kw)
    return RelationalGNNEvaluator(device=CPU, **cfg).eval()


def reference(model, fens, mc_samples):
    batch = GNNAdapterV3(model, RelationalChessGraph(device=CPU)).build_batch(fens)
    return model.evaluate(batch, mc_samples=mc_samples, legal=batch.legal_v3)


FENS = SPECIAL + [b.fen() for b in random_positions(40, 3, tactical=True)]


@pytest.mark.parametrize("mc_samples", [1, 3])
@pytest.mark.parametrize("cfg", [{}, {"ffn_mult": 0, "aux_score": False}, {"dropout": 0.1, "value_dropout": 0.3}])
def test_torch_backend_is_bit_identical(mc_samples, cfg):
    model = model_of(1, **cfg)
    backend = TorchBackendV3(model)
    for chunk in (FENS[:1], FENS[:7], FENS):
        torch.manual_seed(42)
        ref = reference(model, chunk, mc_samples)
        torch.manual_seed(42)
        got = backend.run(build_batch(chunk), mc_samples=mc_samples)
        for r, g in zip(ref, got):
            assert r.dtype == g.dtype and torch.equal(r, g)
    assert not model.training


def test_torch_backend_leaves_train_mode_alone():
    model = model_of(2)
    model.train()
    TorchBackendV3(model)
    assert not model.training  # the backend owns an eval-mode model
    backend = TorchBackendV3(model)
    backend.run(build_batch(FENS[:3]), mc_samples=3)
    assert not model.training and not any(m.training for m in model.modules())


@pytest.fixture(scope="module")
def onnx_model(tmp_path_factory):
    model = model_of(3)
    path = cached_onnx(model, str(tmp_path_factory.mktemp("onnx")))
    return model, path


@needs_onnx
def test_onnx_matches_torch(onnx_model):
    model, path = onnx_model
    backend = OnnxBackendV3(model, onnx_path=path)
    assert backend.providers[-1] == "CPUExecutionProvider"
    for chunk in (FENS[:1], FENS[:5], FENS):
        ref_p, ref_v, _ = reference(model, chunk, 1)
        p, v, s = backend.run(build_batch(chunk), mc_samples=1)
        assert p.shape == ref_p.shape and torch.all(s == 0)
        assert float((p - ref_p).abs().max()) < 1e-5 and float((v - ref_v).abs().max()) < 1e-5
        # Illegal moves get exactly zero, as with the torch masking (rows without legal moves aside).
        legal = build_batch(chunk).legal
        has_moves = legal.any(axis=1)
        assert float(p.numpy()[has_moves][~legal[has_moves]].max(initial=0.0)) == 0.0
        if (~has_moves).any():
            assert torch.allclose(p[~torch.from_numpy(has_moves)].sum(dim=1), torch.ones(int((~has_moves).sum())))


@needs_onnx
def test_onnx_mc_dropout_has_the_right_spread(onnx_model):
    model, path = onnx_model
    backend = OnnxBackendV3(model, onnx_path=path, seed=1)
    fens = FENS[:20]
    _, v1, s1 = backend.run(build_batch(fens), mc_samples=1)
    _, vm, sm = backend.run(build_batch(fens), mc_samples=64)
    torch.manual_seed(0)
    _, tm, ts = reference(model, fens, 64)
    assert torch.all(sm > 0)
    # Same distribution as torch's MC dropout: means and spreads agree within sampling error.
    assert float((vm - tm).abs().max()) < 4 * float(ts.max()) / 8 + 1e-3
    assert float((sm / ts).log().abs().mean()) < 0.35


@needs_onnx
def test_cached_export_is_reused_and_keyed_by_weights(tmp_path):
    a = cached_onnx(model_of(4), str(tmp_path))
    mtime = __import__("os").path.getmtime(a)
    assert cached_onnx(model_of(4), str(tmp_path)) == a and __import__("os").path.getmtime(a) == mtime
    assert cached_onnx(model_of(5), str(tmp_path)) != a


@needs_onnx
def test_make_backend(onnx_model):
    model, path = onnx_model
    assert isinstance(make_backend(model, "torch"), TorchBackendV3)
    assert isinstance(make_backend(model, "onnx", onnx_path=path, providers=["tensorrt", "cuda", "cpu"]), OnnxBackendV3)
    with pytest.raises(ValueError):
        make_backend(model, "tpu")


def test_reduced_precision_path_runs():
    # --fp16 uses CUDA autocast; the CPU's bfloat16 autocast takes the same code path here.
    model = model_of(6)
    backend = TorchBackendV3(model)
    backend.fp16 = True
    backend.autocast_device, backend.autocast_dtype = "cpu", torch.bfloat16
    fens = FENS[:12]
    p, v, s = backend.run(build_batch(fens), mc_samples=3)
    ref_p, ref_v, _ = reference(model, fens, 1)
    assert p.dtype == torch.float32 and v.dtype == torch.float32 and torch.all(s >= 0)
    assert torch.allclose(p.sum(dim=1), torch.ones(len(fens)), atol=1e-4)
    legal = torch.from_numpy(build_batch(fens).legal)
    assert float(p[legal.any(dim=1)][~legal[legal.any(dim=1)]].abs().max()) == 0.0
    assert float((p - ref_p).abs().max()) < 0.05 and float((v - ref_v).abs().max()) < 0.2
