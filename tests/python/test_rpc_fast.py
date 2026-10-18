"""rpc_server.py's fast paths: the gnn_infer backends, cross-connection batching with
pipelined requests, and the reply cache. Replies keep their fields and meaning."""

import json
import socket
import threading

import chess
import numpy as np
import pytest
import torch

import rpc_server
from gnn_infer import TorchBackendV3, make_backend
from gnn_v3 import GNNAdapterV3, RelationalGNNEvaluator, save_checkpoint_v3
from relational_graph import RelationalChessGraph
from test_gnn_infer import needs_onnx
from test_relational_fast import SPECIAL, random_positions

CPU = torch.device("cpu")
FENS = SPECIAL + [b.fen() for b in random_positions(30, 5, tactical=True)]


def tiny_v3(seed=0):
    torch.manual_seed(seed)
    return RelationalGNNEvaluator(hidden_dim=16, layers=2, num_bases=2, device=CPU).eval()


def legal_moves(fen):
    return [m.uci() for m in chess.Board(fen).legal_moves]


def services(mc_samples, **kw):
    model = tiny_v3()
    adapter = GNNAdapterV3(model, RelationalChessGraph(device=CPU))
    legacy = rpc_server.EvaluationService(adapter, mc_samples=mc_samples)
    fast = rpc_server.EvaluationService(adapter, mc_samples=mc_samples, backend=TorchBackendV3(model), **kw)
    return legacy, fast


@pytest.mark.parametrize("mc_samples", [1, 2])
@pytest.mark.parametrize("with_moves", [True, False])
def test_torch_backend_replies_are_byte_identical(mc_samples, with_moves):
    legacy, fast = services(mc_samples)
    for chunk in ([FENS[0]], FENS[:6], FENS):
        payload = {"fens": chunk}
        if with_moves:
            payload["moves"] = [list(reversed(legal_moves(f))) for f in chunk]
        torch.manual_seed(9)
        a = json.dumps(legacy.handle(payload))
        torch.manual_seed(9)
        b = json.dumps(fast.handle(payload))
        assert a == b


def test_errors_are_the_same():
    legacy, fast = services(1)
    for bad in ["x", {"fens": ["garbage"]}, {"fens": [chess.STARTING_FEN], "moves": []}, {"fens": "nope", "moves": 3},
                {"fens": [chess.STARTING_FEN] * 257}]:
        errors = []
        for service in (legacy, fast):
            with pytest.raises(ValueError) as e:
                service.handle(bad)
            errors.append(str(e.value))
        assert errors[0] == errors[1]
    assert fast.handle({"fens": []}) == legacy.handle({"fens": []})


def test_backend_refuses_v2():
    from chess_graph import ChessGraph
    from gnn_evaluator import GNNEvaluator

    torch.manual_seed(0)
    v2 = GNNEvaluator(hidden_dim=16, gnn_layers=1, policy_layers=1, value_layers=1, device=CPU)
    with pytest.raises(ValueError):
        rpc_server.EvaluationService(v2, ChessGraph(device=CPU), backend=TorchBackendV3(tiny_v3()))


def test_cache():
    legacy, fast = services(1, cache_size=8)
    payload = {"fens": FENS[:5], "moves": [legal_moves(f) for f in FENS[:5]]}
    first = fast.handle(payload)
    assert fast.cache_hits == 0 and len(fast.cache) == 5
    mixed = {"fens": FENS[3:8], "moves": [legal_moves(f) for f in FENS[3:8]]}
    second = fast.handle(mixed)
    assert fast.cache_hits == 2
    # Two of these were computed in the first request's batch, three in a batch of their own; a
    # network's float results can depend on the batch size (GEMM blocking), so only nearly equal.
    for a, b in zip(second["results"], legacy.handle(mixed)["results"]):
        assert np.allclose(a["move_priors"], b["move_priors"], atol=1e-6) and abs(a["value"] - b["value"]) < 1e-6
    assert json.dumps(fast.handle(payload)) == json.dumps(first) == json.dumps(legacy.handle(payload))
    assert len(fast.cache) == 8  # bounded, least recently used out
    # Different moves for the same FEN are a different request.
    sub = fast.handle({"fens": [FENS[0]], "moves": [legal_moves(FENS[0])[:2]]})
    assert len(sub["results"][0]["move_priors"]) == 2


def _client(port):
    sock = socket.create_connection(("127.0.0.1", port), timeout=60)
    return sock, sock.makefile("rwb")


def test_batched_pipelined_connections():
    legacy, fast = services(1, batch_wait_ms=5.0, max_batch=16)
    server = rpc_server.make_server(fast, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    expected = {f: legacy.handle({"fens": [f], "moves": [legal_moves(f)]})["results"][0] for f in FENS}
    failures = []

    def run(offset):
        try:
            sock, f = _client(port)
            order = FENS[offset:] + FENS[:offset]
            # All requests first, then all replies: they must come back in order.
            for fen in order:
                f.write((json.dumps({"fens": [fen], "moves": [legal_moves(fen)]}) + "\n").encode())
            f.write(b"not json\n")
            f.write((json.dumps({"fens": ["garbage"]}) + "\n").encode())
            f.write((json.dumps({"fens": order[:3], "moves": [legal_moves(x) for x in order[:3]]}) + "\n").encode())
            f.flush()
            for fen in order:
                item = json.loads(f.readline())["results"][0]
                ref = expected[fen]
                assert len(item["move_priors"]) == len(ref["move_priors"])
                assert np.allclose(item["move_priors"], ref["move_priors"], atol=1e-6)
                assert abs(item["value"] - ref["value"]) < 1e-6
            assert "error" in json.loads(f.readline())
            assert "error" in json.loads(f.readline())
            multi = json.loads(f.readline())
            assert [len(r["move_priors"]) for r in multi["results"]] == [len(legal_moves(x)) for x in order[:3]]
            sock.close()
        except Exception as e:  # pragma: no cover - reported below
            failures.append(repr(e))

    threads = [threading.Thread(target=run, args=(i * 7,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    server.shutdown()
    server.server_close()
    assert not failures, failures
    assert fast.batcher.positions >= 4 * (len(FENS) + 3)
    assert fast.batcher.batches < fast.batcher.positions  # requests were combined


@needs_onnx
def test_server_process_with_onnx_batching_and_cache(tmp_path):
    import training_pipeline as tp

    path = tmp_path / "v3.pth"
    save_checkpoint_v3(tiny_v3(), str(path))
    args = ["--model", str(path), "--device", "cpu", "--mc-samples", "1", "--backend", "onnx", "--batch-wait-ms", "1",
            "--cache-size", "100", "--onnx-cache", str(tmp_path / "onnx")]
    reference = rpc_server.EvaluationService(GNNAdapterV3(tiny_v3(), RelationalChessGraph(device=CPU)), mc_samples=1)
    with tp.ServiceProcess("rpc_server.py", args, tmp_path / "rpc.log") as rpc:
        sock, f = _client(rpc.port)
        for fen in FENS[:6] + FENS[:2]:
            f.write((json.dumps({"fens": [fen], "moves": [legal_moves(fen)]}) + "\n").encode())
            f.flush()
            reply = json.loads(f.readline())
            ref = reference.handle({"fens": [fen], "moves": [legal_moves(fen)]})
            assert reply["model_format"] == "nags-gnn-v3" and reply["policy_dim"] == 4228
            assert np.allclose(reply["results"][0]["move_priors"], ref["results"][0]["move_priors"], atol=1e-5)
        sock.close()
    log = (tmp_path / "rpc.log").read_text()
    assert "Backend onnx (CPUExecutionProvider)" in log, log
    assert list((tmp_path / "onnx").glob("*.onnx"))


@needs_onnx
def test_make_backend_onnx_matches_service(tmp_path):
    model = tiny_v3(3)
    adapter = GNNAdapterV3(model, RelationalChessGraph(device=CPU))
    onnx = rpc_server.EvaluationService(adapter, mc_samples=1,
                                        backend=make_backend(model, "onnx", cache_dir=str(tmp_path)))
    legacy = rpc_server.EvaluationService(adapter, mc_samples=1)
    payload = {"fens": FENS, "moves": [legal_moves(f) for f in FENS]}
    for a, b in zip(onnx.handle(payload)["results"], legacy.handle(payload)["results"]):
        assert np.allclose(a["move_priors"], b["move_priors"], atol=1e-5) and abs(a["value"] - b["value"]) < 1e-5
