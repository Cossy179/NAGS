"""rpc_server.py serving 'nags-gnn-v3' checkpoints, and the legacy replies staying as they were."""

import json
import socket
import threading
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

import rpc_server
from chess_graph import ChessGraph
from gnn_evaluator import GNNEvaluator, legal_move_priors
from gnn_v3 import GNNAdapterV2, GNNAdapterV3, RelationalGNNEvaluator, legal_move_priors_v3, save_checkpoint_v3
from relational_graph import RelationalChessGraph

CPU = torch.device("cpu")
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "data" / "compat"
PROMO_W = "1n2k1n1/PPP3PP/8/8/8/8/ppp3pp/1N2K1N1 w - - 0 1"
PROMO_B = "4k3/1P6/8/8/8/8/6p1/4K3 b - - 0 1"


def tiny_v3(seed=0):
    torch.manual_seed(seed)
    return RelationalGNNEvaluator(hidden_dim=16, layers=2, num_bases=2, device=CPU)


def request(port, payloads):
    with socket.create_connection(("127.0.0.1", port), timeout=60) as sock, sock.makefile("rwb") as f:
        replies = []
        for p in payloads:
            f.write(((p if isinstance(p, str) else json.dumps(p)) + "\n").encode())
            f.flush()
            replies.append(json.loads(f.readline()))
        return replies


@pytest.fixture(scope="module")
def v3_service():
    return rpc_server.EvaluationService(GNNAdapterV3(tiny_v3(), RelationalChessGraph(device=CPU)), mc_samples=2)


def test_v3_priors_align_with_requested_moves(v3_service):
    fens = [PROMO_W, PROMO_B, chess.STARTING_FEN]
    moves = [[m.uci() for m in chess.Board(f).legal_moves] for f in fens]
    moves[0] = list(reversed(moves[0]))  # any order: priors follow the request
    reply = v3_service.handle({"fens": fens, "moves": moves})
    assert reply["policy_dim"] == 4228 and reply["model_format"] == "nags-gnn-v3"
    model = v3_service.model
    builder = RelationalChessGraph(device=CPU)
    for fen, mv, item in zip(fens, moves, reply["results"]):
        probs, _, _ = model.evaluate(builder.fen_to_graph(fen), mc_samples=1)
        assert len(item["move_priors"]) == len(mv) and abs(sum(item["move_priors"]) - 1) < 1e-5
        assert np.allclose(item["move_priors"], legal_move_priors_v3(probs[0], mv), atol=1e-6)
        assert -1 <= item["value"] <= 1 and item["uncertainty"] >= 0
    by_move = dict(zip(moves[1], reply["results"][1]["move_priors"]))
    assert len({by_move["g2g1" + p] for p in "qrbn"}) == 4  # underpromotions are not collapsed
    full = v3_service.handle({"fens": [PROMO_W]})
    assert len(full["results"][0]["policy"]) == 4228
    assert v3_service.handle({"fens": []}) == {"results": [], "policy_dim": 4228, "model_format": "nags-gnn-v3"}


def test_v3_over_a_socket(v3_service):
    server = rpc_server.make_server(v3_service, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        moves = [m.uci() for m in chess.Board(PROMO_B).legal_moves]
        replies = request(server.server_address[1], [
            {"fens": [PROMO_B], "moves": [moves]}, "not json", {"fens": ["garbage"]},
            {"fens": [PROMO_B], "moves": [["g2g1n", "g2g1q"]]}])
        assert len(replies[0]["results"][0]["move_priors"]) == len(moves)
        assert "error" in replies[1] and "error" in replies[2]
        assert len(replies[3]["results"][0]["move_priors"]) == 2
    finally:
        server.shutdown()
        server.server_close()


def legacy_reply(model, fens, moves, mc_samples):
    """The reply exactly as rpc_server.py built it before v3 existed."""
    batch = ChessGraph(device=CPU).fens_to_batch(fens)
    probs, values, unc = model.evaluate(batch, mc_samples=mc_samples)
    results = []
    for i in range(len(fens)):
        item = {"value": float(values[i]), "uncertainty": float(unc[i])}
        if moves is not None:
            item["move_priors"] = legal_move_priors(probs[i], [str(m) for m in moves[i]])
        else:
            item["policy"] = probs[i].tolist()
        results.append(item)
    return {"results": results, "policy_dim": 4096}


@pytest.mark.parametrize("with_moves", [True, False])
def test_legacy_replies_are_unchanged(with_moves):
    torch.manual_seed(0)
    model = GNNEvaluator(hidden_dim=32, gnn_layers=2, policy_layers=1, value_layers=1, device=CPU)
    model.eval()
    fens = [chess.STARTING_FEN, PROMO_W]
    moves = [[m.uci() for m in chess.Board(f).legal_moves] for f in fens] if with_moves else None
    service = rpc_server.EvaluationService(model, ChessGraph(device=CPU), mc_samples=2)
    torch.manual_seed(5)
    reply = service.handle({"fens": fens, **({"moves": moves} if with_moves else {})})
    torch.manual_seed(5)
    expected = legacy_reply(model, fens, moves, 2)
    assert list(reply) == ["results", "policy_dim", "model_format"] and reply["model_format"] == "nags-gnn-v2"
    reply.pop("model_format")
    assert json.dumps(reply) == json.dumps(expected)  # byte for byte, apart from the added field


def test_load_served_model(tmp_path):
    fallback = rpc_server.load_served_model(str(tmp_path / "missing.pth"), CPU, hidden_dim=16)
    assert isinstance(fallback, GNNAdapterV2) and fallback.policy_dim == 4096
    assert fallback.model.config["hidden_dim"] == 16 and not fallback.model.training
    assert isinstance(rpc_server.load_served_model("", CPU, hidden_dim=16), GNNAdapterV2)
    v2 = rpc_server.load_served_model(str(FIXTURES / "gnn_v2.pt"), CPU)
    assert isinstance(v2, GNNAdapterV2)
    cases = json.loads((FIXTURES / "gnn_v2_expected.json").read_text())
    reply = rpc_server.EvaluationService(v2, mc_samples=1).handle(
        {"fens": [c["fen"] for c in cases], "moves": [c["moves"] for c in cases]})
    assert reply["policy_dim"] == 4096 and reply["model_format"] == "nags-gnn-v2"
    for case, r in zip(cases, reply["results"]):
        assert np.allclose(r["move_priors"], case["priors"], atol=1e-5) and abs(r["value"] - case["value"]) < 1e-5
    path = tmp_path / "v3.pth"
    save_checkpoint_v3(tiny_v3(), str(path))
    assert isinstance(rpc_server.load_served_model(str(path), CPU), GNNAdapterV3)


def test_server_process_serves_a_v3_checkpoint(tmp_path):
    import training_pipeline as tp

    path = tmp_path / "v3.pth"
    save_checkpoint_v3(tiny_v3(), str(path))
    with tp.ServiceProcess("rpc_server.py", ["--model", str(path), "--device", "cpu", "--mc-samples", "1"],
                           tmp_path / "rpc.log") as rpc:
        moves = [m.uci() for m in chess.Board(PROMO_W).legal_moves]
        reply = request(rpc.port, [{"fens": [PROMO_W], "moves": [moves]}])[0]
    assert reply["policy_dim"] == 4228 and reply["model_format"] == "nags-gnn-v3"
    assert len(reply["results"][0]["move_priors"]) == len(moves)
    assert "Loaded nags-gnn-v3 model" in (tmp_path / "rpc.log").read_text()


def test_v3_full_policy_is_on_legal_moves(v3_service):
    from policy_encoding import legal_mask_v3

    fens = [PROMO_W, PROMO_B, chess.STARTING_FEN]
    reply = v3_service.handle({"fens": fens})
    for fen, item in zip(fens, reply["results"]):
        policy = torch.tensor(item["policy"])
        legal = legal_mask_v3(chess.Board(fen))
        assert abs(float(policy.sum()) - 1) < 1e-5 and float(policy[~legal].sum()) == 0.0


def test_v2_is_served_when_gnn_v3_cannot_be_imported(tmp_path):
    """A broken or missing gnn_v3 (or one of its dependencies) must not stop v2 networks being served."""
    import subprocess
    import sys

    (tmp_path / "gnn_v3.py").write_text("raise ImportError('simulated')\n")
    root = Path(__file__).resolve().parents[2]
    script = f"""
import json, sys
sys.path[:0] = [{str(tmp_path)!r}, {str(root)!r}]
import torch, rpc_server
assert rpc_server._V3_IMPORT_ERROR is not None
cases = json.loads(open({str(FIXTURES / 'gnn_v2_expected.json')!r}).read())
adapter = rpc_server.load_served_model({str(FIXTURES / 'gnn_v2.pt')!r}, torch.device('cpu'))
reply = rpc_server.EvaluationService(adapter, mc_samples=1).handle(
    {{"fens": [c["fen"] for c in cases], "moves": [c["moves"] for c in cases]}})
assert reply["policy_dim"] == 4096 and reply["model_format"] == "nags-gnn-v2", reply
for c, r in zip(cases, reply["results"]):
    assert max(abs(a - b) for a, b in zip(r["move_priors"], c["priors"])) < 1e-5 and abs(r["value"] - c["value"]) < 1e-5
print("ok")
"""
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=tmp_path, timeout=300)
    assert out.returncode == 0 and out.stdout.strip().endswith("ok"), out.stderr
