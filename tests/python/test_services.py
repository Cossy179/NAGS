"""Protocol tests for rpc_server.py and meta_learner.py over real sockets."""

import json
import socket
import threading

import chess
import pytest
import torch

import meta_learner
import rpc_server
from chess_graph import ChessGraph
from gnn_evaluator import GNNEvaluator


class LineClient:
    def __init__(self, port):
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=30)
        self.file = self.sock.makefile("rwb")

    def request(self, payload):
        line = payload if isinstance(payload, str) else json.dumps(payload)
        self.file.write((line + "\n").encode())
        self.file.flush()
        return json.loads(self.file.readline())

    def close(self):
        self.file.close()
        self.sock.close()


def serve(server):
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server.server_address[1]


@pytest.fixture(scope="module")
def rpc_port():
    torch.manual_seed(0)
    model = GNNEvaluator(hidden_dim=32, gnn_layers=2, policy_layers=1, value_layers=1, device=torch.device("cpu"))
    service = rpc_server.EvaluationService(model, ChessGraph(device=torch.device("cpu")), mc_samples=2)
    server = rpc_server.make_server(service, "127.0.0.1", 0)
    yield serve(server)
    server.shutdown()
    server.server_close()


def test_rpc_persistent_connection_and_priors(rpc_port):
    client = LineClient(rpc_port)
    try:
        fen = chess.STARTING_FEN
        moves = [m.uci() for m in chess.Board().legal_moves]
        for _ in range(3):  # several requests on one connection
            resp = client.request({"fens": [fen], "moves": [moves]})
            item = resp["results"][0]
            assert -1.0 <= item["value"] <= 1.0
            assert item["uncertainty"] >= 0.0
            assert len(item["move_priors"]) == len(moves)
            assert abs(sum(item["move_priors"]) - 1.0) < 1e-5
        full = client.request({"fens": [fen, fen]})
        assert len(full["results"]) == 2 and len(full["results"][0]["policy"]) == 4096
    finally:
        client.close()


def test_rpc_errors_keep_connection_open(rpc_port):
    client = LineClient(rpc_port)
    try:
        assert "error" in client.request("not json")
        assert "error" in client.request({"fens": ["garbage fen"]})
        assert "error" in client.request({"fens": [chess.STARTING_FEN], "moves": []})
        ok = client.request({"fens": [chess.STARTING_FEN], "moves": [["e2e4", "d2d4"]]})
        assert len(ok["results"][0]["move_priors"]) == 2
    finally:
        client.close()


@pytest.fixture()
def meta_port(tmp_path):
    meta = meta_learner.MetaLearner(model_path=str(tmp_path / "meta.pth"), data_path=str(tmp_path / "meta.json"))
    server = meta_learner.make_server(meta, "127.0.0.1", 0)
    yield serve(server), tmp_path
    server.shutdown()
    server.server_close()


def test_meta_protocol(meta_port):
    port, tmp_path = meta_port
    client = LineClient(port)
    try:
        req = {"fen": chess.STARTING_FEN, "time_left_ms": 60000, "last_uncertainty": 0.1, "tactical_shot_ratio": 0.2}
        pred = client.request({"command": "predict", **req})
        assert pred["status"] == "ok"
        assert set(pred["deltas"]) == set(meta_learner.DELTA_KEYS)
        assert all(-1.0 <= v <= 1.0 for v in pred["deltas"].values())
        for i in range(10):  # same connection: the engine reuses its socket
            r = client.request({"command": "add_sample", **req, "chosen_deltas": pred["deltas"], "reward": i % 3 - 1})
            assert r["status"] == "ok" and r["samples"] == i + 1
        legacy = client.request({"command": "add_sample", **req, "chosen_deltas": pred["deltas"], "elo_gain_per_sec": 0.5})
        assert legacy["status"] == "ok"
        trained = client.request({"command": "train", "steps": 5})
        assert trained["status"] == "ok" and trained["loss"] is not None
        assert (tmp_path / "meta.pth").exists() and (tmp_path / "meta.json").exists()
        assert client.request({"command": "nope"})["status"] == "error"
    finally:
        client.close()


def test_meta_learner_moves_towards_rewarded_deltas():
    # Untrained predictions sit near 0, about 0.4 from the rewarded target.
    assert meta_learner.selftest(steps=300) < 0.3
