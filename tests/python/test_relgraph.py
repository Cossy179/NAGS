"""The engine's relational-v1 graphs (src/RelGraph.cpp, through relgraph_dump) are exactly
relational_graph.build_graph's: the same float32 features, the same edges in the same order and
the same legal moves. The server's optional "graphs" field gives byte-identical replies, and the
engine (NNGraphs) sends graphs only to a server that uses them. Skipped when relgraph_dump / nags
are not built."""

import base64
import json
import random
import socketserver
import subprocess
import threading
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

import rpc_server
from gnn_infer import TorchBackendV3
from gnn_v3 import GNNAdapterV3, RelationalGNNEvaluator
from policy_encoding import POLICY_V3_DIM
from relational_fast import build_arrays, decode_graph
from relational_graph import RelationalChessGraph
from test_relational_fast import SPECIAL

ROOT = Path(__file__).resolve().parents[2]

UNDERPROMOTIONS = [
    "1n5k/P7/8/8/8/8/7p/K5N1 w - - 0 1",   # push and capture promotions, both colours
    "1n5k/P7/8/8/8/8/7p/K5N1 b - - 0 1",
    "4k3/1P6/8/8/8/8/8/4K3 w - - 0 1",
]


def _binary(name):
    for c in (ROOT / "build" / name, ROOT / "build" / "Release" / f"{name}.exe"):
        if c.exists():
            return c
    return None


DUMP = _binary("relgraph_dump")
ENGINE = _binary("nags")
needs_dump = pytest.mark.skipif(DUMP is None, reason="relgraph_dump is not built")
needs_engine = pytest.mark.skipif(ENGINE is None, reason="nags is not built")
CPU = torch.device("cpu")


def _random_positions(count, seed):
    rng = random.Random(seed)
    fens = []
    while len(fens) < count:
        board = chess.Board()
        for _ in range(rng.randint(0, 200)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
            if rng.random() < 0.1:
                fens.append(board.fen())
        fens.append(board.fen())
    return fens[:count]


def _engine_graphs(fens):
    out = subprocess.run([str(DUMP)], input="\n".join(fens) + "\n", capture_output=True, text=True,
                         timeout=300, check=True).stdout.split("\n")
    assert len(out) == len(fens) + 1 and out[-1] == ""
    return out[:-1]


def _assert_same(fen, encoded):
    ref = build_arrays(fen)
    got = decode_graph(encoded)
    assert got.x.dtype == np.float32 and got.x.tobytes() == ref.x.tobytes(), fen
    for field in ("src", "dst", "edge_type", "node_kind"):
        assert np.array_equal(getattr(got, field), getattr(ref, field)), (fen, field)
    assert sorted(got.legal.tolist()) == sorted(ref.legal.tolist()), fen


@needs_dump
def test_special_positions_match_the_reference():
    fens = [f for f in SPECIAL + UNDERPROMOTIONS if chess.Board(f).is_valid()]
    for fen, encoded in zip(fens, _engine_graphs(fens)):
        assert encoded != "error", fen
        _assert_same(fen, encoded)


@needs_dump
def test_random_positions_match_the_reference():
    fens = _random_positions(3000, seed=7)
    for fen, encoded in zip(fens, _engine_graphs(fens)):
        assert encoded != "error", fen
        _assert_same(fen, encoded)


@needs_dump
def test_underpromotions_have_their_own_policy_indices():
    fen = UNDERPROMOTIONS[0]
    got = decode_graph(_engine_graphs([fen])[0])
    assert sum(i >= 64 * 64 for i in got.legal.tolist()) == 6  # a8 and xb8, three underpromotions each
    assert max(got.legal.tolist()) < POLICY_V3_DIM


@pytest.mark.parametrize("bad", ["", "!!!!", "AAAA", "AAAAAAAA"])
def test_malformed_graphs_are_rejected(bad):
    with pytest.raises(ValueError):
        decode_graph(bad)


@needs_dump
def test_truncated_graph_is_rejected():
    encoded = _engine_graphs([chess.STARTING_FEN])[0]
    raw = base64.b64decode(encoded)
    with pytest.raises(ValueError):
        decode_graph(base64.b64encode(raw[:-1]).decode())


# The server's "graphs" field.

def _service(fast=True, history=False, **kw):
    torch.manual_seed(0)
    model = RelationalGNNEvaluator(hidden_dim=16, layers=2, num_bases=2, history=history, device=CPU).eval()
    return rpc_server.EvaluationService(GNNAdapterV3(model, RelationalChessGraph(device=CPU)),
                                        backend=TorchBackendV3(model) if fast else None, **kw)


def _payload(fens, graphs=None):
    payload = {"fens": fens, "moves": [[m.uci() for m in chess.Board(f).legal_moves] for f in fens]}
    if graphs is not None:
        payload["graphs"] = graphs
    return payload


def _replies(service, fens, graphs):
    torch.manual_seed(5)
    plain = service.handle(_payload(fens))
    torch.manual_seed(5)
    return plain, service.handle(_payload(fens, graphs))


@needs_dump
@pytest.mark.parametrize("mc_samples", [1, 3])
def test_server_replies_are_byte_identical_with_engine_graphs(mc_samples):
    fens = [f for f in SPECIAL + UNDERPROMOTIONS if chess.Board(f).is_valid()] + _random_positions(40, seed=3)
    graphs = _engine_graphs(fens)
    service = _service(mc_samples=mc_samples)
    plain, with_graphs = _replies(service, fens, graphs)
    assert "graphs" not in plain and with_graphs.pop("graphs") is True
    assert json.dumps(with_graphs) == json.dumps(plain)
    # Some positions without a graph: the server builds those.
    partial = [g if i % 3 else None for i, g in enumerate(graphs)]
    plain, mixed = _replies(service, fens, partial)
    assert mixed.pop("graphs") is True and json.dumps(mixed) == json.dumps(plain)


@needs_dump
@pytest.mark.parametrize("fast,history", [(False, False), (True, True), (False, True)])
def test_servers_that_build_their_own_graphs_ignore_them(fast, history):
    fens = _random_positions(6, seed=4)
    service = _service(fast=fast, history=history, mc_samples=1)
    plain, with_graphs = _replies(service, fens, _engine_graphs(fens))
    assert "graphs" not in with_graphs and json.dumps(with_graphs) == json.dumps(plain)
    # Not even a broken graph is looked at.
    _, broken = _replies(service, fens, ["AAAA"] * len(fens))
    assert json.dumps(broken) == json.dumps(plain)


def test_bad_graphs_field_is_an_error():
    service = _service(mc_samples=1)
    fens = [chess.STARTING_FEN]
    for bad in ({"fens": fens, "graphs": "x"}, {"fens": fens, "graphs": []}, {"fens": fens, "graphs": [3]},
                _payload(fens, ["AAAA"])):
        with pytest.raises(ValueError):
            service.handle(bad)
    assert service.handle(_payload(fens, [None]))["graphs"] is True


# The engine's NNGraphs option.

def _run_engine(cmds, accept):
    seen = []

    class Gnn(socketserver.StreamRequestHandler):
        disable_nagle_algorithm = True

        def handle(self):
            for line in self.rfile:
                req = json.loads(line)
                seen.append(req)
                results = [{"value": 0.0, "uncertainty": 0.0, "move_priors": [1.0 / len(m)] * len(m)}
                           for m in req["moves"]]
                reply = {"results": results, "policy_dim": 4228, "model_format": "nags-gnn-v3"}
                if accept and "graphs" in req:
                    reply["graphs"] = True
                self.wfile.write((json.dumps(reply) + "\n").encode())
                self.wfile.flush()

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Gnn)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        script = (f"setoption name UseNN value true\nsetoption name NNPort value {server.server_address[1]}\n"
                  "setoption name UseMetaLearner value false\nsetoption name MctsMinTime value 0\n" + cmds)
        out = subprocess.run([str(ENGINE)], input=script, capture_output=True, text=True, timeout=60).stdout
    finally:
        server.shutdown()
        server.server_close()
    return out, seen


GO = "setoption name NNBatch value 4\nposition startpos moves e2e4 e7e5 g1f3\ngo movetime 400\n"


@needs_engine
def test_engine_sends_graphs_to_a_server_that_uses_them():
    out, seen = _run_engine("setoption name NNGraphs value true\n" + GO, accept=True)
    assert "bestmove" in out and len(seen) > 3
    assert all("graphs" in req and len(req["graphs"]) == len(req["fens"]) for req in seen)
    checked = 0
    for req in seen[:20]:
        for fen, encoded in zip(req["fens"], req["graphs"]):
            ref, got = build_arrays(fen), decode_graph(encoded)
            assert got.x.tobytes() == ref.x.tobytes() and np.array_equal(got.src, ref.src), fen
            assert np.array_equal(got.edge_type, ref.edge_type), fen
            assert sorted(got.legal.tolist()) == sorted(ref.legal.tolist()), fen
            checked += 1
    assert checked > 10


@needs_engine
def test_engine_stops_sending_graphs_to_a_server_that_ignores_them():
    out, seen = _run_engine("setoption name NNGraphs value true\n" + GO, accept=False)
    assert "bestmove" in out and len(seen) > 3
    assert "graphs" in seen[0] and all("graphs" not in req for req in seen[1:])


@needs_engine
def test_engine_sends_no_graphs_by_default():
    out, seen = _run_engine(GO, accept=True)
    assert "bestmove" in out and seen and all("graphs" not in req for req in seen)
