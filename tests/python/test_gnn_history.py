"""Game histories for v3 networks: history_board, the checkpoint flag, the server's optional
'history' field and history rows in distillation. Without the flag nothing changes."""

import json

import chess
import pytest
import torch

import rpc_server
from gnn_infer import TorchBackendV3
from gnn_v3 import (GNNAdapterV3, RelationalGNNEvaluator, history_board, is_history_aware, load_checkpoint_v3,
                    save_checkpoint_v3)
from relational_graph import COL, KIND_GLOBAL, RelationalChessGraph, build_graph

CPU = torch.device("cpu")
SHUFFLE = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]  # the position after it occurred twice before


def board_after(moves):
    b = chess.Board()
    for m in moves:
        b.push_uci(m)
    return b


def test_history_board():
    target = board_after(SHUFFLE)
    b = history_board(target.fen(), {"fen": chess.STARTING_FEN, "moves": SHUFFLE})
    assert b.fen() == target.fen() and len(b.move_stack) == 6 and b.is_repetition(2)
    g = build_graph(b)
    rep = g.x[g.node_kind == KIND_GLOBAL][0, COL["repetitions"]]
    assert float(rep) == 0.5  # one earlier occurrence (Nf3 Nf6 with White to move)
    assert float(build_graph(chess.Board(target.fen())).x[-1, COL["repetitions"]]) == 0.0  # a bare FEN: none
    for bad in ({"fen": chess.STARTING_FEN, "moves": ["e2e5"]},                  # illegal
                {"fen": chess.STARTING_FEN, "moves": SHUFFLE[:5]},               # leads elsewhere
                {"moves": SHUFFLE}, "e2e4"):
        with pytest.raises(ValueError):
            history_board(target.fen(), bad)


def tiny(history=False, seed=0):
    torch.manual_seed(seed)
    return RelationalGNNEvaluator(hidden_dim=16, layers=2, num_bases=2, history=history, device=CPU).eval()


def test_checkpoint_flag(tmp_path):
    plain = tiny()
    assert "history" not in plain.config and not is_history_aware(plain)  # checkpoints as before
    aware = tiny(history=True)
    assert aware.config["history"] is True and is_history_aware(aware)
    for model, name in ((plain, "plain.pth"), (aware, "aware.pth")):
        save_checkpoint_v3(model, str(tmp_path / name))
        loaded = load_checkpoint_v3(str(tmp_path / name), device=CPU)
        assert loaded.config == model.config
    payload = torch.load(tmp_path / "plain.pth", weights_only=True)
    assert set(payload["config"]) == {"in_dim", "hidden_dim", "layers", "num_bases", "ffn_mult", "dropout",
                                      "value_dropout", "aux_score", "num_relations", "policy_encoding", "graph",
                                      "policy_dim"}


def request(fen, history=None):
    payload = {"fens": [fen], "moves": [[m.uci() for m in chess.Board(fen).legal_moves]]}
    if history is not None:
        payload["history"] = [history]
    return payload


@pytest.mark.parametrize("fast", [False, True])
def test_server_ignores_history_for_networks_without_it(fast):
    model = tiny()
    service = rpc_server.EvaluationService(GNNAdapterV3(model, RelationalChessGraph(device=CPU)), mc_samples=1,
                                           backend=TorchBackendV3(model) if fast else None)
    fen = board_after(SHUFFLE).fen()
    plain = service.handle(request(fen))
    assert "history" not in plain
    assert json.dumps(service.handle(request(fen, {"fen": chess.STARTING_FEN, "moves": SHUFFLE}))) == json.dumps(plain)
    # Not even a broken history is looked at.
    assert json.dumps(service.handle(request(fen, {"fen": "garbage", "moves": []}))) == json.dumps(plain)


@pytest.mark.parametrize("fast", [False, True])
def test_history_aware_network(fast):
    model = tiny(history=True, seed=3)
    service = rpc_server.EvaluationService(GNNAdapterV3(model, RelationalChessGraph(device=CPU)), mc_samples=1,
                                           backend=TorchBackendV3(model) if fast else None, cache_size=16)
    fen = board_after(SHUFFLE).fen()
    bare = service.handle(request(fen))
    with_history = service.handle(request(fen, {"fen": chess.STARTING_FEN, "moves": SHUFFLE}))
    assert bare["history"] is True and with_history["history"] is True
    # The repetition the history reveals is a network input.
    assert bare["results"][0]["value"] != with_history["results"][0]["value"]
    # The cache keeps the two apart.
    assert service.handle(request(fen))["results"] == bare["results"]
    with pytest.raises(ValueError):
        service.handle(request(fen, {"fen": chess.STARTING_FEN, "moves": SHUFFLE[:4]}))
    with pytest.raises(ValueError):
        service.handle({"fens": [fen], "history": []})


def test_distillation_rows_with_history():
    import distillation
    import training_pipeline

    target = board_after(SHUFFLE)
    legal = [m.uci() for m in target.legal_moves]
    row = {"fen": target.fen(), "legal": legal, "scored": [{"move": legal[0], "cp": 10}, {"move": legal[1], "cp": 0}],
           "history": {"fen": chess.STARTING_FEN, "moves": SHUFFLE}}
    params = dict(training_pipeline.DISTILL_DEFAULTS)
    assert params["use_history"] is False
    for fast in (False, True):
        without = distillation.DistillDataset([row], {**params, "fast_graphs": fast})[0]
        with_ = distillation.DistillDataset([row], {**params, "fast_graphs": fast, "use_history": True})[0]
        glob = lambda d: d.x[d.node_kind == KIND_GLOBAL][0, COL["repetitions"]]  # noqa: E731
        assert float(glob(without)) == 0.0 and float(glob(with_)) == 0.5
        assert torch.equal(without.legal, with_.legal) and torch.equal(without.target_policy, with_.target_policy)


def test_new_model_carries_the_flag(tmp_path):
    import training_pipeline

    cfg = json.loads(open(training_pipeline.ROOT / "training_config.json").read())
    cfg.update(data_dir=str(tmp_path / "d"), model_dir=str(tmp_path / "m"), logs_dir=str(tmp_path / "l"))
    cfg["distill_params"] = {"use_history": True, "model": {"hidden_dim": 16, "layers": 1, "num_bases": 2}}
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    pipeline = training_pipeline.TrainingPipeline(str(tmp_path / "c.json"))
    assert is_history_aware(pipeline.new_v3_model())
    cfg["distill_params"]["use_history"] = False
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    assert not is_history_aware(training_pipeline.TrainingPipeline(str(tmp_path / "c.json")).new_v3_model())


def _engine():
    from pathlib import Path
    root = Path(__file__).resolve().parents[2]
    for c in (root / "build" / "nags", root / "build" / "Release" / "nags.exe"):
        if c.exists():
            return c
    return None


@pytest.mark.skipif(_engine() is None, reason="nags not built")
@pytest.mark.parametrize("announce", [True, False])
def test_engine_sends_histories_when_asked(announce):
    import socketserver
    import subprocess
    import threading

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
                if announce:
                    reply["history"] = True
                self.wfile.write((json.dumps(reply) + "\n").encode())
                self.wfile.flush()

    server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Gnn)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        moves = " ".join(SHUFFLE[:5])  # Black to move; Nf6-g8 would repeat the start position
        cmds = (f"setoption name UseNN value true\nsetoption name NNPort value {server.server_address[1]}\n"
                "setoption name UseMetaLearner value false\nsetoption name MctsMinTime value 0\n"
                f"setoption name NNBatch value 4\nposition startpos moves {moves}\ngo movetime 500\n"
                f"position startpos moves {moves} e7e5\ngo movetime 300\n")
        out = subprocess.run([str(_engine())], input=cmds, capture_output=True, text=True, timeout=60).stdout
    finally:
        server.shutdown()
        server.server_close()
    assert out.count("bestmove") == 2
    with_history = [(f, h) for req in seen for f, h in zip(req["fens"], req.get("history") or [])]
    if not announce:
        assert all("history" not in req for req in seen)
        return
    # The first request comes before the engine knows; after the first reply, every one has histories.
    assert "history" not in seen[0] and all("history" in req for req in seen[1:]) and len(seen) > 3
    checked = 0
    for fen, h in with_history:
        if h is None:
            continue  # right after a capture or pawn move there is nothing to send
        b = history_board(fen, h)  # raises if the history does not lead to the position
        assert b.fen().split()[:5] == chess.Board(fen).fen().split()[:5]
        checked += 1
    assert checked > 10
    # The second search's root: the game since the last pawn move (e7e5 resets it, so nothing before it).
    roots = [(f, h) for f, h in with_history if f.startswith("rnbqkbnr/pppp1ppp/8/4p3/8/5N2")]
    assert roots and all(h is None or h["moves"] == [] or h["fen"].split()[4] == "0" for _, h in roots)
