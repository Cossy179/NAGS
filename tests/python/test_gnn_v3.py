"""v3 relational GNN evaluator, its checkpoints and the format-dispatching loader (gnn_v3.py)."""

import json
from pathlib import Path

import chess
import numpy as np
import pytest
import torch

from chess_graph import ChessGraph
from gnn_evaluator import legal_move_priors, load_checkpoint, save_checkpoint, GNNEvaluator
from gnn_v3 import (CHECKPOINT_FORMAT_V3, GNNAdapterV2, GNNAdapterV3, RelationalGNNEvaluator, as_adapter,
                    legal_move_priors_v3, load_any_checkpoint, load_checkpoint_v3, save_checkpoint_v3)
from policy_encoding import POLICY_V3_DIM, legal_mask_v3, move_index_v3
from relational_graph import RelationalChessGraph

CPU = torch.device("cpu")
FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "data" / "compat"
PROMO = "1n2k1n1/PPP3PP/8/8/8/8/ppp3pp/1N2K1N1 w - - 0 1"
FENS = [chess.STARTING_FEN, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", PROMO,
        "4k3/1P6/8/8/8/8/6p1/4K3 b - - 0 1"]


def tiny(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = dict(hidden_dim=16, layers=2, num_bases=2, ffn_mult=2, value_dropout=0.2)
    cfg.update(kw)
    return RelationalGNNEvaluator(device=CPU, **cfg)


@pytest.fixture(scope="module")
def model():
    return tiny()


@pytest.fixture(scope="module")
def builder():
    return RelationalChessGraph(device=CPU)


def test_forward_shapes(model, builder):
    policy, wdl, score = model(builder.fens_to_batch(FENS))
    assert policy.shape == (4, POLICY_V3_DIM) and wdl.shape == (4, 3) and score.shape == (4,)
    assert ((score >= -1) & (score <= 1)).all()
    assert tiny(aux_score=False)(builder.fen_to_graph(FENS[0]))[2] is None


def test_evaluate_probabilities_and_mode(model, builder):
    model.train()
    probs, mean, std = model.evaluate(builder.fens_to_batch(FENS), mc_samples=4)
    assert model.training, "evaluate() must restore the previous mode"
    assert probs.shape == (4, POLICY_V3_DIM)
    assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-5)
    assert ((mean >= -1) & (mean <= 1)).all() and (std >= 0).all() and (std > 0).any()
    model.eval()
    _, v1, s1 = model.evaluate(builder.fens_to_batch(FENS), mc_samples=1)
    assert torch.equal(s1, torch.zeros(4))
    wdl = model.evaluate_wdl(builder.fens_to_batch(FENS))
    assert torch.allclose(wdl.sum(dim=1), torch.ones(4), atol=1e-6)
    assert torch.allclose(v1, wdl[:, 0] - wdl[:, 2], atol=1e-6)


def test_batched_matches_single(model, builder):
    model.eval()
    with torch.no_grad():
        policy, wdl, score = model(builder.fens_to_batch(FENS))
        for i, fen in enumerate(FENS):
            p1, w1, s1 = model(builder.fen_to_graph(fen))
            assert torch.allclose(policy[i], p1[0], atol=1e-5)
            assert torch.allclose(wdl[i], w1[0], atol=1e-5)
            assert torch.allclose(score[i], s1[0], atol=1e-5)
    probs, values, _ = model.evaluate(builder.fens_to_batch(FENS), mc_samples=1)
    for i, fen in enumerate(FENS):
        p1, v1, _ = model.evaluate(builder.fen_to_graph(fen), mc_samples=1)
        assert torch.allclose(probs[i], p1[0], atol=1e-6) and torch.allclose(values[i], v1[0], atol=1e-6)


def test_legal_priors_distinguish_underpromotions(model, builder):
    board = chess.Board(PROMO)
    moves = [m.uci() for m in board.legal_moves]
    probs, _, _ = model.evaluate(builder.fen_to_graph(PROMO), mc_samples=1)
    priors = legal_move_priors_v3(probs[0], moves)
    assert len(priors) == len(moves) and abs(sum(priors) - 1.0) < 1e-5 and min(priors) > 0
    expected = probs[0][legal_mask_v3(board)].sum()
    for m, p in zip(moves, priors):
        assert p == pytest.approx(float(probs[0][move_index_v3(m)] / expected), rel=1e-5)
    by_move = dict(zip(moves, priors))
    assert len({by_move["a7a8" + p] for p in "qrbn"}) == 4  # four different priors, not one shared entry
    assert legal_move_priors_v3(probs[0], []) == []
    assert legal_move_priors_v3(torch.zeros(POLICY_V3_DIM), ["e2e4", "d2d4"]) == [0.5, 0.5]


def test_checkpoint_round_trip(model, builder, tmp_path):
    path = tmp_path / "v3.pth"
    save_checkpoint_v3(model, str(path), {"stage": "test", "metrics": {"top1": 0.5}})
    payload = torch.load(path, weights_only=True)
    assert payload["format"] == CHECKPOINT_FORMAT_V3 == "nags-gnn-v3"
    assert payload["config"]["policy_encoding"] == "v3" and payload["config"]["graph"] == "relational-v1"
    assert not any(k.startswith("promo_") for k in payload["state_dict"])  # encoding constants are not weights
    restored = load_checkpoint_v3(str(path), device=CPU)
    assert restored.config == model.config and not restored.training
    batch = builder.fens_to_batch(FENS)
    model.eval()
    with torch.no_grad():
        for a, b in zip(model(batch), restored(batch)):
            assert torch.equal(a, b)


def test_v3_loader_rejects_other_files(tmp_path):
    with pytest.raises(ValueError):
        load_checkpoint_v3(str(FIXTURES / "gnn_v2.pt"))
    raw = tmp_path / "raw.pth"
    torch.save({"weight": torch.zeros(1)}, raw)
    with pytest.raises(ValueError):
        load_checkpoint_v3(str(raw))
    m = tiny()
    for key, value in (("policy_encoding", "v4"), ("graph", "relational-v9"), ("in_dim", 7), ("new_option", 1)):
        path = tmp_path / f"bad_{key}.pth"
        save_checkpoint_v3(m, str(path))
        payload = torch.load(path, weights_only=True)
        payload["config"][key] = value
        torch.save(payload, path)
        with pytest.raises(ValueError):
            load_checkpoint_v3(str(path))


def test_legacy_loader_rejects_v3(tmp_path):
    path = tmp_path / "v3.pth"
    save_checkpoint_v3(tiny(), str(path))
    with pytest.raises(ValueError):
        load_checkpoint(str(path))


def test_load_any_on_the_frozen_v2_fixture_gives_the_legacy_outputs():
    adapter = load_any_checkpoint(str(FIXTURES / "gnn_v2.pt"), device=CPU)
    assert isinstance(adapter, GNNAdapterV2) and isinstance(adapter.model, GNNEvaluator)
    assert isinstance(adapter.builder, ChessGraph)
    assert adapter.format == "nags-gnn-v2" and adapter.policy_dim == 4096
    legacy = load_checkpoint(str(FIXTURES / "gnn_v2.pt"), device=CPU)
    legacy_builder = ChessGraph(device=CPU)
    cases = json.loads((FIXTURES / "gnn_v2_expected.json").read_text())
    for case in cases:
        probs, value, unc = adapter.evaluate(adapter.build_batch([case["fen"]]), mc_samples=1)
        lp, lv, lu = legacy.evaluate(legacy_builder.fens_to_batch([case["fen"]]), mc_samples=1)
        assert torch.equal(probs, lp) and torch.equal(value, lv) and torch.equal(unc, lu)
        priors = adapter.move_priors(probs[0], case["moves"])
        assert priors == legal_move_priors(lp[0], case["moves"])
        assert np.allclose(priors, case["priors"], atol=1e-5) and abs(float(value[0]) - case["value"]) < 1e-5
    out = adapter.evaluate_fens([c["fen"] for c in cases], [c["moves"] for c in cases])
    assert all(np.allclose(o["move_priors"], c["priors"], atol=1e-5) for o, c in zip(out, cases))


def test_load_any_on_v3_and_unknown_files(tmp_path):
    path = tmp_path / "v3.pth"
    save_checkpoint_v3(tiny(), str(path))
    adapter = load_any_checkpoint(str(path), device=CPU)
    assert isinstance(adapter, GNNAdapterV3) and isinstance(adapter.builder, RelationalChessGraph)
    assert adapter.format == "nags-gnn-v3" and adapter.policy_dim == 4228
    v2 = tmp_path / "v2.pth"
    save_checkpoint(GNNEvaluator(hidden_dim=16, gnn_layers=1, policy_layers=1, value_layers=1, device=CPU), str(v2))
    assert load_any_checkpoint(str(v2), device=CPU).format == "nags-gnn-v2"
    raw = tmp_path / "raw.pth"
    torch.save({"weight": torch.zeros(1)}, raw)
    with pytest.raises(ValueError):
        load_any_checkpoint(str(raw))
    other = tmp_path / "other.pth"
    torch.save({"format": "nags-gnn-v9", "config": {}, "state_dict": {}}, other)
    with pytest.raises(ValueError, match="unknown checkpoint format"):
        load_any_checkpoint(str(other))


def test_as_adapter():
    assert isinstance(as_adapter(tiny()), GNNAdapterV3)
    legacy = GNNEvaluator(hidden_dim=16, gnn_layers=1, policy_layers=1, value_layers=1, device=CPU)
    assert isinstance(as_adapter(legacy), GNNAdapterV2)
    with pytest.raises(TypeError):
        as_adapter(torch.nn.Linear(1, 1))


def test_gradients_reach_every_head(builder):
    m = tiny(seed=1)
    m.train()
    board = chess.Board(PROMO)
    policy, wdl, score = m(builder.fen_to_graph(PROMO))
    target = move_index_v3("a7a8n")
    logp = torch.log_softmax(policy.masked_fill(~legal_mask_v3(board).unsqueeze(0), -1e9), dim=1)
    loss = -logp[0, target] - torch.log_softmax(wdl, dim=1)[0, 0] + (score[0] - 0.5) ** 2
    loss.backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
