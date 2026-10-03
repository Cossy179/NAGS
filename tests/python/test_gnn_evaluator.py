import chess
import pytest
import torch

from chess_graph import ChessGraph
from gnn_evaluator import GNNEvaluator, legal_move_priors, load_checkpoint, move_index, save_checkpoint

FENS = [chess.STARTING_FEN, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"]


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return GNNEvaluator(hidden_dim=32, gnn_layers=2, policy_layers=1, value_layers=1, device=torch.device("cpu"))


def test_move_index():
    assert move_index("a1a2") == 8
    assert move_index("e2e4") == 12 * 64 + 28
    assert move_index("e7e8q") == move_index("e7e8n")
    with pytest.raises(ValueError):
        move_index("z9a1")


def test_batched_matches_single(model):
    builder = ChessGraph(device=torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        logits, values = model(builder.fens_to_batch(FENS))
        for i, fen in enumerate(FENS):
            l1, v1 = model(builder.fen_to_graph(fen))
            assert torch.allclose(logits[i], l1[0], atol=1e-5)
            assert torch.allclose(values[i], v1[0], atol=1e-5)


def test_evaluate_shapes_and_mode(model):
    builder = ChessGraph(device=torch.device("cpu"))
    model.train()
    probs, mean, std = model.evaluate(builder.fens_to_batch(FENS), mc_samples=4)
    assert model.training, "evaluate() must restore the previous mode"
    assert probs.shape == (2, 4096)
    assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1e-4)
    assert ((mean >= -1) & (mean <= 1)).all() and (std >= 0).all()


def test_legal_move_priors(model):
    builder = ChessGraph(device=torch.device("cpu"))
    probs, _, _ = model.evaluate(builder.fen_to_graph(FENS[0]), mc_samples=1)
    moves = [m.uci() for m in chess.Board().legal_moves]
    priors = legal_move_priors(probs[0], moves)
    assert len(priors) == 20 and abs(sum(priors) - 1.0) < 1e-5


def test_checkpoint_round_trip(model, tmp_path):
    path = tmp_path / "m.pth"
    save_checkpoint(model, str(path), {"note": "test"})
    restored = load_checkpoint(str(path), device=torch.device("cpu"))
    assert restored.config == model.config
    batch = ChessGraph(device=torch.device("cpu")).fens_to_batch(FENS)
    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(batch)[0], restored(batch)[0], atol=1e-6)


def test_rejects_foreign_checkpoint(tmp_path):
    path = tmp_path / "raw.pth"
    torch.save({"weight": torch.zeros(1)}, path)
    with pytest.raises(ValueError):
        load_checkpoint(str(path))
