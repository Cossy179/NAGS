import pytest
import torch

from chess_graph import F_EN_PASSANT, F_SIDE_TO_MOVE, FEATURE_DIM, ChessGNN, ChessGraph, parse_fen

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


def test_graph_dims_and_forward():
    builder = ChessGraph(device=torch.device("cpu"))
    data = builder.fen_to_graph(START)
    # 64 squares + 32 pieces + side-to-move + 4 castling + 8 pawn files
    assert data.x.shape == (64 + 32 + 1 + 4 + 8, FEATURE_DIM)
    model = ChessGNN(in_dim=FEATURE_DIM, hidden_dim=32, num_layers=2)
    global_emb, node_emb = model(data)
    assert global_emb.shape == (1, 32)
    assert node_emb.shape == (data.num_nodes, 32)


def test_side_to_move_and_en_passant_are_encoded():
    builder = ChessGraph(device=torch.device("cpu"))
    white = builder.fen_to_graph("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
    black = builder.fen_to_graph("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2")
    assert white.x[:, F_SIDE_TO_MOVE].sum() == 1.0
    assert black.x[:, F_SIDE_TO_MOVE].sum() == 0.0
    e6 = 5 * 8 + 4
    assert white.x[e6, F_EN_PASSANT] == 1.0
    assert black.x[:, F_EN_PASSANT].sum() == 0.0


def test_batching_keeps_graphs_separate():
    builder = ChessGraph(device=torch.device("cpu"))
    batch = builder.fens_to_batch([START, "8/8/8/K2pP2q/8/8/8/7k w - d6 0 1"])
    assert batch.num_graphs == 2
    assert int((batch.batch == 1).sum()) == 64 + 5 + 13


@pytest.mark.parametrize("fen", ["", "8/8/8 w - - 0 1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX w KQkq - 0 1",
                                 "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1"])
def test_invalid_fen_rejected(fen):
    with pytest.raises(ValueError):
        parse_fen(fen)


def test_device_matches_request():
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data = ChessGraph(device=dev).fen_to_graph(START)
    model = ChessGNN(in_dim=FEATURE_DIM, hidden_dim=16, num_layers=1).to(dev)
    g, n = model(data)
    # Compare device types: torch.device('cuda') != torch.device('cuda:0').
    assert g.device.type == dev.type and n.device.type == dev.type
