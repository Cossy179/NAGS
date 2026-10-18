"""relational_fast.build_arrays / collate give exactly what relational_graph.build_graph and
torch_geometric's Batch.from_data_list give (the graphs every v3 checkpoint was trained on)."""

import random

import chess
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch

from gnn_v3 import relation_inv_degree
from policy_encoding import legal_mask_v3
from relational_fast import build_arrays, build_batch, collate
from relational_graph import KIND_GLOBAL, KIND_SQUARE, NUM_RELATIONS, RelationalChessGraph, build_graph

SPECIAL = [
    chess.STARTING_FEN,
    "1n2k1n1/PPP3PP/8/8/8/8/ppp3pp/1N2K1N1 w - - 0 1",        # promotions both ways
    "4k3/1P6/8/8/8/8/6p1/4K3 b - - 0 1",
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # legal en passant
    "8/8/8/KPp4r/8/8/8/6k1 w - c6 0 1",                         # en passant square, capture illegal (pin)
    "4k3/8/8/2KPp2r/8/8/8/8 w - e6 0 1",                        # en passant along a pinned rank
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                     # all castling
    "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 5 20",
    "4k3/8/8/8/8/8/8/4K2R w KQkq - 0 1",                         # rights without the rooks (cleaned)
    "4k3/4r3/8/8/4B3/8/4N3/4K3 w - - 0 1",                       # pinned knight on the file
    "7k/8/8/3q4/8/1B6/8/K7 w - - 0 1",                           # diagonal, nothing pinned
    "k7/1q6/8/8/4B3/8/8/7K w - - 0 1",                           # pinned bishop on the diagonal
    "4k3/8/8/8/1b6/8/3P4/4K3 w - - 0 1",                         # pinned pawn
    "4k3/8/8/8/8/8/r2NK3/8 w - - 0 1",                           # rank pin
    "3qk3/8/8/8/8/8/3R4/3QK3 b - - 0 1",                         # x-ray, own queen behind
    "4k3/8/8/1b6/8/3N4/4K3/8 w - - 0 1",
    "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1",                         # back rank: not in check
    "6k1/5ppp/8/8/8/8/8/6KR w - - 99 80",                        # half-move clock near the limit
    "8/8/8/8/8/5k2/6q1/7K w - - 0 1",                            # checkmated: no legal moves
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",                            # stalemate
    "4k3/8/8/8/8/8/8/4K3 w - - 120 200",                         # bare kings, clock above 100
    "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",  # mate
    "4k3/8/8/8/8/2n5/8/R3K2r w Q - 0 1",                         # double check
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "rnbqkb1r/pp1p1ppp/5n2/2pPp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 4",
]


def random_positions(n, seed, start=chess.STARTING_FEN, tactical=False, keep_history=False):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board(start)
        for _ in range(rng.randint(0, 120)):
            moves = list(board.legal_moves)
            if not moves:
                break
            if tactical and rng.random() < 0.6:
                sharp = [m for m in moves if board.is_capture(m) or m.promotion or board.gives_check(m)]
                if sharp:
                    moves = sharp
            board.push(rng.choice(moves))
        out.append(board if keep_history else chess.Board(board.fen()))
    return out


def same_graph(position):
    ref = build_graph(position if isinstance(position, chess.Board) else chess.Board(position))
    got = build_arrays(position)
    assert got.x.dtype == np.float32 and got.x.shape == tuple(ref.x.shape)
    assert torch.equal(torch.from_numpy(got.x), ref.x), np.argwhere(got.x != ref.x.numpy())[:5]
    assert torch.equal(torch.from_numpy(got.src), ref.edge_index[0])
    assert torch.equal(torch.from_numpy(got.dst), ref.edge_index[1])
    assert torch.equal(torch.from_numpy(got.edge_type), ref.edge_type)
    assert torch.equal(torch.from_numpy(got.node_kind), ref.node_kind)
    board = position if isinstance(position, chess.Board) else chess.Board(position)
    mask = np.zeros(4228, dtype=bool)
    mask[got.legal] = True
    assert np.array_equal(mask, legal_mask_v3(board).numpy())
    return got


@pytest.mark.parametrize("fen", SPECIAL)
def test_special_positions(fen):
    same_graph(fen)


@pytest.mark.parametrize("seed", range(4))
def test_random_positions(seed):
    boards = (random_positions(400, seed) + random_positions(300, 100 + seed, tactical=True) +
              random_positions(150, 200 + seed, start=SPECIAL[1], tactical=True) +
              random_positions(150, 300 + seed, start=SPECIAL[6], tactical=True))
    for b in boards:
        same_graph(b)


def test_boards_with_history():
    # The repetitions feature reads the board's move stack; pieces shuffling back and forth repeat positions.
    boards = random_positions(150, 7, keep_history=True)
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8", "e2e4"]:
        board.push_uci(uci)
        boards.append(board.copy())
    assert any(b.is_repetition(2) for b in boards) and any(b.is_repetition(3) for b in boards)
    for b in boards:
        same_graph(b)


def test_errors_match_the_reference():
    builder = RelationalChessGraph(device=torch.device("cpu"))
    for bad in ["garbage", "8/8/8/8/8/8/8/8 w - - 0 1", "4k3/8/8/8/8/8/8/8 w - - 0 1"]:
        with pytest.raises(ValueError):
            build_arrays(bad)
        with pytest.raises(ValueError):
            builder.fen_to_graph(bad)
    with pytest.raises(ValueError):
        build_arrays(chess.Board("4k3/8/8/8/8/8/8/8 w - - 0 1"))


def test_collate_matches_batch_from_data_list():
    boards = random_positions(40, 11, tactical=True) + [chess.Board(f) for f in SPECIAL]
    ref = Batch.from_data_list([build_graph(b) for b in boards])
    got = collate([build_arrays(b) for b in boards])
    assert torch.equal(torch.from_numpy(got.x), ref.x)
    assert torch.equal(torch.from_numpy(np.stack([got.src, got.dst])), ref.edge_index)
    assert torch.equal(torch.from_numpy(got.edge_type), ref.edge_type)
    assert torch.equal(torch.from_numpy(got.node_kind), ref.node_kind)
    assert torch.equal(torch.from_numpy(got.batch), ref.batch)
    inv = relation_inv_degree(ref.edge_index[1], ref.edge_type, ref.x.size(0), NUM_RELATIONS)
    assert torch.equal(torch.from_numpy(got.inv_deg), inv)
    kind = ref.node_kind
    assert torch.equal(torch.from_numpy(got.square_idx), torch.nonzero(kind == KIND_SQUARE).squeeze(1))
    assert torch.equal(torch.from_numpy(got.global_idx), torch.nonzero(kind == KIND_GLOBAL).squeeze(1))
    assert np.array_equal(got.node_count, np.bincount(got.batch).astype(np.float32))
    assert np.array_equal(got.legal, torch.stack([legal_mask_v3(b) for b in boards]).numpy())
    one = build_batch([boards[0]])
    assert torch.equal(torch.from_numpy(one.x), build_graph(boards[0]).x)
    with pytest.raises(ValueError):
        collate([])
