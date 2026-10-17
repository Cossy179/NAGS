"""v3 promotion-aware move encoding (policy_encoding.py)."""

import random

import chess
import pytest
import torch

from gnn_evaluator import move_index
from policy_encoding import (FROM_TO_DIM, NUM_PROMOTION_PAIRS, POLICY_V3_DIM, PROMOTION_PAIRS, index_to_uci_v3,
                             legal_indices_v3, legal_mask_v3, move_index_v3, move_to_index_v3)

# Start positions with many promotions (and underpromotion captures) available.
PROMOTION_STARTS = [
    "1n2k1n1/PPP3PP/8/8/8/8/ppp3pp/1N2K1N1 w - - 0 1",
    "r1b1k1nr/1P1P1P1P/8/8/8/8/1p1p1p1p/R1B1K1NR b - - 0 1",
    "8/P6k/8/8/8/8/6Kp/8 w - - 0 1",
]


def random_positions(n_games, seed, start=chess.STARTING_FEN, max_plies=120):
    rng = random.Random(seed)
    for _ in range(n_games):
        board = chess.Board(start)
        for _ in range(rng.randint(1, max_plies)):
            moves = list(board.legal_moves)
            if not moves:
                break
            yield board
            board.push(rng.choice(moves))


def all_positions():
    yield from random_positions(30, 1)
    for i, fen in enumerate(PROMOTION_STARTS):
        yield from random_positions(25, 100 + i, start=fen, max_plies=40)


def test_dimensions_and_pair_order():
    assert POLICY_V3_DIM == 4228 == FROM_TO_DIM + 3 * NUM_PROMOTION_PAIRS
    assert NUM_PROMOTION_PAIRS == 44 and len(set(PROMOTION_PAIRS)) == 44
    names = [chess.square_name(a) + chess.square_name(b) for a, b in PROMOTION_PAIRS]
    assert names[:5] == ["a7a8", "a7b8", "b7a8", "b7b8", "b7c8"]
    assert names[21] == "h7h8" and names[22] == "a2a1" and names[43] == "h2h1"


def test_known_indices():
    assert move_index_v3("e2e4") == move_index("e2e4") == 12 * 64 + 28
    assert move_index_v3("e1g1") == move_index("e1g1")          # castling: the king's from/to
    assert move_index_v3("e7e8q") == move_index_v3("e7e8") == move_index("e7e8q")
    pair = PROMOTION_PAIRS.index((chess.E7, chess.E8))
    assert [move_index_v3("e7e8" + p) for p in "nbr"] == [4096 + pair * 3 + i for i in range(3)]
    assert move_index_v3("g2h1r") == 4096 + PROMOTION_PAIRS.index((chess.G2, chess.H1)) * 3 + 2
    assert move_index_v3(chess.Move.from_uci("b7a8n")) == move_index_v3("b7a8n")


@pytest.mark.parametrize("bad", ["", "e2", "z9a1", "e2e4n", "e7e8k", "e7e8x", "a7c8n", "e6e7q", "e7d6q", "0000", "e7e8qq"])
def test_rejects_invalid_moves(bad):
    with pytest.raises(ValueError):
        move_index_v3(bad)


def test_index_to_uci_without_board():
    assert index_to_uci_v3(move_index_v3("e7e8n")) == "e7e8n"
    assert index_to_uci_v3(move_index_v3("e2e4")) == "e2e4"
    assert index_to_uci_v3(move_index_v3("e7e8q")) == "e7e8"  # a rook could make that move too: needs a board
    with pytest.raises(ValueError):
        index_to_uci_v3(POLICY_V3_DIM)


def test_every_legal_move_is_unique_and_round_trips():
    positions = promotions = underpromotions = 0
    for board in all_positions():
        legal = list(board.legal_moves)
        indices = legal_indices_v3(board)
        assert len(set(indices)) == len(legal), board.fen()
        assert all(0 <= i < POLICY_V3_DIM for i in indices)
        mask = legal_mask_v3(board)
        assert mask.dtype == torch.bool and mask.shape == (POLICY_V3_DIM,)
        assert int(mask.sum()) == len(legal)
        assert mask[torch.tensor(indices)].all()
        for move, idx in zip(legal, indices):
            assert move_index_v3(move.uci()) == idx == move_to_index_v3(move)
            assert index_to_uci_v3(idx, board) == move.uci()
            promotions += move.promotion is not None
            underpromotions += move.promotion in (chess.KNIGHT, chess.BISHOP, chess.ROOK)
        positions += 1
    assert positions >= 2000 and underpromotions >= 300 and promotions > underpromotions


def test_legacy_encoding_still_collapses_promotions():
    board = chess.Board("8/P6k/8/8/8/8/6K1/8 w - - 0 1")
    legacy = {move_index(m.uci()) for m in board.legal_moves if m.from_square == chess.A7}
    v3 = {move_index_v3(m.uci()) for m in board.legal_moves if m.from_square == chess.A7}
    assert len(legacy) == 1 and len(v3) == 4
