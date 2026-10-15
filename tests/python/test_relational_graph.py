"""Typed relational graph for the v3 GNN (relational_graph.py)."""

import random

import chess
import pytest
import torch

from relational_graph import (COL, FEATURE_DIM_V3, KIND_GLOBAL, KIND_PIECE, KIND_SQUARE, NUM_RELATIONS, REL,
                              RELATIONS, RelationalChessGraph, build_graph)

CPU = torch.device("cpu")
# Bishop b5 pins the knight d3 to the king e2; rook a8 attacks the passed pawn a6;
# pawn d4 supports e5; pawns h2 / h7 share a file.
TACTICAL = "r3k3/7p/P7/1b2P3/3P4/3N4/4K2P/8 w - - 0 1"
CASTLE_EP = "r3k2r/8/8/8/3pP3/8/8/R3K2R b KQkq e3 0 1"


def piece_nodes(board):
    """{square: piece node id}: piece nodes follow the squares in ascending order."""
    return {sq: 64 + i for i, sq in enumerate(sorted(board.piece_map()))}


def edges(data):
    return {(int(a), int(b), int(t)) for a, b, t in zip(data.edge_index[0], data.edge_index[1], data.edge_type)}


def random_boards(n, seed):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        board = chess.Board()
        for _ in range(rng.randint(0, 80)):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if any(board.legal_moves):
            out.append(board)
    return out


def test_layout_and_types():
    data = RelationalChessGraph(device=CPU).fen_to_graph(chess.STARTING_FEN)
    n = 64 + 32 + 1
    assert data.num_nodes == n and data.x.shape == (n, FEATURE_DIM_V3)
    assert data.node_kind[:64].eq(KIND_SQUARE).all() and data.node_kind[64:96].eq(KIND_PIECE).all()
    assert int(data.node_kind[96]) == KIND_GLOBAL
    assert data.edge_type.dtype == torch.long and data.edge_index.shape == (2, data.edge_type.numel())
    assert int(data.edge_type.min()) >= 0 and int(data.edge_type.max()) < NUM_RELATIONS
    assert torch.equal(data.edge_index[1], data.edge_index[1].sort().values)  # sorted by destination
    assert len(RELATIONS) == NUM_RELATIONS >= 21 and FEATURE_DIM_V3 == len(COL)


def test_tactical_relations_and_features():
    board = chess.Board(TACTICAL)
    data = build_graph(board)
    e = edges(data)
    pn = piece_nodes(board)
    D3, B5, A8, A6, D4, E5, H2, H7, E2 = (chess.D3, chess.B5, chess.A8, chess.A6, chess.D4, chess.E5, chess.H2,
                                          chess.H7, chess.E2)
    assert (pn[D3], pn[B5], REL["pinned_to"]) in e and (pn[B5], pn[D3], REL["pins"]) in e
    assert (pn[A8], pn[A6], REL["attacks"]) in e and (pn[A6], pn[A8], REL["attacked_by"]) in e
    assert (pn[D3], pn[E5], REL["defends"]) in e and (pn[E5], pn[D3], REL["defended_by"]) in e
    assert (pn[D3], chess.B4, REL["controls"]) in e and (chess.B4, pn[D3], REL["controlled_by"]) in e
    assert (pn[D4], pn[E5], REL["pawn_supports"]) in e and (pn[E5], pn[D4], REL["pawn_supported_by"]) in e
    assert (pn[H2], pn[H7], REL["pawn_same_file"]) in e and (pn[H7], pn[H2], REL["pawn_same_file"]) in e
    assert (pn[A6], chess.A7, REL["passed_path"]) in e and (pn[A6], chess.A8, REL["passed_path"]) in e
    assert (chess.A7, pn[A6], REL["passed_path_of"]) in e
    assert not any(t == REL["passed_path"] and a == pn[H2] for a, _, t in e)  # h2 is not passed
    zone = {b for a, b, t in e if a == pn[E2] and t == REL["king_zone"]}
    assert zone == set(chess.SquareSet(chess.BB_KING_ATTACKS[E2]))
    assert not any(a == pn[D3] and t == REL["legal_move"] for a, _, t in e)  # pinned knight has no legal move
    assert (pn[E2], chess.E3, REL["legal_move"]) in e and (chess.E3, pn[E2], REL["legal_move_from"]) in e
    g = data.num_nodes - 1
    assert (g, pn[B5], REL["global_to_piece"]) in e and (pn[B5], g, REL["piece_to_global"]) in e
    for sq, piece in board.piece_map().items():
        assert (pn[sq], sq, REL["piece_on"]) in e and (sq, pn[sq], REL["occupied_by"]) in e

    x = data.x
    assert x[pn[D3], COL["pinned"]] == 1 and x[pn[B5], COL["pinned"]] == 0
    assert [float(x[pn[s], COL["passed"]]) for s in (A6, D4, E5, H2, H7)] == [1, 1, 1, 0, 0]
    assert x[pn[D3], COL["own"]] == 1 and x[pn[B5], COL["opponent"]] == 1
    assert x[E5, COL["attackers_own"]] == pytest.approx(2 / 8)  # knight d3 and pawn d4
    assert x[pn[E5], COL["attackers_own"]] == pytest.approx(2 / 8)
    assert x[E5, COL["type_pawn"]] == 1 and x[E5, COL["own"]] == 1
    assert x[g, COL["legal_move_count"]] == pytest.approx(board.legal_moves.count() / 64)
    assert x[g, COL["white_to_move"]] == 1 and x[g, COL["in_check"]] == 0
    assert x[pn[D3], COL["legal_moves_from"]] == 0 and x[pn[E2], COL["legal_moves_from"]] > 0


def test_castling_en_passant_and_check():
    board = chess.Board(CASTLE_EP)
    data = build_graph(board)
    e, pn, x = edges(data), piece_nodes(board), data.x
    g = data.num_nodes - 1
    assert (pn[chess.D4], chess.E3, REL["legal_move"]) in e          # en passant
    assert (pn[chess.E8], chess.G8, REL["legal_move"]) in e          # castling, UCI king target
    assert (pn[chess.E8], chess.C8, REL["legal_move"]) in e
    assert x[chess.E3, COL["en_passant"]] == 1 and x[:64, COL["en_passant"]].sum() == 1
    for sq in (chess.A8, chess.H8, chess.E8, chess.A1, chess.H1, chess.E1):
        assert x[pn[sq], COL["castling_piece"]] == 1
    assert [float(x[g, COL[c]]) for c in ("castle_own_k", "castle_own_q", "castle_opp_k", "castle_opp_q")] == [1] * 4
    assert x[g, COL["white_to_move"]] == 0
    # Without a legal en-passant capture the square is not marked.
    assert build_graph(chess.Board("4k3/8/8/8/4P3/8/8/4K3 b - e3 0 1")).x[:64, COL["en_passant"]].sum() == 0
    check = chess.Board("4k3/8/8/8/8/8/4q3/4K3 w - - 7 40")
    xc = build_graph(check).x
    gc = xc.size(0) - 1
    assert xc[gc, COL["in_check"]] == 1 and xc[64 + sorted(check.piece_map()).index(chess.E1), COL["in_check"]] == 1
    assert xc[gc, COL["halfmove_clock"]] == pytest.approx(0.07)


def test_features_are_relative_to_the_side_to_move():
    skip = {COL["white_to_move"]}
    keep = [c for c in range(FEATURE_DIM_V3) if c not in skip]
    for board in random_boards(40, 3) + [chess.Board(TACTICAL), chess.Board(CASTLE_EP)]:
        mirror = board.mirror()
        a, b = build_graph(board), build_graph(mirror)
        for sq in range(64):
            assert torch.equal(a.x[sq, keep], b.x[sq ^ 56, keep]), (board.fen(), sq)
        pa, pb = piece_nodes(board), piece_nodes(mirror)
        for sq in pa:
            assert torch.equal(a.x[pa[sq], keep], b.x[pb[sq ^ 56], keep])
        assert torch.equal(a.x[-1, keep], b.x[-1, keep])
        assert torch.equal(torch.bincount(a.edge_type, minlength=NUM_RELATIONS),
                           torch.bincount(b.edge_type, minlength=NUM_RELATIONS))


def test_attack_counts_match_python_chess():
    for board in random_boards(30, 5):
        x = build_graph(board).x
        for sq in range(64):
            assert x[sq, COL["attackers_own"]] * 8 == len(board.attackers(board.turn, sq))
            assert x[sq, COL["attackers_opp"]] * 8 == len(board.attackers(not board.turn, sq))


def test_repetitions_from_history():
    board = chess.Board()
    rep = lambda b: float(build_graph(b).x[-1, COL["repetitions"]])  # noqa: E731
    assert rep(board) == 0
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    assert rep(board) == 0.5
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    assert rep(board) == 1.0
    assert rep(chess.Board(board.fen())) == 0  # a FEN has no history


def test_every_relation_occurs():
    seen = torch.zeros(NUM_RELATIONS, dtype=torch.long)
    for board in random_boards(20, 7) + [chess.Board(TACTICAL), chess.Board(CASTLE_EP)]:
        seen += torch.bincount(build_graph(board).edge_type, minlength=NUM_RELATIONS)
    assert (seen > 0).all(), [RELATIONS[i] for i in range(NUM_RELATIONS) if seen[i] == 0]


def test_batch_keeps_graphs_separate():
    builder = RelationalChessGraph(device=CPU)
    fens = [chess.STARTING_FEN, TACTICAL, CASTLE_EP]
    batch = builder.fens_to_batch(fens)
    singles = [builder.fen_to_graph(f) for f in fens]
    assert batch.num_graphs == 3
    assert torch.equal(batch.x, torch.cat([s.x for s in singles]))
    assert torch.equal(batch.edge_type, torch.cat([s.edge_type for s in singles]))
    offset = 0
    parts = []
    for s in singles:
        parts.append(s.edge_index + offset)
        offset += s.num_nodes
    assert torch.equal(batch.edge_index, torch.cat(parts, dim=1))
    assert int((batch.node_kind == KIND_GLOBAL).sum()) == 3


@pytest.mark.parametrize("fen", ["", "8/8/8 w - - 0 1", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX w KQkq - 0 1",
                                 "8/8/8/8/8/8/8/4K3 w - - 0 1"])
def test_invalid_positions_rejected(fen):
    with pytest.raises(ValueError):
        RelationalChessGraph(device=CPU).fen_to_graph(fen)
