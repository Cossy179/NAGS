"""Fast construction of 'relational-v1' graphs (relational_graph.py) as numpy arrays.

relational_graph.build_graph is the reference and stays as it is: every
v3 checkpoint was trained on its output. build_arrays() computes the same
graph with less Python per edge and feature (edges collected in flat
lists, features filled column by column, pins found from the snipers on
the king's lines), and returns plain arrays instead of a torch_geometric
Data object, which is what the inference backends (gnn_infer.py) need.

The output is identical to the reference, not merely close: the same
float32 features, the same edges in the same order (insertion order, then
a stable sort by destination, as build_graph does), so a model computes
bit-identical results from either. tests/python/test_relational_fast.py
compares them on thousands of positions.

collate() joins graphs into one batch with everything the backends need
precomputed: edge offsets, the per-edge relation in-degree, the indices of
the square and global nodes, the graph of every node and the legal-move
masks.
"""

from __future__ import annotations

from typing import List, NamedTuple, Sequence

import chess
import numpy as np

from policy_encoding import POLICY_V3_DIM, move_to_index_v3
from relational_graph import (COL, FEATURE_DIM_V3, KIND_GLOBAL, KIND_PIECE, KIND_SQUARE, NUM_RELATIONS, REL,
                              Position, repetitions, to_board)


class GraphArrays(NamedTuple):
    """One relational-v1 graph: what build_graph's Data holds, as numpy arrays."""
    x: np.ndarray           # [N, FEATURE_DIM_V3] float32
    src: np.ndarray         # [E] int64, edges sorted by destination (stable)
    dst: np.ndarray         # [E] int64
    edge_type: np.ndarray   # [E] int64
    node_kind: np.ndarray   # [N] int64
    legal: np.ndarray       # [L] int64: v3 policy indices of the legal moves (board.legal_moves order)


class BatchArrays(NamedTuple):
    """A batch of graphs, ready for gnn_infer's backends."""
    x: np.ndarray           # [N, F] float32
    src: np.ndarray         # [E] int64 (node indices into the batch)
    dst: np.ndarray         # [E] int64
    edge_type: np.ndarray   # [E] int64
    inv_deg: np.ndarray     # [E] float32: 1 / edges of the same relation into the same node
    node_kind: np.ndarray   # [N] int64
    batch: np.ndarray       # [N] int64: graph of each node
    square_idx: np.ndarray  # [B * 64] int64: the square nodes of every graph, in order
    global_idx: np.ndarray  # [B] int64: the global node of every graph
    node_count: np.ndarray  # [B] float32: nodes per graph
    legal: np.ndarray       # [B, POLICY_V3_DIM] bool


# Precomputed per-square values (computed in double, as the reference does, then stored as float32).
_FILE = np.array([chess.square_file(s) / 7.0 for s in range(64)], dtype=np.float64)
_RANK = {chess.WHITE: np.array([chess.square_rank(s) / 7.0 for s in range(64)], dtype=np.float64),
         chess.BLACK: np.array([(7 - chess.square_rank(s)) / 7.0 for s in range(64)], dtype=np.float64)}
_FILE_L = _FILE.tolist()
_RANK_L = {c: v.tolist() for c, v in _RANK.items()}

# Squares an enemy pawn must not be on for a pawn to be passed (relational_graph._passed).
_PASSED_MASK = {chess.WHITE: [0] * 64, chess.BLACK: [0] * 64}
for _sq in range(64):
    _f, _r = chess.square_file(_sq), chess.square_rank(_sq)
    for _color, _ranks in ((chess.WHITE, range(_r + 1, 8)), (chess.BLACK, range(0, _r))):
        _m = 0
        for _ff in (_f - 1, _f, _f + 1):
            if 0 <= _ff < 8:
                for _rr in _ranks:
                    _m |= chess.BB_SQUARES[_rr * 8 + _ff]
        _PASSED_MASK[_color][_sq] = _m

_TYPE_COL = [0, COL["type_pawn"], COL["type_knight"], COL["type_bishop"], COL["type_rook"], COL["type_queen"],
             COL["type_king"]]  # by chess piece type (1-6)
_MATERIAL = [0, 1, 3, 3, 5, 9, 0]
_ROOK_LINES = [chess.BB_RANK_ATTACKS[s][0] | chess.BB_FILE_ATTACKS[s][0] for s in range(64)]
_DIAG_LINES = [chess.BB_DIAG_ATTACKS[s][0] for s in range(64)]
_POPCOUNT = [bin(i).count("1") for i in range(256)]

_R_PIECE_ON, _R_OCCUPIED_BY = REL["piece_on"], REL["occupied_by"]
_R_ATTACKS, _R_ATTACKED_BY = REL["attacks"], REL["attacked_by"]
_R_DEFENDS, _R_DEFENDED_BY = REL["defends"], REL["defended_by"]
_R_CONTROLS, _R_CONTROLLED_BY = REL["controls"], REL["controlled_by"]
_R_LEGAL, _R_LEGAL_FROM = REL["legal_move"], REL["legal_move_from"]
_R_SAME_FILE = REL["pawn_same_file"]
_R_SUPPORTS, _R_SUPPORTED_BY = REL["pawn_supports"], REL["pawn_supported_by"]
_R_PASSED, _R_PASSED_OF = REL["passed_path"], REL["passed_path_of"]
_R_KING_ZONE, _R_KING_ZONE_OF = REL["king_zone"], REL["king_zone_of"]
_R_PINNED_TO, _R_PINS = REL["pinned_to"], REL["pins"]
_R_G2P, _R_P2G = REL["global_to_piece"], REL["piece_to_global"]

_C_IS_SQUARE, _C_IS_PIECE, _C_IS_GLOBAL = COL["is_square"], COL["is_piece"], COL["is_global"]
_C_OWN, _C_OPP, _C_FILE, _C_RANK = COL["own"], COL["opponent"], COL["file"], COL["rank"]
_C_ATT_OWN, _C_ATT_OPP = COL["attackers_own"], COL["attackers_opp"]


def _bits(mask: int) -> List[int]:
    """Set squares of a bitboard in ascending order (chess.scan_forward as a list)."""
    out = []
    while mask:
        low = mask & -mask
        out.append(low.bit_length() - 1)
        mask ^= low
    return out


def _popcount(mask: int) -> int:
    return mask.bit_count() if hasattr(mask, "bit_count") else bin(mask).count("1")


def _pinned(board: chess.Board, color: chess.Color, king: int) -> int:
    """Pieces of `color` absolutely pinned to its king, as a bitboard (board.is_pinned for every piece).

    A piece is pinned when it is the only piece between its king and an
    enemy slider that moves along that line (python-chess's pin_mask).
    """
    enemy = board.occupied_co[not color]
    snipers = ((_ROOK_LINES[king] & (board.rooks | board.queens)) |
               (_DIAG_LINES[king] & (board.bishops | board.queens))) & enemy
    pinned = 0
    occupied = board.occupied
    while snipers:
        low = snipers & -snipers
        snipers ^= low
        blockers = chess.between(king, low.bit_length() - 1) & occupied
        if blockers and blockers & (blockers - 1) == 0:
            pinned |= blockers
    return pinned & board.occupied_co[color] & ~chess.BB_SQUARES[king]


def build_arrays(position: Position) -> GraphArrays:
    """relational_graph.build_graph(board) as arrays, bit for bit (ValueError like to_board)."""
    board = to_board(position)
    stm = board.turn
    occupied = board.occupied
    occ_co = board.occupied_co
    piece_type_at = board.piece_type_at
    attacks_mask = board.attacks_mask
    squares = _bits(occupied)
    num_pieces = len(squares)
    gnode = 64 + num_pieces
    num_nodes = gnode + 1
    node_of = [0] * 64
    for i, sq in enumerate(squares):
        node_of[sq] = 64 + i
    white = occ_co[chess.WHITE]

    types = [piece_type_at(sq) for sq in squares]
    colors = [bool(white & chess.BB_SQUARES[sq]) for sq in squares]
    masks = [attacks_mask(sq) for sq in squares]

    # Attack counts per square for each colour.
    att = {chess.WHITE: [0] * 64, chess.BLACK: [0] * 64}
    target_lists = []
    for color, mask in zip(colors, masks):
        targets = _bits(mask)
        target_lists.append(targets)
        counts = att[color]
        for t in targets:
            counts[t] += 1

    legal = list(board.legal_moves)
    legal_from = [0] * 64
    move_pairs = set()
    for mv in legal:
        legal_from[mv.from_square] += 1
        move_pairs.add((mv.from_square, mv.to_square))

    king_sq = {chess.WHITE: board.king(chess.WHITE), chess.BLACK: board.king(chess.BLACK)}
    in_check = board.is_check()
    castling = board.clean_castling_rights()
    has_rights = {c: board.has_castling_rights(c) for c in (chess.WHITE, chess.BLACK)}
    pinned = _pinned(board, chess.WHITE, king_sq[chess.WHITE]) | _pinned(board, chess.BLACK, king_sq[chess.BLACK])

    # Edges in the reference's insertion order: flat (src, dst, type) triples.
    E: List[int] = []
    ext = E.extend
    material = {chess.WHITE: 0, chess.BLACK: 0}
    own_rows: List[int] = []
    opp_rows: List[int] = []
    mobility: List[float] = []
    for i, sq in enumerate(squares):
        n = 64 + i
        color = colors[i]
        material[color] += _MATERIAL[types[i]]
        (own_rows if color == stm else opp_rows).append(i)
        mask = masks[i]
        mobility.append(_popcount(mask & ~occ_co[color]) / 27.0)
        ext((n, sq, _R_PIECE_ON, sq, n, _R_OCCUPIED_BY))
        own_occ = occ_co[color]
        for t in target_lists[i]:
            bit = chess.BB_SQUARES[t]
            if not occupied & bit:
                ext((n, t, _R_CONTROLS, t, n, _R_CONTROLLED_BY))
            elif own_occ & bit:
                m = node_of[t]
                ext((n, m, _R_DEFENDS, m, n, _R_DEFENDED_BY))
            else:
                m = node_of[t]
                ext((n, m, _R_ATTACKS, m, n, _R_ATTACKED_BY))
        ext((gnode, n, _R_G2P, n, gnode, _R_P2G))

    for frm, to in sorted(move_pairs):
        m = node_of[frm]
        ext((m, to, _R_LEGAL, to, m, _R_LEGAL_FROM))

    # Pawn structure.
    pawns = [(sq, colors[i]) for i, sq in enumerate(squares) if types[i] == chess.PAWN]
    passed_rows = []
    all_pawns = board.pawns
    for sq, color in pawns:
        n = node_of[sq]
        f = sq & 7
        for sq2, _ in pawns:
            if sq2 != sq and sq2 & 7 == f:
                ext((n, node_of[sq2], _R_SAME_FILE))
        for t in _bits(masks[n - 64] & all_pawns & occ_co[color]):
            m = node_of[t]
            ext((n, m, _R_SUPPORTS, m, n, _R_SUPPORTED_BY))
        if not all_pawns & occ_co[not color] & _PASSED_MASK[color][sq]:
            passed_rows.append(n - 64)
            step = 8 if color == chess.WHITE else -8
            t = sq + step
            while 0 <= t < 64:
                ext((n, t, _R_PASSED, t, n, _R_PASSED_OF))
                t += step

    # King zones.
    for color in (chess.WHITE, chess.BLACK):
        k = king_sq[color]
        kn = node_of[k]
        for t in _bits(chess.BB_KING_ATTACKS[k]):
            ext((kn, t, _R_KING_ZONE, t, kn, _R_KING_ZONE_OF))

    # Absolute pins: pinned piece <-> the enemy slider pinning it to its king.
    pinned_rows = []
    for sq in _bits(pinned):
        n = node_of[sq]
        pinned_rows.append(n - 64)
        color = colors[n - 64]
        k = king_sq[color]
        for s in _bits(board.attackers_mask(not color, sq)):
            if piece_type_at(s) in (chess.BISHOP, chess.ROOK, chess.QUEEN) and chess.between(k, s) & chess.BB_SQUARES[sq]:
                m = node_of[s]
                ext((n, m, _R_PINNED_TO, m, n, _R_PINS))

    # Features, column by column (values in double, stored as float32 like torch.tensor(..., dtype=float)).
    x = np.zeros((num_nodes, FEATURE_DIM_V3), dtype=np.float32)
    att_own = np.array(att[stm], dtype=np.float64) / 8.0
    att_opp = np.array(att[not stm], dtype=np.float64) / 8.0
    x[:64, _C_IS_SQUARE] = 1.0
    x[:64, _C_FILE] = _FILE
    x[:64, _C_RANK] = _RANK[stm]
    x[:64, _C_ATT_OWN] = att_own
    x[:64, _C_ATT_OPP] = att_opp
    if num_pieces:
        sq_arr = np.array(squares, dtype=np.int64)
        rows = np.arange(64, gnode)
        type_cols = np.array([_TYPE_COL[t] for t in types], dtype=np.int64)
        x[64:gnode, _C_IS_PIECE] = 1.0
        x[rows, type_cols] = 1.0
        x[sq_arr, type_cols] = 1.0
        if own_rows:
            own_idx = np.array(own_rows, dtype=np.int64)
            x[64 + own_idx, _C_OWN] = 1.0
            x[sq_arr[own_idx], _C_OWN] = 1.0
            x[64 + own_idx, COL["legal_moves_from"]] = np.array([legal_from[squares[i]] for i in own_rows],
                                                               dtype=np.float64) / 27.0
        if opp_rows:
            opp_idx = np.array(opp_rows, dtype=np.int64)
            x[64 + opp_idx, _C_OPP] = 1.0
            x[sq_arr[opp_idx], _C_OPP] = 1.0
        x[64:gnode, _C_FILE] = _FILE[sq_arr]
        x[64:gnode, _C_RANK] = _RANK[stm][sq_arr]
        x[64:gnode, _C_ATT_OWN] = att_own[sq_arr]
        x[64:gnode, _C_ATT_OPP] = att_opp[sq_arr]
        x[64:gnode, COL["mobility"]] = mobility
        for i, sq in enumerate(squares):
            t = types[i]
            if t == chess.ROOK:
                if castling & chess.BB_SQUARES[sq]:
                    x[64 + i, COL["castling_piece"]] = 1.0
            elif t == chess.KING:
                if has_rights[colors[i]]:
                    x[64 + i, COL["castling_piece"]] = 1.0
                if in_check and colors[i] == stm:
                    x[64 + i, COL["in_check"]] = 1.0
        for i in passed_rows:
            x[64 + i, COL["passed"]] = 1.0
        for i in pinned_rows:
            x[64 + i, COL["pinned"]] = 1.0
    if board.ep_square is not None and board.has_legal_en_passant():
        x[board.ep_square, COL["en_passant"]] = 1.0

    g = x[gnode]
    g[_C_IS_GLOBAL] = 1.0
    g[COL["in_check"]] = 1.0 if in_check else 0.0
    g[COL["white_to_move"]] = 1.0 if stm == chess.WHITE else 0.0
    g[COL["castle_own_k"]] = float(board.has_kingside_castling_rights(stm))
    g[COL["castle_own_q"]] = float(board.has_queenside_castling_rights(stm))
    g[COL["castle_opp_k"]] = float(board.has_kingside_castling_rights(not stm))
    g[COL["castle_opp_q"]] = float(board.has_queenside_castling_rights(not stm))
    g[COL["halfmove_clock"]] = min(board.halfmove_clock, 100) / 100.0
    g[COL["repetitions"]] = repetitions(board) / 2.0
    g[COL["legal_move_count"]] = len(legal) / 64.0
    g[COL["material_own"]] = material[stm] / 39.0
    g[COL["material_opp"]] = material[not stm] / 39.0

    e = np.array(E, dtype=np.int64).reshape(-1, 3)
    order = np.argsort(e[:, 1], kind="stable")
    e = e[order]
    kind = np.full(num_nodes, KIND_PIECE, dtype=np.int64)
    kind[:64] = KIND_SQUARE
    kind[gnode] = KIND_GLOBAL
    legal_idx = np.array([move_to_index_v3(mv) for mv in legal], dtype=np.int64)
    return GraphArrays(x, np.ascontiguousarray(e[:, 0]), np.ascontiguousarray(e[:, 1]),
                       np.ascontiguousarray(e[:, 2]), kind, legal_idx)


def inv_relation_degree(dst: np.ndarray, edge_type: np.ndarray, num_nodes: int) -> np.ndarray:
    """gnn_v3.relation_inv_degree in numpy: 1 / (edges of the same relation into the same node), float32."""
    key = dst * NUM_RELATIONS + edge_type
    counts = np.bincount(key, minlength=num_nodes * NUM_RELATIONS)
    return (np.float32(1.0) / counts[key].astype(np.float32)).astype(np.float32)


def collate(graphs: Sequence[GraphArrays]) -> BatchArrays:
    """Joins graphs into one batch (the layout torch_geometric's Batch.from_data_list gives)."""
    if not graphs:
        raise ValueError("collate needs at least one graph")
    sizes = np.array([g.x.shape[0] for g in graphs], dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    x = np.concatenate([g.x for g in graphs]) if len(graphs) > 1 else graphs[0].x
    src = np.concatenate([g.src + o for g, o in zip(graphs, offsets)])
    dst = np.concatenate([g.dst + o for g, o in zip(graphs, offsets)])
    edge_type = np.concatenate([g.edge_type for g in graphs])
    node_kind = np.concatenate([g.node_kind for g in graphs])
    total = int(sizes.sum())
    batch = np.repeat(np.arange(len(graphs), dtype=np.int64), sizes)
    square_idx = (offsets[:, None] + np.arange(64, dtype=np.int64)[None, :]).reshape(-1)
    global_idx = offsets + sizes - 1
    legal = np.zeros((len(graphs), POLICY_V3_DIM), dtype=bool)
    for i, g in enumerate(graphs):
        legal[i, g.legal] = True
    return BatchArrays(x, src, dst, edge_type, inv_relation_degree(dst, edge_type, total), node_kind, batch,
                       square_idx, global_idx, sizes.astype(np.float32), legal)


def build_batch(positions: Sequence[Position]) -> BatchArrays:
    return collate([build_arrays(p) for p in positions])
