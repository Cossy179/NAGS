"""Relational (typed, directed edge) graph of a chess position for the v3 GNN.

Built from python-chess, so attacks, pins, castling, en passant and legality
follow the rules exactly. This module is new; the legacy encoding
(chess_graph.ChessGraph, 27 features, untyped edges) is unchanged and is
what 'nags-gnn-v2' checkpoints use.

Nodes of one position, in this order:
  * 64 square nodes (node i is square i, a1 = 0 ... h8 = 63);
  * one node per piece, in ascending order of its square;
  * one global context node.
Each graph also carries node_kind (0 square, 1 piece, 2 global), so a model
can pick out the square and global nodes of every graph in a batch.

Features are expressed relative to the side to move ("own" = the side to
move, ranks counted from its side of the board), so a position and its
colour-flipped mirror image give the same square/piece features (see
FEATURE_COLUMNS). Edges are directed and typed (edge_type, see RELATIONS)
and sorted by destination node.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import chess
import torch

try:
    from torch_geometric.data import Batch, Data
except Exception as e:  # pragma: no cover
    raise RuntimeError("torch_geometric is required.") from e

GRAPH_VERSION = "relational-v1"

KIND_SQUARE, KIND_PIECE, KIND_GLOBAL = 0, 1, 2

# Relations (edge types). "X -> Y" is the direction messages flow.
RELATIONS: Tuple[str, ...] = (
    "piece_on",           # 0  piece -> the square it stands on
    "occupied_by",        # 1  square -> the piece standing on it
    "attacks",            # 2  piece -> enemy piece it attacks
    "attacked_by",        # 3  attacked piece -> its (enemy) attacker
    "defends",            # 4  piece -> own piece it defends (attacks its square)
    "defended_by",        # 5  defended piece -> its defender
    "controls",           # 6  piece -> empty square it attacks
    "controlled_by",      # 7  empty square -> piece attacking it
    "legal_move",         # 8  side-to-move piece -> square it can legally move to (castling: king -> g/c file)
    "legal_move_from",    # 9  that target square -> the piece
    "pawn_same_file",     # 10 pawn -> every other pawn (either colour) on its file
    "pawn_supports",      # 11 pawn -> own pawn it protects (pawn chain)
    "pawn_supported_by",  # 12 protected pawn -> protecting pawn
    "passed_path",        # 13 passed pawn -> each square in front of it up to promotion
    "passed_path_of",     # 14 such a square -> the passed pawn
    "king_zone",          # 15 king -> each square adjacent to it
    "king_zone_of",       # 16 such a square -> the king
    "pinned_to",          # 17 absolutely pinned piece -> the enemy slider pinning it
    "pins",               # 18 pinning slider -> the pinned piece
    "global_to_piece",    # 19 global node -> every piece
    "piece_to_global",    # 20 every piece -> global node
)
REL: Dict[str, int] = {name: i for i, name in enumerate(RELATIONS)}
NUM_RELATIONS = len(RELATIONS)

# Node feature columns (FEATURE_DIM_V3). "squares" = square nodes,
# "pieces" = piece nodes, "global" = the context node.
FEATURE_COLUMNS: Tuple[str, ...] = (
    "is_square",          # 0  node kind one-hot
    "is_piece",           # 1
    "is_global",          # 2
    "type_pawn",          # 3  piece type one-hot P N B R Q K: on pieces and on the square they occupy
    "type_knight",        # 4
    "type_bishop",        # 5
    "type_rook",          # 6
    "type_queen",         # 7
    "type_king",          # 8
    "own",                # 9  piece of the side to move (pieces, occupied squares)
    "opponent",           # 10 piece of the other side (pieces, occupied squares)
    "file",               # 11 file / 7 (squares, pieces)
    "rank",               # 12 rank / 7 counted from the side to move's first rank (squares, pieces)
    "attackers_own",      # 13 side-to-move pieces attacking this square / 8 (squares; pieces: their square)
    "attackers_opp",      # 14 opponent pieces attacking this square / 8 (same; for a piece, attackers of its
                          #    colour are its defenders)
    "en_passant",         # 15 square a legal en-passant capture moves to (squares)
    "pinned",             # 16 piece is absolutely pinned to its own king (pieces)
    "mobility",           # 17 squares the piece attacks not holding a piece of its colour / 27 (pieces)
    "legal_moves_from",   # 18 legal moves of this piece / 27 (side-to-move pieces; promotions count once each)
    "passed",             # 19 passed pawn (pieces)
    "castling_piece",     # 20 rook with a castling right, and the king of a side with any (pieces)
    "in_check",           # 21 side to move is in check (global, and the checked king)
    "white_to_move",      # 22 1 if White is to move (global)
    "castle_own_k",       # 23 side to move may castle kingside (global)
    "castle_own_q",       # 24 side to move may castle queenside (global)
    "castle_opp_k",       # 25 opponent may castle kingside (global)
    "castle_opp_q",       # 26 opponent may castle queenside (global)
    "halfmove_clock",     # 27 half-move clock / 100, capped at 1 (global)
    "repetitions",        # 28 earlier occurrences of this position in the board's history / 2 (0, 0.5, 1);
                          #    0 for a FEN, which carries no history (global)
    "legal_move_count",   # 29 number of legal moves of the side to move / 64 (global)
    "material_own",       # 30 own material (P1 N3 B3 R5 Q9) / 39 (global)
    "material_opp",       # 31 opponent material / 39 (global)
)
COL: Dict[str, int] = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
FEATURE_DIM_V3 = len(FEATURE_COLUMNS)

_TYPE_COL = {chess.PAWN: 3, chess.KNIGHT: 4, chess.BISHOP: 5, chess.ROOK: 6, chess.QUEEN: 7, chess.KING: 8}
_MATERIAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)

Position = Union[str, chess.Board]


def to_board(position: Position) -> chess.Board:
    """A chess.Board from a FEN or a board; ValueError for malformed FENs or positions without both kings."""
    if isinstance(position, chess.Board):
        board = position
    elif isinstance(position, str):
        board = chess.Board(position)  # raises ValueError on malformed FENs
    else:
        raise ValueError(f"expected a FEN string or chess.Board, got {type(position).__name__}")
    if board.king(chess.WHITE) is None or board.king(chess.BLACK) is None:
        raise ValueError(f"position needs one king per side: {board.fen()!r}")
    return board


def _passed(board: chess.Board, sq: int, color: chess.Color) -> bool:
    f, r = chess.square_file(sq), chess.square_rank(sq)
    enemy_pawns = board.pieces_mask(chess.PAWN, not color)
    ranks = range(r + 1, 8) if color == chess.WHITE else range(0, r)
    for ff in (f - 1, f, f + 1):
        if 0 <= ff < 8:
            for rr in ranks:
                if enemy_pawns & chess.BB_SQUARES[rr * 8 + ff]:
                    return False
    return True


class RelationalChessGraph:
    """Builds RELATIONS-typed graphs (torch_geometric Data / Batch) from FENs or boards."""

    feature_dim = FEATURE_DIM_V3
    num_relations = NUM_RELATIONS
    version = GRAPH_VERSION

    def __init__(self, device: torch.device | None = None):
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def fen_to_graph(self, position: Position) -> Data:
        return self.board_to_graph(to_board(position))

    def fens_to_batch(self, positions: Sequence[Position]) -> Batch:
        return Batch.from_data_list([self.fen_to_graph(p) for p in positions])

    def board_to_graph(self, board: chess.Board) -> Data:
        data = build_graph(board)
        return data.to(self.device)


def build_graph(board: chess.Board) -> Data:
    stm = board.turn
    pieces = sorted(board.piece_map().items())
    num_pieces = len(pieces)
    gnode = 64 + num_pieces
    num_nodes = gnode + 1
    node_of = {sq: 64 + i for i, (sq, _) in enumerate(pieces)}
    D = FEATURE_DIM_V3
    feat = [0.0] * (num_nodes * D)

    def setf(node: int, col: int, value: float) -> None:
        feat[node * D + col] = value

    def rel_rank(sq: int) -> float:
        r = chess.square_rank(sq)
        return (r if stm == chess.WHITE else 7 - r) / 7.0

    src: List[int] = []
    dst: List[int] = []
    typ: List[int] = []

    def edge(a: int, b: int, r: int) -> None:
        src.append(a)
        dst.append(b)
        typ.append(r)

    # Attack counts per square for each colour.
    att = {chess.WHITE: [0] * 64, chess.BLACK: [0] * 64}
    attacks_of: Dict[int, int] = {}
    for sq, piece in pieces:
        mask = board.attacks_mask(sq)
        attacks_of[sq] = mask
        counts = att[piece.color]
        for t in chess.scan_forward(mask):
            counts[t] += 1

    # Square nodes.
    for sq in range(64):
        setf(sq, COL["is_square"], 1.0)
        setf(sq, COL["file"], chess.square_file(sq) / 7.0)
        setf(sq, COL["rank"], rel_rank(sq))
        setf(sq, COL["attackers_own"], att[stm][sq] / 8.0)
        setf(sq, COL["attackers_opp"], att[not stm][sq] / 8.0)

    legal = list(board.legal_moves)
    if board.has_legal_en_passant() and board.ep_square is not None:
        setf(board.ep_square, COL["en_passant"], 1.0)
    legal_from = [0] * 64
    move_pairs = set()
    for mv in legal:
        legal_from[mv.from_square] += 1
        move_pairs.add((mv.from_square, mv.to_square))

    king_sq = {c: board.king(c) for c in (chess.WHITE, chess.BLACK)}
    in_check = board.is_check()
    castling = board.clean_castling_rights()  # rook squares that still carry a castling right
    material = {chess.WHITE: 0, chess.BLACK: 0}

    # Piece nodes and their occupancy / attack edges.
    for sq, piece in pieces:
        n = node_of[sq]
        color, ptype = piece.color, piece.piece_type
        material[color] += _MATERIAL[ptype]
        own = color == stm
        setf(n, COL["is_piece"], 1.0)
        setf(n, _TYPE_COL[ptype], 1.0)
        setf(n, COL["own"] if own else COL["opponent"], 1.0)
        setf(n, COL["file"], chess.square_file(sq) / 7.0)
        setf(n, COL["rank"], rel_rank(sq))
        setf(n, COL["attackers_own"], att[stm][sq] / 8.0)
        setf(n, COL["attackers_opp"], att[not stm][sq] / 8.0)
        # The square it stands on shows the occupant too.
        setf(sq, _TYPE_COL[ptype], 1.0)
        setf(sq, COL["own"] if own else COL["opponent"], 1.0)

        mask = attacks_of[sq]
        own_mask = board.occupied_co[color]
        setf(n, COL["mobility"], chess.popcount(mask & ~own_mask) / 27.0)
        if own:
            setf(n, COL["legal_moves_from"], legal_from[sq] / 27.0)
        if ptype == chess.ROOK and castling & chess.BB_SQUARES[sq]:
            setf(n, COL["castling_piece"], 1.0)
        if ptype == chess.KING:
            if board.has_castling_rights(color):
                setf(n, COL["castling_piece"], 1.0)
            if in_check and color == stm:
                setf(n, COL["in_check"], 1.0)
        if ptype == chess.PAWN and _passed(board, sq, color):
            setf(n, COL["passed"], 1.0)

        edge(n, sq, REL["piece_on"])
        edge(sq, n, REL["occupied_by"])
        for t in chess.scan_forward(mask):
            other = board.color_at(t)
            if other is None:
                edge(n, t, REL["controls"])
                edge(t, n, REL["controlled_by"])
            elif other == color:
                edge(n, node_of[t], REL["defends"])
                edge(node_of[t], n, REL["defended_by"])
            else:
                edge(n, node_of[t], REL["attacks"])
                edge(node_of[t], n, REL["attacked_by"])
        edge(gnode, n, REL["global_to_piece"])
        edge(n, gnode, REL["piece_to_global"])

    # Legal moves of the side to move (promotions to different pieces share one edge).
    for frm, to in sorted(move_pairs):
        edge(node_of[frm], to, REL["legal_move"])
        edge(to, node_of[frm], REL["legal_move_from"])

    # Pawn structure.
    pawns = [(sq, p.color) for sq, p in pieces if p.piece_type == chess.PAWN]
    for sq, color in pawns:
        n = node_of[sq]
        f = chess.square_file(sq)
        for sq2, color2 in pawns:
            if sq2 != sq and chess.square_file(sq2) == f:
                edge(n, node_of[sq2], REL["pawn_same_file"])
        for t in chess.scan_forward(attacks_of[sq] & board.pawns & board.occupied_co[color]):
            edge(n, node_of[t], REL["pawn_supports"])
            edge(node_of[t], n, REL["pawn_supported_by"])
        if feat[n * D + COL["passed"]]:
            step = 8 if color == chess.WHITE else -8
            t = sq + step
            while 0 <= t < 64:
                edge(n, t, REL["passed_path"])
                edge(t, n, REL["passed_path_of"])
                t += step

    # King zones.
    for color, k in king_sq.items():
        kn = node_of[k]
        for t in chess.scan_forward(chess.BB_KING_ATTACKS[k]):
            edge(kn, t, REL["king_zone"])
            edge(t, kn, REL["king_zone_of"])

    # Absolute pins: pinned piece <-> the enemy slider pinning it to its king.
    for sq, piece in pieces:
        color = piece.color
        if piece.piece_type == chess.KING or not board.is_pinned(color, sq):
            continue
        setf(node_of[sq], COL["pinned"], 1.0)
        k = king_sq[color]
        for s in chess.scan_forward(board.attackers_mask(not color, sq)):
            if board.piece_type_at(s) in _SLIDERS and chess.between(k, s) & chess.BB_SQUARES[sq]:
                edge(node_of[sq], node_of[s], REL["pinned_to"])
                edge(node_of[s], node_of[sq], REL["pins"])

    # Global node.
    setf(gnode, COL["is_global"], 1.0)
    setf(gnode, COL["in_check"], 1.0 if in_check else 0.0)
    setf(gnode, COL["white_to_move"], 1.0 if stm == chess.WHITE else 0.0)
    setf(gnode, COL["castle_own_k"], float(board.has_kingside_castling_rights(stm)))
    setf(gnode, COL["castle_own_q"], float(board.has_queenside_castling_rights(stm)))
    setf(gnode, COL["castle_opp_k"], float(board.has_kingside_castling_rights(not stm)))
    setf(gnode, COL["castle_opp_q"], float(board.has_queenside_castling_rights(not stm)))
    setf(gnode, COL["halfmove_clock"], min(board.halfmove_clock, 100) / 100.0)
    setf(gnode, COL["repetitions"], repetitions(board) / 2.0)
    setf(gnode, COL["legal_move_count"], len(legal) / 64.0)
    setf(gnode, COL["material_own"], material[stm] / 39.0)
    setf(gnode, COL["material_opp"], material[not stm] / 39.0)

    x = torch.tensor(feat, dtype=torch.float).view(num_nodes, D)
    dst_t = torch.tensor(dst, dtype=torch.long)
    order = torch.argsort(dst_t, stable=True)
    edge_index = torch.stack([torch.tensor(src, dtype=torch.long)[order], dst_t[order]])
    edge_type = torch.tensor(typ, dtype=torch.long)[order]
    node_kind = torch.full((num_nodes,), KIND_PIECE, dtype=torch.long)
    node_kind[:64] = KIND_SQUARE
    node_kind[gnode] = KIND_GLOBAL
    return Data(x=x, edge_index=edge_index, edge_type=edge_type, node_kind=node_kind, num_nodes=num_nodes)


def repetitions(board: chess.Board) -> int:
    """Earlier occurrences of the current position in the board's move history (0, 1 or 2 = two or more)."""
    if not board.move_stack:
        return 0
    if board.is_repetition(3):
        return 2
    return 1 if board.is_repetition(2) else 0
