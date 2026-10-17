"""Promotion-aware move encoding for the v3 GNN policy ('policy_encoding': 'v3').

The legacy encoding (gnn_evaluator.move_index, 4096 entries) gives a move the
index from * 64 + to, so the four promotions on one from/to pair share an
index. The v3 encoding keeps those 4096 indices and adds one slot per
underpromotion, so every legal chess move has its own index:

* index = from * 64 + to for every move that is not an underpromotion,
  including queen promotions, castling (the king's from/to in UCI, e.g.
  e1g1) and en passant (the capturing pawn's from/to). Squares are numbered
  a1 = 0 ... h8 = 63, as in the engine and python-chess.
* Underpromotions get the 132 slots after 4096:
      index = 4096 + pair * 3 + {'n': 0, 'b': 1, 'r': 2}[piece]
  where pair numbers the 44 from/to pairs a promotion can use, in this order
  (PROMOTION_PAIRS):
      white pawns (rank 7 -> rank 8), then black pawns (rank 2 -> rank 1);
      within a colour by the pawn's file a..h;
      for one pawn by the target file from left to right (capture towards
      the a-file, straight push, capture towards the h-file).
  So pair 0 is a7a8, pair 1 a7b8, pair 2 b7a8, pair 3 b7b8, pair 4 b7c8, ...,
  pair 21 h7h8, pair 22 a2a1, ..., pair 43 h2h1.

POLICY_V3_DIM = 4096 + 44 * 3 = 4228. An index below 4096 does not say
whether a move is a queen promotion (e7e8q) or a piece move on the same
squares (a rook on e7 moving to e8); index_to_uci_v3 takes the board to tell
them apart.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import chess
import torch

FROM_TO_DIM = 64 * 64
UNDERPROMOTION_PIECES = "nbr"


def _promotion_pairs() -> List[Tuple[int, int]]:
    pairs: List[Tuple[int, int]] = []
    for from_rank, to_rank in ((6, 7), (1, 0)):          # white 7 -> 8, black 2 -> 1
        for f in range(8):
            for df in (-1, 0, 1):
                if 0 <= f + df < 8:
                    pairs.append((from_rank * 8 + f, to_rank * 8 + f + df))
    return pairs


PROMOTION_PAIRS: List[Tuple[int, int]] = _promotion_pairs()
NUM_PROMOTION_PAIRS = len(PROMOTION_PAIRS)               # 44
NUM_UNDERPROMOTIONS = NUM_PROMOTION_PAIRS * len(UNDERPROMOTION_PIECES)  # 132
POLICY_V3_DIM = FROM_TO_DIM + NUM_UNDERPROMOTIONS        # 4228

_PAIR_INDEX: Dict[Tuple[int, int], int] = {p: i for i, p in enumerate(PROMOTION_PAIRS)}
_PIECE_SLOT = {p: i for i, p in enumerate(UNDERPROMOTION_PIECES)}
_SLOT_OF_PIECE_TYPE = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}

# From / to squares of the 44 pairs as tensors (used by the model's underpromotion head).
PROMOTION_FROM = torch.tensor([p[0] for p in PROMOTION_PAIRS], dtype=torch.long)
PROMOTION_TO = torch.tensor([p[1] for p in PROMOTION_PAIRS], dtype=torch.long)

assert NUM_PROMOTION_PAIRS == 44 and POLICY_V3_DIM == 4228


def _square(file_ch: str, rank_ch: str) -> int:
    f, r = ord(file_ch) - 97, ord(rank_ch) - 49
    if not (0 <= f < 8 and 0 <= r < 8):
        raise ValueError
    return r * 8 + f


def move_index_v3(move: Union[str, chess.Move]) -> int:
    """v3 policy index of a move given as UCI ('e2e4', 'e7e8q', 'e7e8n') or a chess.Move."""
    uci = move.uci() if isinstance(move, chess.Move) else move
    if not isinstance(uci, str) or len(uci) not in (4, 5):
        raise ValueError(f"invalid UCI move {uci!r}")
    try:
        frm, to = _square(uci[0], uci[1]), _square(uci[2], uci[3])
    except (ValueError, IndexError):
        raise ValueError(f"invalid UCI move {uci!r}") from None
    if len(uci) == 4:
        return frm * 64 + to
    promo = uci[4]
    pair = _PAIR_INDEX.get((frm, to))
    if pair is None or promo not in "qnbr":
        raise ValueError(f"invalid promotion {uci!r}")
    if promo == "q":
        return frm * 64 + to
    return FROM_TO_DIM + pair * 3 + _PIECE_SLOT[promo]


def move_to_index_v3(move: chess.Move) -> int:
    """Same as move_index_v3 for a chess.Move, without going through UCI text."""
    if move.promotion in _SLOT_OF_PIECE_TYPE:
        pair = _PAIR_INDEX.get((move.from_square, move.to_square))
        if pair is None:
            raise ValueError(f"invalid promotion {move.uci()!r}")
        return FROM_TO_DIM + pair * 3 + _SLOT_OF_PIECE_TYPE[move.promotion]
    return move.from_square * 64 + move.to_square


def index_to_uci_v3(index: int, board: Optional[chess.Board] = None) -> str:
    """Inverse of move_index_v3.

    For an index below 4096 the result is the plain from/to UCI string; with
    a board, 'q' is appended when a pawn on the from-square reaches its last
    rank, so legal moves round-trip exactly (index_to_uci_v3(move_index_v3(m),
    board) == m for every legal m).
    """
    index = int(index)
    if not 0 <= index < POLICY_V3_DIM:
        raise ValueError(f"policy index {index} out of range [0, {POLICY_V3_DIM})")
    if index >= FROM_TO_DIM:
        pair, slot = divmod(index - FROM_TO_DIM, 3)
        frm, to = PROMOTION_PAIRS[pair]
        return chess.square_name(frm) + chess.square_name(to) + UNDERPROMOTION_PIECES[slot]
    frm, to = divmod(index, 64)
    uci = chess.square_name(frm) + chess.square_name(to)
    if board is not None and (frm, to) in _PAIR_INDEX:
        piece = board.piece_at(frm)
        if piece is not None and piece.piece_type == chess.PAWN and \
                chess.square_rank(to) == (7 if piece.color == chess.WHITE else 0):
            uci += "q"
    return uci


def legal_indices_v3(board: chess.Board) -> List[int]:
    """v3 indices of the legal moves, in board.legal_moves order."""
    return [move_to_index_v3(m) for m in board.legal_moves]


def legal_mask_v3(board: chess.Board) -> torch.Tensor:
    """Boolean mask [POLICY_V3_DIM] with True exactly at the legal moves."""
    mask = torch.zeros(POLICY_V3_DIM, dtype=torch.bool)
    idx = legal_indices_v3(board)
    if idx:
        mask[torch.tensor(idx, dtype=torch.long)] = True
    return mask


def move_indices_v3(moves: Sequence[Union[str, chess.Move]]) -> torch.Tensor:
    """Long tensor of the v3 indices of the given moves."""
    return torch.tensor([move_index_v3(m) for m in moves], dtype=torch.long)
