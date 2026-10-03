"""Graph encoding of chess positions and the GNN backbone.

A position becomes a graph with
  * 64 square nodes,
  * one node per piece,
  * metadata nodes: side to move (1), castling rights (4), pawn files (8),
and edges for occupancy (piece <-> square), attacks (piece <-> attacked
square), pawn files, pawn chains and side-to-move membership. Squares are
numbered a1 = 0 ... h8 = 63, matching the C++ engine.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as e:  # pragma: no cover - helpful error when PyG is missing
    raise RuntimeError("torch_geometric is required. Please install PyTorch Geometric.") from e


PIECE_TO_IDX = {
    'P': (0, 1), 'N': (1, 1), 'B': (2, 1), 'R': (3, 1), 'Q': (4, 1), 'K': (5, 1),
    'p': (0, 0), 'n': (1, 0), 'b': (2, 0), 'r': (3, 0), 'q': (4, 0), 'k': (5, 0),
}

# Feature layout (FEATURE_DIM columns):
F_IS_SQUARE, F_IS_PIECE, F_IS_META = 0, 1, 2
F_PIECE_TYPE = 3          # 6 columns, one-hot P N B R Q K
F_COLOR = 9               # 2 columns, one-hot black / white
F_FILE, F_RANK = 11, 12   # square coordinates scaled to [0, 1]
F_SIDE_TO_MOVE = 13       # 1.0 if white to move (on the side-to-move node)
F_CASTLING = 14           # 4 columns K Q k q
F_PAWN_FILE = 18          # 8 columns a..h
F_EN_PASSANT = 26         # 1.0 on the en passant target square node
FEATURE_DIM = 27


def sq_index(file_idx: int, rank_idx: int) -> int:
    return rank_idx * 8 + file_idx


def file_of(sq: int) -> int:
    return sq & 7


def rank_of(sq: int) -> int:
    return sq >> 3


def in_board(f: int, r: int) -> bool:
    return 0 <= f < 8 and 0 <= r < 8


def parse_fen(fen: str):
    """Returns (pieces, side_to_move, castles, ep_square).

    pieces is a list of (type 0..5, color 1=white/0=black, square).
    Raises ValueError for malformed FENs.
    """
    fields = fen.strip().split()
    if len(fields) < 4:
        raise ValueError(f"Invalid FEN (need at least 4 fields): {fen!r}")
    board_f, stm_f, castling_f, ep_f = fields[:4]
    if stm_f not in ('w', 'b'):
        raise ValueError(f"Invalid side to move in FEN: {fen!r}")

    pieces: List[Tuple[int, int, int]] = []
    ranks = board_f.split('/')
    if len(ranks) != 8:
        raise ValueError(f"Invalid FEN (need 8 ranks): {fen!r}")
    for i, rank_str in enumerate(ranks):
        r = 7 - i
        f = 0
        for ch in rank_str:
            if ch.isdigit():
                f += int(ch)
                continue
            if ch not in PIECE_TO_IDX:
                raise ValueError(f"Invalid piece character {ch!r} in FEN: {fen!r}")
            if f >= 8:
                raise ValueError(f"Rank overflow in FEN: {fen!r}")
            t_idx, color = PIECE_TO_IDX[ch]
            pieces.append((t_idx, color, sq_index(f, r)))
            f += 1
        if f != 8:
            raise ValueError(f"Rank {r + 1} does not have 8 files in FEN: {fen!r}")

    side_to_move = 1 if stm_f == 'w' else 0
    castles = {c: c in castling_f for c in 'KQkq'}

    ep_sq = -1
    if ep_f != '-' and len(ep_f) == 2:
        file = ord(ep_f[0]) - ord('a')
        rank = ord(ep_f[1]) - ord('1')
        if in_board(file, rank):
            ep_sq = sq_index(file, rank)

    return pieces, side_to_move, castles, ep_sq


class ChessGraph:
    def __init__(self, device: torch.device | None = None):
        self.device = device if device is not None else (
            torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    def fen_to_graph(self, fen: str) -> Data:
        pieces, side_to_move, castles, ep_sq = parse_fen(fen)

        piece_node_offset = 64
        num_piece_nodes = len(pieces)
        side_node_id = piece_node_offset + num_piece_nodes
        castling_ids = {c: side_node_id + 1 + i for i, c in enumerate('KQkq')}
        pawn_file_ids = [side_node_id + 5 + i for i in range(8)]
        num_nodes = 64 + num_piece_nodes + 1 + 4 + 8

        x = torch.zeros((num_nodes, FEATURE_DIM), dtype=torch.float)

        for s in range(64):
            x[s, F_IS_SQUARE] = 1.0
            x[s, F_FILE] = file_of(s) / 7.0
            x[s, F_RANK] = rank_of(s) / 7.0
        if ep_sq >= 0:
            x[ep_sq, F_EN_PASSANT] = 1.0

        piece_node_ids: Dict[int, int] = {}
        for idx, (t_idx, color, sq) in enumerate(pieces):
            i = piece_node_offset + idx
            x[i, F_IS_PIECE] = 1.0
            x[i, F_PIECE_TYPE + t_idx] = 1.0
            x[i, F_COLOR + color] = 1.0
            x[i, F_FILE] = file_of(sq) / 7.0
            x[i, F_RANK] = rank_of(sq) / 7.0
            piece_node_ids[sq] = i

        x[side_node_id, F_IS_META] = 1.0
        x[side_node_id, F_SIDE_TO_MOVE] = float(side_to_move)
        for flag_idx, (key, nid) in enumerate(castling_ids.items()):
            x[nid, F_IS_META] = 1.0
            x[nid, F_CASTLING + flag_idx] = 1.0 if castles[key] else 0.0
        for f, nid in enumerate(pawn_file_ids):
            x[nid, F_IS_META] = 1.0
            x[nid, F_PAWN_FILE + f] = 1.0

        src: List[int] = []
        dst: List[int] = []

        def link(a: int, b: int) -> None:
            src.extend((a, b))
            dst.extend((b, a))

        occ = {sq for (_, _, sq) in pieces}
        for (_, _, sq) in pieces:
            link(piece_node_ids[sq], sq)
        for (t_idx, color, sq) in pieces:
            for a in self._attacks_from(t_idx, color, sq, occ):
                link(piece_node_ids[sq], a)
        for (t_idx, _, sq) in pieces:
            if t_idx == 0:
                link(pawn_file_ids[file_of(sq)], piece_node_ids[sq])

        pawn_sets = {c: {sq for (t, col, sq) in pieces if t == 0 and col == c} for c in (0, 1)}
        for color, deltas in ((1, (7, 9)), (0, (-7, -9))):
            for sq in pawn_sets[color]:
                for d in deltas:
                    nb = sq + d
                    if 0 <= nb < 64 and abs(file_of(nb) - file_of(sq)) == 1 and nb in pawn_sets[color]:
                        link(piece_node_ids[sq], piece_node_ids[nb])

        for (_, color, sq) in pieces:
            if color == side_to_move:
                link(side_node_id, piece_node_ids[sq])

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, num_nodes=num_nodes)
        return data.to(self.device)

    def fens_to_batch(self, fens: Sequence[str]) -> Batch:
        return Batch.from_data_list([self.fen_to_graph(f) for f in fens])

    @staticmethod
    def _attacks_from(t_idx: int, color: int, sq: int, occ: set) -> List[int]:
        f, r = file_of(sq), rank_of(sq)
        attacked: List[int] = []
        if t_idx == 0:
            dr = 1 if color == 1 else -1
            for df in (-1, 1):
                if in_board(f + df, r + dr):
                    attacked.append(sq_index(f + df, r + dr))
        elif t_idx == 1:
            for df, dr in ((1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)):
                if in_board(f + df, r + dr):
                    attacked.append(sq_index(f + df, r + dr))
        elif t_idx in (2, 3, 4):
            dirs = []
            if t_idx in (2, 4):
                dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            if t_idx in (3, 4):
                dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for df, dr in dirs:
                nf, nr = f + df, r + dr
                while in_board(nf, nr):
                    to = sq_index(nf, nr)
                    attacked.append(to)
                    if to in occ:
                        break
                    nf += df
                    nr += dr
        elif t_idx == 5:
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if (df or dr) and in_board(f + df, r + dr):
                        attacked.append(sq_index(f + df, r + dr))
        return attacked


class ChessGNN(torch.nn.Module):
    """Residual GCN stack. Returns (graph embeddings [B, H], node embeddings [N, H])."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 6, out_dim: int | None = None):
        super().__init__()
        self.num_layers = num_layers
        self.proj_in = torch.nn.Linear(in_dim, hidden_dim)
        self.convs = torch.nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True) for _ in range(num_layers)])
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.act = torch.nn.ReLU()
        self.out = torch.nn.Identity() if out_dim is None else torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        h = self.proj_in(x)
        for conv, norm in zip(self.convs, self.norms):
            h = self.act(norm(h + conv(h, edge_index)))
        node_emb = self.out(h)
        global_emb = global_mean_pool(node_emb, batch)
        return global_emb, node_emb
