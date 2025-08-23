import math
from typing import List, Tuple, Dict

import torch

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as e:  # pragma: no cover - helpful error when PyG is missing
    raise RuntimeError("torch_geometric is required. Please install PyTorch Geometric.") from e


# -----------------------------
# Utility: FEN parsing and board helpers
# -----------------------------

PIECE_TO_IDX = {
    'P': (0, 1),  # (type idx 0..5, color 1=white, -1=black replaced below)
    'N': (1, 1),
    'B': (2, 1),
    'R': (3, 1),
    'Q': (4, 1),
    'K': (5, 1),
    'p': (0, -1),
    'n': (1, -1),
    'b': (2, -1),
    'r': (3, -1),
    'q': (4, -1),
    'k': (5, -1),
}


def sq_index(file_idx: int, rank_idx: int) -> int:
    return rank_idx * 8 + file_idx


def file_of(sq: int) -> int:
    return sq & 7


def rank_of(sq: int) -> int:
    return sq >> 3


def in_board(f: int, r: int) -> bool:
    return 0 <= f < 8 and 0 <= r < 8


def parse_fen(fen: str):
    fields = fen.strip().split()
    if len(fields) < 4:
        raise ValueError("Invalid FEN")
    board_f, stm_f, castling_f, ep_f = fields[:4]

    pieces: List[Tuple[int, int, int]] = []  # (type_idx 0..5, color 1 white/0 black, square)
    r = 7
    f = 0
    for ch in board_f:
        if ch == '/':
            r -= 1
            f = 0
            continue
        if ch.isdigit():
            f += int(ch)
            continue
        if ch not in PIECE_TO_IDX:
            raise ValueError("Invalid piece char in FEN")
        t_idx, color_sign = PIECE_TO_IDX[ch]
        color = 1 if color_sign > 0 else 0
        sq = sq_index(f, r)
        pieces.append((t_idx, color, sq))
        f += 1

    side_to_move = 1 if stm_f == 'w' else 0

    castles = {
        'K': 'K' in castling_f,
        'Q': 'Q' in castling_f,
        'k': 'k' in castling_f,
        'q': 'q' in castling_f,
    }

    ep_sq = -1
    if ep_f != '-' and len(ep_f) == 2:
        file = ord(ep_f[0]) - ord('a')
        rank = ord(ep_f[1]) - ord('1')
        if in_board(file, rank):
            ep_sq = sq_index(file, rank)

    return pieces, side_to_move, castles, ep_sq


# -----------------------------
# Graph Builder
# -----------------------------

class ChessGraph:
    def __init__(self, device: torch.device | None = None):
        self.device = device if device is not None else (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    def fen_to_graph(self, fen: str) -> Data:
        pieces, side_to_move, castles, ep_sq = parse_fen(fen)
        # Node indexing
        # 0..63: square nodes
        # next: piece nodes (variable)
        # then: metadata nodes: side_to_move (1), castling 4, pawn files 8 (always present)

        square_node_offset = 0
        piece_node_offset = 64
        piece_nodes: List[Tuple[int, int, int]] = []  # (type_idx, color, sq)
        piece_nodes.extend(pieces)
        num_piece_nodes = len(piece_nodes)

        meta_offset = piece_node_offset + num_piece_nodes
        side_node_id = meta_offset
        castling_ids = {
            'K': side_node_id + 1,
            'Q': side_node_id + 2,
            'k': side_node_id + 3,
            'q': side_node_id + 4,
        }
        pawn_file_offset = side_node_id + 5
        pawn_file_ids = [pawn_file_offset + i for i in range(8)]

        num_nodes = 64 + num_piece_nodes + 1 + 4 + 8

        # Build node features
        # Feature schema:
        # [is_square, is_piece, is_meta, piece_type_onehot(6), color_onehot(2), square_file, square_rank,
        #  side_to_move_flag, castling_flags(4), pawn_file_onehot(8)]
        feat_dim = 3 + 6 + 2 + 2 + 1 + 4 + 8
        x = torch.zeros((num_nodes, feat_dim), dtype=torch.float)

        # Squares
        for s in range(64):
            i = square_node_offset + s
            x[i, 0] = 1.0
            f = file_of(s) / 7.0
            r = rank_of(s) / 7.0
            x[i, 3 + 6 + 2 + 0] = f  # square_file
            x[i, 3 + 6 + 2 + 1] = r  # square_rank

        # Pieces
        piece_node_ids: Dict[int, int] = {}  # by square
        for idx, (t_idx, color, sq) in enumerate(piece_nodes):
            i = piece_node_offset + idx
            x[i, 1] = 1.0
            # piece type onehot
            x[i, 3 + t_idx] = 1.0
            # color onehot
            x[i, 3 + 6 + color] = 1.0
            # attach file/rank of its square (positional hint)
            x[i, 3 + 6 + 2 + 0] = file_of(sq) / 7.0
            x[i, 3 + 6 + 2 + 1] = rank_of(sq) / 7.0
            piece_node_ids[sq] = i

        # Metadata
        # side-to-move node
        x[side_node_id, 2] = 1.0
        x[side_node_id, 3 + 6 + 2 + 2] = float(side_to_move)  # side_to_move_flag
        # castling nodes
        for key, nid in castling_ids.items():
            x[nid, 2] = 1.0
            base = 3 + 6 + 2 + 2 + 1
            flag_idx = {'K': 0, 'Q': 1, 'k': 2, 'q': 3}[key]
            x[nid, base + flag_idx] = 1.0 if castles[key] else 0.0
        # pawn file nodes (a..h)
        for f in range(8):
            nid = pawn_file_ids[f]
            x[nid, 2] = 1.0
            base = 3 + 6 + 2 + 2 + 1 + 4
            x[nid, base + f] = 1.0

        # Edges
        edges_src: List[int] = []
        edges_dst: List[int] = []

        occ = {sq for (_, _, sq) in piece_nodes}

        # Occupancy edges: piece <-> square
        for (_, _, sq) in piece_nodes:
            p_id = piece_node_ids[sq]
            s_id = square_node_offset + sq
            edges_src.extend([p_id, s_id])
            edges_dst.extend([s_id, p_id])

        # Attack edges: piece -> attacked squares (bidirectional for message flow)
        for (t_idx, color, sq) in piece_nodes:
            attacked = self._attacks_from(t_idx, color, sq, occ)
            p_id = piece_node_ids[sq]
            for a in attacked:
                s_id = square_node_offset + a
                edges_src.extend([p_id, s_id])
                edges_dst.extend([s_id, p_id])

        # Pawn-file edges: file node <-> pawn piece node
        for (t_idx, color, sq) in piece_nodes:
            if t_idx != 0:
                continue
            pf = file_of(sq)
            pf_id = pawn_file_ids[pf]
            p_id = piece_node_ids[sq]
            edges_src.extend([pf_id, p_id])
            edges_dst.extend([p_id, pf_id])

        # Pawn-chain adjacency: connect pawn piece nodes diagonally adjacent by color
        pawn_by_color: Dict[int, List[int]] = {0: [], 1: []}
        for (t_idx, color, sq) in piece_nodes:
            if t_idx == 0:
                pawn_by_color[color].append(sq)
        pawn_sets = {c: set(lst) for c, lst in pawn_by_color.items()}
        # White: sq -> sq+7, sq+9; Black: sq -> sq-7, sq-9
        for sq in pawn_by_color[1]:
            for d in (7, 9):
                nb = sq + d
                if 0 <= nb < 64 and abs(file_of(nb) - file_of(sq)) == 1 and nb in pawn_sets[1]:
                    a = piece_node_ids[sq]
                    b = piece_node_ids[nb]
                    edges_src.extend([a, b])
                    edges_dst.extend([b, a])
        for sq in pawn_by_color[0]:
            for d in (-7, -9):
                nb = sq + d
                if 0 <= nb < 64 and abs(file_of(nb) - file_of(sq)) == 1 and nb in pawn_sets[0]:
                    a = piece_node_ids[sq]
                    b = piece_node_ids[nb]
                    edges_src.extend([a, b])
                    edges_dst.extend([b, a])

        # Connect side-to-move node to all pieces of that side
        for (t_idx, color, sq) in piece_nodes:
            if color == side_to_move:
                p_id = piece_node_ids[sq]
                edges_src.extend([side_node_id, p_id])
                edges_dst.extend([p_id, side_node_id])

        edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index)
        data.num_nodes = num_nodes
        data.to(self.device)
        return data

    def _attacks_from(self, t_idx: int, color: int, sq: int, occ: set) -> List[int]:
        f = file_of(sq)
        r = rank_of(sq)
        attacked: List[int] = []
        if t_idx == 0:  # pawn
            if color == 1:  # white
                for df in (-1, 1):
                    nf, nr = f + df, r + 1
                    if in_board(nf, nr):
                        attacked.append(sq_index(nf, nr))
            else:
                for df in (-1, 1):
                    nf, nr = f + df, r - 1
                    if in_board(nf, nr):
                        attacked.append(sq_index(nf, nr))
        elif t_idx == 1:  # knight
            for df, dr in [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]:
                nf, nr = f + df, r + dr
                if in_board(nf, nr):
                    attacked.append(sq_index(nf, nr))
        elif t_idx == 2 or t_idx == 3 or t_idx == 4:  # bishop, rook, queen
            dirs = []
            if t_idx in (2, 4):
                dirs += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            if t_idx in (3, 4):
                dirs += [(1, 0), (-1, 0), (0, 1), (0, -1)]
            for df, dr in dirs:
                nf, nr = f + df, r + dr
                while in_board(nf, nr):
                    sq_to = sq_index(nf, nr)
                    attacked.append(sq_to)
                    if sq_to in occ:
                        break
                    nf += df
                    nr += dr
        elif t_idx == 5:  # king
            for df in (-1, 0, 1):
                for dr in (-1, 0, 1):
                    if df == 0 and dr == 0:
                        continue
                    nf, nr = f + df, r + dr
                    if in_board(nf, nr):
                        attacked.append(sq_index(nf, nr))
        return attacked


class ChessGNN(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 6, out_dim: int | None = None):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.proj_in = torch.nn.Linear(in_dim, hidden_dim)
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim, add_self_loops=True, normalize=True))
        self.norms = torch.nn.ModuleList([torch.nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.act = torch.nn.ReLU()
        self.out = torch.nn.Identity() if out_dim is None else torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data):
        x, edge_index = data.x, data.edge_index
        batch = getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        h = self.proj_in(x)
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index)
            h = self.norms[i](h + h_new)
            h = self.act(h)
        node_emb = self.out(h)
        global_emb = global_mean_pool(node_emb, batch)
        return global_emb, node_emb


# -----------------------------
# Unit tests (can be run via pytest -q)
# -----------------------------

def _device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def test_graph_dims_and_forward():
    device = _device()
    builder = ChessGraph(device=device)
    # Starting position
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    data = builder.fen_to_graph(fen)
    assert data.x.dim() == 2
    num_nodes = data.x.size(0)
    # startpos: 64 squares + 32 pieces + 1 side + 4 castling + 8 pawn files
    assert num_nodes == 64 + 32 + 1 + 4 + 8
    in_dim = data.x.size(1)
    model = ChessGNN(in_dim=in_dim, hidden_dim=64, num_layers=6).to(device)
    global_emb, node_emb = model(data)
    assert global_emb.shape[0] == 1
    assert node_emb.shape[0] == num_nodes
    assert node_emb.shape[1] == 64


def test_cpu_cuda_fallback():
    dev = _device()
    builder = ChessGraph(device=dev)
    fen = "8/8/8/8/8/8/8/8 w - - 0 1"
    data = builder.fen_to_graph(fen)
    model = ChessGNN(in_dim=data.x.size(1), hidden_dim=32, num_layers=6).to(dev)
    g, n = model(data)
    assert g.device == dev and n.device == dev


