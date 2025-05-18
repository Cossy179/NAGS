import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.data import Data
import numpy as np
import chess

class ChessHypergraphBuilder:
    """Constructs a hypergraph representation of a chess position."""
    
    def __init__(self):
        # Node types
        self.SQUARE_NODE = 0
        self.PIECE_NODE = 1
        self.META_NODE = 2  # castling, en-passant, etc.
        
        # Edge types
        self.OCCUPANCY_EDGE = 0
        self.ATTACK_EDGE = 1
        self.DEFENSE_EDGE = 2
        self.PAWN_CHAIN_EDGE = 3
        self.KING_SAFETY_EDGE = 4
        
    def board_to_graph(self, board):
        """Convert a python-chess Board to a PyTorch Geometric graph."""
        # Create nodes
        num_square_nodes = 64
        num_piece_nodes = 12  # 6 piece types x 2 colors
        num_meta_nodes = 5    # 4 castling rights + en-passant
        
        num_nodes = num_square_nodes + num_piece_nodes + num_meta_nodes
        
        # Node features
        node_features = torch.zeros((num_nodes, 16), dtype=torch.float)
        
        # Square nodes (0-63)
        for square in range(64):
            node_idx = square
            rank = square // 8
            file = square % 8
            
            # Basic square features
            node_features[node_idx, 0] = 1.0  # Square node type
            node_features[node_idx, 1] = rank / 7.0  # Normalized rank
            node_features[node_idx, 2] = file / 7.0  # Normalized file
            node_features[node_idx, 3] = (rank + file) % 2  # Square color
            
            # Center distance
            center_dist = max(abs(rank - 3.5), abs(file - 3.5)) / 3.5
            node_features[node_idx, 4] = center_dist
            
            # Square content
            piece = board.piece_at(square)
            if piece:
                piece_idx = self._piece_to_idx(piece)
                
                # Occupancy feature
                node_features[node_idx, 5] = 1.0
                
                # Piece type and color
                node_features[node_idx, 6 + piece_idx] = 1.0
        
        # Piece nodes (64-75)
        for piece_idx in range(12):
            node_idx = num_square_nodes + piece_idx
            piece_type = (piece_idx % 6) + 1  # 1-6
            color = piece_idx // 6  # 0 or 1
            
            # Basic piece features
            node_features[node_idx, 0] = 0.0  # Not a square node
            node_features[node_idx, 1] = 1.0  # Piece node type
            
            # Piece type and color
            node_features[node_idx, 6 + piece_idx] = 1.0
            
            # Count of pieces on board
            piece_count = sum(1 for square in chess.SQUARES 
                             if board.piece_at(square) and 
                                board.piece_at(square).piece_type == piece_type and
                                board.piece_at(square).color == (color == 1))
            
            node_features[node_idx, 2] = piece_count / 8.0  # Normalized count
        
        # Meta nodes (76-80)
        meta_start_idx = num_square_nodes + num_piece_nodes
        
        # Castling rights nodes
        for i, castle_right in enumerate(["K", "Q", "k", "q"]):
            node_idx = meta_start_idx + i
            node_features[node_idx, 0] = 0.0  # Not a square node
            node_features[node_idx, 1] = 0.0  # Not a piece node
            node_features[node_idx, 2] = 1.0  # Meta node
            
            # Castling right status
            has_right = board.has_castling_rights(i < 2)  # White for 0,1; Black for 2,3
            kingside = i % 2 == 0
            queenside = i % 2 == 1
            
            if (kingside and has_right and 'K' in board.castling_rights) or \
               (queenside and has_right and 'Q' in board.castling_rights) or \
               (kingside and has_right and 'k' in board.castling_rights) or \
               (queenside and has_right and 'q' in board.castling_rights):
                node_features[node_idx, 3] = 1.0
        
        # En-passant node
        ep_node_idx = meta_start_idx + 4
        node_features[ep_node_idx, 0] = 0.0  # Not a square node
        node_features[ep_node_idx, 1] = 0.0  # Not a piece node
        node_features[ep_node_idx, 2] = 1.0  # Meta node
        
        # En-passant status
        if board.ep_square is not None:
            node_features[ep_node_idx, 4] = 1.0
            rank = board.ep_square // 8
            file = board.ep_square % 8
            node_features[ep_node_idx, 5] = rank / 7.0
            node_features[ep_node_idx, 6] = file / 7.0
        
        # Create edges
        edge_index_list = []
        edge_attr_list = []
        
        # Occupancy edges: connect squares to pieces
        for square in range(64):
            piece = board.piece_at(square)
            if piece:
                piece_idx = self._piece_to_idx(piece)
                piece_node_idx = num_square_nodes + piece_idx
                
                # Square -> Piece
                edge_index_list.append([square, piece_node_idx])
                edge_attr = torch.zeros(5)
                edge_attr[self.OCCUPANCY_EDGE] = 1.0
                edge_attr_list.append(edge_attr)
                
                # Piece -> Square
                edge_index_list.append([piece_node_idx, square])
                edge_attr_list.append(edge_attr.clone())
        
        # Attack and defense edges
        for from_square in range(64):
            from_piece = board.piece_at(from_square)
            if from_piece:
                # Get all squares the piece attacks
                for to_square in self._get_attacked_squares(board, from_square):
                    to_piece = board.piece_at(to_square)
                    
                    # Attack edge
                    edge_index_list.append([from_square, to_square])
                    edge_attr = torch.zeros(5)
                    edge_attr[self.ATTACK_EDGE] = 1.0
                    edge_attr_list.append(edge_attr)
                    
                    # If square has opponent's piece, create attack edge to piece node too
                    if to_piece and to_piece.color != from_piece.color:
                        to_piece_idx = self._piece_to_idx(to_piece)
                        to_piece_node_idx = num_square_nodes + to_piece_idx
                        
                        edge_index_list.append([from_square, to_piece_node_idx])
                        edge_attr_list.append(edge_attr.clone())
                    
                    # If same color piece, it's a defense
                    if to_piece and to_piece.color == from_piece.color:
                        edge_index_list.append([from_square, to_square])
                        edge_attr = torch.zeros(5)
                        edge_attr[self.DEFENSE_EDGE] = 1.0
                        edge_attr_list.append(edge_attr)
        
        # Pawn chain edges
        for square in range(64):
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                # Add pawn chain connections
                pawn_defenders = self._get_pawn_chain_squares(board, square)
                for defender_square in pawn_defenders:
                    edge_index_list.append([square, defender_square])
                    edge_attr = torch.zeros(5)
                    edge_attr[self.PAWN_CHAIN_EDGE] = 1.0
                    edge_attr_list.append(edge_attr)
        
        # King safety zone edges
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is not None:
                # Create king safety zone
                safety_squares = self._get_king_safety_squares(board, king_square)
                for safety_square in safety_squares:
                    edge_index_list.append([king_square, safety_square])
                    edge_attr = torch.zeros(5)
                    edge_attr[self.KING_SAFETY_EDGE] = 1.0
                    edge_attr_list.append(edge_attr)
        
        # Create PyG graph
        if edge_index_list:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.stack(edge_attr_list)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 5), dtype=torch.float)
        
        # Add global features
        global_features = torch.zeros(8, dtype=torch.float)
        global_features[0] = 1.0 if board.turn == chess.WHITE else 0.0  # Side to move
        global_features[1] = board.fullmove_number / 100.0  # Normalized move number
        global_features[2] = board.halfmove_clock / 100.0  # Normalized halfmove clock
        
        # Material balance
        material_balance = self._calculate_material_balance(board)
        global_features[3] = material_balance / 39.0  # Normalized by queen + 2 rooks value
        
        # Phase of game (opening, middlegame, endgame)
        phase = self._calculate_game_phase(board)
        global_features[4:7] = phase  # One-hot encoding of phase
        
        # Result graph
        graph = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
            global_features=global_features,
            num_nodes=num_nodes
        )
        
        return graph
    
    def _piece_to_idx(self, piece):
        """Convert a chess.Piece to an index (0-11)."""
        base = 0 if piece.color == chess.WHITE else 6
        type_idx = piece.piece_type - 1
        return base + type_idx
    
    def _get_attacked_squares(self, board, square):
        """Get all squares attacked by the piece at the given square."""
        piece = board.piece_at(square)
        if not piece:
            return []
        
        attacks = set()
        for target in range(64):
            if board.is_attacked_by(piece.color, target) and \
               board.is_attacked_by(piece.color, target, from_mask=chess.BB_SQUARES[square]):
                attacks.add(target)
        
        return list(attacks)
    
    def _get_pawn_chain_squares(self, board, square):
        """Get squares that form pawn chains with the given pawn."""
        piece = board.piece_at(square)
        if not piece or piece.piece_type != chess.PAWN:
            return []
        
        rank = square // 8
        file = square % 8
        
        chain_squares = []
        directions = [-9, -7, 7, 9] if piece.color == chess.WHITE else [-9, -7, 7, 9]
        
        for direction in directions:
            target = square + direction
            if 0 <= target < 64:
                target_rank = target // 8
                target_file = target % 8
                
                # Check valid diagonal move
                if abs(target_file - file) == 1 and \
                   ((piece.color == chess.WHITE and target_rank - rank == 1) or
                    (piece.color == chess.BLACK and rank - target_rank == 1)):
                    target_piece = board.piece_at(target)
                    if target_piece and target_piece.piece_type == chess.PAWN and \
                       target_piece.color == piece.color:
                        chain_squares.append(target)
        
        return chain_squares
    
    def _get_king_safety_squares(self, board, king_square):
        """Get squares that form the king safety zone."""
        if king_square is None:
            return []
        
        rank = king_square // 8
        file = king_square % 8
        
        safety_squares = []
        for r in range(max(0, rank - 1), min(8, rank + 2)):
            for f in range(max(0, file - 1), min(8, file + 2)):
                square = r * 8 + f
                if square != king_square:
                    safety_squares.append(square)
        
        return safety_squares
    
    def _calculate_material_balance(self, board):
        """Calculate material balance from white's perspective."""
        values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0  # King has infinite value but not counted for material
        }
        
        white_material = 0
        black_material = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                if piece.color == chess.WHITE:
                    white_material += values[piece.piece_type]
                else:
                    black_material += values[piece.piece_type]
        
        return white_material - black_material
    
    def _calculate_game_phase(self, board):
        """Calculate game phase (opening, middlegame, endgame)."""
        # Count total pieces
        total_pieces = sum(1 for square in chess.SQUARES if board.piece_at(square) is not None)
        
        # Check if queens are present
        has_queens = any(board.pieces(chess.QUEEN, chess.WHITE)) or any(board.pieces(chess.QUEEN, chess.BLACK))
        
        # Calculate phase
        if board.fullmove_number <= 10:
            return torch.tensor([1.0, 0.0, 0.0])  # Opening
        elif not has_queens or total_pieces <= 12:
            return torch.tensor([0.0, 0.0, 1.0])  # Endgame
        else:
            return torch.tensor([0.0, 1.0, 0.0])  # Middlegame


class GNNEncoder(nn.Module):
    """Graph Neural Network encoder for chess positions."""
    
    def __init__(self, node_features=16, edge_features=5, hidden_dim=128, num_layers=6):
        super(GNNEncoder, self).__init__()
        
        self.node_encoder = nn.Linear(node_features, hidden_dim)
        self.edge_encoder = nn.Linear(edge_features, hidden_dim)
        
        # Message passing layers with residual connections and batch norm
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.gnn_layers.append(
                gnn.GraphConv(hidden_dim, hidden_dim)
            )
        
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Global attention pooling
        self.global_attention = gnn.GlobalAttention(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        )
        
        # Global feature integration
        self.global_feature_encoder = nn.Linear(8, hidden_dim)
        self.global_integrator = nn.Linear(2 * hidden_dim, hidden_dim)
        
    def forward(self, data):
        x, edge_index, edge_attr, global_features = (
            data.x, data.edge_index, data.edge_attr, data.global_features
        )
        
        # Initial encodings
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr) if edge_attr is not None else None
        
        # Message passing with residual connections
        for i, gnn_layer in enumerate(self.gnn_layers):
            identity = x
            if edge_attr is not None:
                x = gnn_layer(x, edge_index, edge_attr)
            else:
                x = gnn_layer(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x + identity)  # Residual connection
        
        # Global graph embedding
        global_embedding = self.global_attention(x, batch=data.batch)
        
        # Integrate global features
        global_features_encoded = self.global_feature_encoder(global_features)
        global_embedding = self.global_integrator(
            torch.cat([global_embedding, global_features_encoded], dim=1)
        )
        
        return x, global_embedding 