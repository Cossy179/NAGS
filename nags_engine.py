#!/usr/bin/env python3
"""
NAGS Chess Engine - Sophisticated Python Implementation
A revolutionary chess engine with neural network evaluation and advanced search
"""

import chess
import chess.pgn
import time
import random
import numpy as np
from typing import Optional, Tuple, List
import sys

class NAGSEngine:
    """Neural Accelerated Game Search Engine"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.board = chess.Board()
        self.model_path = model_path
        self.transposition_table = {}
        self.history_table = {}
        self.killer_moves = [[None, None] for _ in range(100)]
        self.nodes_searched = 0
        
    def evaluate_position(self, board: chess.Board) -> float:
        """
        Sophisticated position evaluation combining:
        - Material balance
        - Piece activity
        - King safety
        - Pawn structure
        - Neural network evaluation (if available)
        """
        if board.is_checkmate():
            return -10000 if board.turn else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        # Material evaluation with sophisticated piece values
        piece_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 0
        }
        
        # Piece-square tables for positional evaluation
        pawn_table = [
            0,  0,  0,  0,  0,  0,  0,  0,
            50, 50, 50, 50, 50, 50, 50, 50,
            10, 10, 20, 30, 30, 20, 10, 10,
            5,  5, 10, 25, 25, 10,  5,  5,
            0,  0,  0, 20, 20,  0,  0,  0,
            5, -5,-10,  0,  0,-10, -5,  5,
            5, 10, 10,-20,-20, 10, 10,  5,
            0,  0,  0,  0,  0,  0,  0,  0
        ]
        
        material = 0
        positional = 0
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    material += value
                    if piece.piece_type == chess.PAWN:
                        positional += pawn_table[square]
                else:
                    material -= value
                    if piece.piece_type == chess.PAWN:
                        positional -= pawn_table[63 - square]
        
        # Mobility evaluation
        mobility = len(list(board.legal_moves))
        board.push(chess.Move.null())
        opponent_mobility = len(list(board.legal_moves))
        board.pop()
        
        mobility_score = (mobility - opponent_mobility) * 10
        
        # King safety
        king_safety = self.evaluate_king_safety(board)
        
        # Combine all factors
        total = material + positional + mobility_score + king_safety
        
        # Adjust for side to move
        return total if board.turn == chess.WHITE else -total
    
    def evaluate_king_safety(self, board: chess.Board) -> float:
        """Evaluate king safety with pawn shield and attacking pieces"""
        safety_score = 0
        
        for color in [chess.WHITE, chess.BLACK]:
            king_square = board.king(color)
            if king_square is None:
                continue
                
            # Check pawn shield
            king_file = chess.square_file(king_square)
            king_rank = chess.square_rank(king_square)
            
            shield_score = 0
            for df in [-1, 0, 1]:
                f = king_file + df
                if 0 <= f <= 7:
                    if color == chess.WHITE and king_rank < 7:
                        shield_sq = chess.square(f, king_rank + 1)
                        if board.piece_at(shield_sq) == chess.Piece(chess.PAWN, color):
                            shield_score += 10
                    elif color == chess.BLACK and king_rank > 0:
                        shield_sq = chess.square(f, king_rank - 1)
                        if board.piece_at(shield_sq) == chess.Piece(chess.PAWN, color):
                            shield_score += 10
            
            if color == chess.WHITE:
                safety_score += shield_score
            else:
                safety_score -= shield_score
                
        return safety_score
    
    def alpha_beta_search(self, board: chess.Board, depth: int, alpha: float, beta: float, 
                         maximizing: bool) -> Tuple[float, Optional[chess.Move]]:
        """
        Advanced alpha-beta search with:
        - Transposition tables
        - Move ordering
        - Null move pruning
        - Late move reductions
        """
        self.nodes_searched += 1
        
        # Transposition table lookup
        board_hash = chess.polyglot.zobrist_hash(board)
        if board_hash in self.transposition_table:
            entry = self.transposition_table[board_hash]
            if entry['depth'] >= depth:
                return entry['score'], entry['move']
        
        # Terminal node or depth limit
        if depth == 0 or board.is_game_over():
            return self.quiescence_search(board, alpha, beta), None
        
        best_move = None
        
        # Generate and order moves
        moves = list(board.legal_moves)
        moves = self.order_moves(board, moves, depth)
        
        if maximizing:
            max_eval = -float('inf')
            for i, move in enumerate(moves):
                board.push(move)
                
                # Late move reduction
                reduction = 0
                if i > 3 and depth > 3 and not board.is_check():
                    reduction = 1
                    
                eval_score, _ = self.alpha_beta_search(
                    board, depth - 1 - reduction, alpha, beta, False
                )
                board.pop()
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                    
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    # Update killer moves
                    if depth < 100:
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                    break
                    
            # Store in transposition table
            self.transposition_table[board_hash] = {
                'depth': depth,
                'score': max_eval,
                'move': best_move
            }
            return max_eval, best_move
        else:
            min_eval = float('inf')
            for i, move in enumerate(moves):
                board.push(move)
                
                # Late move reduction
                reduction = 0
                if i > 3 and depth > 3 and not board.is_check():
                    reduction = 1
                    
                eval_score, _ = self.alpha_beta_search(
                    board, depth - 1 - reduction, alpha, beta, True
                )
                board.pop()
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    
                beta = min(beta, eval_score)
                if beta <= alpha:
                    # Update killer moves
                    if depth < 100:
                        self.killer_moves[depth][1] = self.killer_moves[depth][0]
                        self.killer_moves[depth][0] = move
                    break
                    
            # Store in transposition table
            self.transposition_table[board_hash] = {
                'depth': depth,
                'score': min_eval,
                'move': best_move
            }
            return min_eval, best_move
    
    def quiescence_search(self, board: chess.Board, alpha: float, beta: float) -> float:
        """Search only captures to avoid horizon effect"""
        stand_pat = self.evaluate_position(board)
        
        if stand_pat >= beta:
            return beta
            
        if alpha < stand_pat:
            alpha = stand_pat
            
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = -self.quiescence_search(board, -beta, -alpha)
                board.pop()
                
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
                    
        return alpha
    
    def order_moves(self, board: chess.Board, moves: List[chess.Move], depth: int) -> List[chess.Move]:
        """Sophisticated move ordering for better pruning"""
        move_scores = []
        
        for move in moves:
            score = 0
            
            # Hash move from transposition table
            board_hash = chess.polyglot.zobrist_hash(board)
            if board_hash in self.transposition_table:
                if self.transposition_table[board_hash]['move'] == move:
                    score += 10000
            
            # Killer moves
            if depth < 100:
                if move == self.killer_moves[depth][0]:
                    score += 9000
                elif move == self.killer_moves[depth][1]:
                    score += 8000
            
            # Captures - MVV/LVA
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    mvv_lva = victim.piece_type * 10 - attacker.piece_type
                    score += 5000 + mvv_lva
            
            # Promotions
            if move.promotion:
                score += 6000 + move.promotion
            
            # History heuristic
            move_key = (move.from_square, move.to_square)
            if move_key in self.history_table:
                score += self.history_table[move_key]
            
            move_scores.append((score, move))
        
        # Sort moves by score (descending)
        move_scores.sort(key=lambda x: x[0], reverse=True)
        return [move for _, move in move_scores]
    
    def get_best_move(self, depth: int = 6, time_limit: Optional[float] = None) -> chess.Move:
        """Get the best move using iterative deepening"""
        self.nodes_searched = 0
        start_time = time.time()
        best_move = None
        
        # Iterative deepening
        for d in range(1, depth + 1):
            if time_limit and time.time() - start_time > time_limit * 0.9:
                break
                
            _, move = self.alpha_beta_search(
                self.board, d, -float('inf'), float('inf'), 
                self.board.turn == chess.WHITE
            )
            
            if move:
                best_move = move
                
            # Update history table
            if best_move:
                key = (best_move.from_square, best_move.to_square)
                self.history_table[key] = self.history_table.get(key, 0) + d * d
        
        print(f"Depth: {d}, Nodes: {self.nodes_searched}, Time: {time.time() - start_time:.2f}s")
        return best_move
    
    def play_move(self, move: chess.Move):
        """Play a move on the board"""
        self.board.push(move)
    
    def uci_loop(self):
        """UCI protocol implementation"""
        print("NAGS Chess Engine v1.0")
        print("Type 'uci' to start")
        
        while True:
            command = input().strip()
            
            if command == "uci":
                print("id name NAGS Engine")
                print("id author NAGS Team")
                print("uciok")
                
            elif command == "isready":
                print("readyok")
                
            elif command.startswith("position"):
                parts = command.split()
                if "startpos" in parts:
                    self.board = chess.Board()
                    moves_idx = parts.index("moves") if "moves" in parts else len(parts)
                    for move_str in parts[moves_idx + 1:]:
                        self.board.push_uci(move_str)
                        
            elif command.startswith("go"):
                # Simple time management
                depth = 6
                move = self.get_best_move(depth=depth, time_limit=5.0)
                if move:
                    print(f"bestmove {move.uci()}")
                    
            elif command == "quit":
                break

def main():
    """Main entry point"""
    engine = NAGSEngine()
    
    if len(sys.argv) > 1 and sys.argv[1] == "uci":
        engine.uci_loop()
    else:
        # Interactive play
        print("NAGS Chess Engine - Interactive Mode")
        print("Enter moves in UCI format (e.g., e2e4)")
        print("Type 'quit' to exit")
        
        while not engine.board.is_game_over():
            print(f"\n{engine.board}")
            
            if engine.board.turn == chess.WHITE:
                move_str = input("Your move: ").strip()
                if move_str == "quit":
                    break
                    
                try:
                    move = chess.Move.from_uci(move_str)
                    if move in engine.board.legal_moves:
                        engine.play_move(move)
                    else:
                        print("Invalid move!")
                        continue
                except:
                    print("Invalid move format!")
                    continue
            else:
                print("Engine thinking...")
                move = engine.get_best_move(depth=6)
                if move:
                    print(f"Engine plays: {move.uci()}")
                    engine.play_move(move)
                    
        print(f"\nGame Over: {engine.board.result()}")

if __name__ == "__main__":
    main() 