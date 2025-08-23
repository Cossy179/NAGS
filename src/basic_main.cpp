#include "Board.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <iomanip>

static std::vector<std::string> split(const std::string &s) {
    std::istringstream iss(s);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

// Search parameters
struct SearchInfo {
    std::chrono::steady_clock::time_point start_time;
    long long allocated_time_ms = 0;
    int max_depth = 6;
    bool time_up = false;
    int nodes_searched = 0;
    std::vector<Move> pv; // Principal variation
};

// Piece values for evaluation
static const int PIECE_VALUES[] = {100, 320, 330, 500, 900, 20000}; // P, N, B, R, Q, K

// Piece-square tables (from white's perspective)
static const int PST_PAWN[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
    50, 50, 50, 50, 50, 50, 50, 50,
    10, 10, 20, 30, 30, 20, 10, 10,
     5,  5, 10, 25, 25, 10,  5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5, -5,-10,  0,  0,-10, -5,  5,
     5, 10, 10,-20,-20, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0
};

static const int PST_KNIGHT[64] = {
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50
};

static const int PST_BISHOP[64] = {
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -20,-10,-10,-10,-10,-10,-10,-20
};

static const int PST_ROOK[64] = {
     0,  0,  0,  0,  0,  0,  0,  0,
     5, 10, 10, 10, 10, 10, 10,  5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     0,  0,  0,  5,  5,  0,  0,  0
};

static const int PST_QUEEN[64] = {
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -10,  0,  5,  5,  5,  5,  0,-10,
     -5,  0,  5,  5,  5,  5,  0, -5,
      0,  0,  5,  5,  5,  5,  0, -5,
    -10,  5,  5,  5,  5,  5,  0,-10,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20
};

static const int PST_KING[64] = {
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -10,-20,-20,-20,-20,-20,-20,-10,
     20, 20,  0,  0,  0,  0, 20, 20,
     20, 30, 10,  0,  0, 10, 30, 20
};

static const int* PST_TABLES[] = {PST_PAWN, PST_KNIGHT, PST_BISHOP, PST_ROOK, PST_QUEEN, PST_KING};

// Forward declarations
class SearchEngine;
int evaluate(const Board& board);
int getPieceValue(Piece piece);
int getPieceSquareValue(Piece piece, int square);
void orderMoves(std::vector<Move>& moves, const Board& board);
int getMVVLVAScore(const Move& move, const Board& board);

// Search engine class
class SearchEngine {
public:
    SearchEngine() = default;
    
    Move search(Board& board, SearchInfo& info) {
        info.nodes_searched = 0;
        info.pv.clear();
        
        auto legal_moves = board.generateLegalMoves();
        if (legal_moves.empty()) {
            return Move{-1, -1, Piece::None, false, false};
        }
        
        if (legal_moves.size() == 1) {
            return legal_moves[0];
        }
        
        Move best_move = legal_moves[0];
        int best_score = -999999;
        
        // Iterative deepening
        for (int depth = 1; depth <= info.max_depth; ++depth) {
            if (isTimeUp(info)) break;
            
            int alpha = -999999;
            int beta = 999999;
            Move current_best;
            std::vector<Move> current_pv;
            
            orderMoves(legal_moves, board);
            
            for (const auto& move : legal_moves) {
                if (isTimeUp(info)) break;
                
                board.makeMove(move);
                int score = -negamax(board, depth - 1, -beta, -alpha, info, current_pv);
                board.unmakeMove();
                
                if (score > best_score) {
                    best_score = score;
                    best_move = move;
                    info.pv.clear();
                    info.pv.push_back(move);
                    info.pv.insert(info.pv.end(), current_pv.begin(), current_pv.end());
                }
                
                alpha = std::max(alpha, score);
                if (alpha >= beta) break;
            }
            
            // Print search info
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - info.start_time).count();
            
            std::cout << "info depth " << depth 
                      << " score cp " << best_score
                      << " nodes " << info.nodes_searched
                      << " time " << elapsed
                      << " pv ";
            for (const auto& pv_move : info.pv) {
                std::cout << Board::moveToUci(pv_move) << " ";
            }
            std::cout << std::endl;
        }
        
        return best_move;
    }
    
private:
    int negamax(Board& board, int depth, int alpha, int beta, SearchInfo& info, std::vector<Move>& pv) {
        info.nodes_searched++;
        pv.clear();
        
        if (isTimeUp(info)) return alpha;
        
        if (depth <= 0) {
            return quiescence(board, alpha, beta, info);
        }
        
        auto legal_moves = board.generateLegalMoves();
        if (legal_moves.empty()) {
            return board.inCheck() ? -999999 + (info.max_depth - depth) : 0; // Checkmate or stalemate
        }
        
        orderMoves(legal_moves, board);
        
        int best_score = -999999;
        std::vector<Move> best_pv;
        
        for (const auto& move : legal_moves) {
            if (isTimeUp(info)) break;
            
            board.makeMove(move);
            std::vector<Move> child_pv;
            int score = -negamax(board, depth - 1, -beta, -alpha, info, child_pv);
            board.unmakeMove();
            
            if (score > best_score) {
                best_score = score;
                best_pv.clear();
                best_pv.push_back(move);
                best_pv.insert(best_pv.end(), child_pv.begin(), child_pv.end());
            }
            
            alpha = std::max(alpha, score);
            if (alpha >= beta) {
                break; // Beta cutoff
            }
        }
        
        pv = best_pv;
        return best_score;
    }
    
    int quiescence(Board& board, int alpha, int beta, SearchInfo& info) {
        info.nodes_searched++;
        
        if (isTimeUp(info)) return alpha;
        
        int stand_pat = evaluate(board);
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
        
        auto legal_moves = board.generateLegalMoves();
        std::vector<Move> captures;
        
        // Only consider captures and checks in quiescence
        for (const auto& move : legal_moves) {
            if (board.pieceAt(move.to) != Piece::None || move.isEnPassant) {
                captures.push_back(move);
            }
        }
        
        orderMoves(captures, board);
        
        for (const auto& move : captures) {
            if (isTimeUp(info)) break;
            
            board.makeMove(move);
            int score = -quiescence(board, -beta, -alpha, info);
            board.unmakeMove();
            
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        
        return alpha;
    }
    
    bool isTimeUp(const SearchInfo& info) {
        if (info.allocated_time_ms <= 0) return false;
        
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - info.start_time).count();
        
        return elapsed >= info.allocated_time_ms;
    }
};

// Evaluation function
int evaluate(const Board& board) {
    int score = 0;
    
    for (int square = 0; square < 64; ++square) {
        Piece piece = board.pieceAt(square);
        if (piece == Piece::None) continue;
        
        int piece_value = getPieceValue(piece);
        int pst_value = getPieceSquareValue(piece, square);
        
        if (Board::isWhite(piece)) {
            score += piece_value + pst_value;
        } else {
            score -= piece_value + pst_value;
        }
    }
    
    // Return from current side's perspective
    return board.sideToMove() == Color::White ? score : -score;
}

int getPieceValue(Piece piece) {
    int type = static_cast<int>(piece) - 1;
    return PIECE_VALUES[type % 6];
}

int getPieceSquareValue(Piece piece, int square) {
    int type = static_cast<int>(piece) - 1;
    int piece_type = type % 6;
    
    // Flip square for black pieces
    int sq = Board::isBlack(piece) ? (square ^ 56) : square;
    
    return PST_TABLES[piece_type][sq];
}

// Move ordering with MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
void orderMoves(std::vector<Move>& moves, const Board& board) {
    std::stable_sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        int score_a = getMVVLVAScore(a, board);
        int score_b = getMVVLVAScore(b, board);
        return score_a > score_b;
    });
}

int getMVVLVAScore(const Move& move, const Board& board) {
    Piece victim = board.pieceAt(move.to);
    Piece attacker = board.pieceAt(move.from);
    
    if (victim == Piece::None && !move.isEnPassant) {
        return 0; // Not a capture
    }
    
    if (move.isEnPassant) {
        return getPieceValue(Piece::WP) * 100 - getPieceValue(attacker);
    }
    
    return getPieceValue(victim) * 100 - getPieceValue(attacker);
}

long long calculateSearchTime(long long our_time, long long opp_time, int moves_to_go) {
    // Simple time management: use 1/30th of remaining time if no movestogo
    // If movestogo is specified, use time/(movestogo + 5) to leave some buffer
    
    if (moves_to_go > 0) {
        return our_time / (moves_to_go + 5);
    } else {
        return our_time / 30;
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    Board board;
    SearchEngine engine;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        auto tokens = split(line);
        if (tokens.empty()) continue;
        const std::string &cmd = tokens[0];

        if (cmd == "uci") {
            std::cout << "id name NAGS Search\n";
            std::cout << "id author Alex\n";
            std::cout << "uciok\n" << std::flush;
        } else if (cmd == "isready") {
            std::cout << "readyok\n" << std::flush;
        } else if (cmd == "ucinewgame") {
            board.setStartPos();
        } else if (cmd == "position") {
            // position [fen <fen> | startpos ]  moves ...
            size_t i = 1;
            if (i < tokens.size() && tokens[i] == "startpos") {
                board.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                i++;
            } else if (i < tokens.size() && tokens[i] == "fen") {
                std::string fen;
                // FEN is 6 fields; gather until we have 6 fields collected
                int fields = 0;
                i++;
                for (; i < tokens.size() && fields < 6; ++i) {
                    if (!fen.empty()) fen += ' ';
                    fen += tokens[i];
                    fields++;
                }
                board.setFromFEN(fen);
            }
            if (i < tokens.size() && tokens[i] == "moves") {
                ++i;
                std::vector<std::string> moves;
                for (; i < tokens.size(); ++i) moves.push_back(tokens[i]);
                board.applyMovesUCI(moves);
            }
        } else if (cmd == "go") {
            // Parse UCI time controls
            SearchInfo info;
            info.start_time = std::chrono::steady_clock::now();
            info.max_depth = 6; // Default depth
            
            long long wtime = 0, btime = 0;
            int moves_to_go = 0;
            int depth = 0;
            long long movetime = 0;
            
            for (size_t i = 1; i < tokens.size(); ++i) {
                const std::string &param = tokens[i];
                if (param == "wtime" && i + 1 < tokens.size()) {
                    wtime = std::stoll(tokens[++i]);
                } else if (param == "btime" && i + 1 < tokens.size()) {
                    btime = std::stoll(tokens[++i]);
                } else if (param == "movestogo" && i + 1 < tokens.size()) {
                    moves_to_go = std::stoi(tokens[++i]);
                } else if (param == "depth" && i + 1 < tokens.size()) {
                    depth = std::stoi(tokens[++i]);
                } else if (param == "movetime" && i + 1 < tokens.size()) {
                    movetime = std::stoll(tokens[++i]);
                }
            }
            
            // Calculate time allocation
            if (movetime > 0) {
                info.allocated_time_ms = movetime;
            } else if (wtime > 0 || btime > 0) {
                long long our_time = (board.sideToMove() == Color::White) ? wtime : btime;
                long long opp_time = (board.sideToMove() == Color::White) ? btime : wtime;
                info.allocated_time_ms = calculateSearchTime(our_time, opp_time, moves_to_go);
                
                // Minimum and maximum time bounds
                info.allocated_time_ms = std::max(100LL, std::min(info.allocated_time_ms, our_time / 2));
            } else {
                info.allocated_time_ms = 1000; // Default 1 second
            }
            
            if (depth > 0) {
                info.max_depth = depth;
                info.allocated_time_ms = 0; // Ignore time when depth is specified
            }
            
            std::cout << "info string Searching with time " << info.allocated_time_ms 
                      << "ms, max depth " << info.max_depth << std::endl;
            
            Move best_move = engine.search(board, info);
            
            if (best_move.from == -1) {
                std::cout << "bestmove 0000\n" << std::flush;
            } else {
                std::cout << "bestmove " << Board::moveToUci(best_move) << "\n" << std::flush;
            }
        } else if (cmd == "stop") {
            // In a full implementation, this would stop the search
            // For now, we ignore it since our search is synchronous
        } else if (cmd == "quit") {
            break;
        } else if (cmd == "d") {
            // Debug command: display current position
            std::cout << board.getFEN() << "\n";
        }
    }
    return 0;
}
