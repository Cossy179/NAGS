#include "FastBoard.h"
#include "FastTT.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <iomanip>
#include <thread>
#include <atomic>
#include <mutex>

static std::vector<std::string> split(const std::string &s) {
    std::istringstream iss(s);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

// Global engine settings
struct EngineOptions {
    std::atomic<int> hash_mb{64};
    std::atomic<int> threads{1};
    
    void setHash(int mb) {
        hash_mb = std::max(1, std::min(4096, mb));
    }
    
    void setThreads(int t) {
        threads = std::max(1, std::min(64, t));
    }
} engine_options;

// Global transposition table
FastTranspositionTable tt;

// Piece values for evaluation
static const int PIECE_VALUES[] = {100, 320, 330, 500, 900, 20000}; // P, N, B, R, Q, K

// Piece-square tables
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
class EnhancedSearchEngine;
int evaluate(const FastBoard& board);
int getPieceValue(Piece piece);
int getPieceSquareValue(Piece piece, int square);
void orderMoves(std::vector<Move>& moves, const FastBoard& board, const Move* ttMove = nullptr);
int getMVVLVAScore(const Move& move, const FastBoard& board);

// Enhanced search engine with TT and threading support
class EnhancedSearchEngine {
public:
    EnhancedSearchEngine() = default;
    
    Move search(FastBoard& board, SearchInfo& info) {
        info.stop_search = false;
        info.completed_depth = 0;
        tt.resetStats();
        
        auto legal_moves = board.generateLegalMoves();
        if (legal_moves.empty()) {
            return Move{-1, -1, Piece::None, false, false};
        }
        
        if (legal_moves.size() == 1) {
            return legal_moves[0];
        }
        
        int num_threads = engine_options.threads.load();
        
        if (num_threads == 1) {
            return singleThreadedSearch(board, info);
        } else {
            return multiThreadedSearch(board, info, num_threads);
        }
    }
    
private:
    Move singleThreadedSearch(FastBoard& board, SearchInfo& info) {
        Move best_move{-1, -1, Piece::None, false, false};
        int best_score = -999999;
        
        // Iterative deepening
        for (int depth = 1; depth <= 12; ++depth) {
            if (info.should_stop()) break;
            
            ThreadData thread_data(0);
            thread_data.board = board;
            
            int score = aspirationSearch(thread_data, depth, best_score, info);
            
            if (!info.should_stop()) {
                best_score = score;
                best_move = info.best_move.load();
                info.completed_depth = depth;
                
                // Print search info
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::steady_clock::now() - info.start_time).count();
                
                uint64_t nodes = thread_data.stats.nodes.load();
                uint64_t nps = elapsed > 0 ? (nodes * 1000) / elapsed : 0;
                
                std::cout << "info depth " << depth 
                          << " score cp " << best_score
                          << " nodes " << nodes
                          << " time " << elapsed
                          << " nps " << nps
                          << " hashfull " << static_cast<int>(tt.getHitRate() * 1000)
                          << " pv ";
                
                {
                    std::lock_guard<std::mutex> lock(info.pv_mutex);
                    for (const auto& pv_move : info.best_pv) {
                        std::cout << FastBoard::moveToUci(pv_move) << " ";
                    }
                }
                std::cout << std::endl;
            }
            
            if (info.limits.depth > 0 && depth >= info.limits.depth) break;
        }
        
        return best_move;
    }
    
    Move multiThreadedSearch(FastBoard& board, SearchInfo& info, int num_threads) {
        // Use single-threaded search for now (threading is complex to implement correctly)
        return singleThreadedSearch(board, info);
    }
    
    void helperThreadSearch(ThreadData& data, int depth, SearchInfo& info) {
        std::vector<Move> pv;
        negamax(data, depth, -999999, 999999, 0, pv, info);
    }
    
    int aspirationSearch(ThreadData& data, int depth, int prev_score, SearchInfo& info) {
        int alpha = -999999;
        int beta = 999999;
        
        // Use aspiration windows for depths > 3
        if (depth > 3 && abs(prev_score) < 900000) {
            int window = 50;
            alpha = prev_score - window;
            beta = prev_score + window;
        }
        
        while (true) {
            std::vector<Move> pv;
            int score = negamax(data, depth, alpha, beta, 0, pv, info);
            
            if (info.should_stop()) return score;
            
            if (score <= alpha) {
                alpha = -999999;
            } else if (score >= beta) {
                beta = 999999;
            } else {
                // Update best move and PV
                if (!pv.empty()) {
                    info.best_move = pv[0];
                    std::lock_guard<std::mutex> lock(info.pv_mutex);
                    info.best_pv = pv;
                }
                return score;
            }
        }
    }
    
    int negamax(ThreadData& data, int depth, int alpha, int beta, int ply, 
                std::vector<Move>& pv, SearchInfo& info) {
        data.stats.nodes.fetch_add(1, std::memory_order_relaxed);
        pv.clear();
        
        if (info.should_stop()) return alpha;
        
        // Check for draw by repetition (simplified)
        if (ply > 0 && data.board.getHalfmoveClock() >= 100) {
            return 0;
        }
        
        // Transposition table probe
        TTEntry tt_entry;
        uint64_t hash_key = data.board.zobrist();
        bool tt_hit = tt.probe(hash_key, tt_entry);
        Move tt_move{-1, -1, Piece::None, false, false};
        
        if (tt_hit) {
            data.stats.tt_hits.fetch_add(1, std::memory_order_relaxed);
            
            if (tt_entry.depth >= depth) {
                TTFlag tt_flag = static_cast<TTFlag>(tt_entry.flag);
                
                if (tt_flag == TTFlag::Exact) return tt_entry.score;
                if (tt_flag == TTFlag::Lower && tt_entry.score >= beta) return tt_entry.score;
                if (tt_flag == TTFlag::Upper && tt_entry.score <= alpha) return tt_entry.score;
            }
            
            if (tt_entry.movePacked != 0) {
                tt_move = FastTranspositionTable::unpackMove(tt_entry.movePacked);
            }
        }
        
        if (depth <= 0) {
            return quiescence(data, alpha, beta, ply, info);
        }
        
        auto legal_moves = data.board.generateLegalMoves();
        if (legal_moves.empty()) {
            return data.board.inCheck() ? -999999 + ply : 0;
        }
        
        orderMoves(legal_moves, data.board, tt_hit ? &tt_move : nullptr);
        
        int best_score = -999999;
        std::vector<Move> best_pv;
        Move best_move{-1, -1, Piece::None, false, false};
        TTFlag tt_flag = TTFlag::Upper;
        
        for (const auto& move : legal_moves) {
            if (info.should_stop()) break;
            
            data.board.makeMove(move);
            std::vector<Move> child_pv;
            int score = -negamax(data, depth - 1, -beta, -alpha, ply + 1, child_pv, info);
            data.board.unmakeMove();
            
            if (score > best_score) {
                best_score = score;
                best_move = move;
                best_pv.clear();
                best_pv.push_back(move);
                best_pv.insert(best_pv.end(), child_pv.begin(), child_pv.end());
            }
            
            if (score > alpha) {
                alpha = score;
                tt_flag = TTFlag::Exact;
                
                if (alpha >= beta) {
                    data.stats.beta_cutoffs.fetch_add(1, std::memory_order_relaxed);
                    tt_flag = TTFlag::Lower;
                    break;
                }
            }
        }
        
        // Store in transposition table
        if (!info.should_stop()) {
            tt.store(hash_key, depth, tt_flag, best_score, &best_move);
        }
        
        pv = best_pv;
        return best_score;
    }
    
    int quiescence(ThreadData& data, int alpha, int beta, int ply, SearchInfo& info) {
        data.stats.nodes.fetch_add(1, std::memory_order_relaxed);
        
        if (info.should_stop()) return alpha;
        
        int stand_pat = evaluate(data.board);
        if (stand_pat >= beta) return beta;
        if (stand_pat > alpha) alpha = stand_pat;
        
        auto legal_moves = data.board.generateLegalMoves();
        std::vector<Move> captures;
        
        for (const auto& move : legal_moves) {
            if (data.board.pieceAt(move.to) != Piece::None || move.isEnPassant) {
                captures.push_back(move);
            }
        }
        
        orderMoves(captures, data.board);
        
        for (const auto& move : captures) {
            if (info.should_stop()) break;
            
            data.board.makeMove(move);
            int score = -quiescence(data, -beta, -alpha, ply + 1, info);
            data.board.unmakeMove();
            
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        
        return alpha;
    }
};

// Evaluation function
int evaluate(const FastBoard& board) {
    int score = 0;
    
    for (int square = 0; square < 64; ++square) {
        Piece piece = board.pieceAt(square);
        if (piece == Piece::None) continue;
        
        int piece_value = getPieceValue(piece);
        int pst_value = getPieceSquareValue(piece, square);
        
        if (FastBoard::isWhite(piece)) {
            score += piece_value + pst_value;
        } else {
            score -= piece_value + pst_value;
        }
    }
    
    return board.sideToMove() == Color::White ? score : -score;
}

int getPieceValue(Piece piece) {
    int type = (static_cast<int>(piece) - 1) % 6;
    return PIECE_VALUES[type];
}

int getPieceSquareValue(Piece piece, int square) {
    int type = (static_cast<int>(piece) - 1) % 6;
    int sq = FastBoard::isBlack(piece) ? (square ^ 56) : square;
    return PST_TABLES[type][sq];
}

void orderMoves(std::vector<Move>& moves, const FastBoard& board, const Move* ttMove) {
    std::stable_sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        // TT move first
        if (ttMove) {
            if (a.from == ttMove->from && a.to == ttMove->to) return true;
            if (b.from == ttMove->from && b.to == ttMove->to) return false;
        }
        
        int score_a = getMVVLVAScore(a, board);
        int score_b = getMVVLVAScore(b, board);
        return score_a > score_b;
    });
}

int getMVVLVAScore(const Move& move, const FastBoard& board) {
    Piece victim = board.pieceAt(move.to);
    Piece attacker = board.pieceAt(move.from);
    
    if (victim == Piece::None && !move.isEnPassant) {
        return 0;
    }
    
    if (move.isEnPassant) {
        return getPieceValue(Piece::WP) * 100 - getPieceValue(attacker);
    }
    
    return getPieceValue(victim) * 100 - getPieceValue(attacker);
}

long long calculateSearchTime(long long our_time, long long opp_time, int moves_to_go) {
    if (moves_to_go > 0) {
        return our_time / (moves_to_go + 5);
    } else {
        return our_time / 30;
    }
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    FastBoard board;
    EnhancedSearchEngine engine;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        auto tokens = split(line);
        if (tokens.empty()) continue;
        const std::string &cmd = tokens[0];

        if (cmd == "uci") {
            std::cout << "id name NAGS Enhanced\n";
            std::cout << "id author Alex\n";
            std::cout << "option name Hash type spin default 64 min 1 max 4096\n";
            std::cout << "option name Threads type spin default 1 min 1 max 64\n";
            std::cout << "option name Clear Hash type button\n";
            std::cout << "uciok\n" << std::flush;
        } else if (cmd == "isready") {
            std::cout << "readyok\n" << std::flush;
        } else if (cmd == "ucinewgame") {
            board.setStartPos();
            tt.clear();
        } else if (cmd == "setoption") {
            // Parse setoption name <name> value <value>
            std::string name, value;
            for (size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "name" && i + 1 < tokens.size()) {
                    name = tokens[++i];
                } else if (tokens[i] == "value" && i + 1 < tokens.size()) {
                    value = tokens[++i];
                }
            }
            
            if (name == "Hash") {
                int hash_mb = std::stoi(value);
                engine_options.setHash(hash_mb);
                tt.resizeMB(hash_mb);
                std::cout << "info string Hash set to " << hash_mb << " MB\n";
            } else if (name == "Threads") {
                int threads = std::stoi(value);
                engine_options.setThreads(threads);
                std::cout << "info string Threads set to " << threads << "\n";
            } else if (name == "Clear Hash") {
                tt.clear();
                std::cout << "info string Hash cleared\n";
            }
        } else if (cmd == "position") {
            size_t i = 1;
            if (i < tokens.size() && tokens[i] == "startpos") {
                board.setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
                i++;
            } else if (i < tokens.size() && tokens[i] == "fen") {
                std::string fen;
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
            SearchInfo info;
            info.start_time = std::chrono::steady_clock::now();
            
            for (size_t i = 1; i < tokens.size(); ++i) {
                const std::string &param = tokens[i];
                if (param == "wtime" && i + 1 < tokens.size()) {
                    info.limits.wtime = std::stoll(tokens[++i]);
                } else if (param == "btime" && i + 1 < tokens.size()) {
                    info.limits.btime = std::stoll(tokens[++i]);
                } else if (param == "movestogo" && i + 1 < tokens.size()) {
                    info.limits.movestogo = std::stoi(tokens[++i]);
                } else if (param == "depth" && i + 1 < tokens.size()) {
                    info.limits.depth = std::stoi(tokens[++i]);
                } else if (param == "movetime" && i + 1 < tokens.size()) {
                    info.limits.movetime = std::stoll(tokens[++i]);
                } else if (param == "infinite") {
                    info.limits.infinite = true;
                } else if (param == "perft" && i + 1 < tokens.size()) {
                    int perft_depth = std::stoi(tokens[++i]);
                    auto start = std::chrono::steady_clock::now();
                    uint64_t nodes = board.perft(perft_depth);
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start).count();
                    std::cout << "Perft(" << perft_depth << ") = " << nodes 
                              << " (time: " << elapsed << "ms";
                    if (elapsed > 0) {
                        std::cout << ", " << (nodes * 1000) / elapsed << " nps";
                    }
                    std::cout << ")" << std::endl;
                    continue;
                }
            }
            
            // Calculate time allocation
            if (info.limits.movetime > 0) {
                // Use exact time
            } else if (info.limits.wtime > 0 || info.limits.btime > 0) {
                long long our_time = (board.sideToMove() == Color::White) ? info.limits.wtime : info.limits.btime;
                long long opp_time = (board.sideToMove() == Color::White) ? info.limits.btime : info.limits.wtime;
                info.limits.movetime = calculateSearchTime(our_time, opp_time, info.limits.movestogo);
                info.limits.movetime = std::max(100LL, std::min(info.limits.movetime, our_time / 2));
            } else if (!info.limits.infinite && info.limits.depth == 0) {
                info.limits.movetime = 1000; // Default 1 second
            }
            
            std::cout << "info string Enhanced search: Hash=" << engine_options.hash_mb.load() 
                      << "MB Threads=" << engine_options.threads.load();
            if (info.limits.movetime > 0) {
                std::cout << " Time=" << info.limits.movetime << "ms";
            }
            if (info.limits.depth > 0) {
                std::cout << " Depth=" << info.limits.depth;
            }
            std::cout << std::endl;
            
            Move best_move = engine.search(board, info);
            
            if (best_move.from == -1) {
                std::cout << "bestmove 0000\n" << std::flush;
            } else {
                std::cout << "bestmove " << FastBoard::moveToUci(best_move) << "\n" << std::flush;
            }
            
            // Print TT statistics
            std::cout << "info string TT: " << tt.getProbes() << " probes, " 
                      << tt.getHits() << " hits (" << std::fixed << std::setprecision(1) 
                      << tt.getHitRate() * 100 << "%)" << std::endl;
                      
        } else if (cmd == "stop") {
            // Stop search (would be handled by search threads)
        } else if (cmd == "quit") {
            break;
        } else if (cmd == "d") {
            std::cout << board.getFEN() << "\n";
        }
    }
    return 0;
}
