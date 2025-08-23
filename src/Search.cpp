#include "Search.h"

#include <algorithm>
#include <chrono>
#include <cstdint>

using Clock = std::chrono::steady_clock;

static int pieceValue(Piece p) {
    switch (p) {
        case Piece::WP: case Piece::BP: return 100;
        case Piece::WN: case Piece::BN: return 320;
        case Piece::WB: case Piece::BB: return 330;
        case Piece::WR: case Piece::BR: return 500;
        case Piece::WQ: case Piece::BQ: return 900;
        case Piece::WK: case Piece::BK: return 0; // king evaluated via safety
        default: return 0;
    }
}

Search::Search(Board &board) : b(board) {}

int Search::eval() const {
    // Simple material count
    int score = 0;
    for (int sq = 0; sq < 128; ++sq) {
        if (!Board::isValidSquare(sq)) { sq = (sq | 15) + 1; continue; }
        Piece p = b.pieceAt(sq);
        if (p == Piece::None) continue;
        int val = pieceValue(p);
        if (Board::isWhite(p)) score += val; else score -= val;
    }
    // From side to move perspective (negamax convention expects eval from side to move)
    return (b.sideToMove() == Color::White) ? score : -score;
}

int Search::mvvLvaScore(Piece victim, Piece attacker) {
    static int victimScore[7] = {0, 100, 320, 330, 500, 900, 10000};
    auto idx = [&](Piece p) {
        switch (p) {
            case Piece::WP: case Piece::BP: return 1;
            case Piece::WN: case Piece::BN: return 2;
            case Piece::WB: case Piece::BB: return 3;
            case Piece::WR: case Piece::BR: return 4;
            case Piece::WQ: case Piece::BQ: return 5;
            case Piece::WK: case Piece::BK: return 6;
            default: return 0;
        }
    };
    return victimScore[idx(victim)] * 10 - victimScore[idx(attacker)];
}

void Search::orderMoves(std::vector<Move> &moves) {
    // Simple: captures first using MVV/LVA; others after
    std::stable_sort(moves.begin(), moves.end(), [&](const Move &a, const Move &bmv) {
        Piece av = b.pieceAt(a.to);
        Piece bv = b.pieceAt(bmv.to);
        bool aCap = av != Piece::None || a.isEnPassant;
        bool bCap = bv != Piece::None || bmv.isEnPassant;
        if (aCap != bCap) return aCap; // captures first
        if (aCap && bCap) {
            Piece aa = b.pieceAt(a.from);
            Piece bb = b.pieceAt(bmv.from);
            int as = mvvLvaScore(av, aa);
            int bs = mvvLvaScore(bv, bb);
            return as > bs;
        }
        // Non-captures: prefer promotions and castling
        if (a.promotion != bmv.promotion) return a.promotion != Piece::None;
        if (a.isCastling != bmv.isCastling) return a.isCastling;
        return false;
    });
}

int Search::quiescence(int alpha, int beta, int ply) {
    if (stopFlag.load(std::memory_order_relaxed) || timeUp()) return alpha;
    if (!b.inCheck()) {
        int stand = eval();
        if (stand >= beta) return beta;
        if (stand > alpha) alpha = stand;

        // Generate capture moves only
        auto all = b.generateLegalMoves();
        std::vector<Move> caps;
        caps.reserve(all.size());
        for (const auto &m : all) {
            if (m.isEnPassant) caps.push_back(m);
            else if (b.pieceAt(m.to) != Piece::None) caps.push_back(m);
        }
        orderMoves(caps);
        for (const auto &m : caps) {
            b.makeMove(m);
            int score = -quiescence(-beta, -alpha, ply + 1);
            b.unmakeMove();
            if (score >= beta) return beta;
            if (score > alpha) alpha = score;
        }
        return alpha;
    } else {
        // If in check, we must consider all legal evasion moves (no stand-pat)
        auto moves = b.generateLegalMoves();
        if (moves.empty()) return -100000 + ply; // checkmate
        orderMoves(moves);
        int best = -1000000;
        for (const auto &m : moves) {
            b.makeMove(m);
            int score = -quiescence(-beta, -alpha, ply + 1);
            b.unmakeMove();
            if (score > best) best = score;
            if (score > alpha) {
                alpha = score;
                if (alpha >= beta) break;
            }
        }
        return alpha;
    }
}

int Search::alphaBeta(int depth, int alpha, int beta, int ply) {
    if (stopFlag.load(std::memory_order_relaxed) || timeUp()) return alpha;
    if (depth <= 0) {
        return quiescence(alpha, beta, ply);
    }
    // TT probe
    uint64_t key = b.zobrist();
    TTEntry tte;
    if (tt.probe(key, tte) && tte.depth >= depth) {
        int ttScore = tte.score;
        if (tte.flag == static_cast<uint8_t>(TTFlag::Exact)) return ttScore;
        if (tte.flag == static_cast<uint8_t>(TTFlag::Lower) && ttScore > alpha) alpha = ttScore;
        else if (tte.flag == static_cast<uint8_t>(TTFlag::Upper) && ttScore < beta) beta = ttScore;
        if (alpha >= beta) return ttScore;
    }

    auto moves = b.generateLegalMoves();
    if (moves.empty()) {
        if (b.inCheck()) return -100000 + ply; // checkmated (avoid horizon artifacts with ply)
        return 0; // stalemate
    }
    orderMoves(moves);

    int best = -1000000;
    Move bestMove{ -1, -1, Piece::None, false, false };
    for (const auto &m : moves) {
        b.makeMove(m);
        int score = -alphaBeta(depth - 1, -beta, -alpha, ply + 1);
        b.unmakeMove();
        if (score > best) best = score;
        if (score > alpha) {
            alpha = score;
            bestMove = m;
            if (alpha >= beta) break; // cutoff
        }
        if (stopFlag.load(std::memory_order_relaxed) || timeUp()) break;
    }
    // Store in TT
    TTFlag flag = TTFlag::Exact;
    // Note: In a full implementation we should track original alpha/beta; simplified here
    tt.store(key, depth, flag, best, bestMove.from != -1 ? &bestMove : nullptr);
    return alpha;
}

void Search::startTimer(int64_t msBudget) {
    budgetMs = msBudget;
    startTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
}

bool Search::timeUp() const {
    if (budgetMs <= 0) return false;
    int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
    return (now - startTimeMs) >= budgetMs;
}

void Search::stop() { stopFlag.store(true, std::memory_order_relaxed); }

SearchResult Search::go(const SearchLimits &limits) {
    stopFlag.store(false, std::memory_order_relaxed);

    int64_t timeBudget = 0;
    if (limits.depth > 0) {
        timeBudget = 0;
    } else if (limits.timeMs > 0) {
        int div = limits.movesToGo > 0 ? limits.movesToGo : 30;
        timeBudget = std::max<int64_t>(1, limits.timeMs / div);
        // Keep a small safety margin
        timeBudget = (timeBudget * 9) / 10;
    }
    startTimer(timeBudget);

    SearchResult res;
    Move bestLocal{ -1, -1, Piece::None, false, false };

    int maxDepth = limits.depth > 0 ? limits.depth : 64;
    for (int depth = 1; depth <= maxDepth; ++depth) {
        int alpha = -1000000, beta = 1000000;
        auto moves = b.generateLegalMoves();
        orderMoves(moves);
        int bestScore = -1000000;
        Move bestAtDepth = bestLocal;
        for (const auto &m : moves) {
            b.makeMove(m);
            int score = -alphaBeta(depth - 1, -beta, -alpha, 1);
            b.unmakeMove();
            if (stopFlag.load(std::memory_order_relaxed) || timeUp()) break;
            if (score > bestScore) {
                bestScore = score;
                bestAtDepth = m;
                if (score > alpha) alpha = score;
            }
        }
        if (bestAtDepth.from != -1) bestLocal = bestAtDepth;
        if (stopFlag.load(std::memory_order_relaxed) || timeUp()) break;
    }

    res.bestMove = bestLocal;
    return res;
}


