#pragma once

#include "Board.h"
#include "TT.h"

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

struct SearchLimits {
    // Time in milliseconds for the side to move
    int64_t timeMs = -1;
    int   movesToGo = 0; // optional
    int   depth = 0;     // if >0, limit by depth else by time
};

struct SearchResult {
    Move bestMove{ -1, -1, Piece::None, false, false };
};

class Search {
public:
    explicit Search(Board &board);

    SearchResult go(const SearchLimits &limits);
    void stop();

    // Config
    void setHashMB(int mb) { tt.resizeMB(static_cast<size_t>(mb)); }

private:
    Board &b;
    TranspositionTable tt;

    std::atomic<bool> stopFlag{false};

    // Iterative deepening with alpha-beta
    int eval() const;
    int quiescence(int alpha, int beta, int ply);
    int alphaBeta(int depth, int alpha, int beta, int ply);

    // Ordering helpers
    static int mvvLvaScore(Piece victim, Piece attacker);
    void orderMoves(std::vector<Move> &moves);

    // Timing
    void startTimer(int64_t msBudget);
    bool timeUp() const;
    int64_t startTimeMs = 0;
    int64_t budgetMs = 0;
};


