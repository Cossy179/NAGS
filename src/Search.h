#pragma once

// Iterative-deepening alpha-beta search shared by every engine. Templated on
// the board type (Board or FastBoard). Features: principal variation search,
// aspiration windows, check and singular extensions, reverse futility and
// null-move pruning, late-move pruning/reductions, futility pruning, SEE,
// killer/countermove/history move ordering, quiescence search (with check
// evasions and the TT), repetition / fifty-move draws, optional transposition table and
// Lazy SMP helper threads, MultiPV, pondering and Syzygy tablebases. See
// docs/ARCHITECTURE.md and, for the tested gain of each, docs/TESTING.md.
//
// Searcher drives a whole UCI "go"; SearchWorker can also be driven one
// iteration at a time (the NAGS hybrid controller does this for its DFS arm).

#include "ChessTypes.h"
#include "Eval.h"
#include "Syzygy.h"
#include "TT.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct SearchLimits {
    int depth = 0;        // 0 = no depth limit
    uint64_t nodes = 0;   // 0 = no node limit
    int64_t softMs = 0;   // 0 = no time limit; no new iteration is started after this
    int64_t hardMs = 0;   // 0 = no time limit; the search is aborted after this
    bool infinite = false; // keep going (and hold bestmove) until stopped
};

struct SearchInfo {
    int depth = 0;
    int selDepth = 0;
    int score = 0;
    uint64_t nodes = 0;
    int64_t timeMs = 0;
    int hashfull = 0;
    int multiPv = 0; // line number (1 = best) when MultiPV > 1, otherwise 0
    uint64_t tbHits = 0;
    std::vector<Move> pv;
};

struct SearchResult {
    Move bestMove;
    Move ponderMove;
    int score = 0;
    int depth = 0;
    uint64_t nodes = 0;
    std::vector<Move> pv;
};

// "cp 35" or "mate -3", as UCI expects.
inline std::string formatScore(int score) {
    if (score >= MATE_BOUND) return "mate " + std::to_string((MATE_SCORE - score + 1) / 2);
    if (score <= -MATE_BOUND) return "mate " + std::to_string(-(MATE_SCORE + score + 1) / 2);
    return "cp " + std::to_string(score);
}

// Score of a tablebase result `ply` plies from the root, from the side to
// move's view. Wins and losses that the fifty-move rule turns into draws
// score just off zero.
inline int tbScore(syzygy::Wdl wdl, int ply) {
    switch (wdl) {
    case syzygy::Wdl::Win: return TB_WIN_SCORE - ply;
    case syzygy::Wdl::Loss: return -TB_WIN_SCORE + ply;
    case syzygy::Wdl::CursedWin: return 1;
    case syzygy::Wdl::BlessedLoss: return -1;
    default: return 0;
    }
}

// Shared stop/limit state for one search.
//
// Pondering: while the GUI's ponder flag is set, time limits are ignored. The
// first check that sees the flag cleared (ponderhit) restarts the clock, so
// the time limits count from the ponderhit.
class SearchControl {
public:
    using Clock = std::chrono::steady_clock;

    void begin(const SearchLimits &l, const std::atomic<bool> *externalStop,
               const std::atomic<bool> *ponderFlag = nullptr) {
        limits = l;
        external = externalStop;
        ponder = ponderFlag;
        startNs.store(nowNs(), std::memory_order_relaxed);
        stop.store(false, std::memory_order_relaxed);
        nodes.store(0, std::memory_order_relaxed);
        ponderActive.store(ponder && ponder->load(std::memory_order_acquire), std::memory_order_release);
    }

    int64_t elapsedMs() const { return (nowNs() - startNs.load(std::memory_order_relaxed)) / 1000000; }

    // True while pondering. Restarts the clock at the ponderhit.
    bool pondering() {
        if (!ponderActive.load(std::memory_order_acquire)) return false;
        if (ponder->load(std::memory_order_acquire)) return true;
        std::lock_guard<std::mutex> lock(ponderMutex);
        if (ponderActive.load(std::memory_order_relaxed)) {
            startNs.store(nowNs(), std::memory_order_relaxed);
            ponderActive.store(false, std::memory_order_release);
        }
        return false;
    }

    // Polled periodically by the workers; latches `stop`.
    bool poll() {
        if (stop.load(std::memory_order_relaxed)) return true;
        bool s = (external && external->load(std::memory_order_relaxed)) ||
                 (limits.hardMs > 0 && !pondering() && elapsedMs() >= limits.hardMs) ||
                 (limits.nodes > 0 && nodes.load(std::memory_order_relaxed) >= limits.nodes);
        if (s) stop.store(true, std::memory_order_relaxed);
        return s;
    }

    bool externallyStopped() const { return external && external->load(std::memory_order_relaxed); }

    // UCI: bestmove must not be sent during an infinite search or while
    // pondering until "stop" (or "ponderhit").
    void holdBestMove() {
        while ((limits.infinite || pondering()) && !externallyStopped())
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    SearchLimits limits;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> nodes{0};

private:
    static int64_t nowNs() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    }

    const std::atomic<bool> *external = nullptr;
    const std::atomic<bool> *ponder = nullptr;
    std::atomic<int64_t> startNs{0};
    std::atomic<bool> ponderActive{false};
    std::mutex ponderMutex;
};

struct IterationResult {
    Move bestMove;
    int score = 0;
    std::vector<Move> pv;
    bool improved = false; // an aborted iteration still found a reliably better root move
};

// One MultiPV line.
struct RootLine {
    Move move;
    int score = 0;
    std::vector<Move> pv;
};

template <class BoardT>
class SearchWorker {
public:
    SearchWorker(TranspositionTable *tt, SearchControl *control) : tt(tt), ctl(control) { clearHistory(); }

    void setRoot(const BoardT &root) {
        board = root;
        board.refreshAccumulator(); // the network may have changed since `root` was set up
        rootMoves = board.generateLegalMoves();
        orderMoves(rootMoves, Move{}, 0);
        nodes = 0;
        unflushed = 0;
        selDepth = 0;
        aborted = false;
        tbHits.store(0, std::memory_order_relaxed);
    }

    // Puts a move found elsewhere (e.g. the TT or another arm) first at the root.
    void preferRootMove(const Move &m) {
        auto it = std::find_if(rootMoves.begin(), rootMoves.end(), [&](const Move &x) { return sameMove(x, m); });
        if (it != rootMoves.end()) std::rotate(rootMoves.begin(), it, it + 1);
    }

    const std::vector<Move> &legalRootMoves() const { return rootMoves; }
    BoardT &rootBoard() { return board; }

    void clearHistory() {
        for (auto &k : killers) k[0] = k[1] = Move{};
        for (auto &side : history) for (auto &from : side) for (auto &v : from) v = 0;
        for (auto &piece : counterMoves) for (auto &mv : piece) mv = Move{};
    }

    // Searches the root to `depth` (with an aspiration window around
    // prevScore). Returns true if the iteration completed; on abort `out` may
    // still carry a better move (out.improved).
    bool iterate(int depth, int prevScore, IterationResult &out) {
        out = IterationResult{};
        if (rootMoves.empty()) return true;
        int window = (depth >= 5 && std::abs(prevScore) < MATE_BOUND) ? 25 : INF_SCORE;
        int alpha = window >= INF_SCORE ? -INF_SCORE : std::max(-INF_SCORE, prevScore - window);
        int beta = window >= INF_SCORE ? INF_SCORE : std::min(INF_SCORE, prevScore + window);
        for (;;) {
            IterationResult attempt;
            int score = rootSearch(depth, alpha, beta, attempt);
            if (aborted) {
                if (attempt.improved) out = attempt;
                return false;
            }
            if (score <= alpha && alpha > -INF_SCORE) {
                beta = (alpha + beta) / 2;
                window *= 2;
                alpha = window >= 1000 ? -INF_SCORE : std::max(-INF_SCORE, score - window);
                continue;
            }
            if (score >= beta && beta < INF_SCORE) {
                window *= 2;
                beta = window >= 1000 ? INF_SCORE : std::min(INF_SCORE, score + window);
                out = attempt; // the fail-high move is genuinely better than the old best
                out.improved = true;
                preferRootMove(attempt.bestMove);
                continue;
            }
            out = attempt;
            out.score = score;
            preferRootMove(out.bestMove);
            return true;
        }
    }

    // MultiPV: finds the best `count` root moves one after another, each with
    // a full-window search over the moves not chosen yet. Lines come back best
    // first, and the root moves are reordered to match. Returns false if
    // aborted.
    bool iterateLines(int depth, int count, std::vector<RootLine> &lines) {
        lines.clear();
        count = std::min(count, static_cast<int>(rootMoves.size()));
        for (int k = 0; k < count; ++k) {
            IterationResult attempt;
            int score = rootSearch(depth, -INF_SCORE, INF_SCORE, attempt, static_cast<size_t>(k));
            if (aborted) return false;
            lines.push_back({attempt.bestMove, score, attempt.pv});
            moveRootMove(attempt.bestMove, static_cast<size_t>(k));
        }
        std::stable_sort(lines.begin(), lines.end(), [](const RootLine &a, const RootLine &b) { return a.score > b.score; });
        for (int k = 0; k < count; ++k) moveRootMove(lines[k].move, static_cast<size_t>(k));
        return true;
    }

    // Score of `m` from the root side's view, searched to depth-1 after the
    // move within (alpha, beta) (a full window by default; with a null window
    // the result is only a bound). Used to verify a candidate from another
    // source. Returns false if aborted.
    bool scoreRootMove(const Move &m, int depth, int &score, int alpha = -INF_SCORE, int beta = INF_SCORE) {
        board.makeMove(m);
        int s = -negamax(std::max(0, depth - 1), -beta, -alpha, 1, beta - alpha > 1);
        board.unmakeMove();
        if (aborted) return false;
        score = s;
        return true;
    }

    uint64_t nodeCount() const { return nodes; }
    uint64_t tbHitCount() const { return tbHits.load(std::memory_order_relaxed); }
    int selectiveDepth() const { return selDepth; }
    bool wasAborted() const { return aborted; }
    void flushNodes() {
        if (unflushed) {
            ctl->nodes.fetch_add(unflushed, std::memory_order_relaxed);
            unflushed = 0;
        }
    }

private:
    BoardT board;
    TranspositionTable *tt;
    SearchControl *ctl;
    std::vector<Move> rootMoves;
    Move killers[MAX_PLY][2];
    int history[2][64][64];
    Move counterMoves[12][64]; // [piece that moved][its target square]
    Move pvTable[MAX_PLY][MAX_PLY];
    int pvLength[MAX_PLY] = {};
    uint64_t nodes = 0;
    uint64_t unflushed = 0;
    int selDepth = 0;
    bool aborted = false;
    std::atomic<uint64_t> tbHits{0}; // written by this worker only, read by the reporting thread

    // Moves `m` to position `pos` of the root move list, shifting the moves
    // in between down by one.
    void moveRootMove(const Move &m, size_t pos) {
        auto it = std::find_if(rootMoves.begin() + pos, rootMoves.end(), [&](const Move &x) { return sameMove(x, m); });
        if (it != rootMoves.end()) std::rotate(rootMoves.begin() + pos, it, it + 1);
    }

    bool checkStop() {
        if (aborted) return true;
        if (++unflushed >= 1024) {
            flushNodes();
            if (ctl->poll()) aborted = true;
        } else if (ctl->stop.load(std::memory_order_relaxed)) {
            aborted = true;
        }
        return aborted;
    }

    // Order: TT move, winning/equal captures and promotions (MVV/LVA), killers
    // and the countermove, losing captures (negative SEE), quiet moves by history.
    int moveScore(const Move &m, const Move &ttMove, int ply, const Move &counter = Move{}) const {
        if (!ttMove.isNull() && sameMove(m, ttMove)) return 1 << 30;
        if (eval::isNoisy(board, m)) {
            int victim = eval::capturedValue(board, m);
            int attacker = eval::pieceValue(board.pieceAt(m.from));
            int score = victim * 16 - attacker / 16;
            if (m.promotion != Piece::None) score += eval::pieceValue(m.promotion) * 16;
            return (eval::see(board, m) >= 0 ? (1 << 24) : (1 << 22)) + score;
        }
        if (ply < MAX_PLY) {
            if (sameMove(m, killers[ply][0])) return (1 << 23) + 2;
            if (sameMove(m, killers[ply][1])) return (1 << 23) + 1;
            if (!counter.isNull() && sameMove(m, counter)) return 1 << 23;
        }
        return history[colorIndex(board.sideToMove())][m.from][m.to];
    }

    // Quiet move that refuted the opponent's previous move last time.
    Move counterMoveFor(const Move &prev) const {
        if (prev.isNull()) return Move{};
        Piece p = board.pieceAt(prev.to);
        return p == Piece::None ? Move{} : counterMoves[static_cast<int>(p) - 1][prev.to];
    }

    void orderMoves(std::vector<Move> &moves, const Move &ttMove, int ply) const {
        std::vector<std::pair<int, size_t>> keyed;
        keyed.reserve(moves.size());
        for (size_t i = 0; i < moves.size(); ++i) keyed.emplace_back(moveScore(moves[i], ttMove, ply), i);
        std::stable_sort(keyed.begin(), keyed.end(), [](const auto &a, const auto &b) { return a.first > b.first; });
        std::vector<Move> sorted;
        sorted.reserve(moves.size());
        for (const auto &k : keyed) sorted.push_back(moves[k.second]);
        moves.swap(sorted);
    }

    // Late-move reduction in plies: 0.75 + ln(depth) * ln(moveNumber) / 2.25.
    static int lmrReduction(int depth, int moveNumber) {
        static const auto table = [] {
            std::array<std::array<int, 64>, 64> t{};
            for (int d = 1; d < 64; ++d)
                for (int m = 1; m < 64; ++m) t[d][m] = static_cast<int>(0.75 + std::log(d) * std::log(m) / 2.25);
            return t;
        }();
        return table[std::min(depth, 63)][std::min(moveNumber, 63)];
    }
    static void updateHistory(int &h, int bonus) { h += bonus - h * std::abs(bonus) / kHistoryMax; }
    static constexpr int kHistoryMax = 16384;

    bool hasNonPawnMaterial(Color c) const {
        return (board.pieceBB(c, KNIGHT) | board.pieceBB(c, BISHOP) | board.pieceBB(c, ROOK) | board.pieceBB(c, QUEEN)) != 0;
    }

    // Lazy move ordering: brings the highest-scored remaining move to index n.
    // Ties keep generation order (the same order as a stable sort), and moves
    // after a cutoff are never sorted at all.
    static void pickNext(MoveList &moves, int *scores, int n) {
        int best = n;
        for (int k = n + 1; k < moves.size(); ++k)
            if (scores[k] > scores[best]) best = k;
        if (best == n) return;
        Move m = moves[best];
        int sc = scores[best];
        for (int k = best; k > n; --k) {
            moves[k] = moves[k - 1];
            scores[k] = scores[k - 1];
        }
        moves[n] = m;
        scores[n] = sc;
    }

    void updatePv(int ply, const Move &m) {
        pvTable[ply][ply] = m;
        int childLen = (ply + 1 < MAX_PLY) ? pvLength[ply + 1] : ply + 1;
        for (int i = ply + 1; i < childLen; ++i) pvTable[ply][i] = pvTable[ply + 1][i];
        pvLength[ply] = std::max(childLen, ply + 1);
    }

    // Searches root moves [first, end) (MultiPV skips the lines already found).
    int rootSearch(int depth, int alpha, int beta, IterationResult &out, size_t first = 0) {
        pvLength[0] = 0;
        int bestScore = -INF_SCORE;
        int originalAlpha = alpha;
        for (size_t i = first; i < rootMoves.size(); ++i) {
            const Move m = rootMoves[i];
            board.makeMove(m);
            int score;
            if (i == first) {
                score = -negamax(depth - 1, -beta, -alpha, 1, true);
            } else {
                score = -negamax(depth - 1, -alpha - 1, -alpha, 1, false);
                if (!aborted && score > alpha && score < beta) score = -negamax(depth - 1, -beta, -alpha, 1, true);
            }
            board.unmakeMove();
            if (aborted) break;
            if (score > bestScore) {
                bestScore = score;
                if (score > alpha) {
                    alpha = score;
                    updatePv(0, m);
                    out.bestMove = m;
                    out.score = score;
                    out.pv.assign(pvTable[0], pvTable[0] + pvLength[0]);
                    // Beating the window's lower bound means this move is at
                    // least as good as anything searched so far.
                    out.improved = score > originalAlpha;
                    if (alpha >= beta) break;
                }
            }
        }
        if (out.bestMove.isNull()) {
            // Fail low: everything scored at or below alpha.
            out.bestMove = rootMoves[first];
            out.pv = {rootMoves[first]};
            out.score = bestScore;
            out.improved = false;
        }
        return bestScore;
    }

    // `excluded`: a move to leave out (singular extension verification). Such
    // a search neither uses nor stores TT and tablebase results for the node.
    int negamax(int depth, int alpha, int beta, int ply, bool isPv, const Move &excluded = Move{}) {
        pvLength[ply] = ply;
        if (checkStop()) return 0;
        if (board.isDraw()) return 0;
        if (ply >= MAX_PLY - 1) return eval::evaluate(board);

        // Mate distance pruning.
        alpha = std::max(alpha, -MATE_SCORE + ply);
        beta = std::min(beta, MATE_SCORE - ply - 1);
        if (alpha >= beta) return alpha;

        bool inCheck = board.inCheck();
        if (inCheck && ply < MAX_PLY / 2) ++depth;
        if (depth <= 0) return quiescence(alpha, beta, ply);

        ++nodes;
        selDepth = std::max(selDepth, ply);

        const bool excluding = !excluded.isNull();
        Move ttMove;
        TTHit hit;
        bool ttHit = false;
        uint64_t key = board.zobrist();
        if (tt && !excluding) {
            if (tt->probe(key, hit)) {
                ttHit = true;
                ttMove = hit.move;
                if (!isPv && hit.depth >= depth) {
                    int s = scoreFromTT(hit.score, ply);
                    if (hit.bound == Bound::Exact) return s;
                    if (hit.bound == Bound::Lower && s >= beta) return s;
                    if (hit.bound == Bound::Upper && s <= alpha) return s;
                }
            }
        }

        // Endgame tablebases. The WDL tables only apply right after a capture
        // or pawn move (half-move clock 0), without castling rights. A result
        // that does not cut off still bounds this node's score.
        int tbFloor = -INF_SCORE, tbCeiling = INF_SCORE;
        if (!excluding && syzygy::cardinality() > 0 && board.getHalfmoveClock() == 0 && board.getCastlingRights() == 0 &&
            popcount(board.occupancy()) <= syzygy::cardinality()) {
            syzygy::Wdl wdl;
            if (syzygy::probeWdl(syzygy::position(board), wdl)) {
                tbHits.store(tbHits.load(std::memory_order_relaxed) + 1, std::memory_order_relaxed);
                int score = tbScore(wdl, ply);
                // A win is a lower bound (a mate may be found), a loss an upper bound.
                Bound bound = wdl == syzygy::Wdl::Win ? Bound::Lower : wdl == syzygy::Wdl::Loss ? Bound::Upper : Bound::Exact;
                if (bound == Bound::Exact || (bound == Bound::Lower && score >= beta) ||
                    (bound == Bound::Upper && score <= alpha)) {
                    if (tt) tt->store(key, std::min(depth + 6, MAX_PLY - 1), bound, scoreToTT(score, ply), Move{});
                    return score;
                }
                if (bound == Bound::Lower) tbFloor = score;
                else tbCeiling = score;
            }
        }

        // Static evaluation for the pruning decisions below (not needed at PV
        // nodes or in check, where nothing is pruned).
        const int staticEval = (!isPv && !inCheck) ? eval::evaluate(board) : 0;

        // Reverse futility pruning: at shallow depth, a static evaluation that
        // beats beta by a depth-scaled margin is assumed to hold.
        if (!isPv && !inCheck && !excluding && depth <= 6 && std::abs(beta) < MATE_BOUND && staticEval - 80 * depth >= beta)
            return staticEval;

        // Null-move pruning: if the side to move could pass and a reduced
        // search still fails high, the position is good enough to cut. Not in
        // check, at PV nodes, right after another null move, near mate scores,
        // or with only king and pawns (zugzwang is common there).
        if (!isPv && !inCheck && !excluding && depth >= 3 && !board.lastMoveWasNull() && std::abs(beta) < MATE_BOUND &&
            hasNonPawnMaterial(board.sideToMove()) && staticEval >= beta) {
            int reduction = 3 + depth / 6;
            board.makeNullMove();
            int score = -negamax(depth - 1 - reduction, -beta, -beta + 1, ply + 1, false);
            board.unmakeNullMove();
            if (aborted) return 0;
            if (score >= beta) return score >= MATE_BOUND ? beta : score; // don't trust unproven mates
        }

        // Pseudo-legal moves; legality is checked only for moves actually
        // tried (most nodes cut off after a few moves).
        MoveList moves;
        board.generatePseudoLegalMoves(moves);
        const Move prevMove = board.lastMove();
        const Move counter = counterMoveFor(prevMove);
        int scores[MoveList::kCapacity];
        for (int k = 0; k < moves.size(); ++k) scores[k] = moveScore(moves[k], ttMove, ply, counter);

        const Color us = board.sideToMove();
        int originalAlpha = alpha;
        int best = -INF_SCORE;
        Move bestMove;
        // Singular extension candidate: a deep enough TT entry whose move
        // failed high (or was exact).
        const int ttScore = ttHit ? scoreFromTT(hit.score, ply) : 0;
        const bool trySingular = !excluding && depth >= 8 && ttHit && !ttMove.isNull() && hit.depth >= depth - 3 &&
                                 hit.bound != Bound::Upper && std::abs(ttScore) < TB_BOUND && ply < MAX_PLY / 2;

        int i = 0; // number of legal moves tried so far
        int quietsTried = 0;
        MoveList quietsSearched; // quiet moves searched without a cutoff
        for (int n = 0; n < moves.size(); ++n) {
            pickNext(moves, scores, n);
            const Move m = moves[n];
            if (excluding && sameMove(m, excluded)) continue;
            bool quiet = !eval::isNoisy(board, m);
            // Shallow quiet-move pruning. Only once a legal move has been
            // searched, so checkmate / stalemate detection is unaffected.
            if (!isPv && !inCheck && quiet && i > 0 && depth <= 3 && std::abs(alpha) < MATE_BOUND) {
                if (quietsTried >= 3 + depth * depth) continue;  // late-move pruning
                if (staticEval + 120 * depth <= alpha) continue; // futility pruning
            }
            // Singular extension: if every other move fails low against a
            // margin below the TT score, the TT move is singular and gets one
            // more ply. If even the alternatives beat beta, cut (multi-cut).
            int extension = 0;
            if (trySingular && sameMove(m, ttMove)) {
                int singularBeta = ttScore - 2 * depth;
                int s = negamax((depth - 1) / 2, singularBeta - 1, singularBeta, ply, false, m);
                if (aborted) return 0;
                if (s < singularBeta) extension = 1;
                else if (singularBeta >= beta) return singularBeta;
            }
            board.makeMove(m);
            if (board.inCheck(us)) { // illegal: leaves our own king in check
                board.unmakeMove();
                continue;
            }
            bool givesCheck = board.inCheck();
            int score;
            if (i == 0) {
                score = -negamax(depth - 1 + extension, -beta, -alpha, ply + 1, isPv);
            } else {
                int reduction = 0;
                if (depth >= 3 && i >= 2 && quiet && !inCheck && !givesCheck &&
                    !sameMove(m, killers[ply][0]) && !sameMove(m, killers[ply][1])) {
                    reduction = lmrReduction(depth, i + 1) - (isPv ? 1 : 0);
                    reduction = std::clamp(reduction, 0, depth - 2);
                }
                score = -negamax(depth - 1 + extension - reduction, -alpha - 1, -alpha, ply + 1, false);
                if (!aborted && reduction && score > alpha)
                    score = -negamax(depth - 1 + extension, -alpha - 1, -alpha, ply + 1, false);
                if (!aborted && score > alpha && score < beta)
                    score = -negamax(depth - 1 + extension, -beta, -alpha, ply + 1, true);
            }
            board.unmakeMove();
            if (aborted) return 0;
            ++i;
            if (quiet) ++quietsTried;

            if (score > best) {
                best = score;
                bestMove = m;
                if (score > alpha) {
                    alpha = score;
                    if (isPv) updatePv(ply, m);
                    if (alpha >= beta) {
                        if (quiet) {
                            if (!sameMove(m, killers[ply][0])) {
                                killers[ply][1] = killers[ply][0];
                                killers[ply][0] = m;
                            }
                            // Reward the cutoff move and penalise the quiet
                            // moves tried before it; values decay towards
                            // zero as they approach the bound.
                            int bonus = std::min(depth * depth, 1200);
                            auto &table = history[colorIndex(board.sideToMove())];
                            updateHistory(table[m.from][m.to], bonus);
                            for (int q = 0; q < quietsSearched.size(); ++q)
                                updateHistory(table[quietsSearched[q].from][quietsSearched[q].to], -bonus);
                            if (!prevMove.isNull()) {
                                Piece p = board.pieceAt(prevMove.to);
                                if (p != Piece::None) counterMoves[static_cast<int>(p) - 1][prevMove.to] = m;
                            }
                        }
                        break;
                    }
                }
            }
            if (quiet) quietsSearched.push_back(m);
        }

        if (i == 0) {
            if (excluding) return alpha; // the excluded move was the only one
            return inCheck ? -MATE_SCORE + ply : 0; // checkmate or stalemate
        }

        Bound bound = best >= beta ? Bound::Lower : (best > originalAlpha ? Bound::Exact : Bound::Upper);
        if (best < tbFloor) {
            best = tbFloor;
            bound = Bound::Lower;
        } else if (best > tbCeiling) {
            best = tbCeiling;
            bound = Bound::Upper;
        }
        if (tt && !excluding) tt->store(key, depth, bound, scoreToTT(best, ply), bestMove);
        return best;
    }

    // Captures and promotions only (every evasion when in check). Results go
    // to the TT at depth 0; non-PV nodes take cutoffs from it.
    int quiescence(int alpha, int beta, int ply) {
        if (checkStop()) return 0;
        ++nodes;
        selDepth = std::max(selDepth, ply);
        if (ply >= MAX_PLY - 1) return eval::evaluate(board);

        const bool isPv = beta - alpha > 1;
        const uint64_t key = board.zobrist();
        Move ttMove;
        TTHit hit;
        bool ttHit = false;
        if (tt && tt->probe(key, hit)) {
            ttHit = true;
            ttMove = hit.move;
            if (!isPv) {
                int s = scoreFromTT(hit.score, ply);
                if (hit.bound == Bound::Exact || (hit.bound == Bound::Lower && s >= beta) ||
                    (hit.bound == Bound::Upper && s <= alpha))
                    return s;
            }
        }

        bool inCheck = board.inCheck();
        int best = -INF_SCORE;
        int stand = 0;
        const int originalAlpha = alpha;
        if (!inCheck) {
            stand = eval::evaluate(board);
            // A TT score whose bound points the right way is a better estimate.
            if (ttHit && std::abs(hit.score) < TB_BOUND) {
                int s = hit.score;
                if (hit.bound == Bound::Exact || (hit.bound == Bound::Lower && s > stand) ||
                    (hit.bound == Bound::Upper && s < stand))
                    stand = s;
            }
            if (stand >= beta) {
                if (tt && !ttHit) tt->store(key, 0, Bound::Lower, scoreToTT(stand, ply), Move{});
                return stand;
            }
            if (stand > alpha) alpha = stand;
            best = stand;
        }

        // In check every evasion must be considered (no stand-pat).
        MoveList moves;
        board.generatePseudoLegalMoves(moves, !inCheck);
        // The TT move only counts if this generator produced it (a quiet TT
        // move is not searched outside check).
        int scores[MoveList::kCapacity];
        for (int k = 0; k < moves.size(); ++k) scores[k] = moveScore(moves[k], ttMove, MAX_PLY);

        const Color us = board.sideToMove();
        int legal = 0;
        Move bestMove;
        for (int n = 0; n < moves.size(); ++n) {
            pickNext(moves, scores, n);
            const Move m = moves[n];
            if (!inCheck && m.promotion == Piece::None && stand + eval::capturedValue(board, m) + 200 <= alpha)
                continue; // delta pruning
            if (!inCheck && eval::see(board, m) < 0) continue; // losing capture
            board.makeMove(m);
            if (board.inCheck(us)) {
                board.unmakeMove();
                continue;
            }
            ++legal;
            int score = -quiescence(-beta, -alpha, ply + 1);
            board.unmakeMove();
            if (aborted) return 0;
            if (score > best) {
                best = score;
                if (score > alpha) {
                    alpha = score;
                    bestMove = m;
                    if (alpha >= beta) break;
                }
            }
        }
        if (inCheck && legal == 0) return -MATE_SCORE + ply; // checkmate
        if (tt) {
            Bound bound = best >= beta ? Bound::Lower : (best > originalAlpha ? Bound::Exact : Bound::Upper);
            tt->store(key, 0, bound, scoreToTT(best, ply), bestMove);
        }
        return best;
    }
};

// Runs a complete search: iterative deepening on the calling thread plus
// (threads - 1) Lazy SMP helpers that share the transposition table.
template <class BoardT>
class Searcher {
public:
    explicit Searcher(TranspositionTable *tt) : tt(tt) { setThreads(1); }

    // Helper threads only make sense with a shared transposition table.
    void setThreads(int n) {
        n = std::max(1, tt ? n : 1);
        workers.clear();
        for (int i = 0; i < n; ++i) workers.push_back(std::make_unique<SearchWorker<BoardT>>(tt, &control));
    }
    int threadCount() const { return static_cast<int>(workers.size()); }

    // Number of best lines reported (UCI MultiPV).
    void setMultiPv(int n) { multiPv = std::max(1, n); }
    int multiPvCount() const { return multiPv; }

    void clearHistory() {
        for (auto &w : workers) w->clearHistory();
    }

    // ponderFlag: set by the GUI during "go ponder" and cleared at ponderhit
    // (may be null).
    SearchResult search(const BoardT &root, const SearchLimits &limits, const std::atomic<bool> &externalStop,
                        const std::function<void(const SearchInfo &)> &onInfo,
                        const std::atomic<bool> *ponderFlag = nullptr) {
        control.begin(limits, &externalStop, ponderFlag);
        if (tt) tt->newSearch();
        for (auto &w : workers) w->setRoot(root);

        SearchWorker<BoardT> &main = *workers[0];
        SearchResult result;
        const auto &rootMoves = main.legalRootMoves();
        if (rootMoves.empty()) {
            control.holdBestMove();
            return result; // checkmate or stalemate: bestmove 0000
        }
        result.bestMove = rootMoves.front(); // always have a legal move to play
        result.pv = {result.bestMove};

        auto report = [&](int depth, int score, const std::vector<Move> &pv, int line, uint64_t extraTbHits = 0) {
            if (!onInfo) return;
            main.flushNodes();
            SearchInfo info;
            info.depth = depth;
            info.selDepth = main.selectiveDepth();
            info.score = score;
            info.nodes = control.nodes.load(std::memory_order_relaxed);
            info.timeMs = control.elapsedMs();
            info.hashfull = tt ? tt->hashfull() : 0;
            info.multiPv = line;
            info.tbHits = extraTbHits;
            for (const auto &w : workers) info.tbHits += w->tbHitCount();
            info.pv = pv;
            onInfo(info);
        };

        // Tablebase position at the root: play the move with the best
        // distance to zeroing, which keeps the result under the fifty-move
        // rule and makes progress. Needs the DTZ tables.
        if (syzygy::cardinality() > 0 && root.getCastlingRights() == 0 &&
            popcount(root.occupancy()) <= syzygy::cardinality()) {
            syzygy::RootResult tb;
            if (syzygy::probeRoot(syzygy::position(root), tb)) {
                for (const Move &m : rootMoves) {
                    if (m.from != tb.from || m.to != tb.to || pieceTypeOf(m.promotion) != tb.promotion) continue;
                    result.bestMove = m;
                    result.pv = {m};
                    result.score = tbScore(tb.wdl, 0);
                    result.depth = 1;
                    report(1, result.score, result.pv, multiPv > 1 ? 1 : 0, 1);
                    control.holdBestMove();
                    return result;
                }
            }
        }

        std::vector<std::thread> helpers;
        for (size_t i = 1; i < workers.size(); ++i) {
            helpers.emplace_back([this, i] {
                SearchWorker<BoardT> &w = *workers[i];
                int prev = 0;
                for (int depth = 1 + static_cast<int>(i % 2); depth < MAX_PLY - 8; ++depth) {
                    IterationResult r;
                    if (!w.iterate(depth, prev, r)) break;
                    prev = r.score;
                }
                w.flushNodes();
            });
        }

        const bool multi = multiPv > 1 && rootMoves.size() > 1;
        int maxDepth = limits.depth > 0 ? std::min(limits.depth, MAX_PLY - 8) : MAX_PLY - 8;
        int prevScore = 0;
        int stableIterations = 0; // completed iterations since the best move last changed
        for (int depth = 1; depth <= maxDepth; ++depth) {
            IterationResult r;
            if (multi) {
                std::vector<RootLine> lines;
                if (!main.iterateLines(depth, multiPv, lines)) break; // keep the last complete iteration
                r.bestMove = lines[0].move;
                r.score = lines[0].score;
                r.pv = lines[0].pv;
                for (size_t k = 0; k < lines.size(); ++k)
                    report(depth, lines[k].score, lines[k].pv, static_cast<int>(k) + 1);
            } else {
                bool completed = main.iterate(depth, prevScore, r);
                if (!completed) {
                    if (r.improved && !r.bestMove.isNull()) {
                        // A root move was fully searched in the interrupted
                        // iteration and beat the previous best: report and use it.
                        result.bestMove = r.bestMove;
                        result.pv = r.pv;
                        result.score = r.score;
                        report(depth, r.score, r.pv, 0);
                    }
                    break;
                }
                report(depth, r.score, r.pv, 0);
            }
            stableIterations = depth > 1 && sameMove(r.bestMove, result.bestMove) ? stableIterations + 1 : 0;
            int scoreDrop = depth > 1 ? prevScore - r.score : 0;
            prevScore = r.score;
            result.bestMove = r.bestMove;
            result.score = r.score;
            result.depth = depth;
            result.pv = r.pv;

            if (control.stop.load(std::memory_order_relaxed)) break;
            if (!limits.infinite) {
                if (rootMoves.size() == 1 && (limits.softMs > 0 || limits.hardMs > 0)) break; // forced move
                // Soft limit: stop earlier when the best move has been stable
                // for several iterations, later when it just changed or the
                // score is dropping (the hard limit still applies).
                static constexpr double kStability[5] = {2.0, 1.5, 1.2, 1.0, 0.85};
                double scale = kStability[std::min(stableIterations, 4)];
                if (scoreDrop > 20) scale *= 1.0 + std::min(scoreDrop, 120) / 240.0;
                if (limits.softMs > 0 && !control.pondering() &&
                    control.elapsedMs() >= static_cast<int64_t>(limits.softMs * scale))
                    break; // next depth won't fit
                if (std::abs(r.score) >= MATE_BOUND && depth >= (MATE_SCORE - std::abs(r.score)) + 4) break;
            }
        }

        control.holdBestMove();
        control.stop.store(true, std::memory_order_relaxed);
        for (auto &t : helpers) t.join();
        main.flushNodes();
        result.nodes = control.nodes.load(std::memory_order_relaxed);
        if (result.pv.size() >= 2) result.ponderMove = result.pv[1];
        return result;
    }

    // Searches root move `m` to `depth` with a full window and returns its
    // score from the root side's view (to verify a move proposed by another
    // search). Returns false if stopped or out of time first.
    bool scoreRootMove(const BoardT &root, const Move &m, int depth, const SearchLimits &limits,
                       const std::atomic<bool> &externalStop, int &score, int alpha = -INF_SCORE,
                       int beta = INF_SCORE) {
        control.begin(limits, &externalStop);
        SearchWorker<BoardT> &w = *workers[0];
        w.setRoot(root);
        bool ok = w.scoreRootMove(m, depth, score, alpha, beta);
        w.flushNodes();
        return ok;
    }

    // Nodes of the last search or scoreRootMove call.
    uint64_t nodesSearched() const { return control.nodes.load(std::memory_order_relaxed); }

private:
    TranspositionTable *tt;
    SearchControl control;
    std::vector<std::unique_ptr<SearchWorker<BoardT>>> workers;
    int multiPv = 1;
};
