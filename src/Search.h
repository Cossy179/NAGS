#pragma once

// Iterative-deepening alpha-beta search shared by every engine. Templated on
// the board type (Board or FastBoard). Features: principal variation search,
// aspiration windows, check extension, late-move reductions, killer/history
// move ordering, quiescence search (with check evasions), repetition /
// fifty-move draws, optional transposition table and Lazy SMP helper threads.
//
// Searcher drives a whole UCI "go"; SearchWorker can also be driven one
// iteration at a time (the NAGS hybrid controller does this for its DFS arm).

#include "ChessTypes.h"
#include "Eval.h"
#include "TT.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
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

// Shared stop/limit state for one search.
class SearchControl {
public:
    using Clock = std::chrono::steady_clock;

    void begin(const SearchLimits &l, const std::atomic<bool> *externalStop) {
        limits = l;
        external = externalStop;
        start = Clock::now();
        stop.store(false, std::memory_order_relaxed);
        nodes.store(0, std::memory_order_relaxed);
    }

    int64_t elapsedMs() const {
        return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count();
    }

    // Polled periodically by the workers; latches `stop`.
    bool poll() {
        if (stop.load(std::memory_order_relaxed)) return true;
        bool s = (external && external->load(std::memory_order_relaxed)) ||
                 (limits.hardMs > 0 && elapsedMs() >= limits.hardMs) ||
                 (limits.nodes > 0 && nodes.load(std::memory_order_relaxed) >= limits.nodes);
        if (s) stop.store(true, std::memory_order_relaxed);
        return s;
    }

    bool externallyStopped() const { return external && external->load(std::memory_order_relaxed); }

    SearchLimits limits;
    std::atomic<bool> stop{false};
    std::atomic<uint64_t> nodes{0};
    Clock::time_point start;

private:
    const std::atomic<bool> *external = nullptr;
};

struct IterationResult {
    Move bestMove;
    int score = 0;
    std::vector<Move> pv;
    bool improved = false; // an aborted iteration still found a reliably better root move
};

template <class BoardT>
class SearchWorker {
public:
    SearchWorker(TranspositionTable *tt, SearchControl *control) : tt(tt), ctl(control) { clearHistory(); }

    void setRoot(const BoardT &root) {
        board = root;
        rootMoves = board.generateLegalMoves();
        orderMoves(rootMoves, Move{}, 0);
        nodes = 0;
        unflushed = 0;
        selDepth = 0;
        aborted = false;
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

    // Score of `m` from the root side's view, searched to depth-1 after the
    // move with a full window. Used to verify a candidate from another source.
    // Returns false if aborted.
    bool scoreRootMove(const Move &m, int depth, int &score) {
        board.makeMove(m);
        int s = -negamax(std::max(0, depth - 1), -INF_SCORE, INF_SCORE, 1, true);
        board.unmakeMove();
        if (aborted) return false;
        score = s;
        return true;
    }

    uint64_t nodeCount() const { return nodes; }
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
    Move pvTable[MAX_PLY][MAX_PLY];
    int pvLength[MAX_PLY] = {};
    uint64_t nodes = 0;
    uint64_t unflushed = 0;
    int selDepth = 0;
    bool aborted = false;

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

    int moveScore(const Move &m, const Move &ttMove, int ply) const {
        if (!ttMove.isNull() && sameMove(m, ttMove)) return 1 << 30;
        int score = 0;
        if (eval::isNoisy(board, m)) {
            int victim = eval::capturedValue(board, m);
            int attacker = eval::pieceValue(board.pieceAt(m.from));
            score = (1 << 24) + victim * 16 - attacker / 16;
            if (m.promotion != Piece::None) score += eval::pieceValue(m.promotion) * 16;
            return score;
        }
        if (ply < MAX_PLY) {
            if (sameMove(m, killers[ply][0])) return (1 << 23) + 2;
            if (sameMove(m, killers[ply][1])) return (1 << 23) + 1;
        }
        return history[colorIndex(board.sideToMove())][m.from][m.to];
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

    int rootSearch(int depth, int alpha, int beta, IterationResult &out) {
        pvLength[0] = 0;
        int bestScore = -INF_SCORE;
        int originalAlpha = alpha;
        for (size_t i = 0; i < rootMoves.size(); ++i) {
            const Move m = rootMoves[i];
            board.makeMove(m);
            int score;
            if (i == 0) {
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
            out.bestMove = rootMoves.front();
            out.pv = {rootMoves.front()};
            out.score = bestScore;
            out.improved = false;
        }
        return bestScore;
    }

    int negamax(int depth, int alpha, int beta, int ply, bool isPv) {
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

        Move ttMove;
        uint64_t key = board.zobrist();
        if (tt) {
            TTHit hit;
            if (tt->probe(key, hit)) {
                ttMove = hit.move;
                if (!isPv && hit.depth >= depth) {
                    int s = scoreFromTT(hit.score, ply);
                    if (hit.bound == Bound::Exact) return s;
                    if (hit.bound == Bound::Lower && s >= beta) return s;
                    if (hit.bound == Bound::Upper && s <= alpha) return s;
                }
            }
        }

        // Null-move pruning: if the side to move could pass and a reduced
        // search still fails high, the position is good enough to cut. Not in
        // check, at PV nodes, right after another null move, near mate scores,
        // or with only king and pawns (zugzwang is common there).
        if (!isPv && !inCheck && depth >= 3 && !board.lastMoveWasNull() && std::abs(beta) < MATE_BOUND &&
            hasNonPawnMaterial(board.sideToMove()) && eval::evaluate(board) >= beta) {
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
        int scores[MoveList::kCapacity];
        for (int k = 0; k < moves.size(); ++k) scores[k] = moveScore(moves[k], ttMove, ply);

        const Color us = board.sideToMove();
        int originalAlpha = alpha;
        int best = -INF_SCORE;
        Move bestMove;
        int i = 0; // number of legal moves tried so far
        for (int n = 0; n < moves.size(); ++n) {
            pickNext(moves, scores, n);
            const Move m = moves[n];
            bool quiet = !eval::isNoisy(board, m);
            board.makeMove(m);
            if (board.inCheck(us)) { // illegal: leaves our own king in check
                board.unmakeMove();
                continue;
            }
            bool givesCheck = board.inCheck();
            int score;
            if (i == 0) {
                score = -negamax(depth - 1, -beta, -alpha, ply + 1, isPv);
            } else {
                int reduction = 0;
                if (depth >= 3 && i >= 3 && quiet && !inCheck && !givesCheck &&
                    !sameMove(m, killers[ply][0]) && !sameMove(m, killers[ply][1])) {
                    reduction = 1 + (i >= 6 && depth >= 6 ? 1 : 0);
                }
                score = -negamax(depth - 1 - reduction, -alpha - 1, -alpha, ply + 1, false);
                if (!aborted && reduction && score > alpha)
                    score = -negamax(depth - 1, -alpha - 1, -alpha, ply + 1, false);
                if (!aborted && score > alpha && score < beta)
                    score = -negamax(depth - 1, -beta, -alpha, ply + 1, true);
            }
            board.unmakeMove();
            if (aborted) return 0;
            ++i;

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
                            int &h = history[colorIndex(board.sideToMove())][m.from][m.to];
                            h = std::min(h + depth * depth, 1 << 20);
                        }
                        break;
                    }
                }
            }
        }

        if (i == 0) return inCheck ? -MATE_SCORE + ply : 0; // checkmate or stalemate

        if (tt) {
            Bound bound = best >= beta ? Bound::Lower : (best > originalAlpha ? Bound::Exact : Bound::Upper);
            tt->store(key, depth, bound, scoreToTT(best, ply), bestMove);
        }
        return best;
    }

    int quiescence(int alpha, int beta, int ply) {
        if (checkStop()) return 0;
        ++nodes;
        selDepth = std::max(selDepth, ply);
        if (ply >= MAX_PLY - 1) return eval::evaluate(board);

        bool inCheck = board.inCheck();
        int best = -INF_SCORE;
        int stand = 0;
        if (!inCheck) {
            stand = eval::evaluate(board);
            if (stand >= beta) return stand;
            if (stand > alpha) alpha = stand;
            best = stand;
        }

        // In check every evasion must be considered (no stand-pat).
        MoveList moves;
        board.generatePseudoLegalMoves(moves, !inCheck);
        int scores[MoveList::kCapacity];
        for (int k = 0; k < moves.size(); ++k) scores[k] = moveScore(moves[k], Move{}, MAX_PLY);

        const Color us = board.sideToMove();
        int legal = 0;
        for (int n = 0; n < moves.size(); ++n) {
            pickNext(moves, scores, n);
            const Move m = moves[n];
            if (!inCheck && m.promotion == Piece::None && stand + eval::capturedValue(board, m) + 200 <= alpha)
                continue; // delta pruning
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
                    if (alpha >= beta) break;
                }
            }
        }
        if (inCheck && legal == 0) return -MATE_SCORE + ply; // checkmate
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

    void clearHistory() {
        for (auto &w : workers) w->clearHistory();
    }

    SearchResult search(const BoardT &root, const SearchLimits &limits, const std::atomic<bool> &externalStop,
                        const std::function<void(const SearchInfo &)> &onInfo) {
        control.begin(limits, &externalStop);
        if (tt) tt->newSearch();
        for (auto &w : workers) w->setRoot(root);

        SearchWorker<BoardT> &main = *workers[0];
        SearchResult result;
        const auto &rootMoves = main.legalRootMoves();
        if (rootMoves.empty()) {
            waitIfInfinite(limits, externalStop);
            return result; // checkmate or stalemate: bestmove 0000
        }
        result.bestMove = rootMoves.front(); // always have a legal move to play
        result.pv = {result.bestMove};

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

        int maxDepth = limits.depth > 0 ? std::min(limits.depth, MAX_PLY - 8) : MAX_PLY - 8;
        int prevScore = 0;
        for (int depth = 1; depth <= maxDepth; ++depth) {
            IterationResult r;
            bool completed = main.iterate(depth, prevScore, r);
            if (!completed) {
                if (r.improved && !r.bestMove.isNull()) {
                    // A root move was fully searched in the interrupted
                    // iteration and beat the previous best: report and use it.
                    result.bestMove = r.bestMove;
                    result.pv = r.pv;
                    result.score = r.score;
                    main.flushNodes();
                    SearchInfo info;
                    info.depth = depth;
                    info.selDepth = main.selectiveDepth();
                    info.score = r.score;
                    info.nodes = control.nodes.load(std::memory_order_relaxed);
                    info.timeMs = control.elapsedMs();
                    info.hashfull = tt ? tt->hashfull() : 0;
                    info.pv = r.pv;
                    if (onInfo) onInfo(info);
                }
                break;
            }
            prevScore = r.score;
            result.bestMove = r.bestMove;
            result.score = r.score;
            result.depth = depth;
            result.pv = r.pv;

            main.flushNodes();
            SearchInfo info;
            info.depth = depth;
            info.selDepth = main.selectiveDepth();
            info.score = r.score;
            info.nodes = control.nodes.load(std::memory_order_relaxed);
            info.timeMs = control.elapsedMs();
            info.hashfull = tt ? tt->hashfull() : 0;
            info.pv = r.pv;
            if (onInfo) onInfo(info);

            if (control.stop.load(std::memory_order_relaxed)) break;
            if (!limits.infinite) {
                if (rootMoves.size() == 1 && (limits.softMs > 0 || limits.hardMs > 0)) break; // forced move
                if (limits.softMs > 0 && control.elapsedMs() >= limits.softMs) break;         // next depth won't fit
                if (std::abs(r.score) >= MATE_BOUND && depth >= (MATE_SCORE - std::abs(r.score)) + 4) break;
            }
        }

        waitIfInfinite(limits, externalStop);
        control.stop.store(true, std::memory_order_relaxed);
        for (auto &t : helpers) t.join();
        main.flushNodes();
        result.nodes = control.nodes.load(std::memory_order_relaxed);
        if (result.pv.size() >= 2) result.ponderMove = result.pv[1];
        return result;
    }

private:
    TranspositionTable *tt;
    SearchControl control;
    std::vector<std::unique_ptr<SearchWorker<BoardT>>> workers;

    // UCI: in infinite mode bestmove must not be sent before "stop".
    static void waitIfInfinite(const SearchLimits &limits, const std::atomic<bool> &externalStop) {
        while (limits.infinite && !externalStop.load(std::memory_order_relaxed))
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
};
