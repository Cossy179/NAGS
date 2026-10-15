#pragma once

// Iterative-deepening alpha-beta search shared by every engine. Templated on
// the board type (Board or FastBoard). Features: principal variation search,
// aspiration windows, check and singular extensions, reverse futility and
// null-move pruning, late-move pruning/reductions, futility pruning, internal
// iterative reduction, an "improving" flag, SEE (ordering and pruning),
// killer/countermove/history/continuation-history move ordering, quiescence search (with check
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
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

static_assert(eval::kEvalLimit < TB_BOUND, "evaluations must stay below the tablebase and mate scores");

// Search statistics for profiling (CMake option NAGS_STATS). Compiled out
// otherwise: normal builds search, and run, exactly as without them.
#ifdef NAGS_STATS
inline constexpr bool kSearchStats = true;
#else
inline constexpr bool kSearchStats = false;
#endif
#define NAGS_STAT(stmt)                    \
    do {                                   \
        if constexpr (kSearchStats) { stmt; } \
    } while (0)

struct SearchStats {
    uint64_t nodes = 0, qnodes = 0, evals = 0;
    uint64_t ttProbes = 0, ttHits = 0, ttCutoffs = 0, qttProbes = 0, qttHits = 0, qttCutoffs = 0;
    uint64_t cutNodes = 0, cutoffAt[8] = {}; // beta cutoffs by the number of legal moves tried before (7: 7 or more)
    uint64_t cutoffByStage[5] = {};          // TT move, winning capture, killer/countermove, losing capture, quiet
    uint64_t allNodes = 0, pvNodes = 0;      // nodes without a cutoff: no move beat alpha / one did
    uint64_t movesSearched = 0, illegal = 0;
    uint64_t rfp = 0, nmpTries = 0, nmpCutoffs = 0, lmp = 0, futility = 0, historyPrunes = 0, seePrunes = 0;
    uint64_t lmrSearches = 0, lmrResearches = 0, pvsResearches = 0;
    uint64_t singularTries = 0, singularExtensions = 0, multiCuts = 0;
    uint64_t qStandPat = 0, qDelta = 0, qSee = 0, qCutoffs = 0;

    void add(const SearchStats &o) {
        const uint64_t *src = reinterpret_cast<const uint64_t *>(&o);
        uint64_t *dst = reinterpret_cast<uint64_t *>(this);
        for (size_t i = 0; i < sizeof(SearchStats) / sizeof(uint64_t); ++i) dst[i] += src[i];
    }

    std::vector<std::string> report() const {
        auto pct = [](uint64_t a, uint64_t b) {
            char buf[32];
            std::snprintf(buf, sizeof buf, "%.1f%%", b ? 100.0 * static_cast<double>(a) / static_cast<double>(b) : 0.0);
            return std::string(buf);
        };
        auto n = [](uint64_t v) { return std::to_string(v); };
        std::string at;
        for (int i = 0; i < 8; ++i) at += (i ? " " : "") + pct(cutoffAt[i], cutNodes);
        static const char *stages[5] = {"tt", "good_capture", "killer_counter", "bad_capture", "quiet"};
        std::string byStage;
        for (int i = 0; i < 5; ++i) byStage += std::string(i ? " " : "") + stages[i] + " " + pct(cutoffByStage[i], cutNodes);
        return {
            "stats nodes " + n(nodes) + " qnodes " + n(qnodes) + " (" + pct(qnodes, nodes + qnodes) + ") evals " + n(evals),
            "stats tt probes " + n(ttProbes) + " hits " + pct(ttHits, ttProbes) + " cutoffs " + pct(ttCutoffs, ttProbes) +
                " | qsearch probes " + n(qttProbes) + " hits " + pct(qttHits, qttProbes) + " cutoffs " + pct(qttCutoffs, qttProbes),
            "stats node types cut " + pct(cutNodes, cutNodes + allNodes + pvNodes) + " all " +
                pct(allNodes, cutNodes + allNodes + pvNodes) + " pv " + pct(pvNodes, cutNodes + allNodes + pvNodes) +
                " | moves searched per node " +
                std::to_string(cutNodes + allNodes + pvNodes ? static_cast<double>(movesSearched) / static_cast<double>(cutNodes + allNodes + pvNodes) : 0.0) +
                " illegal " + n(illegal),
            "stats cutoff on move 1..8+ " + at,
            "stats cutoff by stage " + byStage,
            "stats pruning rfp " + n(rfp) + " nmp " + n(nmpCutoffs) + "/" + n(nmpTries) + " lmp " + n(lmp) + " futility " +
                n(futility) + " history " + n(historyPrunes) + " see " + n(seePrunes),
            "stats reductions lmr " + n(lmrSearches) + " researched " + pct(lmrResearches, lmrSearches) + " pvs re-searches " +
                n(pvsResearches) + " | singular " + n(singularExtensions) + "/" + n(singularTries) + " multicut " + n(multiCuts),
            "stats qsearch stand-pat cutoffs " + n(qStandPat) + " delta " + n(qDelta) + " see skips " + n(qSee) +
                " beta cutoffs " + n(qCutoffs),
        };
    }
};

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
        for (int i = 0; i < 12 * 64; ++i) for (auto &p : contHist[i]) for (auto &v : p) v = 0;
        std::fill(corrPawn.get(), corrPawn.get() + 2 * kCorrSize, int16_t{0});
        std::fill(corrNonPawn.get(), corrNonPawn.get() + 4 * kCorrSize, int16_t{0});
        std::fill(corrCont.get(), corrCont.get() + 2 * 12 * 64, int16_t{0});
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
        setPlyMove(0, m);
        board.makeMove(m);
        int s = -negamax(std::max(0, depth - 1), -beta, -alpha, 1, beta - alpha > 1);
        board.unmakeMove();
        if (aborted) return false;
        score = s;
        return true;
    }

    uint64_t nodeCount() const { return nodes; }
    const SearchStats &statistics() const { return stats; }
    void resetStatistics() { stats = SearchStats{}; }
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
    // Continuation history: how well a quiet move (piece, target) did right
    // after a given move (piece, target) one or two plies earlier.
    using ContTable = int16_t[12][64];
    std::unique_ptr<ContTable[]> contHist = std::make_unique<ContTable[]>(12 * 64); // [prev piece * 64 + prev to]
    struct PlyMove {
        int piece = -1; // 0..11 (moving piece), -1 for none or a null move
        int to = 0;
    };
    PlyMove plyMoves[MAX_PLY + 1]; // the move played at each ply

    // Correction history: running averages of how far the static evaluation
    // was from the search result in positions with the same pawn structure,
    // the same pieces of one side, or after the same previous move (per
    // side to move). Their weighted mean corrects the static evaluation
    // used for pruning and stand-pat. Entries are centipawns x kCorrGrain.
    static constexpr int kCorrSize = 16384, kCorrGrain = 64, kCorrMaxCp = 256;
    std::unique_ptr<int16_t[]> corrPawn = std::make_unique<int16_t[]>(2 * kCorrSize);       // [stm][pawn key]
    std::unique_ptr<int16_t[]> corrNonPawn = std::make_unique<int16_t[]>(4 * kCorrSize);    // [stm][colour][key]
    std::unique_ptr<int16_t[]> corrCont = std::make_unique<int16_t[]>(2 * 12 * 64);         // [stm][prev piece][prev to]
    struct CorrSlots {
        int16_t *pawn, *white, *black, *cont; // cont may be null
    };
    static uint64_t mix(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdULL;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ULL;
        return x ^ (x >> 33);
    }
    CorrSlots corrSlots(int ply) const {
        const int stm = colorIndex(board.sideToMove());
        const uint64_t pawns = mix(board.pieceBB(Color::White, PAWN) ^ mix(board.pieceBB(Color::Black, PAWN)));
        uint64_t side[2];
        for (int c = 0; c < 2; ++c) {
            uint64_t h = 0;
            for (int t = KNIGHT; t <= KING; ++t) h = mix(h ^ board.pieceBB(static_cast<Color>(c), t) ^ static_cast<uint64_t>(t));
            side[c] = h;
        }
        CorrSlots c{&corrPawn[stm * kCorrSize + (pawns & (kCorrSize - 1))],
                    &corrNonPawn[(stm * 2 + 0) * kCorrSize + (side[0] & (kCorrSize - 1))],
                    &corrNonPawn[(stm * 2 + 1) * kCorrSize + (side[1] & (kCorrSize - 1))], nullptr};
        if (ply >= 1 && plyMoves[ply - 1].piece >= 0)
            c.cont = &corrCont[(stm * 12 + plyMoves[ply - 1].piece) * 64 + plyMoves[ply - 1].to];
        return c;
    }
    static int correction(const CorrSlots &c) {
        int sum = 2 * *c.pawn + *c.white + *c.black, weight = 4;
        if (c.cont) sum += 2 * *c.cont, weight += 2;
        return sum / (weight * kCorrGrain);
    }
    // Moves every slot towards the search result's distance from the raw
    // static evaluation, faster for deeper results.
    static void updateCorrection(const CorrSlots &c, int diffCp, int depth) {
        const int target = std::clamp(diffCp, -kCorrMaxCp, kCorrMaxCp) * kCorrGrain;
        const int w = std::min(depth + 1, 16);
        for (int16_t *e : {c.pawn, c.white, c.black, c.cont})
            if (e) *e = static_cast<int16_t>((*e * (256 - w) + target * w) / 256);
    }
    // Static evaluation at each ply (kNoEval when in check), for "improving":
    // whether the side to move stands better than two plies earlier.
    static constexpr int kNoEval = -1000000;
    int evalStack[MAX_PLY + 1];
    Move pvTable[MAX_PLY][MAX_PLY];
    int pvLength[MAX_PLY] = {};
    uint64_t nodes = 0;
    uint64_t unflushed = 0;
    SearchStats stats; // only counted with NAGS_STATS
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
    int moveScore(const Move &m, const Move &ttMove, int ply, const Move &counter = Move{},
                  const ContTable *cont1 = nullptr, const ContTable *cont2 = nullptr) const {
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
        int score = history[colorIndex(board.sideToMove())][m.from][m.to];
        if (cont1 || cont2) {
            const int piece = static_cast<int>(board.pieceAt(m.from)) - 1;
            if (cont1) score += (*cont1)[piece][m.to];
            if (cont2) score += (*cont2)[piece][m.to];
        }
        return score;
    }

    // Continuation table for the move played `back` plies before `ply`.
    ContTable *contFor(int ply, int back) const {
        if (ply < back) return nullptr;
        const PlyMove &pm = plyMoves[ply - back];
        return pm.piece < 0 ? nullptr : &contHist[pm.piece * 64 + pm.to];
    }
    void setPlyMove(int ply, const Move &m) {
        plyMoves[ply] = {m.isNull() ? -1 : static_cast<int>(board.pieceAt(m.from)) - 1, m.isNull() ? 0 : m.to};
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
    static void updateContHistory(int16_t &h, int bonus) {
        h = static_cast<int16_t>(h + bonus - h * std::abs(bonus) / kHistoryMax);
    }
    static constexpr int kHistoryMax = 16384;

    bool hasNonPawnMaterial(Color c) const {
        return (board.pieceBB(c, KNIGHT) | board.pieceBB(c, BISHOP) | board.pieceBB(c, ROOK) | board.pieceBB(c, QUEEN)) != 0;
    }

    // Staged move ordering for one node: TT move, winning/equal captures
    // (MVV/LVA), killers and countermove, losing captures, quiet moves by
    // history (ties in generation order). Moves are generated only when their
    // stage is reached: a TT move cutoff needs no move generation at all, and
    // quiet moves are generated and scored (with the history as it is then)
    // only after the captures and killers.
    class MovePicker {
    public:
        MovePicker(const SearchWorker &w, const Move &ttMove, int ply, const Move &counter, const ContTable *cont1,
                   const ContTable *cont2, bool noisyOnly)
            : w(w), cont1(cont1), cont2(cont2), noisyOnly(noisyOnly) {
            if (!ttMove.isNull() && w.board.toPseudoLegal(ttMove, tt) && (!noisyOnly || eval::isNoisy(w.board, tt)))
                hasTt = true;
            if (ply < MAX_PLY) {
                special[0] = w.killers[ply][0];
                special[1] = w.killers[ply][1];
                special[2] = counter;
            }
        }

        // The next move (index into moveAt) or -1 when there are none left.
        // Losing captures are left out with skipLosingCaptures.
        int next(bool skipLosingCaptures = false) {
            if (stage == Stage::Tt) {
                stage = Stage::GoodCaptures;
                if (hasTt) {
                    moves.push_back(tt);
                    return found(moves.size() - 1, false);
                }
            }
            if (stage == Stage::GoodCaptures) {
                if (!capturesGenerated) generateCaptures();
                if (nGood) return found(take(good, nGood), true);
                stage = Stage::Special;
            }
            if (stage == Stage::Special) {
                while (specialPos < 3) {
                    const Move &c = special[specialPos++];
                    Move m;
                    if (c.isNull() || isDuplicate(c, specialPos - 1) || !w.board.toPseudoLegal(c, m) ||
                        eval::isNoisy(w.board, m))
                        continue;
                    emitted[specialPos - 1] = true;
                    moves.push_back(m);
                    return found(moves.size() - 1, false);
                }
                stage = Stage::LosingCaptures;
            }
            if (stage == Stage::LosingCaptures) {
                if (nBad && !skipLosingCaptures) return found(take(bad, nBad), true);
                stage = Stage::Quiets;
            }
            if (!quietsGenerated) generateQuiets();
            if (nQuiet) return found(take(quiet, nQuiet), false);
            return -1;
        }

        const Move &moveAt(int k) const { return moves[k]; }
        // Stage of the move last returned by next(): 0 TT move, 1 winning
        // capture, 2 killer/countermove, 3 losing capture, 4 quiet.
        int lastStage() const { return lastStageIdx; }
        // Static exchange evaluation of the move last returned by next()
        // (computed only if the picker does not know it already).
        int seeOfLast() const { return lastSeeKnown ? see[last] : eval::see(w.board, moves[last]); }

    private:
        enum class Stage { Tt, GoodCaptures, Special, LosingCaptures, Quiets };

        int found(int k, bool seeKnown) {
            last = k;
            lastSeeKnown = seeKnown;
            if constexpr (kSearchStats)
                lastStageIdx = stage == Stage::GoodCaptures ? (hasTt && k == 0 ? 0 : 1)
                             : stage == Stage::Special ? 2 : stage == Stage::LosingCaptures ? 3 : 4;
            return k;
        }
        // Killer/countermove slot `i` repeats the TT move or an earlier slot.
        bool isDuplicate(const Move &c, int i) const {
            if (hasTt && sameMove(c, tt)) return true;
            for (int j = 0; j < i; ++j)
                if (sameMove(c, special[j])) return true;
            return false;
        }
        void generateCaptures() {
            capturesGenerated = true;
            const BoardT &b = w.board;
            const int start = moves.size();
            b.generateMoves(moves, GenMode::Noisy);
            for (int k = start; k < moves.size(); ++k) {
                const Move &m = moves[k];
                if (hasTt && sameMove(m, tt)) continue;
                int s = eval::capturedValue(b, m) * 16 - eval::pieceValue(b.pieceAt(m.from)) / 16;
                if (m.promotion != Piece::None) s += eval::pieceValue(m.promotion) * 16;
                score[k] = s;
                see[k] = static_cast<int16_t>(eval::see(b, m));
                if (see[k] >= 0) good[nGood++] = static_cast<uint8_t>(k);
                else bad[nBad++] = static_cast<uint8_t>(k);
            }
        }
        void generateQuiets() {
            quietsGenerated = true;
            if (noisyOnly) return;
            const BoardT &b = w.board;
            const int stm = colorIndex(b.sideToMove());
            const int start = moves.size();
            b.generateMoves(moves, GenMode::Quiet);
            for (int k = start; k < moves.size(); ++k) {
                const Move &m = moves[k];
                if (hasTt && sameMove(m, tt)) continue;
                bool dup = false;
                for (int j = 0; j < 3; ++j) dup |= emitted[j] && sameMove(m, special[j]);
                if (dup) continue;
                int s = w.history[stm][m.from][m.to];
                if (cont1 || cont2) {
                    const int piece = static_cast<int>(b.pieceAt(m.from)) - 1;
                    if (cont1) s += (*cont1)[piece][m.to];
                    if (cont2) s += (*cont2)[piece][m.to];
                }
                score[k] = s;
                quiet[nQuiet++] = static_cast<uint8_t>(k);
            }
        }
        // Removes and returns the highest-scored entry of list[0..n), the
        // earliest one on ties; the rest keep their order.
        int take(uint8_t *list, int &n) {
            int best = 0, bestScore = score[list[0]];
            for (int i = 1; i < n; ++i) {
                const int sc = score[list[i]];
                if (sc > bestScore) {
                    best = i;
                    bestScore = sc;
                }
            }
            const int k = list[best];
            for (int i = best + 1; i < n; ++i) list[i - 1] = list[i];
            --n;
            return k;
        }

        const SearchWorker &w;
        const ContTable *cont1, *cont2;
        const bool noisyOnly;
        MoveList moves;
        int score[MoveList::kCapacity];
        int16_t see[MoveList::kCapacity];
        uint8_t good[MoveList::kCapacity], bad[MoveList::kCapacity], quiet[MoveList::kCapacity];
        int nGood = 0, nBad = 0, nQuiet = 0;
        bool capturesGenerated = false, quietsGenerated = false;
        Move tt;
        bool hasTt = false;
        Move special[3]; // killer 1, killer 2, countermove
        bool emitted[3] = {false, false, false};
        int specialPos = 0;
        Stage stage = Stage::Tt;
        int last = -1;
        bool lastSeeKnown = false;
        int lastStageIdx = 0;
    };

    void updatePv(int ply, const Move &m) {
        pvTable[ply][ply] = m;
        int childLen = (ply + 1 < MAX_PLY) ? pvLength[ply + 1] : ply + 1;
        for (int i = ply + 1; i < childLen; ++i) pvTable[ply][i] = pvTable[ply + 1][i];
        pvLength[ply] = std::max(childLen, ply + 1);
    }

    // Searches root moves [first, end) (MultiPV skips the lines already found).
    int rootSearch(int depth, int alpha, int beta, IterationResult &out, size_t first = 0) {
        pvLength[0] = 0;
        evalStack[0] = board.inCheck() ? kNoEval : eval::evaluate(board);
        int bestScore = -INF_SCORE;
        int originalAlpha = alpha;
        for (size_t i = first; i < rootMoves.size(); ++i) {
            const Move m = rootMoves[i];
            setPlyMove(0, m);
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
        NAGS_STAT(++stats.nodes);
        selDepth = std::max(selDepth, ply);

        const bool excluding = !excluded.isNull();
        Move ttMove;
        TTHit hit;
        bool ttHit = false;
        uint64_t key = board.zobrist();
        if (tt && !excluding) {
            NAGS_STAT(++stats.ttProbes);
            if (tt->probe(key, hit)) {
                NAGS_STAT(++stats.ttHits);
                ttHit = true;
                ttMove = hit.move;
                if (!isPv && hit.depth >= depth) {
                    int s = scoreFromTT(hit.score, ply);
                    if (hit.bound == Bound::Exact || (hit.bound == Bound::Lower && s >= beta) ||
                        (hit.bound == Bound::Upper && s <= alpha)) {
                        NAGS_STAT(++stats.ttCutoffs);
                        return s;
                    }
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

        // Internal iterative reduction: without a TT move this node was not
        // searched before (or failed low everywhere); search it a ply less.
        if (tt && depth >= 4 && ttMove.isNull() && !excluding) --depth;

        // Static evaluation for the pruning decisions below (a singular
        // extension search reuses its node's).
        const CorrSlots corr = corrSlots(ply);
        const int rawEval = inCheck || excluding ? kNoEval : eval::evaluate(board);
        const int staticEval = inCheck ? kNoEval : excluding ? evalStack[ply]
                                                 : std::clamp(rawEval + correction(corr), -eval::kEvalLimit, eval::kEvalLimit);
        NAGS_STAT(if (!inCheck && !excluding) ++stats.evals);
        evalStack[ply] = staticEval;
        const bool improving = !inCheck && (ply < 2 || evalStack[ply - 2] == kNoEval || staticEval > evalStack[ply - 2]);

        // Reverse futility pruning: at shallow depth, a static evaluation that
        // beats beta by a depth-scaled margin is assumed to hold.
        if (!isPv && !inCheck && !excluding && depth <= 6 && std::abs(beta) < MATE_BOUND &&
            staticEval - 80 * (depth - improving) >= beta) {
            NAGS_STAT(++stats.rfp);
            return staticEval;
        }

        // Null-move pruning: if the side to move could pass and a reduced
        // search still fails high, the position is good enough to cut. Not in
        // check, at PV nodes, right after another null move, near mate scores,
        // or with only king and pawns (zugzwang is common there).
        if (!isPv && !inCheck && !excluding && depth >= 3 && !board.lastMoveWasNull() && std::abs(beta) < MATE_BOUND &&
            hasNonPawnMaterial(board.sideToMove()) && staticEval >= beta) {
            int reduction = 3 + depth / 6;
            NAGS_STAT(++stats.nmpTries);
            plyMoves[ply] = PlyMove{};
            board.makeNullMove();
            int score = -negamax(depth - 1 - reduction, -beta, -beta + 1, ply + 1, false);
            board.unmakeNullMove();
            if (aborted) return 0;
            if (score >= beta) {
                NAGS_STAT(++stats.nmpCutoffs);
                return score >= TB_BOUND ? beta : score; // don't trust unproven mates or TB wins
            }
        }

        // Pseudo-legal moves; legality is checked only for moves actually
        // tried (most nodes cut off after a few moves).
        const Move prevMove = board.lastMove();
        const Move counter = counterMoveFor(prevMove);
        ContTable *cont1 = contFor(ply, 1), *cont2 = contFor(ply, 2);
        MovePicker picker(*this, ttMove, ply, counter, cont1, cont2, false);

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
        uint8_t quietsSearched[MoveList::kCapacity]; // quiet moves searched without a cutoff (picker indices)
        int nQuietsSearched = 0;
        for (int idx; (idx = picker.next()) >= 0;) {
            const Move m = picker.moveAt(idx);
            if (excluding && sameMove(m, excluded)) continue;
            bool quiet = !eval::isNoisy(board, m);
            int histScore = 0;
            if (quiet) {
                histScore = history[colorIndex(us)][m.from][m.to];
                const int piece = static_cast<int>(board.pieceAt(m.from)) - 1;
                if (cont1) histScore += (*cont1)[piece][m.to];
                if (cont2) histScore += (*cont2)[piece][m.to];
            }
            // Shallow quiet-move pruning. Only once a legal move has been
            // searched, so checkmate / stalemate detection is unaffected.
            if (!isPv && !inCheck && quiet && i > 0 && depth <= 3 && std::abs(alpha) < MATE_BOUND) {
                if (quietsTried >= (3 + depth * depth) / (improving ? 1 : 2)) { // late-move pruning
                    NAGS_STAT(++stats.lmp);
                    continue;
                }
                if (staticEval + 120 * depth <= alpha) { // futility pruning
                    NAGS_STAT(++stats.futility);
                    continue;
                }
                if (histScore < -4096 * depth) { // history pruning
                    NAGS_STAT(++stats.historyPrunes);
                    continue;
                }
            }
            // SEE pruning: at low depth, skip moves that lose material by
            // force (more for captures, which win some of it back).
            if (!isPv && !inCheck && i > 0 && depth <= 8 && std::abs(alpha) < MATE_BOUND && !sameMove(m, ttMove) &&
                picker.seeOfLast() < (quiet ? -60 * depth : -20 * depth * depth)) {
                NAGS_STAT(++stats.seePrunes);
                continue;
            }
            // Singular extension: if every other move fails low against a
            // margin below the TT score, the TT move is singular and gets one
            // more ply. If even the alternatives beat beta, cut (multi-cut).
            int extension = 0;
            if (trySingular && sameMove(m, ttMove)) {
                int singularBeta = ttScore - 2 * depth;
                NAGS_STAT(++stats.singularTries);
                int s = negamax((depth - 1) / 2, singularBeta - 1, singularBeta, ply, false, m);
                if (aborted) return 0;
                if (s < singularBeta) {
                    extension = 1;
                    NAGS_STAT(++stats.singularExtensions);
                } else if (singularBeta >= beta) {
                    NAGS_STAT(++stats.multiCuts);
                    return singularBeta;
                }
            }
            setPlyMove(ply, m);
            board.makeMove(m);
            if (board.inCheck(us)) { // illegal: leaves our own king in check
                board.unmakeMove();
                NAGS_STAT(++stats.illegal);
                continue;
            }
            NAGS_STAT(++stats.movesSearched);
            bool givesCheck = board.inCheck();
            int score;
            if (i == 0) {
                score = -negamax(depth - 1 + extension, -beta, -alpha, ply + 1, isPv);
            } else {
                int reduction = 0;
                if (depth >= 3 && i >= 2 && quiet && !inCheck && !givesCheck &&
                    !sameMove(m, killers[ply][0]) && !sameMove(m, killers[ply][1])) {
                    reduction = lmrReduction(depth, i + 1) - (isPv ? 1 : 0) + (improving ? 0 : 1) - histScore / 8192;
                    reduction = std::clamp(reduction, 0, depth - 2);
                }
                NAGS_STAT(if (reduction) ++stats.lmrSearches);
                score = -negamax(depth - 1 + extension - reduction, -alpha - 1, -alpha, ply + 1, false);
                if (!aborted && reduction && score > alpha) {
                    NAGS_STAT(++stats.lmrResearches);
                    score = -negamax(depth - 1 + extension, -alpha - 1, -alpha, ply + 1, false);
                }
                if (!aborted && score > alpha && score < beta) {
                    NAGS_STAT(++stats.pvsResearches);
                    score = -negamax(depth - 1 + extension, -beta, -alpha, ply + 1, true);
                }
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
                        NAGS_STAT(++stats.cutNodes; ++stats.cutoffAt[std::min(i - 1, 7)];
                                  ++stats.cutoffByStage[picker.lastStage()]);
                        if (quiet) {
                            if (!sameMove(m, killers[ply][0])) {
                                killers[ply][1] = killers[ply][0];
                                killers[ply][0] = m;
                            }
                            // Reward the cutoff move and penalise the quiet
                            // moves tried before it; values decay towards
                            // zero as they approach the bound.
                            int bonus = std::min(150 * depth - 100, 1500);
                            auto &table = history[colorIndex(board.sideToMove())];
                            updateHistory(table[m.from][m.to], bonus);
                            for (int q = 0; q < nQuietsSearched; ++q) {
                                const Move &qm = picker.moveAt(quietsSearched[q]);
                                updateHistory(table[qm.from][qm.to], -bonus);
                            }
                            for (ContTable *cont : {cont1, cont2}) {
                                if (!cont) continue;
                                updateContHistory((*cont)[static_cast<int>(board.pieceAt(m.from)) - 1][m.to], bonus);
                                for (int q = 0; q < nQuietsSearched; ++q) {
                                    const Move &qm = picker.moveAt(quietsSearched[q]);
                                    updateContHistory((*cont)[static_cast<int>(board.pieceAt(qm.from)) - 1][qm.to], -bonus);
                                }
                            }
                            if (!prevMove.isNull()) {
                                Piece p = board.pieceAt(prevMove.to);
                                if (p != Piece::None) counterMoves[static_cast<int>(p) - 1][prevMove.to] = m;
                            }
                        }
                        break;
                    }
                }
            }
            if (quiet) quietsSearched[nQuietsSearched++] = static_cast<uint8_t>(idx);
        }

        NAGS_STAT(if (best < beta) ++(best > originalAlpha ? stats.pvNodes : stats.allNodes));
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
        // The search result corrects later static evaluations, when it is a
        // usable signal: not in check, a quiet best move (captures change the
        // material the evaluation saw), and a bound on the right side of it.
        if (!inCheck && !excluding && std::abs(best) < TB_BOUND && (bestMove.isNull() || !eval::isNoisy(board, bestMove)) &&
            !(bound == Bound::Lower && best <= staticEval) && !(bound == Bound::Upper && best >= staticEval))
            updateCorrection(corr, best - rawEval, depth);
        if (tt && !excluding) tt->store(key, depth, bound, scoreToTT(best, ply), bestMove);
        return best;
    }

    // Captures and promotions only (every evasion when in check). Results go
    // to the TT at depth 0; non-PV nodes take cutoffs from it.
    int quiescence(int alpha, int beta, int ply) {
        if (checkStop()) return 0;
        ++nodes;
        NAGS_STAT(++stats.qnodes);
        selDepth = std::max(selDepth, ply);
        if (ply >= MAX_PLY - 1) return eval::evaluate(board);

        const bool isPv = beta - alpha > 1;
        const uint64_t key = board.zobrist();
        Move ttMove;
        TTHit hit;
        bool ttHit = false;
        NAGS_STAT(if (tt) ++stats.qttProbes);
        if (tt && tt->probe(key, hit)) {
            NAGS_STAT(++stats.qttHits);
            ttHit = true;
            ttMove = hit.move;
            if (!isPv) {
                int s = scoreFromTT(hit.score, ply);
                if (hit.bound == Bound::Exact || (hit.bound == Bound::Lower && s >= beta) ||
                    (hit.bound == Bound::Upper && s <= alpha)) {
                    NAGS_STAT(++stats.qttCutoffs);
                    return s;
                }
            }
        }

        bool inCheck = board.inCheck();
        int best = -INF_SCORE;
        int stand = 0;
        const int originalAlpha = alpha;
        if (!inCheck) {
            {
                const CorrSlots corr = corrSlots(0); // without the previous-move slot (not tracked in quiescence)
                stand = std::clamp(eval::evaluate(board) + correction(corr), -eval::kEvalLimit, eval::kEvalLimit);
            }
            NAGS_STAT(++stats.evals);
            // A TT score whose bound points the right way is a better estimate.
            if (ttHit && std::abs(hit.score) < TB_BOUND) {
                int s = hit.score;
                if (hit.bound == Bound::Exact || (hit.bound == Bound::Lower && s > stand) ||
                    (hit.bound == Bound::Upper && s < stand))
                    stand = s;
            }
            if (stand >= beta) {
                NAGS_STAT(++stats.qStandPat);
                if (tt && !ttHit) tt->store(key, 0, Bound::Lower, scoreToTT(stand, ply), Move{});
                return stand;
            }
            if (stand > alpha) alpha = stand;
            best = stand;
        }

        // In check every evasion must be considered (no stand-pat).
        // The TT move only counts if this generator produced it (a quiet TT
        // move is not searched outside check).
        MovePicker picker(*this, ttMove, MAX_PLY, Move{}, nullptr, nullptr, !inCheck);

        const Color us = board.sideToMove();
        int legal = 0;
        Move bestMove;
        // Outside check losing captures are never searched (below), so the
        // picker need not return them.
        for (int idx; (idx = picker.next(!inCheck)) >= 0;) {
            const Move m = picker.moveAt(idx);
            if (!inCheck && m.promotion == Piece::None && stand + eval::capturedValue(board, m) + 200 <= alpha) {
                NAGS_STAT(++stats.qDelta);
                continue; // delta pruning
            }
            if (!inCheck && picker.seeOfLast() < 0) { // losing capture
                NAGS_STAT(++stats.qSee);
                continue;
            }
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
                    if (alpha >= beta) {
                        NAGS_STAT(++stats.qCutoffs);
                        break;
                    }
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

    // Search statistics summed over all threads since the last reset (only
    // counted in NAGS_STATS builds).
    SearchStats statistics() const {
        SearchStats total;
        for (const auto &w : workers) total.add(w->statistics());
        return total;
    }
    void resetStatistics() {
        for (auto &w : workers) w->resetStatistics();
    }

private:
    TranspositionTable *tt;
    SearchControl control;
    std::vector<std::unique_ptr<SearchWorker<BoardT>>> workers;
    int multiPv = 1;
};
