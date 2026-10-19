#pragma once

// NAGS hybrid search controller: "MCTS proposes, alpha-beta verifies".
//
// Two search arms work on each move:
//   * alpha-beta - the engine's full Searcher (the same search, NNUE
//     evaluation, time management, threads and tablebases as nags_enhanced);
//   * MCTS       - PUCT tree search whose priors and leaf values come from the
//                  GNN served by rpc_server.py, on its own thread.
// MCTS only runs when the GNN service answers: with the heuristic fallback
// evaluator it has nothing the alpha-beta search lacks, so without the
// service nags plays exactly like nags_enhanced.
//
// Final move: the alpha-beta result, unless MCTS clearly prefers another
// move and a verification search confirms that move is no worse.
//
// The meta-learner service (meta_learner.py) adjusts, per move, the
// verification depth (dfs_depth_delta), the MCTS simulation budget
// (mcts_budget_delta) and the PUCT exploration constant
// (bandit_exploration_delta; the name is kept for the service protocol).

#include "FastBoard.h"
#include "MetaClient.h"
#include "Net.h"
#include "Search.h"
#include "TT.h"

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

// Value in [-1, 1] from the side to move's point of view and one prior per
// legal move (same order as the move list passed in, summing to 1).
struct EvalResult {
    std::vector<float> priors;
    float value = 0.0f;
    float uncertainty = 0.0f;
};

// Network answers by position, so a position the MCTS meets again (a
// transposition, or the next move's search through the same lines) costs
// no round trip. Keyed by the Zobrist hash and the half-move clock (a
// network input); the legal-move list is checked too, since the priors
// follow its order. Fixed size, always-replace.
class EvalCache {
public:
    void resize(size_t megabytes);
    void clear();
    bool probe(uint64_t key, const std::vector<Move> &legal, EvalResult &out);
    void store(uint64_t key, const std::vector<Move> &legal, const EvalResult &r);
    bool enabled() const { return !table.empty(); }
    static uint64_t keyOf(const FastBoard &board);
    long long hits = 0, misses = 0;

private:
    struct Entry {
        uint64_t key = 0, movesSig = 0;
        float value = 0.0f, uncertainty = 0.0f;
        std::vector<float> priors;
    };
    static uint64_t signature(const std::vector<Move> &legal);
    std::vector<Entry> table;
    size_t mask = 0;
};

// One position for the network: its FEN and legal moves (the priors come
// back in this order).
struct EvalRequest {
    std::string fen;
    std::vector<Move> legal;
    // For a server whose network uses game histories: the position before
    // `historyMoves` (empty: none sent).
    std::string historyFen;
    std::vector<Move> historyMoves;
};

// Fills r's history from the board (the moves since the last capture or pawn
// move, and the position before them); the board is unchanged afterwards.
void fillHistory(FastBoard &board, EvalRequest &r);

class Evaluator {
public:
    virtual ~Evaluator() = default;
    virtual EvalResult evaluate(FastBoard &board, const std::vector<Move> &legal) = 0;
};

class HeuristicEvaluator : public Evaluator {
public:
    EvalResult evaluate(FastBoard &board, const std::vector<Move> &legal) override;
};

// Queries rpc_server.py; falls back to the heuristic when the server is not
// reachable (and then waits a while before trying again).
class RpcEvaluator : public Evaluator {
public:
    explicit RpcEvaluator(Evaluator &fallback) : fallback(fallback) {}
    void configure(bool enabled, const std::string &host, int port);
    // Longest wait for one answer (a slow answer must not overrun the move's time).
    void setRequestTimeout(int ms) { requestTimeoutMs = ms; }
    EvalResult evaluate(FastBoard &board, const std::vector<Move> &legal) override;
    bool lastUsedNetwork() const { return usedNetwork; }

    // Batches: several positions in one request, and requests that are sent
    // before the replies to earlier ones are read (the server answers in
    // order). A failed send or receive closes the connection, drops what is
    // outstanding and backs off like evaluate().
    bool available();
    bool sendBatch(const std::vector<const EvalRequest *> &batch);
    bool receiveBatch(const std::vector<const EvalRequest *> &batch, std::vector<EvalResult> &out);
    // Largest batch expected to come back well within the request timeout,
    // from the measured time per position (at most `wanted`).
    int batchLimit(int wanted) const;
    // The served network uses game histories (its replies say "history": true).
    bool wantsHistory() const { return serverHistory; }

    EvalCache &cache() { return evalCache; }
    void resetCache() { evalCache.clear(); }

    // Round trips to the service since the last reset: count, failures, the
    // positions sent and the time spent waiting for answers.
    struct Timing {
        long long requests = 0, failures = 0, positions = 0;
        double waitMs = 0.0, maxMs = 0.0;
    };
    const Timing &timing() const { return stats; }
    void resetTiming() { stats = Timing{}; }

private:
    void fail(std::chrono::steady_clock::time_point now);
    Evaluator &fallback;
    bool enabled = true;
    std::string host = "127.0.0.1";
    int port = 5555;
    int requestTimeoutMs = 3000;
    LineSocket socket;
    std::chrono::steady_clock::time_point retryAfter{};
    bool usedNetwork = false;
    Timing stats;
    EvalCache evalCache;
    // Requests sent and not yet answered, with their send times.
    std::vector<std::chrono::steady_clock::time_point> outstanding;
    std::chrono::steady_clock::time_point lastReply{};
    double msPerPosition = 0.0; // running estimate (0: unknown)
    bool serverHistory = false;
};

struct MCTSNode {
    Move move;
    MCTSNode *parent = nullptr;
    std::vector<MCTSNode> children; // allocated once at expansion, never resized
    int visits = 0;
    double valueSum = 0.0; // from the point of view of the side that played `move`
    float prior = 0.0f;
    bool expanded = false;
    bool terminal = false;
    bool pending = false;       // queued for the network (batched MCTS)
    float terminalValue = 0.0f; // side to move's view, used when terminal

    double q() const { return visits > 0 ? valueSum / visits : 0.0; }
};

struct NagsSettings {
    int baseMctsBudget = 2000;    // MCTS simulations per move before meta-learner deltas
    float baseExploration = 1.4f; // PUCT constant before meta-learner deltas
    int verifyMargin = 25;        // an MCTS move may score this much below the alpha-beta move
    int minMctsMs = 1000;         // MCTS only runs when the soft time limit is at least this (or unlimited)

    // Both services are off by default: neither has shown a gain in strength
    // (MCTS with an untrained GNN cost -64 +- 62 Elo and takes 15% of the
    // alpha-beta time whenever it runs; the meta-learner was never measured).
    // The training pipeline turns them on explicitly.
    bool useMetaLearner = false;
    std::string metaHost = "127.0.0.1";
    int metaPort = 5556;
    float metaExploration = 0.0f; // std-dev of Gaussian noise added to deltas (self-play)

    bool useNN = false;
    std::string nnHost = "127.0.0.1";
    int nnPort = 5555;

    // Neural inference. nnBatch leaves are selected with virtual loss and
    // evaluated in one request (1: one request per simulation, the original
    // MCTS); with nnPipeline the next batch is sent before the previous
    // one's reply has arrived. nnCacheMb: evaluation cache (0 = off).
    int nnBatch = 1;
    bool nnPipeline = false;
    int nnCacheMb = 32;

    // Hybrid search (all off / at their old values by default; see
    // docs/ARCHITECTURE.md "Hybrid search options").
    bool mctsReuse = false;       // keep the tree and continue from the new root's subtree next move
    int cpuctBase = 0;            // > 0: PUCT constant grows as cpuct + ln((1 + N + base) / base)
    double fpuReduction = 0.2;    // unvisited children start this far below the parent's value
    float policyTemp = 1.0f;      // priors become p^(1/T) (renormalised) at expansion
    bool smartStop = false;       // stop MCTS once its most visited move cannot be overtaken
    int abSharePct = 85;          // alpha-beta's share of the time when MCTS runs
    int ttValueDepth = 0;         // > 0: leaves with an exact TT score this deep use it for their value
    bool nnRootOrder = false;     // alpha-beta searches root moves in the order of the network's prior
};

struct NagsResult {
    Move bestMove;
    Move ponderMove;
    int score = 0;
    uint64_t nodes = 0; // alpha-beta nodes + MCTS simulations
};

class NAGSController {
public:
    // Uses (and shares the transposition table of) the engine's alpha-beta searcher.
    explicit NAGSController(Searcher<FastBoard> &searcher);

    // Resets the RNG, the uncertainty carried between moves and the evaluation cache.
    void newGame();
    NagsSettings &settings() { return config; }
    const NagsSettings &settings() const { return config; }

    // timeLeftMs: our clock (or -1 if unknown); fed to the meta-learner.
    // ponder: the GUI's ponder flag (MCTS does not run while pondering).
    NagsResult search(const FastBoard &root, const SearchLimits &limits, int64_t timeLeftMs,
                      const std::atomic<bool> &stop, const std::atomic<bool> *ponder,
                      const std::function<void(const SearchInfo &)> &onInfo,
                      const std::function<void(const std::string &)> &onString);

private:
    Searcher<FastBoard> &searcher;
    NagsSettings config;
    HeuristicEvaluator heuristic;
    RpcEvaluator network;
    MetaClient meta;
    static constexpr uint32_t kSeed = 0xC0FFEEu;
    std::mt19937 rng;
    float lastUncertainty = 0.1f;
    int cacheMbApplied = -1; // size the evaluation cache was allocated with (-1: not yet)

    // MCTS state for the current move (used by the MCTS thread only).
    double mctsMs = 0.0; // time the MCTS thread ran
    std::unique_ptr<MCTSNode> mctsRoot;
    FastBoard mctsBoard;
    int mctsSims = 0;
    size_t mctsNodes = 0;
    float cpuct = 1.4f;
    bool mctsUsedNetwork = false;
    long long mctsCollisions = 0; // batched MCTS: selections that ran into a leaf already queued
    long long mctsBatches = 0;
    long long mctsTTValues = 0;   // leaves valued by an exact TT score
    int mctsReusedVisits = 0;     // visits the root brought along from the previous search
    bool mctsSmartStopped = false;
    // Tree reuse: the previous search's tree and the FEN of its root.
    std::string reuseRootFen;
    std::chrono::steady_clock::time_point mctsDeadline{}; // the alpha-beta arm's hard limit (smart stop)
    std::chrono::steady_clock::time_point mctsStart{};
    static constexpr size_t kMaxMctsNodes = 2000000; // ~150 MB

    // A leaf waiting for the network, with the moves leading to it (to
    // evaluate it with the fallback if the request fails).
    struct PendingLeaf {
        MCTSNode *node = nullptr;
        uint64_t key = 0; // EvalCache key of its position
        bool hasTTValue = false;
        float ttValue = 0.0f; // its value from an exact TT score (MctsTTValue)
        std::vector<Move> path;
        EvalRequest request;
    };
    enum class Selected { Queued, Resolved, Collision };

    void runMcts(long long budget, const std::atomic<bool> &stopMcts);
    void runMctsBatched(long long budget, const std::atomic<bool> &stopMcts);
    void simulate();
    Selected selectLeaf(std::vector<PendingLeaf> &batch);
    void backup(MCTSNode *leaf, float value, bool virtualLoss);
    void revertVirtualLoss(MCTSNode *leaf);
    void evaluateWithFallback(std::vector<PendingLeaf> &batch);
    bool shouldSmartStop(long long budget) const;
    bool ttValue(const FastBoard &board, float &value) const;
    std::unique_ptr<MCTSNode> reuseSubtree(const FastBoard &root);
    MCTSNode *selectChild(MCTSNode *node) const;
    void expand(MCTSNode *node, const std::vector<Move> &legal, const EvalResult &eval);
    const MCTSNode *mostVisitedRootChild() const;
    float tacticalRatio(const FastBoard &root) const;
};
