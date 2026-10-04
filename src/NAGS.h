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

private:
    Evaluator &fallback;
    bool enabled = true;
    std::string host = "127.0.0.1";
    int port = 5555;
    int requestTimeoutMs = 3000;
    LineSocket socket;
    std::chrono::steady_clock::time_point retryAfter{};
    bool usedNetwork = false;
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
    float terminalValue = 0.0f; // side to move's view, used when terminal

    double q() const { return visits > 0 ? valueSum / visits : 0.0; }
};

struct NagsSettings {
    int baseMctsBudget = 2000;    // MCTS simulations per move before meta-learner deltas
    float baseExploration = 1.4f; // PUCT constant before meta-learner deltas
    int verifyMargin = 25;        // an MCTS move may score this much below the alpha-beta move
    int minMctsMs = 1000;         // MCTS only runs when the soft time limit is at least this (or unlimited)

    bool useMetaLearner = true;
    std::string metaHost = "127.0.0.1";
    int metaPort = 5556;
    float metaExploration = 0.0f; // std-dev of Gaussian noise added to deltas (self-play)

    bool useNN = true;
    std::string nnHost = "127.0.0.1";
    int nnPort = 5555;
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

    // Resets the RNG and the uncertainty carried between moves.
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

    // MCTS state for the current move (used by the MCTS thread only).
    std::unique_ptr<MCTSNode> mctsRoot;
    FastBoard mctsBoard;
    int mctsSims = 0;
    size_t mctsNodes = 0;
    float cpuct = 1.4f;
    bool mctsUsedNetwork = false;
    static constexpr size_t kMaxMctsNodes = 2000000; // ~150 MB

    void runMcts(long long budget, const std::atomic<bool> &stopMcts);
    void simulate();
    MCTSNode *selectChild(MCTSNode *node) const;
    void expand(MCTSNode *node, const std::vector<Move> &legal, const EvalResult &eval);
    const MCTSNode *mostVisitedRootChild() const;
    float tacticalRatio(const FastBoard &root) const;
};
