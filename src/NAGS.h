#pragma once

// NAGS hybrid search controller.
//
// Two search "arms" share the thinking time:
//   * DFS  - the shared alpha-beta SearchWorker, deepened one iteration per pull;
//   * MCTS - PUCT tree search whose priors and leaf values come from an
//            Evaluator (the GNN served by rpc_server.py when it is running,
//            otherwise a deterministic material/PST + capture-search heuristic).
// A Thompson-sampling bandit picks the arm for each pull. An arm "succeeds"
// when the pull changed its recommendation (best move, or a significant value
// shift) - i.e. it is still producing new information - so time flows to the
// arm that is learning the most about the position.
//
// The meta-learner service (meta_learner.py) can adjust the DFS depth cap,
// the MCTS simulation budget and the PUCT exploration constant per move.
//
// Final move: the deepest completed alpha-beta result, unless MCTS strongly
// prefers another move and a verification search confirms that move is no
// worse ("MCTS proposes, alpha-beta verifies").

#include "Board.h"
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

enum class SearchArm : int { DFS = 0, MCTS = 1 };

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
    virtual EvalResult evaluate(Board &board, const std::vector<Move> &legal) = 0;
};

class HeuristicEvaluator : public Evaluator {
public:
    EvalResult evaluate(Board &board, const std::vector<Move> &legal) override;
};

// Queries rpc_server.py; falls back to the heuristic when the server is not
// reachable (and then waits a while before trying again).
class RpcEvaluator : public Evaluator {
public:
    explicit RpcEvaluator(Evaluator &fallback) : fallback(fallback) {}
    void configure(bool enabled, const std::string &host, int port);
    EvalResult evaluate(Board &board, const std::vector<Move> &legal) override;
    bool lastUsedNetwork() const { return usedNetwork; }

private:
    Evaluator &fallback;
    bool enabled = true;
    std::string host = "127.0.0.1";
    int port = 5555;
    LineSocket socket;
    std::chrono::steady_clock::time_point retryAfter{};
    bool usedNetwork = false;
};

class BayesianBandit {
public:
    BayesianBandit() { reset(); }
    void reset();
    SearchArm select(std::mt19937 &rng);
    // Bernoulli update with mild forgetting so the bandit can adapt as the
    // game changes character.
    void update(SearchArm arm, bool success);
    int pulls(SearchArm arm) const { return arms[static_cast<int>(arm)].pulls; }
    int successes(SearchArm arm) const { return arms[static_cast<int>(arm)].successes; }

private:
    struct ArmStats {
        double alpha = 1.0, beta = 1.0;
        int pulls = 0, successes = 0;
    };
    ArmStats arms[2];
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
    int baseDfsDepth = 10;
    int baseMctsBudget = 2000;
    float baseExploration = 1.4f;

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
    explicit NAGSController(TranspositionTable &tt);

    // Resets the bandit, history tables and RNG; searches after newGame() are
    // reproducible when they have no time limit (used by `bench`).
    void newGame();
    NagsSettings &settings() { return config; }
    const NagsSettings &settings() const { return config; }

    // timeLeftMs: our clock (or -1 if unknown); fed to the meta-learner.
    NagsResult search(const Board &root, const SearchLimits &limits, int64_t timeLeftMs,
                      const std::atomic<bool> &stop, const std::function<void(const SearchInfo &)> &onInfo,
                      const std::function<void(const std::string &)> &onString);

private:
    TranspositionTable &tt;
    NagsSettings config;
    SearchControl control;
    std::unique_ptr<SearchWorker<Board>> dfs;
    HeuristicEvaluator heuristic;
    RpcEvaluator network;
    MetaClient meta;
    BayesianBandit bandit;
    static constexpr uint32_t kSeed = 0xC0FFEEu;
    std::mt19937 rng;
    float lastUncertainty = 0.1f;

    // Per-search state
    std::function<void(const SearchInfo &)> infoCallback;
    int dfsDepth = 0;
    int dfsScore = 0;
    Move dfsBest;
    std::vector<Move> dfsPv;
    std::unique_ptr<MCTSNode> mctsRoot;
    Board mctsBoard;
    int mctsSims = 0;
    size_t mctsNodes = 0;
    float cpuct = 1.4f;
    static constexpr size_t kMaxMctsNodes = 2000000; // ~150 MB

    bool runDfsIteration();
    void reportDfs(int depth);
    void runMctsBatch(int simulations, int64_t sliceMs);
    void simulate();
    MCTSNode *selectChild(MCTSNode *node) const;
    void expand(MCTSNode *node, const std::vector<Move> &legal, const EvalResult &eval);
    const MCTSNode *mostVisitedRootChild() const;
    float tacticalRatio(const Board &root) const;
};
