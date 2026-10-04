#include "NAGS.h"

#include "Eval.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>

namespace {

constexpr int kNnConnectTimeoutMs = 150;
constexpr int kNnRequestTimeoutMs = 3000;
constexpr int kNnBackoffSeconds = 60;

// Centipawns <-> [-1, 1] value scale shared by the heuristic evaluator and logs.
inline float cpToValue(int cp) { return static_cast<float>(std::tanh(cp / 400.0)); }
inline int valueToCp(double v) {
    v = std::clamp(v, -0.999, 0.999);
    return static_cast<int>(std::lround(400.0 * std::atanh(v)));
}

void softmaxInPlace(std::vector<float> &x) {
    if (x.empty()) return;
    float mx = *std::max_element(x.begin(), x.end());
    float sum = 0.0f;
    for (float &v : x) {
        v = std::exp(v - mx);
        sum += v;
    }
    for (float &v : x) v /= sum;
}

} // namespace

// --------------------------------------------------------------------------
// Evaluators

EvalResult HeuristicEvaluator::evaluate(FastBoard &board, const std::vector<Move> &legal) {
    EvalResult r;
    r.value = cpToValue(eval::quiescence(board, -INF_SCORE, INF_SCORE, 6));
    r.uncertainty = 0.0f;
    r.priors.reserve(legal.size());
    Color us = board.sideToMove();
    for (const Move &m : legal) {
        float s = 0.0f;
        Piece mover = board.pieceAt(m.from);
        if (eval::isNoisy(board, m)) {
            s += 1.0f + (eval::capturedValue(board, m) - eval::pieceValue(mover) / 10.0f) / 300.0f;
            if (m.promotion != Piece::None) s += pieceTypeOf(m.promotion) == QUEEN ? 2.0f : -1.0f;
        } else {
            int type = pieceTypeOf(mover);
            if (type >= 0) {
                int delta = eval::PST[type][eval::pstIndex(us, m.to)] - eval::PST[type][eval::pstIndex(us, m.from)];
                s += delta / 50.0f;
            }
        }
        if (m.isCastling) s += 0.5f;
        r.priors.push_back(s);
    }
    softmaxInPlace(r.priors);
    return r;
}

void RpcEvaluator::configure(bool on, const std::string &h, int p) {
    if (on != enabled || h != host || p != port) {
        socket.close();
        retryAfter = {};
    }
    enabled = on;
    host = h;
    port = p;
}

EvalResult RpcEvaluator::evaluate(FastBoard &board, const std::vector<Move> &legal) {
    usedNetwork = false;
    auto now = std::chrono::steady_clock::now();
    if (!enabled || now < retryAfter) return fallback.evaluate(board, legal);
    if (!socket.isOpen() && !socket.connect(host, port, kNnConnectTimeoutMs)) {
        retryAfter = now + std::chrono::seconds(kNnBackoffSeconds);
        return fallback.evaluate(board, legal);
    }

    std::string request = "{\"fens\":[\"" + jsonlite::escape(board.getFEN()) + "\"],\"moves\":[[";
    for (size_t i = 0; i < legal.size(); ++i) {
        if (i) request += ',';
        request += '"' + moveToUciString(legal[i]) + '"';
    }
    request += "]]}";

    std::string response;
    std::vector<double> priors;
    double value = 0, uncertainty = 0;
    if (!socket.request(request, response, kNnRequestTimeoutMs) ||
        !jsonlite::findNumberArray(response, "move_priors", priors) || priors.size() != legal.size() ||
        !jsonlite::findNumber(response, "value", value)) {
        socket.close();
        retryAfter = now + std::chrono::seconds(kNnBackoffSeconds);
        return fallback.evaluate(board, legal);
    }
    jsonlite::findNumber(response, "uncertainty", uncertainty);

    EvalResult r;
    r.value = static_cast<float>(std::clamp(value, -1.0, 1.0));
    r.uncertainty = static_cast<float>(std::max(0.0, uncertainty));
    double sum = 0;
    for (double p : priors) sum += std::max(0.0, p);
    r.priors.reserve(legal.size());
    for (double p : priors)
        r.priors.push_back(sum > 0 ? static_cast<float>(std::max(0.0, p) / sum) : 1.0f / legal.size());
    usedNetwork = true;
    return r;
}

// --------------------------------------------------------------------------
// Controller

NAGSController::NAGSController(Searcher<FastBoard> &s) : searcher(s), network(heuristic), rng(kSeed) {}

void NAGSController::newGame() {
    lastUncertainty = 0.1f;
    rng.seed(kSeed);
}

float NAGSController::tacticalRatio(const FastBoard &root) const {
    // Share of legal moves that capture, promote or give check.
    FastBoard b = root;
    auto moves = b.generateLegalMoves();
    if (moves.empty()) return 0.0f;
    int tactical = 0;
    for (const Move &m : moves) {
        if (eval::isNoisy(b, m)) { ++tactical; continue; }
        b.makeMove(m);
        if (b.inCheck()) ++tactical;
        b.unmakeMove();
    }
    return static_cast<float>(tactical) / moves.size();
}

void NAGSController::expand(MCTSNode *node, const std::vector<Move> &legal, const EvalResult &e) {
    node->children.resize(legal.size());
    for (size_t i = 0; i < legal.size(); ++i) {
        MCTSNode &child = node->children[i];
        child.move = legal[i];
        child.parent = node;
        child.prior = i < e.priors.size() ? e.priors[i] : 1.0f / legal.size();
    }
    node->expanded = true;
    mctsNodes += legal.size();
}

MCTSNode *NAGSController::selectChild(MCTSNode *node) const {
    double sqrtN = std::sqrt(std::max(1, node->visits));
    // First-play urgency: unvisited children start slightly below the
    // parent's own value (from the side to move at `node`).
    double fpu = (node->visits > 0 ? -node->q() : 0.0) - 0.2;
    MCTSNode *best = nullptr;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (auto &c : node->children) {
        double q = c.visits > 0 ? c.q() : fpu;
        double u = cpuct * c.prior * sqrtN / (1.0 + c.visits);
        if (q + u > bestScore) {
            bestScore = q + u;
            best = &c;
        }
    }
    return best;
}

void NAGSController::simulate() {
    MCTSNode *node = mctsRoot.get();
    int depth = 0;
    while (node->expanded && !node->terminal && !node->children.empty()) {
        node = selectChild(node);
        mctsBoard.makeMove(node->move);
        ++depth;
    }

    float value; // side to move at `node`
    if (node->terminal) {
        value = node->terminalValue;
    } else if (depth > 0 && mctsBoard.isDraw()) {
        node->terminal = true;
        node->terminalValue = 0.0f;
        value = 0.0f;
    } else {
        auto legal = mctsBoard.generateLegalMoves();
        if (legal.empty()) {
            node->terminal = true;
            node->terminalValue = mctsBoard.inCheck() ? -1.0f : 0.0f; // checkmated / stalemate
            value = node->terminalValue;
        } else {
            EvalResult e = network.evaluate(mctsBoard, legal);
            mctsUsedNetwork = mctsUsedNetwork || network.lastUsedNetwork();
            expand(node, legal, e);
            value = e.value;
        }
    }

    // Each node stores value from the view of the player who moved into it,
    // i.e. the opponent of the side to move there.
    for (MCTSNode *n = node; n; n = n->parent) {
        n->visits += 1;
        n->valueSum += -value;
        value = -value;
    }
    for (int i = 0; i < depth; ++i) mctsBoard.unmakeMove();
}

void NAGSController::runMcts(long long budget, const std::atomic<bool> &stopMcts) {
    while (mctsSims < budget && mctsNodes < kMaxMctsNodes && !stopMcts.load(std::memory_order_relaxed)) {
        simulate();
        ++mctsSims;
    }
}

const MCTSNode *NAGSController::mostVisitedRootChild() const {
    const MCTSNode *best = nullptr;
    if (!mctsRoot) return best;
    for (const auto &c : mctsRoot->children)
        if (!best || c.visits > best->visits) best = &c;
    return best;
}

NagsResult NAGSController::search(const FastBoard &root, const SearchLimits &limits, int64_t timeLeftMs,
                                  const std::atomic<bool> &stop, const std::atomic<bool> *ponder,
                                  const std::function<void(const SearchInfo &)> &onInfo,
                                  const std::function<void(const std::string &)> &onString) {
    NagsResult result;

    // ---- Meta-learner: per-move deltas ----------------------------------------
    network.configure(config.useNN, config.nnHost, config.nnPort);
    meta.configure(config.metaHost, config.metaPort);
    std::string fen = root.getFEN();
    float tactical = tacticalRatio(root);
    int timeLeft = static_cast<int>(timeLeftMs >= 0 ? std::min<int64_t>(timeLeftMs, 1 << 30) : 30000);
    MetaDeltas deltas;
    bool metaUsed = config.useMetaLearner && meta.predict(fen, timeLeft, lastUncertainty, tactical, deltas);
    if (config.metaExploration > 0.0f) {
        std::normal_distribution<float> noise(0.0f, config.metaExploration);
        deltas.dfs_depth_delta = std::clamp(deltas.dfs_depth_delta + noise(rng), -1.0f, 1.0f);
        deltas.mcts_budget_delta = std::clamp(deltas.mcts_budget_delta + noise(rng), -1.0f, 1.0f);
        deltas.bandit_exploration_delta = std::clamp(deltas.bandit_exploration_delta + noise(rng), -1.0f, 1.0f);
    }
    long long mctsBudget = std::clamp<long long>(config.baseMctsBudget + std::lround(deltas.mcts_budget_delta * 1000), 200, 20000);
    if (limits.infinite) mctsBudget = std::numeric_limits<int>::max();
    cpuct = std::clamp(config.baseExploration + deltas.bandit_exploration_delta * 0.5f, 0.5f, 2.5f);

    // ---- MCTS arm: only with a live network (and not while pondering) -------
    mctsRoot.reset();
    mctsBoard = root;
    mctsSims = 0;
    mctsNodes = 1;
    mctsUsedNetwork = false;
    std::vector<Move> legal = mctsBoard.generateLegalMoves();
    bool pondering = ponder && ponder->load();
    bool runMctsArm = false;
    if (!legal.empty() && legal.size() > 1 && !pondering && config.useNN) {
        EvalResult rootEval = network.evaluate(mctsBoard, legal);
        lastUncertainty = rootEval.uncertainty;
        if (network.lastUsedNetwork()) {
            mctsRoot = std::make_unique<MCTSNode>();
            expand(mctsRoot.get(), legal, rootEval);
            mctsUsedNetwork = true;
            runMctsArm = true;
        }
    }

    // Keep part of the time for verifying an MCTS proposal.
    // (0 means "no limit", so a positive limit stays at least 1 ms.)
    SearchLimits abLimits = limits;
    if (runMctsArm) {
        if (limits.softMs > 0) abLimits.softMs = std::max<int64_t>(1, limits.softMs * 85 / 100);
        if (limits.hardMs > 0) abLimits.hardMs = std::max<int64_t>(1, limits.hardMs * 85 / 100);
    }
    std::atomic<bool> stopMcts{false};
    std::thread mctsThread;
    if (runMctsArm) mctsThread = std::thread([this, mctsBudget, &stopMcts] { runMcts(mctsBudget, stopMcts); });

    // ---- Alpha-beta arm: the full search -------------------------------------
    auto start = std::chrono::steady_clock::now();
    SearchResult ab = searcher.search(root, abLimits, stop, onInfo, ponder);
    stopMcts = true;
    if (mctsThread.joinable()) mctsThread.join();
    uint64_t nodes = ab.nodes;

    // ---- Final decision: MCTS proposes, alpha-beta verifies -----------------
    Move best = ab.bestMove;
    std::string reason = runMctsArm ? "alpha-beta" : "alpha-beta (no network: MCTS off)";
    const MCTSNode *top = mostVisitedRootChild();
    if (runMctsArm && mctsUsedNetwork && top && !sameMove(top->move, ab.bestMove) && ab.depth >= 2 &&
        mctsRoot->visits > 0 && top->visits >= 64 && top->visits * 2 >= mctsRoot->visits &&
        !stop.load(std::memory_order_relaxed)) {
        int depth = std::clamp(ab.depth + static_cast<int>(std::lround(deltas.dfs_depth_delta)), 1, MAX_PLY - 8);
        SearchLimits vLimits;
        if (limits.hardMs > 0) {
            auto used = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            vLimits.hardMs = std::max<int64_t>(1, limits.hardMs - used);
        }
        // Null-window test: is the proposal worth at least the alpha-beta
        // score minus the margin? Much cheaper than scoring it exactly, and
        // the main search's TT already holds most of its tree.
        int threshold = ab.score - config.verifyMargin;
        int verified = 0;
        if (std::abs(ab.score) < MATE_BOUND &&
            searcher.scoreRootMove(root, top->move, depth, vLimits, stop, verified, threshold - 1, threshold)) {
            if (verified >= threshold) {
                best = top->move;
                reason = "mcts, verified >= " + formatScore(threshold) + " (alpha-beta " + formatScore(ab.score) + ")";
            } else {
                reason = "alpha-beta, mcts proposal rejected (< " + formatScore(threshold) + ")";
            }
        } else if (std::abs(ab.score) >= MATE_BOUND) {
            reason = "alpha-beta (mate score: no override)";
        } else {
            reason = "alpha-beta, no time to verify the mcts proposal";
        }
        nodes += searcher.nodesSearched();
    }

    result.bestMove = best;
    result.score = ab.score;
    result.nodes = nodes + static_cast<uint64_t>(mctsSims);
    if (sameMove(best, ab.bestMove)) result.ponderMove = ab.ponderMove;

    if (onString) {
        std::ostringstream s;
        s << "nags alpha_beta_depth " << ab.depth << " mcts_sims " << mctsSims;
        if (top) s << " mcts_top " << moveToUciString(top->move) << " visits " << top->visits << " q " << std::fixed
                   << std::setprecision(3) << top->q() << " (" << formatScore(valueToCp(top->q())) << ")";
        s << " evaluator " << (mctsUsedNetwork ? "network" : "none") << " chose " << moveToUciString(best) << " ("
          << reason << ")";
        onString(s.str());

        // Kept as the last info line so the training pipeline can pair the
        // meta-learner inputs/outputs with the game result.
        std::ostringstream m;
        m << std::setprecision(4) << "nags_meta time_left " << timeLeft << " uncertainty " << lastUncertainty
          << " tactical " << tactical << " deltas " << deltas.dfs_depth_delta << "," << deltas.mcts_budget_delta << ","
          << deltas.bandit_exploration_delta << " meta " << (metaUsed ? "on" : "off");
        onString(m.str());
    }
    mctsRoot.reset();
    return result;
}
