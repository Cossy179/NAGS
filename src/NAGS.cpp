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

EvalResult HeuristicEvaluator::evaluate(Board &board, const std::vector<Move> &legal) {
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

EvalResult RpcEvaluator::evaluate(Board &board, const std::vector<Move> &legal) {
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
// Bandit

void BayesianBandit::reset() {
    for (auto &a : arms) a = ArmStats{};
}

SearchArm BayesianBandit::select(std::mt19937 &rng) {
    auto sample = [&](const ArmStats &a) {
        std::gamma_distribution<double> ga(a.alpha, 1.0), gb(a.beta, 1.0);
        double x = ga(rng), y = gb(rng);
        return x / (x + y);
    };
    return sample(arms[0]) >= sample(arms[1]) ? SearchArm::DFS : SearchArm::MCTS;
}

void BayesianBandit::update(SearchArm arm, bool success) {
    constexpr double kDecay = 0.98;
    ArmStats &a = arms[static_cast<int>(arm)];
    a.alpha = 1.0 + (a.alpha - 1.0) * kDecay + (success ? 1.0 : 0.0);
    a.beta = 1.0 + (a.beta - 1.0) * kDecay + (success ? 0.0 : 1.0);
    ++a.pulls;
    if (success) ++a.successes;
}

// --------------------------------------------------------------------------
// Controller

NAGSController::NAGSController(TranspositionTable &table)
    : tt(table), dfs(std::make_unique<SearchWorker<Board>>(&table, &control)), network(heuristic),
      rng(kSeed) {}

void NAGSController::newGame() {
    bandit.reset();
    dfs->clearHistory();
    lastUncertainty = 0.1f;
    rng.seed(kSeed);
}

float NAGSController::tacticalRatio(const Board &root) const {
    // Share of legal moves that capture, promote or give check.
    Board b = root;
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

void NAGSController::reportDfs(int depth) {
    if (!infoCallback) return;
    dfs->flushNodes();
    SearchInfo info;
    info.depth = depth;
    info.selDepth = dfs->selectiveDepth();
    info.score = dfsScore;
    info.nodes = control.nodes.load(std::memory_order_relaxed);
    info.timeMs = control.elapsedMs();
    info.hashfull = tt.hashfull();
    info.pv = dfsPv;
    infoCallback(info);
}

bool NAGSController::runDfsIteration() {
    int depth = dfsDepth + 1;
    dfs->preferRootMove(dfsBest);
    IterationResult r;
    bool completed = dfs->iterate(depth, dfsScore, r);
    if (!completed) {
        if (r.improved && !r.bestMove.isNull()) {
            dfsBest = r.bestMove;
            dfsScore = r.score;
            dfsPv = r.pv;
            reportDfs(depth);
        }
        return false;
    }
    dfsDepth = depth;
    dfsBest = r.bestMove;
    dfsScore = r.score;
    dfsPv = r.pv;
    reportDfs(depth);
    return true;
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

void NAGSController::runMctsBatch(int simulations, int64_t sliceMs) {
    int64_t deadline = control.elapsedMs() + sliceMs;
    for (int i = 0; i < simulations; ++i) {
        if (control.stop.load(std::memory_order_relaxed) || control.poll()) return;
        if (mctsNodes >= kMaxMctsNodes) return;
        if (i > 0 && control.elapsedMs() >= deadline) return; // keep pulls short so DFS gets its turn
        simulate();
        ++mctsSims;
        control.nodes.fetch_add(1, std::memory_order_relaxed);
    }
}

const MCTSNode *NAGSController::mostVisitedRootChild() const {
    const MCTSNode *best = nullptr;
    if (!mctsRoot) return best;
    for (const auto &c : mctsRoot->children)
        if (!best || c.visits > best->visits) best = &c;
    return best;
}

NagsResult NAGSController::search(const Board &root, const SearchLimits &limits, int64_t timeLeftMs,
                                  const std::atomic<bool> &stop,
                                  const std::function<void(const SearchInfo &)> &onInfo,
                                  const std::function<void(const std::string &)> &onString) {
    NagsResult result;
    control.begin(limits, &stop);
    tt.newSearch();
    infoCallback = onInfo;
    auto waitIfInfinite = [&] {
        while (limits.infinite && !stop.load(std::memory_order_relaxed))
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
    };

    dfs->setRoot(root);
    const std::vector<Move> legal = dfs->legalRootMoves();
    if (legal.empty()) {
        waitIfInfinite();
        return result; // checkmate or stalemate: bestmove 0000
    }

    // ---- Meta-learner: per-move hyperparameter deltas -----------------------
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

    int dfsCap = limits.depth > 0
                     ? limits.depth
                     : std::clamp(config.baseDfsDepth + static_cast<int>(std::lround(deltas.dfs_depth_delta * 4)), 4, MAX_PLY - 8);
    long long mctsBudget = std::clamp<long long>(config.baseMctsBudget + std::lround(deltas.mcts_budget_delta * 1000), 200, 20000);
    if (limits.infinite) {
        dfsCap = MAX_PLY - 8;
        mctsBudget = std::numeric_limits<int>::max();
    }
    cpuct = std::clamp(config.baseExploration + deltas.bandit_exploration_delta * 0.5f, 0.5f, 2.5f);

    // ---- Initial state: always finish a depth-1 alpha-beta search ----------
    dfsDepth = 0;
    dfsScore = 0;
    dfsBest = legal.front();
    dfsPv = {dfsBest};
    mctsSims = 0;
    mctsNodes = 1;
    runDfsIteration();

    mctsBoard = root;
    mctsRoot = std::make_unique<MCTSNode>();
    {
        EvalResult rootEval = network.evaluate(mctsBoard, legal);
        expand(mctsRoot.get(), legal, rootEval);
        lastUncertainty = rootEval.uncertainty;
    }
    bool usedNetwork = network.lastUsedNetwork();

    bool forced = legal.size() == 1 && (limits.softMs > 0 || limits.hardMs > 0);
    int batch = static_cast<int>(std::clamp<long long>(mctsBudget / 20, 16, 400));
    // Wall-clock cap per MCTS pull: network evaluations can take tens of ms each.
    // Without a time limit (fixed depth / nodes) pulls are bounded by the
    // simulation count only, which keeps such searches reproducible.
    int64_t sliceMs = limits.softMs > 0 ? std::clamp<int64_t>(limits.softMs / 10, 10, 1000)
                      : limits.infinite ? 100
                                        : std::numeric_limits<int64_t>::max() / 4;

    // ---- Main loop: the bandit allocates pulls between the two arms --------
    while (!forced && !control.poll()) {
        if (!limits.infinite && limits.softMs > 0 && control.elapsedMs() >= limits.softMs) break;
        bool dfsDone = dfsDepth >= dfsCap || (std::abs(dfsScore) >= MATE_BOUND && dfsDepth >= MATE_SCORE - std::abs(dfsScore) + 4);
        bool mctsDone = mctsSims >= mctsBudget || mctsNodes >= kMaxMctsNodes;
        if (limits.depth > 0 && dfsDepth >= limits.depth) break;
        if (dfsDone && mctsDone) break;

        SearchArm arm = dfsDone ? SearchArm::MCTS : mctsDone ? SearchArm::DFS : bandit.select(rng);
        bool success;
        if (arm == SearchArm::DFS) {
            Move before = dfsBest;
            int scoreBefore = dfsScore;
            bool completed = runDfsIteration();
            success = !sameMove(before, dfsBest) || std::abs(dfsScore - scoreBefore) >= 30;
            if (!completed) {
                bandit.update(arm, success);
                break;
            }
        } else {
            const MCTSNode *topBefore = mostVisitedRootChild();
            Move moveBefore = topBefore ? topBefore->move : Move{};
            double qBefore = topBefore ? topBefore->q() : 0.0;
            runMctsBatch(static_cast<int>(std::min<long long>(batch, mctsBudget - mctsSims)), sliceMs);
            const MCTSNode *topAfter = mostVisitedRootChild();
            success = topAfter && (!sameMove(moveBefore, topAfter->move) || std::abs(topAfter->q() - qBefore) >= 0.05);
            usedNetwork = usedNetwork || network.lastUsedNetwork();
        }
        bandit.update(arm, success);
    }

    // ---- Final decision ------------------------------------------------------
    // MCTS may only override alpha-beta when it is guided by the trained
    // network; with the heuristic evaluator it has no knowledge the
    // alpha-beta search lacks.
    Move best = dfsBest;
    std::string reason = "dfs";
    const MCTSNode *top = mostVisitedRootChild();
    if (usedNetwork && top && !sameMove(top->move, dfsBest) && dfsDepth >= 2 && mctsRoot->visits > 0 && top->visits >= 64 &&
        top->visits * 2 >= mctsRoot->visits && !control.stop.load(std::memory_order_relaxed)) {
        int verified = 0;
        if (dfs->scoreRootMove(top->move, dfsDepth, verified) && verified >= dfsScore - 25) {
            best = top->move;
            reason = "mcts, verified " + formatScore(verified) + " vs " + formatScore(dfsScore);
        } else {
            reason = "dfs, mcts proposal rejected";
        }
    }

    result.bestMove = best;
    result.score = dfsScore;
    dfs->flushNodes();
    result.nodes = control.nodes.load(std::memory_order_relaxed);
    if (sameMove(best, dfsBest) && dfsPv.size() >= 2) result.ponderMove = dfsPv[1];

    if (onString) {
        std::ostringstream s;
        s << "nags dfs_depth " << dfsDepth << " mcts_sims " << mctsSims;
        if (top) s << " mcts_top " << moveToUciString(top->move) << " visits " << top->visits << " q " << std::fixed
                   << std::setprecision(3) << top->q() << " (" << formatScore(valueToCp(top->q())) << ")";
        s << " bandit dfs " << bandit.successes(SearchArm::DFS) << "/" << bandit.pulls(SearchArm::DFS) << " mcts "
          << bandit.successes(SearchArm::MCTS) << "/" << bandit.pulls(SearchArm::MCTS) << " evaluator "
          << (usedNetwork ? "network" : "heuristic") << " chose " << moveToUciString(best) << " (" << reason << ")";
        onString(s.str());

        // Kept as the last info line so the training pipeline can pair the
        // meta-learner inputs/outputs with the game result.
        std::ostringstream m;
        m << std::setprecision(4) << "nags_meta time_left " << timeLeft << " uncertainty " << lastUncertainty
          << " tactical " << tactical << " deltas " << deltas.dfs_depth_delta << "," << deltas.mcts_budget_delta << ","
          << deltas.bandit_exploration_delta << " meta " << (metaUsed ? "on" : "off");
        onString(m.str());
    }

    waitIfInfinite();
    mctsRoot.reset();
    infoCallback = nullptr;
    return result;
}
