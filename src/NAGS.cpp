#include "NAGS.h"

#include "Eval.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>

namespace {

constexpr int kNnConnectTimeoutMs = 150;
constexpr int kNnBackoffSeconds = 60;
// Batched MCTS: a selected path counts as a loss (in [-1, 1]) until its leaf is evaluated.
constexpr double kVirtualLoss = 1.0;

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

void fillHistory(FastBoard &board, EvalRequest &r) {
    // Back over the moves to the position before them, then forward again.
    r.historyMoves = board.recentMoves();
    r.historyFen.clear();
    if (r.historyMoves.empty()) return;
    for (size_t i = 0; i < r.historyMoves.size(); ++i) board.unmakeMove();
    r.historyFen = board.getFEN();
    for (const Move &m : r.historyMoves) board.makeMove(m);
}

// --------------------------------------------------------------------------
// Evaluation cache

void EvalCache::resize(size_t megabytes) {
    table.clear();
    table.shrink_to_fit();
    mask = 0;
    hits = misses = 0;
    if (megabytes == 0) return;
    // An entry and its priors take about 200 bytes (a typical position has 30-40 legal moves).
    const size_t perEntry = sizeof(Entry) + 40 * sizeof(float);
    size_t n = 1;
    while (n * 2 * perEntry <= (megabytes << 20)) n *= 2;
    table.resize(n);
    mask = n - 1;
}

void EvalCache::clear() {
    for (Entry &e : table) {
        e.key = e.movesSig = 0;
        e.priors.clear();
    }
    hits = misses = 0;
}

uint64_t EvalCache::keyOf(const FastBoard &board) {
    return board.zobrist() ^ (static_cast<uint64_t>(board.getHalfmoveClock() + 1) * 0x9E3779B97F4A7C15ull);
}

uint64_t EvalCache::signature(const std::vector<Move> &legal) {
    uint64_t h = 0xcbf29ce484222325ull ^ legal.size();
    for (const Move &m : legal) {
        h ^= static_cast<uint64_t>(m.from) | static_cast<uint64_t>(m.to) << 6 |
             static_cast<uint64_t>(static_cast<int>(m.promotion) + 1) << 12;
        h *= 0x100000001b3ull;
    }
    return h;
}

bool EvalCache::probe(uint64_t key, const std::vector<Move> &legal, EvalResult &out) {
    if (table.empty()) return false;
    const Entry &e = table[key & mask];
    if (e.key != key || e.priors.size() != legal.size() || e.movesSig != signature(legal)) {
        ++misses;
        return false;
    }
    out.priors = e.priors;
    out.value = e.value;
    out.uncertainty = e.uncertainty;
    ++hits;
    return true;
}

void EvalCache::store(uint64_t key, const std::vector<Move> &legal, const EvalResult &r) {
    if (table.empty() || r.priors.size() != legal.size()) return;
    Entry &e = table[key & mask];
    e.key = key;
    e.movesSig = signature(legal);
    e.value = r.value;
    e.uncertainty = r.uncertainty;
    e.priors = r.priors;
}

// --------------------------------------------------------------------------
// Network client

namespace {

// The objects of the array under `key` (top level of each object only; strings may hold braces).
bool splitObjects(const std::string &json, const std::string &key, std::vector<std::string> &out) {
    size_t p = json.find("\"" + key + "\"");
    if (p == std::string::npos) return false;
    p = json.find('[', p);
    if (p == std::string::npos) return false;
    out.clear();
    int depth = 0;
    bool inString = false;
    size_t begin = 0;
    for (++p; p < json.size(); ++p) {
        char c = json[p];
        if (inString) {
            if (c == '\\') ++p;
            else if (c == '"') inString = false;
            continue;
        }
        if (c == '"') inString = true;
        else if (c == '{') {
            if (depth++ == 0) begin = p;
        } else if (c == '}') {
            if (--depth == 0) out.push_back(json.substr(begin, p - begin + 1));
            if (depth < 0) return false;
        } else if (c == ']' && depth == 0) {
            return true;
        }
    }
    return false;
}

// ,"history":[{"fen":..., "moves":[...]} or null, ...] for the positions of a request.
void appendHistory(std::string &request, const std::vector<const EvalRequest *> &batch) {
    request += ",\"history\":[";
    for (size_t i = 0; i < batch.size(); ++i) {
        if (i) request += ',';
        const EvalRequest &r = *batch[i];
        if (r.historyFen.empty()) {
            request += "null";
            continue;
        }
        request += "{\"fen\":\"" + jsonlite::escape(r.historyFen) + "\",\"moves\":[";
        for (size_t j = 0; j < r.historyMoves.size(); ++j) {
            if (j) request += ',';
            request += '"' + moveToUciString(r.historyMoves[j]) + '"';
        }
        request += "]}";
    }
    request += ']';
}

// ,"graphs":["<base64>" or null, ...] for the positions of a request (nothing when none has one).
bool appendGraphs(std::string &request, const std::vector<const EvalRequest *> &batch) {
    if (std::none_of(batch.begin(), batch.end(), [](const EvalRequest *r) { return !r->graph.empty(); }))
        return false;
    request += ",\"graphs\":[";
    for (size_t i = 0; i < batch.size(); ++i) {
        if (i) request += ',';
        request += batch[i]->graph.empty() ? "null" : '"' + batch[i]->graph + '"';
    }
    request += ']';
    return true;
}

// A reply to a request with graphs says whether the server used them.
bool repliesUseGraphs(const std::string &response) {
    return response.find("\"graphs\": true") != std::string::npos ||
           response.find("\"graphs\":true") != std::string::npos;
}

// A reply from a network that wants game histories says so.
bool repliesWantHistory(const std::string &response) {
    return response.find("\"history\": true") != std::string::npos ||
           response.find("\"history\":true") != std::string::npos;
}

// One result object of a reply as an EvalResult (priors normalised, value clamped).
bool parseResult(const std::string &obj, size_t moves, EvalResult &r) {
    std::vector<double> priors;
    double value = 0, uncertainty = 0;
    if (!jsonlite::findNumberArray(obj, "move_priors", priors) || priors.size() != moves ||
        !jsonlite::findNumber(obj, "value", value))
        return false;
    jsonlite::findNumber(obj, "uncertainty", uncertainty);
    r.value = static_cast<float>(std::clamp(value, -1.0, 1.0));
    r.uncertainty = static_cast<float>(std::max(0.0, uncertainty));
    double sum = 0;
    for (double p : priors) sum += std::max(0.0, p);
    r.priors.clear();
    r.priors.reserve(moves);
    for (double p : priors)
        r.priors.push_back(sum > 0 ? static_cast<float>(std::max(0.0, p) / sum) : 1.0f / moves);
    return true;
}

} // namespace

void RpcEvaluator::configure(bool on, const std::string &h, int p) {
    if (on != enabled || h != host || p != port) {
        socket.close();
        outstanding.clear();
        retryAfter = {};
        evalCache.clear(); // another server may serve another network
        msPerPosition = 0.0;
        serverHistory = false;
        serverGraphs = 0;
    }
    enabled = on;
    host = h;
    port = p;
}

void RpcEvaluator::fail(std::chrono::steady_clock::time_point now) {
    socket.close();
    outstanding.clear();
    retryAfter = now + std::chrono::seconds(kNnBackoffSeconds);
    ++stats.failures;
}

bool RpcEvaluator::available() {
    auto now = std::chrono::steady_clock::now();
    if (!enabled || now < retryAfter) return false;
    if (!socket.isOpen()) {
        outstanding.clear();
        serverGraphs = 0; // perhaps another server: ask again
        if (!socket.connect(host, port, kNnConnectTimeoutMs)) {
            retryAfter = now + std::chrono::seconds(kNnBackoffSeconds);
            return false;
        }
    }
    return true;
}

int RpcEvaluator::batchLimit(int wanted) const {
    if (wanted <= 1) return 1;
    if (msPerPosition <= 0.0) return std::min(wanted, 8);
    // Keep a batch's expected round trip under 40% of the request timeout.
    const double fits = requestTimeoutMs * 0.4 / msPerPosition;
    return static_cast<int>(std::clamp(fits, 1.0, static_cast<double>(wanted)));
}

bool RpcEvaluator::sendBatch(const std::vector<const EvalRequest *> &batch) {
    if (batch.empty() || !available()) return false;
    std::string request = "{\"fens\":[";
    for (size_t i = 0; i < batch.size(); ++i) {
        if (i) request += ',';
        request += '"' + jsonlite::escape(batch[i]->fen) + '"';
    }
    request += "],\"moves\":[";
    for (size_t i = 0; i < batch.size(); ++i) {
        request += i ? ",[" : "[";
        const std::vector<Move> &legal = batch[i]->legal;
        for (size_t j = 0; j < legal.size(); ++j) {
            if (j) request += ',';
            request += '"' + moveToUciString(legal[j]) + '"';
        }
        request += ']';
    }
    request += ']';
    if (serverHistory) appendHistory(request, batch);
    const bool graphs = appendGraphs(request, batch);
    request += '}';
    const auto now = std::chrono::steady_clock::now();
    if (!socket.sendLine(request, requestTimeoutMs)) {
        fail(now);
        return false;
    }
    outstanding.push_back({now, graphs});
    ++stats.requests;
    stats.positions += static_cast<long long>(batch.size());
    return true;
}

bool RpcEvaluator::receiveBatch(const std::vector<const EvalRequest *> &batch, std::vector<EvalResult> &out) {
    usedNetwork = false;
    if (outstanding.empty() || !socket.isOpen()) return false;
    std::string response;
    const bool answered = socket.recvLine(response, requestTimeoutMs);
    const auto now = std::chrono::steady_clock::now();
    const auto sent = outstanding.front().sent;
    const bool sentGraphs = outstanding.front().graphs;
    outstanding.erase(outstanding.begin());
    const double roundTrip = std::chrono::duration<double, std::milli>(now - sent).count();
    stats.waitMs += roundTrip;
    stats.maxMs = std::max(stats.maxMs, roundTrip);
    std::vector<std::string> objects;
    bool ok = answered && splitObjects(response, "results", objects) && objects.size() == batch.size();
    out.assign(batch.size(), EvalResult{});
    for (size_t i = 0; ok && i < batch.size(); ++i) ok = parseResult(objects[i], batch[i]->legal.size(), out[i]);
    if (!ok) {
        fail(now);
        return false;
    }
    serverHistory = repliesWantHistory(response);
    if (sentGraphs) serverGraphs = repliesUseGraphs(response) ? 1 : -1;
    // The server starts on a pipelined request once it has answered the one before,
    // so the time it took is counted from the later of its sending and that answer.
    const double service = std::chrono::duration<double, std::milli>(now - std::max(sent, lastReply)).count();
    lastReply = now;
    const double perPosition = service / static_cast<double>(batch.size());
    msPerPosition = msPerPosition > 0.0 ? 0.7 * msPerPosition + 0.3 * perPosition : perPosition;
    usedNetwork = true;
    return true;
}

EvalResult RpcEvaluator::evaluate(FastBoard &board, const std::vector<Move> &legal) {
    usedNetwork = false;
    auto now = std::chrono::steady_clock::now();
    if (!enabled || now < retryAfter) return fallback.evaluate(board, legal);
    const uint64_t key = EvalCache::keyOf(board);
    {
        EvalResult cached;
        if (evalCache.probe(key, legal, cached)) {
            usedNetwork = true;
            return cached;
        }
    }
    if (!outstanding.empty()) socket.close(); // unread batch replies would be taken for this one's
    outstanding.clear();
    if (!socket.isOpen()) {
        serverGraphs = 0;
        if (!socket.connect(host, port, kNnConnectTimeoutMs)) {
            retryAfter = now + std::chrono::seconds(kNnBackoffSeconds);
            return fallback.evaluate(board, legal);
        }
    }

    std::string request = "{\"fens\":[\"" + jsonlite::escape(board.getFEN()) + "\"],\"moves\":[[";
    for (size_t i = 0; i < legal.size(); ++i) {
        if (i) request += ',';
        request += '"' + moveToUciString(legal[i]) + '"';
    }
    request += "]]";
    EvalRequest req;
    if (serverHistory) {
        fillHistory(board, req);
        appendHistory(request, {&req});
    }
    if (wantsGraphs()) {
        RelGraph graph;
        buildRelationalGraph(board, graph);
        req.graph = encodeGraph(graph);
    }
    const bool sentGraphs = appendGraphs(request, {&req});
    request += '}';

    std::string response;
    std::vector<double> priors;
    double value = 0, uncertainty = 0;
    const auto sent = std::chrono::steady_clock::now();
    const bool answered = socket.request(request, response, requestTimeoutMs);
    lastReply = std::chrono::steady_clock::now();
    const double waited = std::chrono::duration<double, std::milli>(lastReply - sent).count();
    ++stats.requests;
    ++stats.positions;
    stats.waitMs += waited;
    stats.maxMs = std::max(stats.maxMs, waited);
    if (!answered ||
        !jsonlite::findNumberArray(response, "move_priors", priors) || priors.size() != legal.size() ||
        !jsonlite::findNumber(response, "value", value)) {
        socket.close();
        retryAfter = now + std::chrono::seconds(kNnBackoffSeconds);
        ++stats.failures;
        return fallback.evaluate(board, legal);
    }
    jsonlite::findNumber(response, "uncertainty", uncertainty);
    serverHistory = repliesWantHistory(response);
    if (sentGraphs) serverGraphs = repliesUseGraphs(response) ? 1 : -1;

    EvalResult r;
    r.value = static_cast<float>(std::clamp(value, -1.0, 1.0));
    r.uncertainty = static_cast<float>(std::max(0.0, uncertainty));
    double sum = 0;
    for (double p : priors) sum += std::max(0.0, p);
    r.priors.reserve(legal.size());
    for (double p : priors)
        r.priors.push_back(sum > 0 ? static_cast<float>(std::max(0.0, p) / sum) : 1.0f / legal.size());
    usedNetwork = true;
    msPerPosition = msPerPosition > 0.0 ? 0.7 * msPerPosition + 0.3 * waited : waited;
    evalCache.store(key, legal, r);
    return r;
}

// --------------------------------------------------------------------------
// Controller

NAGSController::NAGSController(Searcher<FastBoard> &s) : searcher(s), network(heuristic), rng(kSeed) {}

void NAGSController::newGame() {
    lastUncertainty = 0.1f;
    rng.seed(kSeed);
    network.resetCache();
    mctsRoot.reset();
    reuseRootFen.clear();
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
    if (config.policyTemp != 1.0f && !node->children.empty()) {
        // Softmax temperature on the priors: T > 1 flattens them, T < 1 sharpens them.
        const double inv = 1.0 / std::max(0.05f, config.policyTemp);
        double sum = 0.0;
        for (MCTSNode &c : node->children) sum += std::pow(std::max(0.0f, c.prior), inv);
        for (MCTSNode &c : node->children)
            c.prior = sum > 0 ? static_cast<float>(std::pow(std::max(0.0f, c.prior), inv) / sum)
                              : 1.0f / static_cast<float>(node->children.size());
    }
    node->expanded = true;
    mctsNodes += legal.size();
}

bool NAGSController::ttValue(const FastBoard &board, float &value) const {
    // An exact alpha-beta score of the position at least ttValueDepth deep,
    // as a value: P(win) - P(loss) under the logistic WDL model the v3
    // network is distilled towards (docs/ARCHITECTURE.md), mates as +-1.
    TranspositionTable *tt = searcher.table();
    TTHit hit;
    if (config.ttValueDepth <= 0 || !tt || !tt->probe(board.zobrist(), hit) || hit.bound != Bound::Exact ||
        hit.depth < config.ttValueDepth)
        return false;
    const int s = scoreFromTT(hit.score, 0);
    if (std::abs(s) >= TB_BOUND) {
        value = s > 0 ? 1.0f : -1.0f;
        return true;
    }
    auto sigmoid = [](double x) { return 1.0 / (1.0 + std::exp(-x)); };
    value = static_cast<float>(sigmoid((s - 400.0) / 250.0) - sigmoid((-s - 400.0) / 250.0));
    return true;
}

MCTSNode *NAGSController::selectChild(MCTSNode *node) const {
    double sqrtN = std::sqrt(std::max(1, node->visits));
    // First-play urgency: unvisited children start slightly below the
    // parent's own value (from the side to move at `node`).
    double fpu = (node->visits > 0 ? -node->q() : 0.0) - config.fpuReduction;
    // With cpuctBase the exploration constant grows with the visits (AlphaZero).
    const double c = config.cpuctBase > 0
                         ? cpuct + std::log((1.0 + node->visits + config.cpuctBase) / config.cpuctBase)
                         : cpuct;
    MCTSNode *best = nullptr;
    double bestScore = -std::numeric_limits<double>::infinity();
    for (auto &child : node->children) {
        double q = child.visits > 0 ? child.q() : fpu;
        double u = c * child.prior * sqrtN / (1.0 + child.visits);
        if (q + u > bestScore) {
            bestScore = q + u;
            best = &child;
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
            float tv;
            if (ttValue(mctsBoard, tv)) { // the alpha-beta arm already searched this position
                value = tv;
                ++mctsTTValues;
            }
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
    mctsStart = std::chrono::steady_clock::now();
    // Batched until the search ends or the network fails; then (as with
    // NNBatch 1) one simulation at a time, the evaluator falling back to the
    // heuristic.
    if (config.nnBatch > 1) runMctsBatched(budget, stopMcts);
    while (mctsSims < budget && mctsNodes < kMaxMctsNodes && !stopMcts.load(std::memory_order_relaxed) &&
           !mctsSmartStopped) {
        simulate();
        ++mctsSims;
        if ((mctsSims & 15) == 0 && shouldSmartStop(budget)) mctsSmartStopped = true;
    }
    mctsMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mctsStart).count();
}

bool NAGSController::shouldSmartStop(long long budget) const {
    // The most visited root move cannot change once its lead over the second
    // is larger than the simulations still possible: those left in the
    // budget, and those the current rate fits in before the alpha-beta arm's
    // hard limit.
    if (!config.smartStop || !mctsRoot || mctsRoot->children.size() < 2) return false;
    int first = 0, second = 0;
    for (const MCTSNode &c : mctsRoot->children) {
        if (c.visits > first) {
            second = first;
            first = c.visits;
        } else if (c.visits > second) {
            second = c.visits;
        }
    }
    double remaining = static_cast<double>(budget - mctsSims);
    if (mctsDeadline != std::chrono::steady_clock::time_point{}) {
        const auto now = std::chrono::steady_clock::now();
        const double ran = std::chrono::duration<double, std::milli>(now - mctsStart).count();
        const double left = std::chrono::duration<double, std::milli>(mctsDeadline - now).count();
        if (ran <= 0.0 || mctsSims == 0) return false;
        remaining = std::min(remaining, mctsSims / ran * std::max(0.0, left));
    }
    return first - second > remaining;
}

std::unique_ptr<MCTSNode> NAGSController::reuseSubtree(const FastBoard &root) {
    // The node of the previous tree whose position is the new root: the
    // root itself (the same position searched again), one of its children,
    // or a grandchild (our move and the reply). Its subtree becomes the new
    // tree; the rest is freed.
    std::unique_ptr<MCTSNode> out;
    FastBoard b;
    if (!mctsRoot || reuseRootFen.empty() || !b.setFromFEN(reuseRootFen)) return out;
    const std::string target = root.getFEN();
    const uint64_t key = root.zobrist();
    MCTSNode *found = nullptr;
    auto matches = [&](const FastBoard &x) { return x.zobrist() == key && x.getFEN() == target; };
    if (matches(b)) {
        return std::move(mctsRoot);
    }
    for (MCTSNode &c : mctsRoot->children) {
        if (found) break;
        b.makeMove(c.move);
        if (matches(b)) {
            found = &c;
        } else {
            for (MCTSNode &g : c.children) {
                b.makeMove(g.move);
                const bool hit = matches(b);
                b.unmakeMove();
                if (hit) {
                    found = &g;
                    break;
                }
            }
        }
        b.unmakeMove();
    }
    if (!found || !found->expanded) return out;
    out = std::make_unique<MCTSNode>();
    out->children = std::move(found->children); // the children keep their addresses
    out->visits = found->visits;
    out->valueSum = found->valueSum;
    out->expanded = true;
    out->terminal = found->terminal;
    out->terminalValue = found->terminalValue;
    for (MCTSNode &c : out->children) c.parent = out.get();
    found->expanded = false;
    return out;
}

void NAGSController::backup(MCTSNode *leaf, float value, bool virtualLoss) {
    // `value` is from the side to move at `leaf`; each node stores the view
    // of the player who moved into it. With virtual loss the visit is
    // already counted and the loss it was charged is taken back.
    double v = value;
    for (MCTSNode *n = leaf; n; n = n->parent) {
        if (virtualLoss) n->valueSum += kVirtualLoss;
        else n->visits += 1;
        n->valueSum += -v;
        v = -v;
    }
}

void NAGSController::revertVirtualLoss(MCTSNode *leaf) {
    for (MCTSNode *n = leaf; n; n = n->parent) {
        n->visits -= 1;
        n->valueSum += kVirtualLoss;
    }
}

NAGSController::Selected NAGSController::selectLeaf(std::vector<PendingLeaf> &batch) {
    // Like simulate(), but every node on the path is charged a virtual loss
    // (a visit that lost), so the next selections of the batch spread out,
    // and a leaf that needs the network is queued instead of evaluated.
    MCTSNode *node = mctsRoot.get();
    std::vector<Move> path;
    node->visits += 1;
    node->valueSum -= kVirtualLoss;
    while (node->expanded && !node->terminal && !node->children.empty()) {
        node = selectChild(node);
        mctsBoard.makeMove(node->move);
        path.push_back(node->move);
        node->visits += 1;
        node->valueSum -= kVirtualLoss;
    }

    const size_t depth = path.size();
    Selected result = Selected::Resolved;
    if (node->pending) {
        revertVirtualLoss(node);
        result = Selected::Collision;
    } else if (node->terminal) {
        backup(node, node->terminalValue, true);
    } else if (!path.empty() && mctsBoard.isDraw()) {
        node->terminal = true;
        node->terminalValue = 0.0f;
        backup(node, 0.0f, true);
    } else {
        std::vector<Move> legal = mctsBoard.generateLegalMoves();
        const uint64_t key = EvalCache::keyOf(mctsBoard);
        EvalResult cached;
        if (legal.empty()) {
            node->terminal = true;
            node->terminalValue = mctsBoard.inCheck() ? -1.0f : 0.0f; // checkmated / stalemate
            backup(node, node->terminalValue, true);
        } else if (network.cache().probe(key, legal, cached)) {
            expand(node, legal, cached);
            mctsUsedNetwork = true;
            float tv;
            const bool fromTT = ttValue(mctsBoard, tv);
            mctsTTValues += fromTT;
            backup(node, fromTT ? tv : cached.value, true);
        } else {
            node->pending = true;
            PendingLeaf leaf;
            leaf.node = node;
            leaf.key = key;
            leaf.hasTTValue = ttValue(mctsBoard, leaf.ttValue);
            leaf.path = std::move(path);
            leaf.request.fen = mctsBoard.getFEN();
            leaf.request.legal = std::move(legal);
            if (network.wantsHistory()) fillHistory(mctsBoard, leaf.request);
            if (network.wantsGraphs()) {
                buildRelationalGraph(mctsBoard, graphScratch);
                leaf.request.graph = encodeGraph(graphScratch);
            }
            batch.push_back(std::move(leaf));
            result = Selected::Queued;
        }
    }
    for (size_t i = 0; i < depth; ++i) mctsBoard.unmakeMove();
    return result;
}

void NAGSController::evaluateWithFallback(std::vector<PendingLeaf> &batch) {
    for (PendingLeaf &leaf : batch) {
        for (const Move &m : leaf.path) mctsBoard.makeMove(m);
        EvalResult e = heuristic.evaluate(mctsBoard, leaf.request.legal);
        for (size_t i = 0; i < leaf.path.size(); ++i) mctsBoard.unmakeMove();
        leaf.node->pending = false;
        expand(leaf.node, leaf.request.legal, e);
        mctsTTValues += leaf.hasTTValue;
        backup(leaf.node, leaf.hasTTValue ? leaf.ttValue : e.value, true);
        ++mctsSims;
    }
}

void NAGSController::runMctsBatched(long long budget, const std::atomic<bool> &stopMcts) {
    const size_t maxInFlight = config.nnPipeline ? 2 : 1;
    std::deque<std::vector<PendingLeaf>> inFlight;
    long long queued = 0; // leaves in flight
    auto running = [&] {
        return !stopMcts.load(std::memory_order_relaxed) && mctsNodes < kMaxMctsNodes && !mctsSmartStopped;
    };
    auto requests = [](const std::vector<PendingLeaf> &batch) {
        std::vector<const EvalRequest *> r;
        r.reserve(batch.size());
        for (const PendingLeaf &leaf : batch) r.push_back(&leaf.request);
        return r;
    };
    // Leaves of requests that will not be answered: evaluated by the
    // heuristic while the search goes on, otherwise just released.
    auto drop = [&](std::vector<PendingLeaf> &batch) {
        if (running()) {
            evaluateWithFallback(batch);
        } else {
            for (PendingLeaf &leaf : batch) {
                leaf.node->pending = false;
                revertVirtualLoss(leaf.node);
            }
        }
    };

    bool networkFailed = false;
    while (!networkFailed) {
        // Fill the pipeline: select leaves (terminal and cached ones are
        // backed up at once) and send them as one request.
        while (running() && inFlight.size() < maxInFlight && mctsSims + queued < budget) {
            if (shouldSmartStop(budget)) { // the leaves in flight are still received
                mctsSmartStopped = true;
                break;
            }
            const int want = network.batchLimit(config.nnBatch);
            std::vector<PendingLeaf> batch;
            int collisions = 0;
            while (static_cast<int>(batch.size()) < want && mctsSims + queued + static_cast<long long>(batch.size()) < budget &&
                   running()) {
                Selected s = selectLeaf(batch);
                if (s == Selected::Resolved) {
                    ++mctsSims;
                } else if (s == Selected::Collision) {
                    ++mctsCollisions;
                    if (++collisions >= want) break; // the tree is narrow: send what there is
                }
            }
            if (batch.empty()) break;
            if (!network.sendBatch(requests(batch))) {
                drop(batch);
                networkFailed = true;
                break;
            }
            ++mctsBatches;
            queued += static_cast<long long>(batch.size());
            inFlight.push_back(std::move(batch));
        }
        if (inFlight.empty()) break;

        std::vector<PendingLeaf> batch = std::move(inFlight.front());
        inFlight.pop_front();
        queued -= static_cast<long long>(batch.size());
        std::vector<EvalResult> results;
        if (network.receiveBatch(requests(batch), results)) {
            for (size_t i = 0; i < batch.size(); ++i) {
                PendingLeaf &leaf = batch[i];
                leaf.node->pending = false;
                network.cache().store(leaf.key, leaf.request.legal, results[i]);
                expand(leaf.node, leaf.request.legal, results[i]);
                mctsTTValues += leaf.hasTTValue;
                backup(leaf.node, leaf.hasTTValue ? leaf.ttValue : results[i].value, true);
                ++mctsSims;
            }
            mctsUsedNetwork = true;
        } else {
            // The connection is gone, and with it every request in flight.
            drop(batch);
            for (auto &b : inFlight) drop(b);
            inFlight.clear();
            queued = 0;
            networkFailed = true;
        }
    }
    // Nothing is left pending: every virtual loss has been taken back.
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
    const auto start = std::chrono::steady_clock::now();

    // ---- Meta-learner: per-move deltas ----------------------------------------
    network.configure(config.useNN, config.nnHost, config.nnPort);
    network.setSendGraphs(config.nnGraphs);
    network.resetTiming();
    if (config.useNN && config.nnCacheMb != cacheMbApplied) { // allocated once the network is in use
        network.cache().resize(static_cast<size_t>(config.nnCacheMb));
        cacheMbApplied = config.nnCacheMb;
    }
    const long long cacheHitsBefore = network.cache().hits;
    mctsMs = 0.0;
    mctsCollisions = 0;
    mctsBatches = 0;
    mctsTTValues = 0;
    mctsReusedVisits = 0;
    mctsSmartStopped = false;
    mctsDeadline = {};
    meta.configure(config.metaHost, config.metaPort);
    std::string fen = root.getFEN();
    float tactical = tacticalRatio(root);
    int timeLeft = static_cast<int>(timeLeftMs >= 0 ? std::min<int64_t>(timeLeftMs, 1 << 30) : 30000);
    MetaDeltas deltas;
    // The request must fit into the move: at most a quarter of the hard
    // limit (and none at all for a move shorter than 40 ms). Without a time
    // limit the usual timeouts apply.
    const bool metaAffordable = limits.hardMs <= 0 || limits.hardMs >= 40;
    meta.setTimeBudget(limits.hardMs > 0 ? static_cast<int>(std::clamp<int64_t>(limits.hardMs / 4, 10, 1200)) : 1200);
    bool metaUsed = config.useMetaLearner && metaAffordable && meta.predict(fen, timeLeft, lastUncertainty, tactical, deltas);
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
    // With MctsReuse the subtree of the new root is taken from the last tree.
    std::unique_ptr<MCTSNode> reused = config.mctsReuse ? reuseSubtree(root) : nullptr;
    mctsRoot.reset();
    reuseRootFen.clear();
    mctsBoard = root;
    mctsBoard.refreshAccumulator(); // `root` may hold accumulators of an earlier network (EvalFile/UseNNUE changed)
    mctsSims = 0;
    mctsNodes = 1;
    mctsUsedNetwork = false;
    std::vector<Move> legal = mctsBoard.generateLegalMoves();
    bool pondering = ponder && ponder->load();
    // Each network evaluation is a round trip to rpc_server.py (milliseconds
    // to tens of milliseconds), so MCTS is pointless with little time.
    bool enoughTime = limits.softMs == 0 || limits.softMs >= config.minMctsMs;
    network.setRequestTimeout(limits.hardMs > 0 ? static_cast<int>(std::clamp<int64_t>(limits.hardMs / 10, 20, 3000)) : 3000);
    bool runMctsArm = false;
    bool rootOrdered = false;
    const bool wantMcts = legal.size() > 1 && !pondering && config.useNN && enoughTime;
    // NNRootOrder needs one evaluation, so it is worth it even when MCTS is not.
    const bool wantOrder = legal.size() > 1 && !pondering && config.useNN && config.nnRootOrder &&
                           (limits.hardMs <= 0 || limits.hardMs >= 50);
    if (wantMcts || wantOrder) {
        EvalResult rootEval = network.evaluate(mctsBoard, legal);
        lastUncertainty = rootEval.uncertainty;
        if (network.lastUsedNetwork()) {
            if (wantOrder) {
                std::vector<size_t> idx(legal.size());
                for (size_t i = 0; i < idx.size(); ++i) idx[i] = i;
                std::stable_sort(idx.begin(), idx.end(),
                                 [&](size_t a, size_t b) { return rootEval.priors[a] > rootEval.priors[b]; });
                std::vector<Move> order;
                order.reserve(idx.size());
                for (size_t i : idx) order.push_back(legal[i]);
                searcher.setRootOrder(std::move(order));
                rootOrdered = true;
            }
            if (wantMcts) {
                bool sameChildren = reused && reused->children.size() == legal.size();
                for (size_t i = 0; sameChildren && i < legal.size(); ++i)
                    sameChildren = sameMove(reused->children[i].move, legal[i]);
                if (sameChildren) {
                    mctsRoot = std::move(reused);
                    mctsReusedVisits = mctsRoot->visits;
                    // Count the nodes the reused tree holds (the node cap).
                    std::vector<const MCTSNode *> stack{mctsRoot.get()};
                    mctsNodes = 0;
                    while (!stack.empty()) {
                        const MCTSNode *n = stack.back();
                        stack.pop_back();
                        ++mctsNodes;
                        for (const MCTSNode &c : n->children) stack.push_back(&c);
                    }
                } else {
                    mctsRoot = std::make_unique<MCTSNode>();
                    expand(mctsRoot.get(), legal, rootEval);
                }
                mctsUsedNetwork = true;
                runMctsArm = true;
            }
        }
    }
    reused.reset();

    // The time used so far (meta-learner, root evaluation) counts against
    // the move, and with MCTS part of it is kept for verifying a proposal.
    // (0 means "no limit", so a positive limit stays at least 1 ms.)
    const int64_t used = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
    const int share = runMctsArm ? config.abSharePct : 100;
    SearchLimits abLimits = limits;
    if (limits.softMs > 0) abLimits.softMs = std::max<int64_t>(1, (limits.softMs - used) * share / 100);
    if (limits.hardMs > 0) abLimits.hardMs = std::max<int64_t>(1, (limits.hardMs - used) * share / 100);
    if (abLimits.hardMs > 0 && !limits.infinite)
        mctsDeadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(abLimits.hardMs);
    std::atomic<bool> stopMcts{false};
    std::thread mctsThread;
    if (runMctsArm) mctsThread = std::thread([this, mctsBudget, &stopMcts] { runMcts(mctsBudget, stopMcts); });

    // ---- Alpha-beta arm: the full search -------------------------------------
    SearchResult ab = searcher.search(root, abLimits, stop, onInfo, ponder);
    stopMcts = true;
    if (mctsThread.joinable()) mctsThread.join();
    uint64_t nodes = ab.nodes;

    // ---- Final decision: MCTS proposes, alpha-beta verifies -----------------
    Move best = ab.bestMove;
    std::string reason = "alpha-beta";
    if (!runMctsArm)
        reason += !config.useNN || legal.size() <= 1 ? " (MCTS off)"
                  : pondering                          ? " (pondering: MCTS off)"
                  : !enoughTime                        ? " (too little time: MCTS off)"
                                                       : " (no network: MCTS off)";
    const MCTSNode *top = mostVisitedRootChild();
    if (runMctsArm && mctsUsedNetwork && top && !sameMove(top->move, ab.bestMove) && ab.depth >= 2 &&
        mctsRoot->visits > 0 && top->visits >= 64 && top->visits * 2 >= mctsRoot->visits &&
        !stop.load(std::memory_order_relaxed)) {
        int depth = std::clamp(ab.depth + static_cast<int>(std::lround(deltas.dfs_depth_delta)), 1, MAX_PLY - 8);
        SearchLimits vLimits;
        if (limits.hardMs > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
            vLimits.hardMs = std::max<int64_t>(1, limits.hardMs - elapsed);
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
        // Cost of the neural arm: simulation rate and time spent waiting for
        // rpc_server.py (including the root evaluation).
        const RpcEvaluator::Timing &rpc = network.timing();
        s << std::setprecision(1) << " mcts_ms " << mctsMs << " mcts_sims_per_s "
          << (mctsMs > 0 ? mctsSims * 1000.0 / mctsMs : 0.0) << " rpc_requests " << rpc.requests << " rpc_failures "
          << rpc.failures << " rpc_wait_ms " << rpc.waitMs << " rpc_mean_ms "
          << (rpc.requests ? rpc.waitMs / rpc.requests : 0.0) << " rpc_max_ms " << rpc.maxMs;
        // Batching and the evaluation cache: positions per request, positions
        // answered from the cache, and selections that met a queued leaf.
        s << " rpc_positions " << rpc.positions << " rpc_batch "
          << (rpc.requests ? static_cast<double>(rpc.positions) / rpc.requests : 0.0) << " cache_hits "
          << network.cache().hits - cacheHitsBefore << " mcts_collisions " << mctsCollisions;
        // Hybrid search options: visits brought from the last tree, leaves
        // valued by the TT, MCTS stopped early, root moves in prior order.
        s << " mcts_reused " << mctsReusedVisits << " tt_values " << mctsTTValues << " smart_stop "
          << (mctsSmartStopped ? 1 : 0) << " root_order " << (rootOrdered ? 1 : 0);
        // The alpha-beta arm's own move (what was played without MCTS), for tools/hybrid_eval.py.
        s << " alpha_beta_move " << moveToUciString(ab.bestMove) << " alpha_beta_score " << ab.score;
        onString(s.str());

        // Kept as the last info line so the training pipeline can pair the
        // meta-learner inputs/outputs with the game result.
        std::ostringstream m;
        m << std::setprecision(4) << "nags_meta time_left " << timeLeft << " uncertainty " << lastUncertainty
          << " tactical " << tactical << " deltas " << deltas.dfs_depth_delta << "," << deltas.mcts_budget_delta << ","
          << deltas.bandit_exploration_delta << " meta " << (metaUsed ? "on" : "off");
        onString(m.str());
    }
    if (config.mctsReuse && mctsRoot) {
        reuseRootFen = root.getFEN(); // the next search may continue in this tree
    } else {
        mctsRoot.reset();
    }
    return result;
}
