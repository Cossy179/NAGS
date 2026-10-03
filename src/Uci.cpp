#include "Uci.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

namespace uci {

namespace {
std::mutex outputMutex;
}

void send(const std::string &line) {
    std::lock_guard<std::mutex> lock(outputMutex);
    std::cout << line << '\n' << std::flush;
}

std::vector<std::string> tokenize(const std::string &line) {
    std::istringstream iss(line);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

bool parseInt(const std::string &s, long long &out) {
    if (s.empty()) return false;
    try {
        size_t used = 0;
        long long v = std::stoll(s, &used);
        if (used != s.size()) return false;
        out = v;
        return true;
    } catch (...) {
        return false;
    }
}

bool parseBool(const std::string &s, bool &out) {
    std::string v = s;
    std::transform(v.begin(), v.end(), v.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (v == "true" || v == "1" || v == "on") { out = true; return true; }
    if (v == "false" || v == "0" || v == "off") { out = false; return true; }
    return false;
}

bool parseGo(const std::vector<std::string> &tokens, GoParams &out, std::string &error) {
    out = GoParams{};
    auto number = [&](size_t &i, long long &v) -> bool {
        if (i + 1 >= tokens.size() || !parseInt(tokens[i + 1], v)) {
            error = "missing or invalid value for '" + tokens[i] + "'";
            return false;
        }
        ++i;
        return true;
    };
    bool anyLimit = false;
    for (size_t i = 1; i < tokens.size(); ++i) {
        const std::string &t = tokens[i];
        long long v = 0;
        if (t == "infinite") { out.infinite = true; anyLimit = true; continue; }
        if (t == "ponder") { out.ponder = true; continue; }
        if (t == "searchmoves") { break; } // not supported; ignore the move list
        if (!number(i, v)) return false;
        if (t == "wtime") { out.wtime = std::max<long long>(0, v); anyLimit = true; }
        else if (t == "btime") { out.btime = std::max<long long>(0, v); anyLimit = true; }
        else if (t == "winc") out.winc = std::max<long long>(0, v);
        else if (t == "binc") out.binc = std::max<long long>(0, v);
        else if (t == "movestogo") out.movestogo = static_cast<int>(std::max<long long>(0, v));
        else if (t == "movetime") { out.movetime = std::max<long long>(1, v); anyLimit = true; }
        else if (t == "depth") { out.depth = static_cast<int>(std::clamp<long long>(v, 1, MAX_PLY - 8)); anyLimit = true; }
        else if (t == "nodes") { out.nodes = static_cast<uint64_t>(std::max<long long>(1, v)); anyLimit = true; }
        else if (t == "perft") out.perft = static_cast<int>(std::clamp<long long>(v, 0, 12));
        else if (t == "mate") { out.depth = static_cast<int>(std::clamp<long long>(2 * v, 1, MAX_PLY - 8)); anyLimit = true; }
        else { error = "unknown go parameter '" + t + "'"; return false; }
    }
    // A bare "go" means search until "stop".
    if (!anyLimit && out.perft == 0) out.infinite = true;
    return true;
}

SearchLimits computeLimits(const GoParams &go, bool whiteToMove, int64_t overhead) {
    SearchLimits limits;
    limits.depth = go.depth;
    limits.nodes = go.nodes;
    limits.infinite = go.infinite;
    if (go.infinite) return limits;

    if (go.movetime > 0) {
        int64_t t = std::max<int64_t>(1, go.movetime - overhead);
        limits.softMs = t;
        limits.hardMs = t;
        return limits;
    }

    int64_t time = whiteToMove ? go.wtime : go.btime;
    int64_t inc = whiteToMove ? go.winc : go.binc;
    if (time < 0) {
        // Only the opponent's clock was sent: assume symmetric clocks rather
        // than searching without any time limit.
        time = whiteToMove ? go.btime : go.wtime;
        inc = whiteToMove ? go.binc : go.winc;
    }
    if (time < 0) return limits; // no clock at all: depth/node limited (or unlimited)

    int64_t available = std::max<int64_t>(1, time - overhead);
    int movesToGo = go.movestogo > 0 ? std::min(go.movestogo, 50) : 30;
    int64_t target = available / movesToGo + inc * 3 / 4;
    int64_t hard = movesToGo == 1 ? available * 9 / 10 : std::min(available / 2, target * 3);
    target = std::min(target, hard);
    limits.hardMs = std::max<int64_t>(1, hard);
    // Iterations roughly double in cost, so stop starting new ones a little
    // after half the target has elapsed.
    limits.softMs = std::max<int64_t>(1, target * 55 / 100);
    return limits;
}

std::string formatInfo(const SearchInfo &info) {
    std::ostringstream oss;
    uint64_t nps = info.timeMs > 0 ? info.nodes * 1000 / static_cast<uint64_t>(info.timeMs) : info.nodes;
    oss << "info depth " << info.depth << " seldepth " << std::max(info.selDepth, info.depth);
    if (info.multiPv > 0) oss << " multipv " << info.multiPv;
    oss << " score " << formatScore(info.score) << " nodes " << info.nodes << " nps " << nps
        << " hashfull " << info.hashfull << " time " << info.timeMs;
    if (!info.pv.empty()) {
        oss << " pv";
        for (const Move &m : info.pv) oss << ' ' << moveToUciString(m);
    }
    return oss.str();
}

std::string bestMoveLine(const Move &best, const Move &ponder) {
    std::string line = "bestmove " + moveToUciString(best);
    if (!best.isNull() && !ponder.isNull()) line += " ponder " + moveToUciString(ponder);
    return line;
}

namespace {
// Depth argument of "bench [depth]"; 0 (engine default) if absent or invalid.
int benchDepth(const std::string &arg) {
    long long d = 0;
    return parseInt(arg, d) ? static_cast<int>(std::clamp<long long>(d, 1, MAX_PLY - 8)) : 0;
}
} // namespace

int run(Engine &engine, int argc, char **argv) {
    std::ios::sync_with_stdio(false);
    // cin is tied to cout by default: every read would flush cout from the
    // input thread without holding outputMutex, racing with the search thread.
    std::cin.tie(nullptr);

    if (argc > 1 && std::string(argv[1]) == "bench") {
        engine.bench(argc > 2 ? benchDepth(argv[2]) : 0);
        return 0;
    }

    std::thread worker;
    std::atomic<bool> stopFlag{false};
    std::atomic<bool> ponderFlag{false};
    bool currentInfinite = false;

    auto waitForSearch = [&] {
        if (worker.joinable()) worker.join();
    };
    auto stopSearch = [&] {
        stopFlag.store(true);
        waitForSearch();
    };

    std::string line;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        auto tokens = tokenize(line);
        if (tokens.empty()) continue;
        const std::string &cmd = tokens[0];

        if (cmd == "uci") {
            send("id name " + engine.name());
            send("id author " + engine.author());
            for (const auto &opt : engine.optionLines()) send(opt);
            send("uciok");
        } else if (cmd == "isready") {
            send("readyok");
        } else if (cmd == "ucinewgame") {
            stopSearch();
            engine.newGame();
        } else if (cmd == "setoption") {
            stopSearch();
            // setoption name <name, may contain spaces> [value <value, may contain spaces>]
            std::string name, value;
            size_t i = 1;
            if (i < tokens.size() && tokens[i] == "name") ++i;
            for (; i < tokens.size() && tokens[i] != "value"; ++i) name += (name.empty() ? "" : " ") + tokens[i];
            if (i < tokens.size() && tokens[i] == "value") ++i;
            for (; i < tokens.size(); ++i) value += (value.empty() ? "" : " ") + tokens[i];
            std::string message;
            bool ok = engine.setOption(name, value, message);
            if (!ok && message.empty()) message = "unknown option '" + name + "'";
            if (!message.empty()) send("info string " + message);
        } else if (cmd == "position") {
            stopSearch();
            std::string fen;
            std::vector<std::string> moves;
            size_t i = 1;
            bool ok = true;
            if (i < tokens.size() && tokens[i] == "startpos") {
                ++i;
            } else if (i < tokens.size() && tokens[i] == "fen") {
                ++i;
                for (; i < tokens.size() && tokens[i] != "moves"; ++i) fen += (fen.empty() ? "" : " ") + tokens[i];
                if (fen.empty()) ok = false;
            } else {
                ok = false;
            }
            if (!ok) {
                send("info string error: expected 'position startpos' or 'position fen <fen>'");
                continue;
            }
            if (i < tokens.size() && tokens[i] == "moves")
                for (++i; i < tokens.size(); ++i) moves.push_back(tokens[i]);
            std::string error;
            if (!engine.setPosition(fen, moves, error)) send("info string error: " + error);
        } else if (cmd == "go" || cmd == "perft") {
            stopSearch();
            GoParams params;
            std::string error;
            std::vector<std::string> goTokens = tokens;
            if (cmd == "perft") goTokens = {"go", "perft", tokens.size() > 1 ? tokens[1] : "1"};
            if (!parseGo(goTokens, params, error)) {
                send("info string error: " + error);
                continue;
            }
            if (params.perft > 0) {
                auto start = std::chrono::steady_clock::now();
                uint64_t n = engine.perft(params.perft);
                auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
                send("info string perft depth " + std::to_string(params.perft) + " nodes " + std::to_string(n) +
                     " time " + std::to_string(ms));
                send("Nodes searched: " + std::to_string(n));
                continue;
            }
            stopFlag.store(false);
            ponderFlag.store(params.ponder);
            currentInfinite = params.infinite;
            worker = std::thread([&engine, params, &stopFlag, &ponderFlag] { engine.go(params, stopFlag, ponderFlag); });
        } else if (cmd == "stop") {
            stopSearch();
        } else if (cmd == "ponderhit") {
            // The predicted move was played: the search continues on the clock.
            ponderFlag.store(false);
        } else if (cmd == "quit") {
            stopSearch();
            return 0;
        } else if (cmd == "bench") {
            stopSearch();
            engine.bench(tokens.size() > 1 ? benchDepth(tokens[1]) : 0);
        } else if (cmd == "d" || cmd == "fen") {
            send(engine.fen());
        } else if (cmd == "debug" || cmd == "register") {
            // Not used.
        } else {
            send("info string unknown command '" + cmd + "'");
        }
    }
    // End of input: let a finite search finish so piped scripts get their
    // bestmove; an infinite or pondering one would never finish, so stop it.
    if (currentInfinite || ponderFlag.load()) stopFlag.store(true);
    waitForSearch();
    return 0;
}

} // namespace uci
