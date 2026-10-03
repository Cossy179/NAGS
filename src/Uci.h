#pragma once

// UCI front end shared by all engines. The search runs on a worker thread so
// that "stop", "quit" and "isready" are handled while it thinks. Each engine
// implements uci::Engine and calls uci::run().

#include "Search.h"

#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace uci {

// Writes one line to stdout (thread-safe, flushed).
void send(const std::string &line);

std::vector<std::string> tokenize(const std::string &line);

struct GoParams {
    int64_t wtime = -1, btime = -1; // -1: not given
    int64_t winc = 0, binc = 0;
    int64_t movetime = 0;
    int movestogo = 0;
    int depth = 0;
    uint64_t nodes = 0;
    int perft = 0;
    bool infinite = false;
    bool ponder = false; // "go ponder": the clock starts at ponderhit
};

// Returns false (with a message) on malformed numbers.
bool parseGo(const std::vector<std::string> &tokens, GoParams &out, std::string &error);

// Converts clock information into search limits. softMs: don't start another
// iteration after this; hardMs: abort the search. Both 0 means "no time limit".
SearchLimits computeLimits(const GoParams &go, bool whiteToMove, int64_t moveOverheadMs);

// "info depth ... pv ..." for a completed iteration.
std::string formatInfo(const SearchInfo &info);

std::string bestMoveLine(const Move &best, const Move &ponder);

class Engine {
public:
    virtual ~Engine() = default;
    virtual std::string name() const = 0;
    virtual std::string author() const { return "NAGS contributors"; }
    // Complete "option name ... type ..." lines.
    virtual std::vector<std::string> optionLines() const = 0;
    // Returns false for unknown options or bad values; `message` is reported as an info string.
    virtual bool setOption(const std::string &name, const std::string &value, std::string &message) = 0;
    virtual void newGame() = 0;
    // fen empty means the start position. On error the previous position is kept.
    virtual bool setPosition(const std::string &fen, const std::vector<std::string> &moves, std::string &error) = 0;
    // Runs on the worker thread and must finish by printing "bestmove".
    // `ponder` is set during "go ponder" until "ponderhit"; engines that
    // support pondering hold bestmove while it is set (as for "infinite").
    virtual void go(const GoParams &params, const std::atomic<bool> &stop, const std::atomic<bool> &ponder) = 0;
    virtual uint64_t perft(int depth) = 0;
    virtual std::string fen() const = 0;
    // Searches the bench positions single-threaded to `depth` (0 = engine
    // default) from a clean state, prints per-position lines and the summary
    // from bench::summary(), then leaves the engine as after ucinewgame.
    virtual void bench(int depth) = 0;
};

// Runs the UCI loop. If argv[1] is "bench" (optionally followed by a depth),
// runs the bench instead and exits.
int run(Engine &engine, int argc = 0, char **argv = nullptr);

// Helpers for option parsing.
bool parseInt(const std::string &s, long long &out);
bool parseBool(const std::string &s, bool &out);

} // namespace uci
