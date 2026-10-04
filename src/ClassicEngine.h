#pragma once

// UCI engine around the shared alpha-beta Searcher. Used by nags_basic
// (Board, no hash table), nags_fast (FastBoard, no hash table) and
// nags_enhanced (FastBoard, hash table + Lazy SMP threads).

#include "Bench.h"
#include "Search.h"
#include "TT.h"
#include "Uci.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>

template <class BoardT>
class ClassicEngine : public uci::Engine {
public:
    ClassicEngine(std::string engineName, bool useHashTable)
        : engineName(std::move(engineName)),
          tt(useHashTable ? std::make_unique<TranspositionTable>(kDefaultHashMB) : nullptr),
          searcher(tt.get()) {}

    std::string name() const override { return engineName; }

    std::vector<std::string> optionLines() const override {
        std::vector<std::string> lines;
        if (tt) {
            lines.push_back("option name Hash type spin default " + std::to_string(kDefaultHashMB) + " min 1 max 4096");
            lines.push_back("option name Threads type spin default 1 min 1 max 64");
            lines.push_back("option name Clear Hash type button");
        }
        lines.push_back("option name Move Overhead type spin default 50 min 0 max 5000");
        lines.push_back("option name MultiPV type spin default 1 min 1 max 256");
        lines.push_back("option name Ponder type check default false");
        for (const auto &l : uci::syzygyOptionLines()) lines.push_back(l);
        if constexpr (eval::HasNnue<BoardT>::value) {
            lines.push_back("option name UseNNUE type check default true");
            lines.push_back("option name EvalFile type string default <embedded>");
        }
        return lines;
    }

    bool setOption(const std::string &optName, const std::string &value, std::string &message) override {
        long long v = 0;
        bool ok = false;
        if (uci::setSyzygyOption(optName, value, message, ok)) return ok;
        if constexpr (eval::HasNnue<BoardT>::value) {
            if (optName == "UseNNUE") {
                bool on = false;
                if (!uci::parseBool(value, on)) { message = "invalid UseNNUE value '" + value + "'"; return false; }
                bool active = nnue::setEnabled(on);
                message = active ? "NNUE evaluation: " + nnue::networkName()
                                 : on ? "no NNUE network loaded; using the classical evaluation" : "classical evaluation";
                return true;
            }
            if (optName == "EvalFile") {
                if (value.empty() || value == "<embedded>") {
                    nnue::useEmbedded();
                } else {
                    std::string error;
                    if (!nnue::load(value, error)) { message = "EvalFile: " + error; return false; }
                }
                message = nnue::network() ? "NNUE evaluation: " + nnue::networkName()
                                          : "no NNUE network; using the classical evaluation";
                return true;
            }
        }
        if (tt && optName == "Hash") {
            if (!uci::parseInt(value, v)) { message = "invalid Hash value '" + value + "'"; return false; }
            v = std::clamp<long long>(v, 1, 4096);
            tt->resize(static_cast<size_t>(v));
            message = "Hash set to " + std::to_string(v) + " MB";
            return true;
        }
        if (tt && optName == "Threads") {
            if (!uci::parseInt(value, v)) { message = "invalid Threads value '" + value + "'"; return false; }
            searcher.setThreads(static_cast<int>(std::clamp<long long>(v, 1, 64)));
            message = "Threads set to " + std::to_string(searcher.threadCount());
            return true;
        }
        if (tt && optName == "Clear Hash") {
            tt->clear();
            message = "Hash cleared";
            return true;
        }
        if (optName == "Move Overhead") {
            if (!uci::parseInt(value, v)) { message = "invalid Move Overhead value '" + value + "'"; return false; }
            moveOverheadMs = std::clamp<long long>(v, 0, 5000);
            return true;
        }
        if (optName == "MultiPV") {
            if (!uci::parseInt(value, v)) { message = "invalid MultiPV value '" + value + "'"; return false; }
            searcher.setMultiPv(static_cast<int>(std::clamp<long long>(v, 1, 256)));
            return true;
        }
        if (optName == "Ponder") {
            // Only tells the engine the GUI may ponder; "go ponder" works either way.
            bool b = false;
            if (!uci::parseBool(value, b)) { message = "invalid Ponder value '" + value + "'"; return false; }
            return true;
        }
        return false;
    }

    void newGame() override {
        if (tt) tt->clear();
        searcher.clearHistory();
        board.setStartPos();
    }

    bool setPosition(const std::string &fen, const std::vector<std::string> &moves, std::string &error) override {
        BoardT next;
        if (!fen.empty() && !next.setFromFEN(fen)) {
            error = "invalid FEN '" + fen + "'";
            return false;
        }
        for (const std::string &m : moves) {
            if (!next.applyMovesUCI({m})) {
                error = "illegal move '" + m + "' in position " + next.getFEN();
                return false;
            }
        }
        board = next;
        return true;
    }

    void go(const uci::GoParams &params, const std::atomic<bool> &stop, const std::atomic<bool> &ponder) override {
        BoardT root = board;
        SearchLimits limits = uci::computeLimits(params, root.sideToMove() == Color::White, moveOverheadMs);
        SearchResult result = searcher.search(
            root, limits, stop, [](const SearchInfo &info) { uci::send(uci::formatInfo(info)); }, &ponder);
        uci::send(uci::bestMoveLine(result.bestMove, result.ponderMove));
    }

    uint64_t perft(int depth) override {
        BoardT copy = board;
        return copy.perft(depth);
    }

    std::string fen() const override { return board.getFEN(); }

    std::string staticEvaluation() const override {
        BoardT b = board;
        b.refreshAccumulator();
        std::string kind = "classical";
        if constexpr (eval::HasNnue<BoardT>::value)
            if (nnue::network()) kind = "nnue " + nnue::networkName();
        return "info string eval " + std::to_string(eval::evaluate(b)) + " (side to move, " + kind + ")";
    }

    void bench(int depth) override {
        if (depth <= 0) depth = defaultBenchDepth;
        int threads = searcher.threadCount();
        int lines = searcher.multiPvCount();
        searcher.setThreads(1); // helper threads would make the node count nondeterministic
        searcher.setMultiPv(1);
        newGame();
        const auto &fens = bench::positions();
        uint64_t total = 0;
        std::atomic<bool> stop{false};
        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < fens.size(); ++i) {
            BoardT b;
            b.setFromFEN(fens[i]);
            SearchLimits limits;
            limits.depth = depth;
            SearchResult r = searcher.search(b, limits, stop, nullptr);
            total += r.nodes;
            uci::send("info string bench " + std::to_string(i + 1) + "/" + std::to_string(fens.size()) + " nodes " +
                      std::to_string(r.nodes) + " bestmove " + moveToUciString(r.bestMove));
        }
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count();
        for (const auto &line : bench::summary(total, ms)) uci::send(line);
        searcher.setThreads(threads);
        searcher.setMultiPv(lines);
        newGame();
    }

    // Depth used by `bench` without an argument (chosen per engine so a run
    // takes a few seconds).
    void setDefaultBenchDepth(int depth) { defaultBenchDepth = depth; }

protected:
    static constexpr int kDefaultHashMB = 64;
    int defaultBenchDepth = 10;
    std::string engineName;
    std::unique_ptr<TranspositionTable> tt;
    Searcher<BoardT> searcher;
    BoardT board;
    int64_t moveOverheadMs = 50;
};
