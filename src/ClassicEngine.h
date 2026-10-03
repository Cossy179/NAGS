#pragma once

// UCI engine around the shared alpha-beta Searcher. Used by nags_basic
// (Board, no hash table), nags_fast (FastBoard, no hash table) and
// nags_enhanced (FastBoard, hash table + Lazy SMP threads).

#include "Search.h"
#include "TT.h"
#include "Uci.h"

#include <algorithm>
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
        return lines;
    }

    bool setOption(const std::string &optName, const std::string &value, std::string &message) override {
        long long v = 0;
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

    void go(const uci::GoParams &params, const std::atomic<bool> &stop) override {
        BoardT root = board;
        SearchLimits limits = uci::computeLimits(params, root.sideToMove() == Color::White, moveOverheadMs);
        SearchResult result = searcher.search(root, limits, stop, [](const SearchInfo &info) {
            uci::send(uci::formatInfo(info));
        });
        uci::send(uci::bestMoveLine(result.bestMove, result.ponderMove));
    }

    uint64_t perft(int depth) override {
        BoardT copy = board;
        return copy.perft(depth);
    }

    std::string fen() const override { return board.getFEN(); }

private:
    static constexpr int kDefaultHashMB = 64;
    std::string engineName;
    std::unique_ptr<TranspositionTable> tt;
    Searcher<BoardT> searcher;
    BoardT board;
    int64_t moveOverheadMs = 50;
};
