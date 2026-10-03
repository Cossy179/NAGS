// nags: the hybrid NAGS engine (alpha-beta + MCTS chosen by a bandit, GNN
// evaluation over RPC, meta-learned hyperparameters).

#include "Board.h"
#include "NAGS.h"
#include "TT.h"
#include "Uci.h"

#include <algorithm>
#include <string>
#include <vector>

class NagsEngine : public uci::Engine {
public:
    NagsEngine() : tt(kDefaultHashMB), controller(tt) {}

    std::string name() const override { return "NAGS"; }

    std::vector<std::string> optionLines() const override {
        const NagsSettings &s = controller.settings();
        return {
            "option name Hash type spin default " + std::to_string(kDefaultHashMB) + " min 1 max 4096",
            "option name Clear Hash type button",
            "option name Move Overhead type spin default 50 min 0 max 5000",
            std::string("option name UseNN type check default ") + (s.useNN ? "true" : "false"),
            "option name NNHost type string default " + s.nnHost,
            "option name NNPort type spin default " + std::to_string(s.nnPort) + " min 1 max 65535",
            std::string("option name UseMetaLearner type check default ") + (s.useMetaLearner ? "true" : "false"),
            "option name MetaHost type string default " + s.metaHost,
            "option name MetaPort type spin default " + std::to_string(s.metaPort) + " min 1 max 65535",
            "option name MetaExploration type spin default 0 min 0 max 100",
        };
    }

    bool setOption(const std::string &opt, const std::string &value, std::string &message) override {
        NagsSettings &s = controller.settings();
        long long v = 0;
        bool b = false;
        auto badValue = [&] {
            message = "invalid value '" + value + "' for " + opt;
            return false;
        };
        if (opt == "Hash") {
            if (!uci::parseInt(value, v)) return badValue();
            v = std::clamp<long long>(v, 1, 4096);
            tt.resize(static_cast<size_t>(v));
            message = "Hash set to " + std::to_string(v) + " MB";
        } else if (opt == "Clear Hash") {
            tt.clear();
            message = "Hash cleared";
        } else if (opt == "Move Overhead") {
            if (!uci::parseInt(value, v)) return badValue();
            moveOverheadMs = std::clamp<long long>(v, 0, 5000);
        } else if (opt == "UseNN") {
            if (!uci::parseBool(value, b)) return badValue();
            s.useNN = b;
        } else if (opt == "NNHost") {
            if (value.empty()) return badValue();
            s.nnHost = value;
        } else if (opt == "NNPort") {
            if (!uci::parseInt(value, v) || v < 1 || v > 65535) return badValue();
            s.nnPort = static_cast<int>(v);
        } else if (opt == "UseMetaLearner") {
            if (!uci::parseBool(value, b)) return badValue();
            s.useMetaLearner = b;
        } else if (opt == "MetaHost") {
            if (value.empty()) return badValue();
            s.metaHost = value;
        } else if (opt == "MetaPort") {
            if (!uci::parseInt(value, v) || v < 1 || v > 65535) return badValue();
            s.metaPort = static_cast<int>(v);
        } else if (opt == "MetaExploration") {
            if (!uci::parseInt(value, v)) return badValue();
            s.metaExploration = static_cast<float>(std::clamp<long long>(v, 0, 100)) / 100.0f;
        } else {
            return false;
        }
        return true;
    }

    void newGame() override {
        tt.clear();
        controller.newGame();
        board.setStartPos();
    }

    bool setPosition(const std::string &fen, const std::vector<std::string> &moves, std::string &error) override {
        Board next;
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
        Board root = board; // the search never touches the engine's own board
        bool white = root.sideToMove() == Color::White;
        SearchLimits limits = uci::computeLimits(params, white, moveOverheadMs);
        int64_t timeLeft = white ? params.wtime : params.btime;
        NagsResult r = controller.search(
            root, limits, timeLeft, stop, [](const SearchInfo &info) { uci::send(uci::formatInfo(info)); },
            [](const std::string &s) { uci::send("info string " + s); });
        uci::send(uci::bestMoveLine(r.bestMove, r.ponderMove));
    }

    uint64_t perft(int depth) override {
        Board copy = board;
        return copy.perft(depth);
    }

    std::string fen() const override { return board.getFEN(); }

private:
    static constexpr int kDefaultHashMB = 64;
    TranspositionTable tt;
    NAGSController controller;
    Board board;
    int64_t moveOverheadMs = 50;
};

int main() {
    NagsEngine engine;
    return uci::run(engine);
}
