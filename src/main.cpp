// nags: the hybrid NAGS engine. The alpha-beta arm is nags_enhanced's full
// search; with the GNN service (rpc_server.py) running, an MCTS arm guided by
// the network searches in parallel and may replace the alpha-beta move after
// a verification search. meta_learner.py adjusts the hybrid's parameters per
// move. See NAGS.h.

#include "ClassicEngine.h"
#include "FastBoard.h"
#include "NAGS.h"

#include <algorithm>
#include <atomic>
#include <string>
#include <vector>

class NagsEngine : public ClassicEngine<FastBoard> {
public:
    NagsEngine() : ClassicEngine<FastBoard>("NAGS", /*useHashTable=*/true), controller(searcher) {
        setDefaultBenchDepth(10);
    }

    std::vector<std::string> optionLines() const override {
        std::vector<std::string> lines = ClassicEngine<FastBoard>::optionLines();
        const NagsSettings &s = controller.settings();
        lines.push_back(std::string("option name UseNN type check default ") + (s.useNN ? "true" : "false"));
        lines.push_back("option name NNHost type string default " + s.nnHost);
        lines.push_back("option name NNPort type spin default " + std::to_string(s.nnPort) + " min 1 max 65535");
        lines.push_back(std::string("option name UseMetaLearner type check default ") + (s.useMetaLearner ? "true" : "false"));
        lines.push_back("option name MetaHost type string default " + s.metaHost);
        lines.push_back("option name MetaPort type spin default " + std::to_string(s.metaPort) + " min 1 max 65535");
        lines.push_back("option name MetaExploration type spin default 0 min 0 max 100");
        lines.push_back("option name MctsMinTime type spin default " + std::to_string(s.minMctsMs) + " min 0 max 600000");
        return lines;
    }

    bool setOption(const std::string &opt, const std::string &value, std::string &message) override {
        NagsSettings &s = controller.settings();
        long long v = 0;
        bool b = false;
        auto badValue = [&] {
            message = "invalid value '" + value + "' for " + opt;
            return false;
        };
        if (opt == "UseNN") {
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
        } else if (opt == "MctsMinTime") {
            if (!uci::parseInt(value, v)) return badValue();
            s.minMctsMs = static_cast<int>(std::clamp<long long>(v, 0, 600000));
        } else {
            return ClassicEngine<FastBoard>::setOption(opt, value, message);
        }
        return true;
    }

    void newGame() override {
        ClassicEngine<FastBoard>::newGame();
        controller.newGame();
    }

    void go(const uci::GoParams &params, const std::atomic<bool> &stop, const std::atomic<bool> &ponder) override {
        FastBoard root = board;
        bool white = root.sideToMove() == Color::White;
        SearchLimits limits = uci::computeLimits(params, white, moveOverheadMs);
        int64_t timeLeft = white ? params.wtime : params.btime;
        NagsResult r = controller.search(
            root, limits, timeLeft, stop, &ponder, [](const SearchInfo &info) { uci::send(uci::formatInfo(info)); },
            [](const std::string &s) { uci::send("info string " + s); });
        uci::send(uci::bestMoveLine(r.bestMove, r.ponderMove));
    }

private:
    NAGSController controller;
};

int main(int argc, char **argv) {
    NagsEngine engine;
    return uci::run(engine, argc, argv);
}
