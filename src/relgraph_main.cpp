// relgraph_dump: the engine's relational-v1 graph of each FEN read from
// stdin (one per line), printed as encodeGraph's base64, or "error" for a FEN
// the board does not accept. tests/python/test_relgraph.py compares the output
// with relational_graph.build_graph; "--bench N" times N builds of each FEN.

#include "FastBoard.h"
#include "RelGraph.h"

#include <chrono>
#include <cstdio>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
    long long benchIters = 0;
    if (argc == 3 && std::string(argv[1]) == "--bench") benchIters = std::stoll(argv[2]);
    std::ios::sync_with_stdio(false);
    std::string line;
    RelGraph g;
    long long builds = 0;
    double seconds = 0;
    while (std::getline(std::cin, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        FastBoard board;
        if (!board.setFromFEN(line)) {
            std::cout << "error\n";
            continue;
        }
        if (benchIters > 0) {
            const auto t0 = std::chrono::steady_clock::now();
            for (long long i = 0; i < benchIters; ++i) {
                buildRelationalGraph(board, g);
                encodeGraph(g);
            }
            seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            builds += benchIters;
            continue;
        }
        buildRelationalGraph(board, g);
        std::cout << encodeGraph(g) << '\n';
    }
    if (benchIters > 0)
        std::printf("%lld graphs in %.3f s: %.2f us per graph (build + encode)\n", builds, seconds,
                    builds ? seconds * 1e6 / builds : 0.0);
    return 0;
}
