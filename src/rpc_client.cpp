// Smoke test for rpc_server.py: sends one batched request (8 positions, with
// their legal moves) and checks that every result carries a value in [-1, 1],
// an uncertainty and one prior per legal move.
//
// Usage: rpc_client [host] [port]

#include "Board.h"
#include "Net.h"

#include <cmath>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main(int argc, char **argv) {
    std::string host = argc > 1 ? argv[1] : "127.0.0.1";
    int port = argc > 2 ? std::atoi(argv[2]) : 5555;

    LineSocket sock;
    bool connected = false;
    for (int attempt = 0; attempt < 50 && !connected; ++attempt) {
        connected = sock.connect(host, port, 200);
        if (!connected) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!connected) {
        std::cerr << "could not connect to " << host << ":" << port << std::endl;
        return 1;
    }

    const std::vector<std::string> fens = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "8/8/8/K2pP2q/8/8/8/7k w - d6 0 1",
        "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1",
    };

    std::vector<size_t> moveCounts;
    std::string request = "{\"fens\":[";
    std::string moves = "\"moves\":[";
    for (size_t i = 0; i < fens.size(); ++i) {
        Board b;
        if (!b.setFromFEN(fens[i])) {
            std::cerr << "bad test FEN " << fens[i] << std::endl;
            return 1;
        }
        auto legal = b.generateLegalMoves();
        moveCounts.push_back(legal.size());
        request += (i ? ",\"" : "\"") + jsonlite::escape(fens[i]) + "\"";
        moves += i ? ",[" : "[";
        for (size_t j = 0; j < legal.size(); ++j) moves += (j ? ",\"" : "\"") + moveToUciString(legal[j]) + "\"";
        moves += "]";
    }
    request += "]," + moves + "]}";

    std::string response;
    if (!sock.request(request, response, 60000)) {
        std::cerr << "no response from server" << std::endl;
        return 1;
    }
    std::string error;
    if (jsonlite::findString(response, "error", error)) {
        std::cerr << "server error: " << error << std::endl;
        return 2;
    }

    // Results come back in order; walk them one "move_priors" array at a time.
    size_t pos = 0;
    for (size_t i = 0; i < fens.size(); ++i) {
        size_t start = response.find("{\"value\"", pos);
        if (start == std::string::npos) start = response.find("\"value\"", pos);
        size_t end = response.find('}', start);
        if (start == std::string::npos || end == std::string::npos) {
            std::cerr << "missing result " << i << std::endl;
            return 2;
        }
        std::string item = response.substr(start, end - start + 1);
        double value = 0, uncertainty = 0;
        std::vector<double> priors;
        if (!jsonlite::findNumber(item, "value", value) || !jsonlite::findNumber(item, "uncertainty", uncertainty) ||
            !jsonlite::findNumberArray(item, "move_priors", priors)) {
            std::cerr << "malformed result " << i << ": " << item.substr(0, 200) << std::endl;
            return 2;
        }
        double sum = 0;
        for (double p : priors) sum += p;
        if (value < -1.0 || value > 1.0 || priors.size() != moveCounts[i] || std::abs(sum - 1.0) > 1e-3) {
            std::cerr << "bad result " << i << ": value " << value << ", " << priors.size() << " priors (expected "
                      << moveCounts[i] << "), sum " << sum << std::endl;
            return 2;
        }
        std::cout << "position " << i << ": value " << value << " uncertainty " << uncertainty << " moves "
                  << priors.size() << std::endl;
        pos = end + 1;
    }
    std::cout << "OK: received " << fens.size() << " evaluations in one RPC call." << std::endl;
    return 0;
}
