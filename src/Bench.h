#pragma once

// Positions searched by the `bench` command. The total node count of a bench
// run is a fingerprint of the engine's search: a change that should not alter
// search behaviour (a refactor, a speed-up) must leave it unchanged, and a
// functional change should record the new value in its commit message
// ("Bench: <nodes>"). Changing this list changes every fingerprint.

#include <chrono>
#include <cstdint>
#include <string>
#include <vector>

namespace bench {

inline const std::vector<std::string> &positions() {
    static const std::vector<std::string> fens = {
        // Openings and early middlegames
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkb1r/pp1p1ppp/4pn2/2p5/2PP4/5N2/PP2PPPP/RNBQKB1R w KQkq - 0 4",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
        "rnbqk2r/ppp1bppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 4 5",
        "r1bqk2r/pp2bppp/2nppn2/8/3NP3/2N1B3/PPP1BPPP/R2QK2R w KQkq - 2 8",
        "rnbq1rk1/pp2ppbp/3p1np1/2p5/2PPP3/2N2N2/PP2BPPP/R1BQK2R w KQ - 0 7",
        // Middlegames
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        "r1bq1rk1/pp3ppp/2n1pn2/2bp4/2P5/2N1PN2/PPQ1BPPP/R1B2RK1 b - - 3 9",
        "2rq1rk1/pb1nbppp/1p2pn2/2pp4/2PP4/1PNBPN2/PB3PPP/2RQ1RK1 w - - 2 12",
        "r2qr1k1/1b1nbppp/p2p1n2/1pp1p3/4P3/2PP1N1P/PPB2PP1/RNBQR1K1 w - - 0 13",
        "1r2r1k1/pp1q1ppp/2p2n2/3p4/3P4/2NQ1N2/PP3PPP/2R2RK1 w - - 4 18",
        "r1b2rk1/2q1bppp/p2ppn2/1p6/3BPP2/2NB4/PPPQ2PP/2KR3R w - - 0 13",
        "3r2k1/pp3ppp/2n1b3/q1pp4/8/P1NPP1P1/1P1Q1PBP/R4RK1 b - - 0 17",
        "r1b1k2r/ppppnppp/2n2q2/2b5/3NP3/2P1B3/PP3PPP/RN1QKB1R w KQkq - 0 7",
        "6k1/pp3pp1/2p1r2p/3p4/3P1q2/2P1R1P1/PP3P1P/4Q1K1 w - - 0 27",
        // Tactics
        "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",
        "2r3k1/p4p2/3Rp2p/1p2P1pK/8/1P4P1/P3Q2P/1q6 b - - 0 1",
        "r2q1rk1/ppp2ppp/2np1n2/2b1p1B1/2B1P1b1/2NP1N2/PPP2PPP/R2Q1RK1 w - - 4 8",
        "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1",
        // Endgames
        "8/8/8/4k3/8/8/4P3/4K3 w - - 0 1",
        "8/8/1k6/8/2KP4/8/8/8 w - - 0 1",
        "1K1k4/1P6/8/8/8/8/r7/2R5 w - - 0 1",
        "3k4/8/3K4/3P4/8/8/7r/R7 b - - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "8/5pk1/6p1/1r5p/7P/5KP1/5P2/R7 w - - 0 40",
        "8/8/3k4/3p4/3P4/3K4/8/8 w - - 0 1",
        "6k1/5p1p/6p1/8/8/1P4P1/5PKP/8 w - - 0 1",
        "8/3k4/8/3N4/8/8/3BK3/8 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K2Q w - - 0 1",
    };
    return fens;
}

// Summary in the format used by Stockfish (and understood by OpenBench),
// followed by a compact "<nodes> nodes <nps> nps" line.
inline std::vector<std::string> summary(uint64_t nodes, int64_t elapsedMs) {
    uint64_t ms = elapsedMs > 0 ? static_cast<uint64_t>(elapsedMs) : 1;
    uint64_t nps = nodes * 1000 / ms;
    return {
        "===========================",
        "Total time (ms) : " + std::to_string(ms),
        "Nodes searched  : " + std::to_string(nodes),
        "Nodes/second    : " + std::to_string(nps),
        std::to_string(nodes) + " nodes " + std::to_string(nps) + " nps",
    };
}

} // namespace bench
