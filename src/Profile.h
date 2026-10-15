#pragma once

// `speedtest`: the cost of each engine component in isolation, in
// nanoseconds per operation, measured on a fixed set of positions (the bench
// positions plus positions from deterministic random playouts of them).
// Complements `bench` (whole-search speed) and NAGS_STATS builds (what the
// search does): together they show where the time per node goes.

#include "Bench.h"
#include "Eval.h"
#include "TT.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace profile {

template <class BoardT>
std::vector<BoardT> samplePositions(size_t perStart = 24) {
    std::vector<BoardT> out;
    std::mt19937_64 rng(20261014);
    for (const std::string &fen : bench::positions()) {
        BoardT b;
        if (!b.setFromFEN(fen)) continue;
        out.push_back(b);
        for (size_t i = 0; i < perStart; ++i) {
            auto moves = b.generateLegalMoves();
            if (moves.empty() || b.isDraw()) break;
            b.makeMove(moves[rng() % moves.size()]);
            BoardT copy;
            copy.setFromFEN(b.getFEN()); // a fresh board: no history, accumulators from scratch
            out.push_back(copy);
        }
    }
    return out;
}

struct Timer {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    double ns() const {
        return static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
    }
};

// Runs `body` over the positions `rounds` times; returns ns per operation
// (body returns the number of operations it did).
template <class BoardT, class Body>
double measure(std::vector<BoardT> &positions, int rounds, Body body) {
    uint64_t ops = 0;
    Timer t;
    for (int r = 0; r < rounds; ++r)
        for (BoardT &b : positions) ops += body(b);
    return ops ? t.ns() / static_cast<double>(ops) : 0.0;
}

template <class BoardT>
std::vector<std::string> run() {
    std::vector<BoardT> positions = samplePositions<BoardT>();
    std::vector<std::string> lines;
    volatile int64_t sink = 0; // keeps the work from being optimised away
    auto line = [&](const char *what, double ns, const char *per) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "speedtest %-34s %9.1f ns per %s", what, ns, per);
        lines.emplace_back(buf);
    };
    const int R = 200;

    line("generate all pseudo-legal moves", measure(positions, R, [&](BoardT &b) {
             MoveList m;
             b.generatePseudoLegalMoves(m);
             sink = sink + m.size();
             return 1;
         }), "position");
    line("generate captures/promotions", measure(positions, R, [&](BoardT &b) {
             MoveList m;
             b.generatePseudoLegalMoves(m, true);
             sink = sink + m.size();
             return 1;
         }), "position");
    line("generateLegalMoves (vector)", measure(positions, R / 4, [&](BoardT &b) {
             sink = sink + static_cast<int64_t>(b.generateLegalMoves().size());
             return 1;
         }), "position");
    line("inCheck", measure(positions, R * 4, [&](BoardT &b) {
             sink = sink + b.inCheck();
             return 1;
         }), "call");
    line("makeMove + unmakeMove", measure(positions, R / 4, [&](BoardT &b) {
             MoveList m;
             b.generatePseudoLegalMoves(m);
             for (int i = 0; i < m.size(); ++i) {
                 b.makeMove(m[i]);
                 b.unmakeMove();
             }
             return m.size();
         }), "move");
    line("legality check (make, inCheck, unmake)", measure(positions, R / 4, [&](BoardT &b) {
             MoveList m;
             b.generatePseudoLegalMoves(m);
             const Color us = b.sideToMove();
             for (int i = 0; i < m.size(); ++i) {
                 b.makeMove(m[i]);
                 sink = sink + b.inCheck(us);
                 b.unmakeMove();
             }
             return m.size();
         }), "move");
    line("static exchange evaluation", measure(positions, R / 2, [&](BoardT &b) {
             MoveList m;
             b.generatePseudoLegalMoves(m, true);
             for (int i = 0; i < m.size(); ++i) sink = sink + eval::see(b, m[i]);
             return m.size();
         }), "capture");
    line("isDraw", measure(positions, R * 2, [&](BoardT &b) {
             sink = sink + b.isDraw();
             return 1;
         }), "call");
    line("evaluate (accumulators ready)", measure(positions, R * 4, [&](BoardT &b) {
             sink = sink + eval::evaluate(b);
             return 1;
         }), "call");
    if constexpr (eval::HasNnue<BoardT>::value) {
        if (nnue::network()) {
            line("make + evaluate (incremental) + unmake", measure(positions, R / 4, [&](BoardT &b) {
                     MoveList m;
                     b.generatePseudoLegalMoves(m);
                     const Color us = b.sideToMove();
                     int n = 0;
                     for (int i = 0; i < m.size(); ++i) {
                         b.makeMove(m[i]);
                         if (!b.inCheck(us)) {
                             sink = sink + eval::evaluate(b);
                             ++n;
                         }
                         b.unmakeMove();
                     }
                     return n;
                 }), "move");
            line("accumulator refresh (both sides)", measure(positions, R, [&](BoardT &b) {
                     b.refreshAccumulator();
                     sink = sink + eval::evaluate(b);
                     return 1;
                 }), "position");
        }
    }
    TranspositionTable tt(64);
    std::mt19937_64 rng(1);
    std::vector<uint64_t> keys(1 << 21); // far more entries than the caches hold, as in a real search
    for (auto &k : keys) k = rng();
    {
        Timer t;
        for (int r = 0; r < 2; ++r)
            for (uint64_t k : keys) tt.store(k, 5, Bound::Exact, static_cast<int>(k & 255), Move{});
        line("TT store (64 MB, random keys)", t.ns() / (2.0 * keys.size()), "store");
    }
    {
        Timer t;
        TTHit hit;
        for (int r = 0; r < 2; ++r)
            for (uint64_t k : keys) sink = sink + tt.probe(k ^ (r & 1), hit);
        line("TT probe (64 MB, random keys)", t.ns() / (2.0 * keys.size()), "probe");
    }
    lines.push_back("speedtest positions " + std::to_string(positions.size()) + " checksum " + std::to_string(sink));
    return lines;
}

} // namespace profile
