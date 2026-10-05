#pragma once

// Transposition table shared by all search threads.
//
// Each entry is two 64-bit words: the packed data and (key XOR data). A reader
// accepts an entry only if the XOR checks out, so a torn read caused by a
// concurrent writer is rejected instead of returning another position's score
// (the usual lockless-hashing scheme). Both words are std::atomic with relaxed
// ordering, so there is no data race in the C++ sense.

#include "ChessTypes.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>

constexpr int MAX_PLY = 128;
constexpr int MATE_SCORE = 32000;
constexpr int MATE_BOUND = MATE_SCORE - MAX_PLY; // |score| >= this means a forced mate
constexpr int INF_SCORE = 32500;
// Tablebase win found `ply` plies from the root: TB_WIN_SCORE - ply. Below
// every mate score and above every evaluation.
constexpr int TB_WIN_SCORE = MATE_BOUND - MAX_PLY - 1;
constexpr int TB_BOUND = TB_WIN_SCORE - MAX_PLY; // |score| >= this: mate or tablebase result

// Mate and tablebase scores are stored relative to the node rather than the
// root so they stay correct when the same position is reached at a
// different ply.
inline int scoreToTT(int score, int ply) {
    if (score >= TB_BOUND) return score + ply;
    if (score <= -TB_BOUND) return score - ply;
    return score;
}

inline int scoreFromTT(int score, int ply) {
    if (score >= TB_BOUND) return score - ply;
    if (score <= -TB_BOUND) return score + ply;
    return score;
}

enum class Bound : uint8_t { None = 0, Upper = 1, Lower = 2, Exact = 3 };

struct TTHit {
    Move move;        // from/to/promotion kind only; isNull() if none stored
    int score = 0;    // node-relative (use scoreFromTT)
    int depth = 0;
    Bound bound = Bound::None;
};

class TranspositionTable {
public:
    explicit TranspositionTable(size_t megabytes = 16);

    // Rounds the entry count down to a power of two so the table never uses
    // more memory than requested. Clears the table.
    void resize(size_t megabytes);
    void clear();
    void newSearch() { generation = static_cast<uint8_t>(generation + 1); }

    bool probe(uint64_t key, TTHit &out) const;
    void store(uint64_t key, int depth, Bound bound, int score, const Move &move);

    // Permille of sampled entries written during the current search (UCI hashfull).
    int hashfull() const;
    size_t entryCount() const { return mask + 1; }
    static constexpr size_t entryBytes() { return sizeof(Entry); }

private:
    struct Entry {
        std::atomic<uint64_t> keyXorData;
        std::atomic<uint64_t> data;
    };
    std::unique_ptr<Entry[]> table;
    size_t mask = 0;
    uint8_t generation = 0;
};
