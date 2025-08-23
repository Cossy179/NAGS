#pragma once

#include <cstdint>
#include <vector>

#include "Board.h"

enum class TTFlag : uint8_t { Exact = 0, Lower = 1, Upper = 2 };

struct TTEntry {
    uint64_t key; // non-atomic; simple single-writer design
    int32_t score;
    int16_t depth; // in plies
    uint8_t flag;  // TTFlag
    uint32_t movePacked; // from(6) | to(6) | promo(4)

    TTEntry() : key(0), score(0), depth(-1), flag(0), movePacked(0) {}
};

class TranspositionTable {
public:
    TranspositionTable();

    void resizeMB(size_t megabytes);
    void clear();

    bool probe(uint64_t key, TTEntry &out) const;
    void store(uint64_t key, int depth, TTFlag flag, int score, const Move *bestMove);

    static uint32_t packMove(const Move &m);
    static Move unpackMove(uint32_t p);

private:
    std::vector<TTEntry> table;
    size_t mask = 0; // table.size()-1 when size is power of two
};


