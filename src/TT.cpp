#include "TT.h"

#include <algorithm>

TranspositionTable::TranspositionTable() {
    resizeMB(16);
}

void TranspositionTable::resizeMB(size_t megabytes) {
    size_t bytes = megabytes * 1024ull * 1024ull;
    size_t entries = std::max<size_t>(1, bytes / sizeof(TTEntry));
    size_t pow2 = 1;
    while (pow2 < entries) pow2 <<= 1;
    table.clear();
    table.reserve(pow2);
    for (size_t i = 0; i < pow2; ++i) table.emplace_back();
    mask = pow2 - 1;
    clear();
}

void TranspositionTable::clear() {
    for (auto &e : table) {
        e.key = 0;
        e.score = 0;
        e.depth = -1;
        e.flag = 0;
        e.movePacked = 0;
    }
}

static inline size_t indexFromKey(uint64_t key, size_t mask) {
    return key & mask;
}

bool TranspositionTable::probe(uint64_t key, TTEntry &out) const {
    const TTEntry &e = table[indexFromKey(key, mask)];
    if (e.key == key) { out = e; return true; }
    return false;
}

void TranspositionTable::store(uint64_t key, int depth, TTFlag flag, int score, const Move *bestMove) {
    TTEntry &e = table[indexFromKey(key, mask)];
    // Replace by depth
    if (e.key != key || depth >= e.depth) {
        e.key = key;
        e.depth = static_cast<int16_t>(depth);
        e.flag = static_cast<uint8_t>(flag);
        e.score = score;
        e.movePacked = bestMove ? packMove(*bestMove) : 0;
    }
}

uint32_t TranspositionTable::packMove(const Move &m) {
    uint32_t promo = 0;
    switch (m.promotion) {
        case Piece::WQ: case Piece::BQ: promo = 1; break;
        case Piece::WR: case Piece::BR: promo = 2; break;
        case Piece::WB: case Piece::BB: promo = 3; break;
        case Piece::WN: case Piece::BN: promo = 4; break;
        default: promo = 0; break;
    }
    return (static_cast<uint32_t>(m.from) & 63) | ((static_cast<uint32_t>(m.to) & 63) << 6) | (promo << 12);
}

Move TranspositionTable::unpackMove(uint32_t p) {
    Move m{ -1, -1, Piece::None, false, false };
    m.from = p & 63;
    m.to = (p >> 6) & 63;
    uint32_t promo = (p >> 12) & 15;
    switch (promo) {
        case 1: m.promotion = Piece::WQ; break;
        case 2: m.promotion = Piece::WR; break;
        case 3: m.promotion = Piece::WB; break;
        case 4: m.promotion = Piece::WN; break;
        default: m.promotion = Piece::None; break;
    }
    return m;
}


