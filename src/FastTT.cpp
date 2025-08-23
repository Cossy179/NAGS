#include "FastTT.h"
#include <algorithm>

FastTranspositionTable::FastTranspositionTable() {
    resizeMB(64); // Default 64MB
}

void FastTranspositionTable::resizeMB(size_t megabytes) {
    size_t bytes = megabytes * 1024ull * 1024ull;
    size_t entries = std::max<size_t>(1, bytes / sizeof(TTEntry));
    
    // Round down to power of 2
    size_t pow2 = 1;
    while (pow2 * 2 <= entries) pow2 <<= 1;
    
    table.clear();
    table.resize(pow2);
    mask = pow2 - 1;
    
    clear();
}

void FastTranspositionTable::clear() {
    for (auto &entry : table) {
        entry.key = 0;
        entry.score = 0;
        entry.depth = -1;
        entry.flag = 0;
        entry.movePacked = 0;
    }
    resetStats();
}

bool FastTranspositionTable::probe(uint64_t key, TTEntry &out) const {
    probes.fetch_add(1, std::memory_order_relaxed);
    
    const TTEntry &entry = table[indexFromKey(key)];
    
    // Simple read (not fully thread-safe but good enough for chess engines)
    if (entry.key == key) {
        out = entry;
        hits.fetch_add(1, std::memory_order_relaxed);
        return true;
    }
    
    return false;
}

void FastTranspositionTable::store(uint64_t key, int depth, TTFlag flag, int score, const Move *bestMove) {
    TTEntry &entry = table[indexFromKey(key)];
    
    // Replace if different position or deeper search
    if (entry.key != key || depth >= entry.depth) {
        entry.movePacked = bestMove ? packMove(*bestMove) : 0;
        entry.score = score;
        entry.depth = static_cast<int16_t>(depth);
        entry.flag = static_cast<uint8_t>(flag);
        entry.key = key; // Store key last
    }
}

uint32_t FastTranspositionTable::packMove(const Move &m) {
    uint32_t promo = 0;
    switch (m.promotion) {
        case Piece::WQ: case Piece::BQ: promo = 1; break;
        case Piece::WR: case Piece::BR: promo = 2; break;
        case Piece::WB: case Piece::BB: promo = 3; break;
        case Piece::WN: case Piece::BN: promo = 4; break;
        default: promo = 0; break;
    }
    
    uint32_t flags = 0;
    if (m.isEnPassant) flags |= 1;
    if (m.isCastling) flags |= 2;
    
    return (static_cast<uint32_t>(m.from) & 63) | 
           ((static_cast<uint32_t>(m.to) & 63) << 6) | 
           (promo << 12) |
           (flags << 16);
}

Move FastTranspositionTable::unpackMove(uint32_t p) {
    Move m;
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
    
    uint32_t flags = (p >> 16) & 3;
    m.isEnPassant = (flags & 1) != 0;
    m.isCastling = (flags & 2) != 0;
    
    return m;
}
