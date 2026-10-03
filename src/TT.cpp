#include "TT.h"

#include <algorithm>

namespace {

// data layout:
//   bits  0..5   from square
//   bits  6..11  to square
//   bits 12..14  promotion kind + 1 (0 = none)
//   bit  15      move present
//   bits 16..31  score (int16)
//   bits 32..39  depth (0..255)
//   bits 40..41  bound
//   bits 42..49  generation
uint64_t pack(int depth, Bound bound, int score, const Move &move, uint8_t generation) {
    uint64_t d = 0;
    if (!move.isNull()) {
        int promo = pieceTypeOf(move.promotion) + 1; // 0 if none
        d |= static_cast<uint64_t>(move.from & 63);
        d |= static_cast<uint64_t>(move.to & 63) << 6;
        d |= static_cast<uint64_t>(promo & 7) << 12;
        d |= 1ULL << 15;
    }
    d |= static_cast<uint64_t>(static_cast<uint16_t>(static_cast<int16_t>(score))) << 16;
    d |= static_cast<uint64_t>(std::clamp(depth, 0, 255)) << 32;
    d |= static_cast<uint64_t>(static_cast<uint8_t>(bound) & 3) << 40;
    d |= static_cast<uint64_t>(generation) << 42;
    return d;
}

inline int depthOf(uint64_t d) { return static_cast<int>((d >> 32) & 0xFF); }
inline uint8_t generationOf(uint64_t d) { return static_cast<uint8_t>((d >> 42) & 0xFF); }

} // namespace

TranspositionTable::TranspositionTable(size_t megabytes) { resize(megabytes); }

void TranspositionTable::resize(size_t megabytes) {
    megabytes = std::max<size_t>(1, megabytes);
    size_t entries = megabytes * 1024ULL * 1024ULL / sizeof(Entry);
    size_t pow2 = 1;
    while (pow2 * 2 <= entries) pow2 <<= 1;
    table.reset(new Entry[pow2]);
    mask = pow2 - 1;
    clear();
}

void TranspositionTable::clear() {
    for (size_t i = 0; i <= mask; ++i) {
        table[i].keyXorData.store(0, std::memory_order_relaxed);
        table[i].data.store(0, std::memory_order_relaxed);
    }
    generation = 0;
}

bool TranspositionTable::probe(uint64_t key, TTHit &out) const {
    const Entry &e = table[key & mask];
    uint64_t data = e.data.load(std::memory_order_relaxed);
    uint64_t check = e.keyXorData.load(std::memory_order_relaxed);
    if (data == 0 || (check ^ data) != key) return false;

    out.move = Move{};
    if (data & (1ULL << 15)) {
        out.move.from = static_cast<int>(data & 63);
        out.move.to = static_cast<int>((data >> 6) & 63);
        int promo = static_cast<int>((data >> 12) & 7) - 1;
        // Colour is irrelevant for comparisons (sameMove ignores it).
        out.move.promotion = promo >= 0 ? makePiece(Color::White, promo) : Piece::None;
    }
    out.score = static_cast<int16_t>(static_cast<uint16_t>((data >> 16) & 0xFFFF));
    out.depth = depthOf(data);
    out.bound = static_cast<Bound>((data >> 40) & 3);
    return true;
}

void TranspositionTable::store(uint64_t key, int depth, Bound bound, int score, const Move &move) {
    Entry &e = table[key & mask];
    uint64_t oldData = e.data.load(std::memory_order_relaxed);
    uint64_t oldCheck = e.keyXorData.load(std::memory_order_relaxed);
    bool sameKey = oldData != 0 && (oldCheck ^ oldData) == key;

    if (sameKey) {
        // Keep a deeper result for the same position unless the new one is exact.
        if (bound != Bound::Exact && depth + 2 < depthOf(oldData) && generationOf(oldData) == generation) return;
    } else if (oldData != 0 && generationOf(oldData) == generation && depth + 4 < depthOf(oldData)) {
        // Don't evict a much deeper entry from the current search.
        return;
    }

    Move m = move;
    if (m.isNull() && sameKey && (oldData & (1ULL << 15))) {
        // Preserve the previously stored best move.
        m.from = static_cast<int>(oldData & 63);
        m.to = static_cast<int>((oldData >> 6) & 63);
        int promo = static_cast<int>((oldData >> 12) & 7) - 1;
        m.promotion = promo >= 0 ? makePiece(Color::White, promo) : Piece::None;
    }
    uint64_t data = pack(depth, bound, score, m, generation);
    e.data.store(data, std::memory_order_relaxed);
    e.keyXorData.store(key ^ data, std::memory_order_relaxed);
}

int TranspositionTable::hashfull() const {
    size_t samples = std::min<size_t>(1000, mask + 1);
    size_t used = 0;
    for (size_t i = 0; i < samples; ++i) {
        uint64_t d = table[i].data.load(std::memory_order_relaxed);
        if (d != 0 && generationOf(d) == generation) ++used;
    }
    return static_cast<int>(used * 1000 / samples);
}
