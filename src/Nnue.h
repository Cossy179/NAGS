#pragma once

// NNUE evaluation: a 768 -> 2x256 -> 1 network trained by tools/nnue/train.py
// (see that file for the layout and the file format).
//
// The two 256-wide accumulators (one per perspective) are kept in the board
// and updated incrementally as pieces are put and removed; evaluation applies
// a clipped ReLU and the output layer. The network is global to the process:
// load()/setEnabled() must not be called while a search is running, and
// boards created before a change must call refresh() (the search does this
// for its root).

#include "ChessTypes.h"

#include <cstdint>
#include <string>

namespace nnue {

#ifndef NAGS_NNUE_HIDDEN
#define NAGS_NNUE_HIDDEN 256 // CMake option NAGS_NNUE_HIDDEN
#endif

constexpr int kFeatures = 768;
constexpr int kHidden = NAGS_NNUE_HIDDEN; // networks of another size are rejected when loaded
constexpr int QA = 255, QB = 64, SCALE = 400;
constexpr int kMaxBuckets = 8;

// Output bucket for a position with `pieces` pieces (1 bucket: always 0).
inline int bucketOf(int pieces, int buckets) {
    int b = (pieces - 1) * buckets / 32;
    return b < 0 ? 0 : b >= buckets ? buckets - 1 : b;
}

struct Network {
    alignas(64) int16_t ftWeights[kFeatures * kHidden];
    alignas(64) int16_t ftBias[kHidden];
    alignas(64) int16_t outWeights[kMaxBuckets][2 * kHidden]; // [bucket][us | them]
    int32_t outBias[kMaxBuckets];
    int buckets; // output layers in use, chosen by the number of pieces
};

struct Accumulator {
    alignas(64) int16_t v[2][kHidden]; // [perspective: White, Black]
};

namespace detail {
inline const Network *active = nullptr; // null: NNUE off
}

// The network in use, or null when NNUE is off.
inline const Network *network() { return detail::active; }

// Loads a network file; on success it becomes the active network (if NNUE
// is enabled). Returns false and sets `error` otherwise.
bool load(const std::string &path, std::string &error);
// The network compiled into the binary, if any (null otherwise).
const Network *embedded();
// Name of the loaded network ("embedded", a file path, or "" if none).
std::string networkName();
// Forgets a network loaded from a file (back to the embedded one, if any).
void useEmbedded();
// Turns NNUE on (with the loaded network, else the embedded one) or off.
// Returns whether NNUE is on afterwards.
bool setEnabled(bool on);

inline int featureIndex(int perspective, Piece p, int sq) {
    int colour = colorIndex(colorOf(p)) ^ perspective; // 0 = the perspective's own pieces
    if (perspective == 1) sq ^= 56;
    return colour * 384 + pieceTypeOf(p) * 64 + sq;
}

void addFeature(Accumulator &acc, const Network &net, Piece p, int sq);

// Pieces a move adds and removes (at most two of each: castling, captures).
struct DirtyPieces {
    int adds = 0, removes = 0;
    Piece addPiece[2], removePiece[2];
    int addSquare[2], removeSquare[2];
    void add(Piece p, int sq) { addPiece[adds] = p; addSquare[adds++] = sq; }
    void remove(Piece p, int sq) { removePiece[removes] = p; removeSquare[removes++] = sq; }
};

// next = prev with the move's pieces added and removed, in one pass.
void update(const Accumulator &prev, Accumulator &next, const Network &net, const DirtyPieces &d);

// Recomputes both accumulators from the pieces on the board.
template <class BoardT>
void refresh(Accumulator &acc, const BoardT &b) {
    const Network *net = network();
    if (!net) return;
    for (int persp = 0; persp < 2; ++persp)
        for (int i = 0; i < kHidden; ++i) acc.v[persp][i] = net->ftBias[i];
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = b.pieceAt(sq);
        if (p != Piece::None) addFeature(acc, *net, p, sq);
    }
}

// Centipawns from the side to move's point of view (with the active
// network); `occupied` selects the output bucket by piece count.
int evaluate(const Accumulator &acc, Color stm, uint64_t occupied);

} // namespace nnue
