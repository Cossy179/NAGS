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

constexpr int kFeatures = 768;
constexpr int kHidden = 256;
constexpr int QA = 255, QB = 64, SCALE = 400;

struct Network {
    alignas(64) int16_t ftWeights[kFeatures * kHidden];
    alignas(64) int16_t ftBias[kHidden];
    alignas(64) int16_t outWeights[2 * kHidden];
    int32_t outBias;
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

// Defined out of line so the board's move code stays small when NNUE is off.
void addFeature(Accumulator &acc, const Network &net, Piece p, int sq);
void removeFeature(Accumulator &acc, const Network &net, Piece p, int sq);

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

// Centipawns from the side to move's point of view.
inline int evaluate(const Accumulator &acc, Color stm) {
    const Network &net = *network();
    const int16_t *us = acc.v[colorIndex(stm)];
    const int16_t *them = acc.v[colorIndex(stm) ^ 1];
    int32_t sum = 0;
    for (int i = 0; i < kHidden; ++i) {
        int a = us[i] < 0 ? 0 : us[i] > QA ? QA : us[i];
        int b = them[i] < 0 ? 0 : them[i] > QA ? QA : them[i];
        sum += a * net.outWeights[i] + b * net.outWeights[kHidden + i];
    }
    return static_cast<int>((static_cast<int64_t>(sum) + net.outBias) * SCALE / (QA * QB));
}

} // namespace nnue
