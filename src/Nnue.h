#pragma once

// NNUE evaluation: a perspective network trained by tools/nnue/train.py (see
// that file for the layout and the file format).
//
// Inputs: for each perspective, 768 piece-square features, optionally in one
// of several sets chosen by that side's king square ("king buckets", with the
// board mirrored left-right when the king is on files e-h). A feature
// transformer turns them into two accumulators (one per perspective), which
// the board keeps up to date incrementally as pieces are put and removed;
// evaluation applies a clipped ReLU (or its square, SCReLU) and one of up to
// 8 output layers, chosen by the number of pieces. The network is global to
// the process: load()/setEnabled() must not be called while a search is
// running, and boards created before a change must call refresh() (the
// search does this for its root).

#include "BitOps.h"
#include "ChessTypes.h"

#include <cstdint>
#include <string>
#include <vector>

namespace nnue {

#ifndef NAGS_NNUE_HIDDEN
#define NAGS_NNUE_HIDDEN 256 // CMake option NAGS_NNUE_HIDDEN
#endif

constexpr int kFeatures = 768;
constexpr int kHidden = NAGS_NNUE_HIDDEN; // networks of another size are rejected when loaded
constexpr int QA = 255, QB = 64, SCALE = 400;
constexpr int kMaxBuckets = 8;
constexpr int kMaxKingBuckets = 32;

// Output bucket for a position with `pieces` pieces (1 bucket: always 0).
inline int bucketOf(int pieces, int buckets) {
    int b = (pieces - 1) * buckets / 32;
    return b < 0 ? 0 : b >= buckets ? buckets - 1 : b;
}

struct Network {
    std::vector<int16_t> ftWeights; // [kingBuckets][768][kHidden]
    alignas(64) int16_t ftBias[kHidden];
    alignas(64) int16_t outWeights[kMaxBuckets][2 * kHidden]; // [bucket][us | them]
    int32_t outBias[kMaxBuckets];
    int buckets = 1;        // output layers in use, chosen by the number of pieces
    int kingBuckets = 1;    // input feature sets, chosen by the perspective's king square
    uint8_t kingLayout[64] = {}; // king square (from the perspective) -> input set
    bool mirror = false;    // king on files e-h: the perspective's squares are mirrored left-right
    bool screlu = false;    // activation: squared clipped ReLU instead of clipped ReLU
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

// Which feature set `perspective` uses with its king on `kingSq`:
// set * 2 + (1 if the squares are mirrored left-right). Accumulators can only
// be updated incrementally while this stays the same.
inline int kingState(const Network &net, int perspective, int kingSq) {
    if (perspective == 1) kingSq ^= 56;
    return net.kingLayout[kingSq] * 2 + (net.mirror && (kingSq & 7) >= 4 ? 1 : 0);
}
inline int kingStates(const Network &net) { return 2 * net.kingBuckets; }

inline int featureIndex(int perspective, Piece p, int sq, int kstate) {
    int colour = colorIndex(colorOf(p)) ^ perspective; // 0 = the perspective's own pieces
    if (perspective == 1) sq ^= 56;
    if (kstate & 1) sq ^= 7;
    return (kstate >> 1) * kFeatures + colour * 384 + pieceTypeOf(p) * 64 + sq;
}

// a += (or -=) the weights of one feature.
void addFeature(int16_t *a, const Network &net, int feature);
void subFeature(int16_t *a, const Network &net, int feature);

// Pieces a move adds and removes (at most two of each: castling, captures).
struct DirtyPieces {
    int adds = 0, removes = 0;
    Piece addPiece[2], removePiece[2];
    int addSquare[2], removeSquare[2];
    void add(Piece p, int sq) { addPiece[adds] = p; addSquare[adds++] = sq; }
    void remove(Piece p, int sq) { removePiece[removes] = p; removeSquare[removes++] = sq; }
};

// out = in with the move's pieces added and removed, in one pass, for one
// perspective whose king state is `kstate` before and after the move.
void update(const int16_t *in, int16_t *out, const Network &net, const DirtyPieces &d, int perspective, int kstate);

// Recomputes one perspective's accumulator from the pieces on the board.
template <class BoardT>
void refresh(int16_t *a, const BoardT &b, int perspective) {
    const Network *net = network();
    if (!net) return;
    const int kstate = kingState(*net, perspective, lsb(b.pieceBB(perspective ? Color::Black : Color::White, KING)));
    for (int i = 0; i < kHidden; ++i) a[i] = net->ftBias[i];
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = b.pieceAt(sq);
        if (p != Piece::None) addFeature(a, *net, featureIndex(perspective, p, sq, kstate));
    }
}

// Recomputes both accumulators.
template <class BoardT>
void refresh(Accumulator &acc, const BoardT &b) {
    refresh(acc.v[0], b, 0);
    refresh(acc.v[1], b, 1);
}

// Centipawns from the side to move's point of view (with the active
// network); `occupied` selects the output bucket by piece count.
int evaluate(const Accumulator &acc, Color stm, uint64_t occupied);

} // namespace nnue
