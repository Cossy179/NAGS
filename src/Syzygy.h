#pragma once

// Syzygy endgame tablebases, probed with Fathom (third_party/fathom).
//
// The tables are global to the process. init() must not be called while a
// search is running. probeWdl() is thread-safe; probeRoot() must only be
// called by one thread at a time.

#include "BitOps.h"
#include "ChessTypes.h"

#include <cstdint>
#include <string>

namespace syzygy {

enum class Wdl { Loss, BlessedLoss, Draw, CursedWin, Win };

// The position in the form Fathom expects (a1 = bit 0).
struct Position {
    uint64_t white = 0, black = 0;
    uint64_t kings = 0, queens = 0, rooks = 0, bishops = 0, knights = 0, pawns = 0;
    unsigned rule50 = 0;   // half-move clock
    unsigned castling = 0; // nonzero: the tables do not apply
    unsigned ep = 0;       // en passant square, 0 if none
    bool whiteToMove = true;
};

// Loads the tables found in `path` (several directories separated by ':', or
// ';' on Windows; "" or "<empty>" unloads them). Returns the largest number of
// pieces the loaded tables cover (0 if none were found).
int init(const std::string &path);

// Largest number of pieces the loaded tables cover (0 = none loaded).
int largest();

// Positions with more pieces than this are not probed (UCI SyzygyProbeLimit).
void setProbeLimit(int pieces);

namespace detail {
inline int cardinality = 0; // min(largest(), probe limit)
}

// Positions with at most this many pieces are probed; 0 = tablebases off.
inline int cardinality() { return detail::cardinality; }

// Win/draw/loss for the side to move. Fails (returns false) unless the
// half-move clock is 0 and no castling rights remain: the WDL tables do not
// know how close the fifty-move rule is.
bool probeWdl(const Position &pos, Wdl &wdl);

// Best root move by distance to zeroing (DTZ): it keeps the tablebase result,
// taking the fifty-move counter into account, and makes progress. Needs the
// DTZ tables. promotion is QUEEN..KNIGHT or -1.
struct RootResult {
    int from = 0, to = 0, promotion = -1;
    Wdl wdl = Wdl::Draw;
    unsigned dtz = 0;
};
bool probeRoot(const Position &pos, RootResult &out);

template <class BoardT>
Position position(const BoardT &b) {
    Position p;
    Bitboard *byType[6] = {&p.pawns, &p.knights, &p.bishops, &p.rooks, &p.queens, &p.kings};
    for (int type = PAWN; type <= KING; ++type) {
        Bitboard w = b.pieceBB(Color::White, type), k = b.pieceBB(Color::Black, type);
        p.white |= w;
        p.black |= k;
        *byType[type] = w | k;
    }
    p.rule50 = static_cast<unsigned>(b.getHalfmoveClock());
    p.castling = b.getCastlingRights();
    p.ep = b.getEpSquare() > 0 ? static_cast<unsigned>(b.getEpSquare()) : 0;
    p.whiteToMove = b.sideToMove() == Color::White;
    return p;
}

} // namespace syzygy
