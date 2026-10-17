#pragma once

// Plausibility checks for positions loaded from a FEN, shared by Board and
// FastBoard. Positions that fail them would break assumptions the engine
// relies on (move lists of at most MoveList::kCapacity moves; an en-passant
// capture removes a pawn that is really there).

#include "BitOps.h"
#include "ChessTypes.h"

#include <algorithm>

// Material reachable from the starting position: at most 8 pawns a side, and
// no more promoted pieces than pawns that are gone.
template <class BoardT>
bool plausibleMaterial(const BoardT &b) {
    for (Color c : {Color::White, Color::Black}) {
        const int pawns = popcount(b.pieceBB(c, PAWN));
        const int extra = std::max(0, popcount(b.pieceBB(c, QUEEN)) - 1) + std::max(0, popcount(b.pieceBB(c, ROOK)) - 2) +
                          std::max(0, popcount(b.pieceBB(c, BISHOP)) - 2) + std::max(0, popcount(b.pieceBB(c, KNIGHT)) - 2);
        if (pawns > 8 || extra > 8 - pawns) return false;
    }
    return true;
}

// Whether `sq` can be the en-passant square with `stm` to move: the opponent
// has just pushed a pawn two squares past it (the square and the one the pawn
// came from are empty, the pawn stands in front of it).
template <class BoardT>
bool plausibleEpSquare(const BoardT &b, int sq, Color stm) {
    if (sq < 0 || sq > 63 || rankOf(sq) != (stm == Color::White ? 5 : 2)) return false;
    const int pawnSq = stm == Color::White ? sq - 8 : sq + 8, fromSq = stm == Color::White ? sq + 8 : sq - 8;
    const Bitboard occ = b.occupancy();
    return !(occ & ((1ULL << sq) | (1ULL << fromSq))) && (b.pieceBB(opposite(stm), PAWN) & (1ULL << pawnSq));
}

// Whether a pawn of `stm` could capture en passant on `sq` (it stands next to
// the pawn that has just been pushed two squares). An en-passant square no
// pawn can use is not part of the position: FEN parsing and makeMove drop
// it, so that it never makes two otherwise equal positions differ (for
// repetitions, the hash and the TT).
template <class BoardT>
bool epCapturable(const BoardT &b, int sq, Color stm) {
    const int pawnSq = stm == Color::White ? sq - 8 : sq + 8;
    Bitboard adjacent = 0;
    if (fileOf(pawnSq) > 0) adjacent |= 1ULL << (pawnSq - 1);
    if (fileOf(pawnSq) < 7) adjacent |= 1ULL << (pawnSq + 1);
    return (b.pieceBB(stm, PAWN) & adjacent) != 0;
}
