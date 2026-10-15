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
