#pragma once

// Legality of a pseudo-legal move without making it, for any board exposing
// sideToMove(), pieceBB(), occupancy() and attackersTo(sq, occupied).
//
// A move is legal if the mover's king is not attacked afterwards. For a
// king move that is: the target square is not attacked once the king has
// left its square (sliders see through it). For any other move: no enemy
// piece except the captured one attacks the king, given the occupancy
// after the move (en passant also empties the captured pawn's square).
// Castling is fully checked when it is generated (not in check, no attacked
// square on the king's path).

#include "BitOps.h"
#include "ChessTypes.h"

template <class BoardT>
bool isLegalMove(const BoardT &b, const Move &m) {
    const Color us = b.sideToMove(), them = opposite(us);
    Bitboard enemies = 0;
    for (int t = PAWN; t <= KING; ++t) enemies |= b.pieceBB(them, t);
    const Bitboard king = b.pieceBB(us, KING);
    const Bitboard fromBB = 1ULL << m.from, toBB = 1ULL << m.to;
    if (king & fromBB) {
        if (m.isCastling) return true;
        return !(b.attackersTo(m.to, b.occupancy() ^ fromBB) & enemies);
    }
    Bitboard occ = (b.occupancy() ^ fromBB) | toBB;
    Bitboard captured = toBB;
    if (m.isEnPassant) {
        captured = 1ULL << (us == Color::White ? m.to - 8 : m.to + 8);
        occ ^= captured;
    }
    return !(b.attackersTo(lsb(king), occ) & enemies & ~captured);
}
