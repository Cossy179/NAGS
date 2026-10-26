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

#if defined(_MSC_VER)
#define NAGS_NOINLINE __declspec(noinline)
#else
#define NAGS_NOINLINE __attribute__((noinline))
#endif

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

namespace legality_detail {
constexpr Bitboard stepAttacks(int sq, const int (*steps)[2], int n, bool slide) {
    Bitboard a = 0;
    for (int i = 0; i < n; ++i)
        for (int f = sq % 8 + steps[i][0], r = sq / 8 + steps[i][1]; f >= 0 && f < 8 && r >= 0 && r < 8;
             f += steps[i][0], r += steps[i][1]) {
            a |= 1ULL << (r * 8 + f);
            if (!slide) break;
        }
    return a;
}
struct StepTables {
    Bitboard knight[64] = {}, king[64] = {}, lines[64] = {}; // lines: a queen's squares on an empty board
    constexpr StepTables() {
        constexpr int kn[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
        constexpr int kg[8][2] = {{1, 0}, {1, 1}, {0, 1}, {-1, 1}, {-1, 0}, {-1, -1}, {0, -1}, {1, -1}};
        for (int sq = 0; sq < 64; ++sq) {
            knight[sq] = stepAttacks(sq, kn, 8, false);
            king[sq] = stepAttacks(sq, kg, 8, false);
            lines[sq] = stepAttacks(sq, kg, 8, true);
        }
    }
};
inline constexpr StepTables kSteps{};

// Whether one of the side's pawns, knights or sliders in `movable` has a
// pseudo-legal move (en passant aside); `own` and `enemies` are the sides'
// pieces.
template <class BoardT>
bool anyPieceMove(const BoardT &b, Color us, Bitboard movable, Bitboard own, Bitboard enemies) {
    const Bitboard occ = own | enemies, pawns = b.pieceBB(us, PAWN) & movable;
    constexpr Bitboard kNotA = ~0x0101010101010101ULL, kNotH = ~0x8080808080808080ULL;
    const Bitboard pawnMoves = us == Color::White
        ? (pawns << 8 & ~occ) | ((pawns & kNotA) << 7 & enemies) | ((pawns & kNotH) << 9 & enemies)
        : (pawns >> 8 & ~occ) | ((pawns & kNotH) >> 7 & enemies) | ((pawns & kNotA) >> 9 & enemies);
    if (pawnMoves) return true;
    for (Bitboard p = b.pieceBB(us, KNIGHT) & movable; p;)
        if (kSteps.knight[popLsb(p)] & ~own) return true;
    const Bitboard queens = b.pieceBB(us, QUEEN);
    for (Bitboard p = (b.pieceBB(us, BISHOP) | queens) & movable; p;)
        if (BoardT::bishopAttacks(popLsb(p), occ) & ~own) return true;
    for (Bitboard p = (b.pieceBB(us, ROOK) | queens) & movable; p;)
        if (BoardT::rookAttacks(popLsb(p), occ) & ~own) return true;
    return false;
}

// hasLegalMoveOutsideCheck after its first test (out of line: the first
// test settles most calls and stays small where it is inlined).
template <class BoardT>
NAGS_NOINLINE bool hasLegalMoveRest(const BoardT &b, int ksq) {
    const Color us = b.sideToMove(), them = opposite(us);
    const Bitboard occ = b.occupancy(), kingBB = 1ULL << ksq, lines = kSteps.lines[ksq];
    Bitboard own = 0;
    for (int t = PAWN; t <= KING; ++t) own |= b.pieceBB(us, t);
    const Bitboard enemies = occ & ~own;
    if (anyPieceMove(b, us, ~lines, own, enemies)) return true;
    // Squares between the king and an enemy slider on one of its lines: a
    // piece there may be pinned (unless another one stands between too).
    Bitboard pinnable = 0;
    const Bitboard queens = b.pieceBB(them, QUEEN);
    for (Bitboard s = BoardT::rookAttacks(ksq, 0) & (b.pieceBB(them, ROOK) | queens); s;) {
        const int sq = popLsb(s);
        pinnable |= BoardT::rookAttacks(ksq, 1ULL << sq) & BoardT::rookAttacks(sq, kingBB);
    }
    for (Bitboard s = BoardT::bishopAttacks(ksq, 0) & (b.pieceBB(them, BISHOP) | queens); s;) {
        const int sq = popLsb(s);
        pinnable |= BoardT::bishopAttacks(ksq, 1ULL << sq) & BoardT::bishopAttacks(sq, kingBB);
    }
    if (anyPieceMove(b, us, lines & ~pinnable, own, enemies)) return true;
    for (Bitboard t = kSteps.king[ksq] & ~own; t;)
        if (!(b.attackersTo(popLsb(t), occ ^ kingBB) & enemies)) return true;
    // Left: pieces that may be pinned (they can still move along the pin)
    // and en passant. Castling needs a safe square next to the king.
    return !b.generateLegalMoves().empty();
}
} // namespace legality_detail

// Whether the side to move, not in check, has a legal move (outside check
// "none" is stalemate), mostly without generating moves. Outside check a
// pseudo-legal move of a piece that is not pinned is legal. A piece off
// every line through its king cannot be pinned, so one such pawn able to
// step forward settles it at once (the common case, checked inline); then
// such knights and sliders with a square to go to, the pieces on those
// lines with no enemy slider behind them, and the king's own squares. Only
// a side whose pieces are all blocked or possibly pinned has its moves
// generated.
template <class BoardT>
inline bool hasLegalMoveOutsideCheck(const BoardT &b) {
    const Color us = b.sideToMove();
    const Bitboard kingBB = b.pieceBB(us, KING);
    if (!kingBB) return !b.generateLegalMoves().empty();
    const int ksq = lsb(kingBB);
    const Bitboard free = b.pieceBB(us, PAWN) & ~legality_detail::kSteps.lines[ksq];
    if ((us == Color::White ? free << 8 : free >> 8) & ~b.occupancy()) return true;
    return legality_detail::hasLegalMoveRest(b, ksq);
}
