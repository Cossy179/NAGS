#pragma once

// Static evaluation shared by every engine: material, piece-square tables
// (tapered between middlegame and endgame for the king) and a bishop-pair
// bonus. Works with any board type exposing pieceBB(Color, type) and
// sideToMove().
//
// The tables are written the way they are usually printed: rank 8 in the
// first row, from White's point of view. Squares are numbered a1 = 0, so a
// White piece on `sq` reads entry `sq ^ 56` and a Black piece reads `sq`.

#include "BitOps.h"
#include "ChessTypes.h"
#include "Nnue.h"

#include <type_traits>
#include <utility>

namespace eval {

constexpr int PIECE_VALUE[6] = {100, 320, 330, 500, 900, 0};

constexpr int PST[6][64] = {
    { // pawn
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0},
    { // knight
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50},
    { // bishop
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20},
    { // rook
          0,  0,  0,  0,  0,  0,  0,  0,
          5, 10, 10, 10, 10, 10, 10,  5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
         -5,  0,  0,  0,  0,  0,  0, -5,
          0,  0,  0,  5,  5,  0,  0,  0},
    { // queen
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20},
    { // king, middlegame
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20},
};

constexpr int KING_ENDGAME[64] = {
    -50,-40,-30,-20,-20,-30,-40,-50,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-30,  0,  0,  0,  0,-30,-30,
    -50,-30,-30,-30,-30,-30,-30,-50,
};

inline int pstIndex(Color c, int sq) { return c == Color::White ? (sq ^ 56) : sq; }

inline int pieceValue(Piece p) { return p == Piece::None ? 0 : PIECE_VALUE[pieceTypeOf(p)]; }

// Value of the piece a move captures (en passant captures a pawn).
template <class BoardT>
int capturedValue(const BoardT &b, const Move &m) {
    if (m.isEnPassant) return PIECE_VALUE[PAWN];
    return pieceValue(b.pieceAt(m.to));
}

template <class BoardT>
bool isNoisy(const BoardT &b, const Move &m) {
    return m.isEnPassant || m.promotion != Piece::None || b.pieceAt(m.to) != Piece::None;
}

// Boards that keep material + piece-square values and the game phase up to
// date incrementally (FastBoard) provide psqScore() and gamePhase().
template <class BoardT, class = void>
struct HasIncrementalEval : std::false_type {};
template <class BoardT>
struct HasIncrementalEval<BoardT, std::void_t<decltype(std::declval<const BoardT &>().psqScore()),
                                              decltype(std::declval<const BoardT &>().gamePhase())>> : std::true_type {};

// Boards that keep NNUE accumulators (FastBoard) provide accumulator().
template <class BoardT, class = void>
struct HasNnue : std::false_type {};
template <class BoardT>
struct HasNnue<BoardT, std::void_t<decltype(std::declval<const BoardT &>().accumulator())>> : std::true_type {};

// Material and piece-square values of everything but the kings (White minus
// Black), and the uncapped game phase, computed from scratch.
template <class BoardT>
void materialAndPhase(const BoardT &b, int &score, int &phase) {
    score = 0;
    phase = 0;
    for (int c = 0; c < 2; ++c) {
        Color color = static_cast<Color>(c);
        int sign = c == 0 ? 1 : -1;
        for (int type = PAWN; type <= QUEEN; ++type) {
            Bitboard bb = b.pieceBB(color, type);
            int count = popcount(bb);
            score += sign * count * PIECE_VALUE[type];
            phase += count * (type == KNIGHT || type == BISHOP ? 1 : type == ROOK ? 2 : type == QUEEN ? 4 : 0);
            while (bb) score += sign * PST[type][pstIndex(color, popLsb(bb))];
        }
    }
}

// Score in centipawns from the side to move's point of view: the NNUE
// network when one is active (FastBoard), otherwise the hand-written terms.
template <class BoardT>
int evaluate(const BoardT &b) {
    if constexpr (HasNnue<BoardT>::value) {
        if (nnue::network()) return nnue::evaluate(b.accumulator(), b.sideToMove());
    }
    int score, phase; // phase: 24 = all pieces on the board, 0 = bare kings and pawns
    if constexpr (HasIncrementalEval<BoardT>::value) {
        score = b.psqScore();
        phase = b.gamePhase();
    } else {
        materialAndPhase(b, score, phase);
    }
    if (popcount(b.pieceBB(Color::White, BISHOP)) >= 2) score += 30;
    if (popcount(b.pieceBB(Color::Black, BISHOP)) >= 2) score -= 30;
    if (phase > 24) phase = 24;
    for (int c = 0; c < 2; ++c) {
        Color color = static_cast<Color>(c);
        Bitboard kings = b.pieceBB(color, KING);
        if (!kings) continue;
        int idx = pstIndex(color, lsb(kings));
        int king = (PST[KING][idx] * phase + KING_ENDGAME[idx] * (24 - phase)) / 24;
        score += c == 0 ? king : -king;
    }
    return b.sideToMove() == Color::White ? score : -score;
}

// Small capture-only search used to give static evaluations some tactical
// sanity (e.g. by the MCTS heuristic evaluator). No time control; the depth
// cap bounds the work.
template <class BoardT>
int quiescence(BoardT &b, int alpha, int beta, int depthLeft) {
    int stand = evaluate(b);
    if (stand >= beta || depthLeft <= 0) return stand;
    if (stand > alpha) alpha = stand;
    int best = stand;
    for (const Move &m : b.generateLegalMoves(true)) {
        if (stand + capturedValue(b, m) + 200 < alpha && m.promotion == Piece::None) continue;
        b.makeMove(m);
        int score = -quiescence(b, -beta, -alpha, depthLeft - 1);
        b.unmakeMove();
        if (score > best) {
            best = score;
            if (score > alpha) {
                alpha = score;
                if (alpha >= beta) break;
            }
        }
    }
    return best;
}

} // namespace eval
