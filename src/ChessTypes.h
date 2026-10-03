#pragma once

// Types shared by both board implementations (Board and FastBoard) and by the
// search, evaluation and UCI layers. Squares are numbered a1 = 0 ... h8 = 63.

#include <cstdint>
#include <string>

using Bitboard = uint64_t;

enum class Piece : uint8_t {
    None = 0,
    WP, WN, WB, WR, WQ, WK,
    BP, BN, BB, BR, BQ, BK
};

enum class Color : uint8_t { White = 0, Black = 1 };

// Piece kinds, independent of colour.
enum PieceType : int { PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5 };

struct Move {
    int from = -1;          // 0..63, -1 for "no move"
    int to = -1;            // 0..63
    Piece promotion = Piece::None; // side-coloured promotion piece, or None
    bool isEnPassant = false;
    bool isCastling = false;

    bool isNull() const { return from < 0 || to < 0; }
};

inline constexpr Color opposite(Color c) { return c == Color::White ? Color::Black : Color::White; }

inline constexpr int colorIndex(Color c) { return static_cast<int>(c); }

// 0..5 (PAWN..KING) or -1 for Piece::None.
inline constexpr int pieceTypeOf(Piece p) {
    return p == Piece::None ? -1 : (static_cast<int>(p) - 1) % 6;
}

inline constexpr bool isWhitePiece(Piece p) { return p >= Piece::WP && p <= Piece::WK; }
inline constexpr bool isBlackPiece(Piece p) { return p >= Piece::BP && p <= Piece::BK; }

inline constexpr Color colorOf(Piece p) { return isBlackPiece(p) ? Color::Black : Color::White; }

inline constexpr Piece makePiece(Color c, int type) {
    return static_cast<Piece>(1 + type + (c == Color::Black ? 6 : 0));
}

inline constexpr int fileOf(int sq) { return sq & 7; }
inline constexpr int rankOf(int sq) { return sq >> 3; }

// Two moves are the same if they share from/to and promotion kind. Promotion
// colour is ignored so moves decoded from the transposition table (which only
// keeps the kind) compare equal to generated moves.
inline bool sameMove(const Move &a, const Move &b) {
    return a.from == b.from && a.to == b.to && pieceTypeOf(a.promotion) == pieceTypeOf(b.promotion);
}

inline std::string squareName(int sq) {
    return std::string{static_cast<char>('a' + fileOf(sq)), static_cast<char>('1' + rankOf(sq))};
}

// -1 if the string is not a square name.
inline int parseSquare(const std::string &s) {
    if (s.size() < 2) return -1;
    int f = s[0] - 'a';
    int r = s[1] - '1';
    if (f < 0 || f > 7 || r < 0 || r > 7) return -1;
    return r * 8 + f;
}

inline std::string moveToUciString(const Move &m) {
    if (m.isNull()) return "0000";
    std::string s = squareName(m.from) + squareName(m.to);
    switch (pieceTypeOf(m.promotion)) {
        case QUEEN: s += 'q'; break;
        case ROOK: s += 'r'; break;
        case BISHOP: s += 'b'; break;
        case KNIGHT: s += 'n'; break;
        default: break;
    }
    return s;
}
