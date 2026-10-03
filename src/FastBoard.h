#pragma once

// Bitboard board with magic-bitboard sliding attacks and a mailbox for O(1)
// piece lookup. Used by `nags_fast` and `nags_enhanced`. Same interface as
// Board so the search/evaluation templates work with either.

#include "ChessTypes.h"

#include <cstdint>
#include <string>
#include <vector>

struct Magic {
    Bitboard mask;
    Bitboard magic;
    Bitboard *attacks;
    int shift;
};

class FastBoard {
public:
    FastBoard();

    // Loads a FEN. On failure the board is left unchanged and false is returned.
    bool setFromFEN(const std::string &fen);
    std::string getFEN() const;
    bool setStartPos();
    // Applies UCI moves in order; stops at (and returns false for) the first illegal one.
    bool applyMovesUCI(const std::vector<std::string> &uciMoves);

    // Legal moves for the side to move; noisyOnly keeps captures and promotions.
    std::vector<Move> generateLegalMoves(bool noisyOnly = false) const;
    void generatePseudoMoves(std::vector<Move> &moves) const;

    bool inCheck() const { return inCheck(side); }
    bool inCheck(Color c) const;
    Color sideToMove() const { return side; }
    Piece pieceAt(int sq) const { return mailbox[sq]; }
    Bitboard pieceBB(Color c, int type) const { return pieces[colorIndex(c)][type]; }
    Bitboard occupancy() const { return all_occupied; }
    uint64_t zobrist() const { return hash; }
    int getHalfmoveClock() const { return halfmoveClock; }
    int getFullmoveNumber() const { return fullmoveNumber; }

    void makeMove(const Move &m);
    void unmakeMove();

    // Fifty-move rule, repetition (one earlier occurrence) or insufficient material.
    bool isDraw() const;
    bool isRepetition() const;
    bool isInsufficientMaterial() const;

    static bool isValidSquare(int sq) { return sq >= 0 && sq < 64; }
    static int fileOf(int sq) { return sq & 7; }
    static int rankOf(int sq) { return sq >> 3; }
    static std::string moveToUci(const Move &m) { return moveToUciString(m); }
    static int algebraicToSquare(const std::string &alg) { return parseSquare(alg); }
    static bool isWhite(Piece p) { return isWhitePiece(p); }
    static bool isBlack(Piece p) { return isBlackPiece(p); }

    static Bitboard getRookAttacks(int sq, Bitboard occupied);
    static Bitboard getBishopAttacks(int sq, Bitboard occupied);
    static Bitboard getQueenAttacks(int sq, Bitboard occupied) {
        return getRookAttacks(sq, occupied) | getBishopAttacks(sq, occupied);
    }

    uint64_t perft(int depth);
    void divide(int depth);

private:
    struct NoHistory {};
    FastBoard(const FastBoard &other, NoHistory);

    Bitboard pieces[2][6] = {}; // [color][PAWN..KING]
    Bitboard occupied[2] = {};
    Bitboard all_occupied = 0;
    Piece mailbox[64] = {};

    Color side = Color::White;
    uint8_t castlingRights = 0; // KQkq bits
    int epSquare = -1;
    int halfmoveClock = 0;
    int fullmoveNumber = 1;
    uint64_t hash = 0;

    struct HistoryEntry {
        Move move;
        Piece moved;
        Piece captured;
        uint8_t castlingRights;
        int epSquare;
        int halfmoveClock;
        int fullmoveNumber;
        uint64_t hash;
    };
    std::vector<HistoryEntry> history;

    static void initTables();
    static Bitboard pawnAttacks[2][64];
    static Bitboard knightAttacks[64];
    static Bitboard kingAttacks[64];
    static Magic rookMagics[64];
    static Magic bishopMagics[64];
    static Bitboard rookTable[102400];
    static Bitboard bishopTable[5248];
    static void initMagics(Magic *magics, Bitboard *table, bool isRook);

    void putPiece(Piece p, int sq);
    void removePiece(int sq);
    void movePiece(int from, int to);

    void generatePawnMoves(std::vector<Move> &moves, Color c) const;
    void generatePieceMoves(std::vector<Move> &moves, Color c) const;
    void generateCastlingMoves(std::vector<Move> &moves, Color c) const;
    bool isSquareAttacked(int sq, Color byColor) const;
    int kingSquare(Color c) const;
    void sanitizeCastlingRights();
    static void addMove(std::vector<Move> &moves, int from, int to,
                        Piece promo = Piece::None, bool ep = false, bool castle = false) {
        moves.push_back(Move{from, to, promo, ep, castle});
    }

    static uint64_t zPiece[12][64];
    static uint64_t zSide;
    static uint64_t zCastle[16];
    static uint64_t zEnpassant[8];
    void hashRecompute();
};
