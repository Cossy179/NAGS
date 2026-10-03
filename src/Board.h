#pragma once

// Bitboard board with ray-based sliding attacks. Used by the `nags` hybrid
// engine and by `nags_basic`. FastBoard (magic bitboards) exposes the same
// interface so the search and evaluation templates work with either.

#include "ChessTypes.h"

#include <array>
#include <cstdint>
#include <string>
#include <vector>

class Board {
public:
    Board();

    // Loads a FEN. On failure the board is left unchanged and false is returned.
    bool setFromFEN(const std::string &fen);
    std::string getFEN() const;

    bool setStartPos();
    // Applies UCI moves in order. Stops at (and returns false for) the first
    // move that is not legal; earlier moves stay applied.
    bool applyMovesUCI(const std::vector<std::string> &uciMoves);

    // Legal moves for the side to move. With noisyOnly, only captures
    // (including en passant) and promotions are returned.
    std::vector<Move> generateLegalMoves(bool noisyOnly = false) const;

    bool inCheck() const { return inCheck(side); }
    bool inCheck(Color c) const;

    void makeMove(const Move &m);
    void unmakeMove();

    Color sideToMove() const { return side; }
    Piece pieceAt(int sq) const;
    Bitboard pieceBB(Color c, int type) const { return pieces[colorIndex(c) * 6 + type]; }
    Bitboard occupancy() const { return occAll; }
    uint64_t zobrist() const { return hash; }
    int getHalfmoveClock() const { return halfmoveClock; }
    int getFullmoveNumber() const { return fullmoveNumber; }

    // Draw by fifty-move rule, repetition (a single earlier occurrence counts,
    // as is usual inside a search) or insufficient material.
    bool isDraw() const;
    bool isRepetition() const;
    bool isInsufficientMaterial() const;

    uint64_t perft(int depth);

    static bool isValidSquare(int sq) { return sq >= 0 && sq < 64; }
    static int fileOf(int sq) { return sq & 7; }
    static int rankOf(int sq) { return sq >> 3; }
    static std::string moveToUci(const Move &m) { return moveToUciString(m); }
    static int algebraicToSquare(const std::string &alg) { return parseSquare(alg); }
    static bool isWhite(Piece p) { return isWhitePiece(p); }
    static bool isBlack(Piece p) { return isBlackPiece(p); }

private:
    struct NoHistory {};
    Board(const Board &other, NoHistory);

    // Piece bitboards: 0..5 white (P,N,B,R,Q,K), 6..11 black (P,N,B,R,Q,K)
    std::array<Bitboard, 12> pieces{};
    Bitboard occWhite = 0, occBlack = 0, occAll = 0;
    Color side = Color::White;
    // bit 0: white O-O, bit 1: white O-O-O, bit 2: black O-O, bit 3: black O-O-O
    uint8_t castlingRights = 0;
    int epSquare = -1; // en passant target square or -1
    int halfmoveClock = 0;
    int fullmoveNumber = 1;
    uint64_t hash = 0;

    struct HistoryEntry {
        Move move;
        Piece captured;
        uint8_t castlingRights;
        int epSquare;
        int halfmoveClock;
        int fullmoveNumber;
        Color side;
        std::array<Bitboard, 12> pieces;
        uint64_t prevHash;
    };
    std::vector<HistoryEntry> history;

    static void initTables();
    static Bitboard knightAttacks[64];
    static Bitboard kingAttacks[64];
    static Bitboard pawnAttacks[2][64]; // squares attacked BY a pawn of that colour on sq

    bool isSquareAttacked(int sq, Color byColor) const;
    int kingSquare(Color c) const;
    void updateOccupancy();
    void sanitizeCastlingRights();

    static uint64_t zPiece[12][64];
    static uint64_t zSide;
    static uint64_t zCastle[16];
    static uint64_t zEnpassant[8];
    void hashRecompute();
    void hashTogglePiece(Piece p, int sq) { hash ^= zPiece[static_cast<int>(p) - 1][sq]; }

    static int pieceIndex(Piece p) { return p == Piece::None ? -1 : static_cast<int>(p) - 1; }
    static Bitboard slidingAttackRook(int sq, Bitboard occ);
    static Bitboard slidingAttackBishop(int sq, Bitboard occ);

    void generatePseudoMoves(std::vector<Move> &moves) const;
    void generatePawnMoves(std::vector<Move> &moves, Color c) const;
    static void addMove(std::vector<Move> &moves, int from, int to, Piece promo = Piece::None, bool ep = false, bool castle = false) {
        moves.push_back(Move{from, to, promo, ep, castle});
    }
};
