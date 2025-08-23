#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

// Bitboard-based board representation
using Bitboard = uint64_t;

enum class Piece : uint8_t {
    None = 0,
    WP, WN, WB, WR, WQ, WK,
    BP, BN, BB, BR, BQ, BK
};

enum class Color : uint8_t { White = 0, Black = 1 };

struct Move {
    int from; // 0..63
    int to;   // 0..63
    Piece promotion; // Piece::None if not a promotion; uses side-colored piece kind
    bool isEnPassant;
    bool isCastling;
};

class Board {
public:
    Board();

    // Load from FEN; returns false if invalid
    bool setFromFEN(const std::string &fen);
    std::string getFEN() const;

    // Apply UCI "position" command inputs
    bool setStartPos();
    bool applyMovesUCI(const std::vector<std::string> &uciMoves);

    // Generate legal moves for current side to move
    std::vector<Move> generateLegalMoves() const;

    // Check utilities
    bool inCheck() const { return inCheck(side); }
    bool inCheck(Color c) const;

    // Make and unmake move (for search or validation)
    void makeMove(const Move &m);
    void unmakeMove();

    // Helpers
    Color sideToMove() const { return side; }
    Piece pieceAt(int sq) const;
    uint64_t zobrist() const { return hash; }

    static bool isValidSquare(int sq) { return sq >= 0 && sq < 64; }
    static int fileOf(int sq) { return sq & 7; }
    static int rankOf(int sq) { return sq >> 3; }

    static std::string moveToUci(const Move &m);
    static int algebraicTo0x88(const std::string &alg);
    static bool isWhite(Piece p);
    static bool isBlack(Piece p);

private:
    // Piece bitboards: 0..5 white (P,N,B,R,Q,K), 6..11 black (P,N,B,R,Q,K)
    std::array<Bitboard, 12> pieces{};
    Bitboard occWhite = 0, occBlack = 0, occAll = 0;
    Color side;
    // Castling rights: bit 0: white K, bit1: white Q, bit2: black K, bit3: black Q
    uint8_t castlingRights;
    // en passant target square (0..63) or -1
    int epSquare;
    int halfmoveClock;
    int fullmoveNumber;
    uint64_t hash = 0; // Zobrist hash

    struct HistoryEntry {
        Move move;
        Piece captured;
        uint8_t castlingRights;
        int epSquare;
        int halfmoveClock;
        Color side;
        Bitboard occWhite, occBlack, occAll;
        std::array<Bitboard, 12> pieces;
        uint64_t prevHash;
    };

    std::vector<HistoryEntry> history;

    static void initAttackTables();
    static bool attackTablesInit;
    static Bitboard knightAttacks[64];
    static Bitboard kingAttacks[64];
    static Bitboard pawnAttacks[2][64];

    static Piece makePiece(Color c, Piece pKind);

    bool isSquareAttacked(int sq, Color byColor) const;
    int kingSquare(Color c) const;

    void setEmpty();
    void updateOccupancy();

    // Zobrist
    static void initZobrist();
    static bool zobristInit;
    static uint64_t zPiece[12][64];
    static uint64_t zSide; // side to move
    static uint64_t zCastle[16];
    static uint64_t zEnpassant[8];
    void hashRecompute();
    inline void hashTogglePiece(Piece p, int sq) { int idx = pieceIndex(p); if (idx >= 0) hash ^= zPiece[idx][sq]; }
    inline void hashToggleCastle(uint8_t rights) { hash ^= zCastle[rights & 0xF]; }
    inline void hashToggleSide() { hash ^= zSide; }
    inline void hashSetEp(int epSqOld, int epSqNew) {
        if (epSqOld != -1) hash ^= zEnpassant[epSqOld & 7];
        if (epSqNew != -1) hash ^= zEnpassant[epSqNew & 7];
    }

    static int pieceIndex(Piece p);
    static int popLSB(Bitboard &bb);
    static int bitScanForward(Bitboard bb);
    static int countBits(Bitboard bb);
    static Bitboard slidingAttackRook(int sq, Bitboard occ);
    static Bitboard slidingAttackBishop(int sq, Bitboard occ);

    void generatePseudoMoves(std::vector<Move> &moves) const;
    void generatePawnMoves(std::vector<Move> &moves, Color c) const;
    void generateKnightMoves(std::vector<Move> &moves, Color c) const;
    void generateSlidingMoves(std::vector<Move> &moves, Color c) const;
    void generateKingMoves(std::vector<Move> &moves, Color c) const;
    void addMove(std::vector<Move> &moves, int from, int to, Piece promo = Piece::None, bool ep = false, bool castle = false) const;
};


