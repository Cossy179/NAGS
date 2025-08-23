#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

// High-performance bitboard-based board representation with magic bitboards
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
    Piece promotion; // Piece::None if not a promotion
    bool isEnPassant;
    bool isCastling;
};

// Magic bitboard structure
struct Magic {
    Bitboard mask;
    Bitboard magic;
    Bitboard* attacks;
    int shift;
};

class FastBoard {
public:
    FastBoard();
    
    // Board setup
    bool setFromFEN(const std::string &fen);
    std::string getFEN() const;
    bool setStartPos();
    bool applyMovesUCI(const std::vector<std::string> &uciMoves);
    
    // Move generation
    std::vector<Move> generateLegalMoves() const;
    void generatePseudoMoves(std::vector<Move> &moves) const;
    
    // Board state
    bool inCheck() const { return inCheck(side); }
    bool inCheck(Color c) const;
    Color sideToMove() const { return side; }
    Piece pieceAt(int sq) const;
    uint64_t zobrist() const { return hash; }
    int getHalfmoveClock() const { return halfmoveClock; }
    
    // Make/unmake moves
    void makeMove(const Move &m);
    void unmakeMove();
    
    // Static utilities
    static bool isValidSquare(int sq) { return sq >= 0 && sq < 64; }
    static int fileOf(int sq) { return sq & 7; }
    static int rankOf(int sq) { return sq >> 3; }
    static std::string moveToUci(const Move &m);
    static int algebraicTo0x88(const std::string &alg);
    static bool isWhite(Piece p);
    static bool isBlack(Piece p);
    
    // Performance testing
    uint64_t perft(int depth);
    void divide(int depth);
    
private:
    // Bitboard representation - one per piece type per color
    Bitboard pieces[2][6]; // [color][piece_type] where piece_type: P=0,N=1,B=2,R=3,Q=4,K=5
    Bitboard occupied[2]; // [color]
    Bitboard all_occupied;
    
    Color side;
    uint8_t castlingRights; // KQkq bits
    int epSquare; // -1 if none
    int halfmoveClock;
    int fullmoveNumber;
    uint64_t hash;
    
    // Move history for unmake
    struct HistoryEntry {
        Move move;
        Piece captured;
        uint8_t castlingRights;
        int epSquare;
        int halfmoveClock;
        uint64_t hash;
    };
    std::vector<HistoryEntry> history;
    
    // Attack tables and magic bitboards
    static bool initialized;
    static void initAttackTables();
    
    // Precomputed attack tables
    static Bitboard pawnAttacks[2][64];
    static Bitboard knightAttacks[64];
    static Bitboard kingAttacks[64];
    
    // Magic bitboards for sliding pieces
    static Magic rookMagics[64];
    static Magic bishopMagics[64];
    static Bitboard rookAttacks[102400]; // Total size for all rook attack tables
    static Bitboard bishopAttacks[5248]; // Total size for all bishop attack tables
    
    // Bitboard manipulation
    static int popLSB(Bitboard &bb);
    static int countBits(Bitboard bb);
    static Bitboard shift(Bitboard bb, int delta);
    
    // Attack generation
    static Bitboard getRookAttacks(int sq, Bitboard occupied);
    static Bitboard getBishopAttacks(int sq, Bitboard occupied);
    static Bitboard getQueenAttacks(int sq, Bitboard occupied);
    
    // Move generation helpers
    void generatePawnMoves(std::vector<Move> &moves, Color c) const;
    void generateKnightMoves(std::vector<Move> &moves, Color c) const;
    void generateBishopMoves(std::vector<Move> &moves, Color c) const;
    void generateRookMoves(std::vector<Move> &moves, Color c) const;
    void generateQueenMoves(std::vector<Move> &moves, Color c) const;
    void generateKingMoves(std::vector<Move> &moves, Color c) const;
    void generateCastlingMoves(std::vector<Move> &moves, Color c) const;
    
    // Utilities
    bool isSquareAttacked(int sq, Color byColor) const;
    int kingSquare(Color c) const;
    void addMove(std::vector<Move> &moves, int from, int to, 
                 Piece promo = Piece::None, bool ep = false, bool castle = false) const;
    void updateOccupancy();
    
    // Zobrist hashing
    static void initZobrist();
    static uint64_t zPiece[2][6][64]; // [color][piece_type][square]
    static uint64_t zSide;
    static uint64_t zCastle[16];
    static uint64_t zEnpassant[8];
    void hashRecompute();
    void hashTogglePiece(Color c, int piece_type, int sq);
    void hashToggleSide();
    void hashToggleCastle(uint8_t rights);
    void hashSetEp(int epSqOld, int epSqNew);
    
    // Magic bitboard initialization
    static void initMagics();
    static Bitboard rookMask(int sq);
    static Bitboard bishopMask(int sq);
    static Bitboard rookAttack(int sq, Bitboard occupied);
    static Bitboard bishopAttack(int sq, Bitboard occupied);
    static uint64_t findMagic(int sq, int bits, bool isRook);
    static Bitboard randomBitboard();
    
    // Helper functions
    friend Bitboard indexToOccupancy(int index, Bitboard mask);
};

// Bitboard constants
constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_B = 0x0202020202020202ULL;
constexpr Bitboard FILE_C = 0x0404040404040404ULL;
constexpr Bitboard FILE_D = 0x0808080808080808ULL;
constexpr Bitboard FILE_E = 0x1010101010101010ULL;
constexpr Bitboard FILE_F = 0x2020202020202020ULL;
constexpr Bitboard FILE_G = 0x4040404040404040ULL;
constexpr Bitboard FILE_H = 0x8080808080808080ULL;

constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
constexpr Bitboard RANK_3 = 0x0000000000FF0000ULL;
constexpr Bitboard RANK_4 = 0x00000000FF000000ULL;
constexpr Bitboard RANK_5 = 0x000000FF00000000ULL;
constexpr Bitboard RANK_6 = 0x0000FF0000000000ULL;
constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;

// Useful bitboard operations
inline int lsb(Bitboard bb) {
#ifdef _MSC_VER
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    return static_cast<int>(idx);
#else
    return __builtin_ctzll(bb);
#endif
}

inline int popcount(Bitboard bb) {
#ifdef _MSC_VER
    return static_cast<int>(__popcnt64(bb));
#else
    return __builtin_popcountll(bb);
#endif
}

inline Bitboard northOne(Bitboard bb) { return bb << 8; }
inline Bitboard southOne(Bitboard bb) { return bb >> 8; }
inline Bitboard eastOne(Bitboard bb) { return (bb << 1) & ~FILE_A; }
inline Bitboard westOne(Bitboard bb) { return (bb >> 1) & ~FILE_H; }
inline Bitboard northEast(Bitboard bb) { return (bb << 9) & ~FILE_A; }
inline Bitboard northWest(Bitboard bb) { return (bb << 7) & ~FILE_H; }
inline Bitboard southEast(Bitboard bb) { return (bb >> 7) & ~FILE_A; }
inline Bitboard southWest(Bitboard bb) { return (bb >> 9) & ~FILE_H; }
