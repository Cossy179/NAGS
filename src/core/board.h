#pragma once

#include <string>
#include <vector>
#include <array>
#include <bitset>
#include <cstdint>

#include "engine.h"

namespace nags {

// Forward declarations
class GraphBuilder;

enum class PieceType : uint8_t {
    NONE = 0,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING
};

enum class Color : uint8_t {
    WHITE = 0,
    BLACK
};

struct Piece {
    PieceType type;
    Color color;
    
    bool operator==(const Piece& other) const {
        return type == other.type && color == other.color;
    }
    
    bool operator!=(const Piece& other) const {
        return !(*this == other);
    }
    
    char toChar() const;
    static Piece fromChar(char c);
};

struct Square {
    int file;  // 0-7, a-h
    int rank;  // 0-7, 1-8
    
    // Convert to/from algebraic notation (e.g., "e4")
    std::string toAlgebraic() const;
    static Square fromAlgebraic(const std::string& algebraic);
    
    // Convert to 0-63 index (a1=0, h8=63)
    int toIndex() const;
    static Square fromIndex(int index);
    
    // Static utility methods
    static std::string toAlgebraic(int index);
    static int fromAlgebraic(const std::string& algebraic);
    
    bool operator==(const Square& other) const {
        return file == other.file && rank == other.rank;
    }
    
    bool operator!=(const Square& other) const {
        return !(*this == other);
    }
};

class Board {
public:
    Board();
    
    // Setup methods
    void resetToStartPosition();
    void setFromFEN(const std::string& fen);
    std::string toFEN() const;
    
    // Move generation and execution
    std::vector<Move> generateLegalMoves() const;
    bool makeMove(const Move& move);
    bool isLegalMove(const Move& move) const;
    
    // Position information
    Piece getPieceAt(int square) const { 
        if (square < 0 || square >= 64) return Piece{PieceType::NONE, Color::WHITE};
        return pieces_[square]; 
    }
    
    Piece getPieceAt(Square square) const { 
        int sq = square.toIndex();
        if (sq < 0 || sq >= 64) return Piece{PieceType::NONE, Color::WHITE};
        return pieces_[sq]; 
    }
    
    Color getSideToMove() const { return sideToMove_; }
    void setSideToMove(Color color) { sideToMove_ = color; }
    int getHalfMoveClock() const { return halfMoveClock_; }
    
    bool isCheck() const;
    bool isCheckmate() const;
    bool isStalemate() const;
    
    // Castling rights
    bool canCastleKingside(Color color) const;
    bool canCastleQueenside(Color color) const;
    
    // En passant
    Square getEnPassantSquare() const;
    
    // Utility
    std::string toString() const;
    
    // Hash
    uint64_t getPositionHash() const { return positionHash_; }
    
    // For graph representation
    friend class GraphBuilder;
    
private:
    // Internal board representation
    std::array<Piece, 64> pieces_;
    
    // Game state
    Color sideToMove_;
    std::bitset<4> castlingRights_; // KQkq
    int enPassantSquare_;  // 0-63, or 64 for none
    int halfMoveClock_;
    int fullMoveNumber_;
    uint64_t positionHash_;
    
    // Move generation helpers
    void generatePieceMoves(int square, std::vector<Move>& moves) const;
    void generatePawnMoves(int square, std::vector<Move>& moves) const;
    void generateKnightMoves(int square, std::vector<Move>& moves) const;
    void generateSlidingMoves(int square, std::vector<Move>& moves, bool diagonal, bool straight) const;
    void generateKingMoves(int square, std::vector<Move>& moves) const;
    bool isAttacking(int from, int to) const;
    bool isInCheck(Color color) const;
    bool isClearPath(int from, int to) const;
    char pieceToChar(const Piece& piece) const;
    void calculateHash();
};

} // namespace nags 