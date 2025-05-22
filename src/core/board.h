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
    Piece getPieceAt(const Square& square) const;
    Color getSideToMove() const;
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
    
    // For graph representation
    friend class GraphBuilder;
    
    int getHalfMoveClock() const { return halfMoveClock_; }
    
private:
    // Internal board representation
    std::array<Piece, 64> pieces_;
    
    // Game state
    Color sideToMove_;
    std::bitset<4> castlingRights_; // KQkq
    Square enPassantSquare_;
    int halfMoveClock_;
    int fullMoveNumber_;
    
    // Move generation helpers
    std::vector<Move> generatePseudoLegalMoves() const;
    bool isAttacked(const Square& square, Color attackerColor) const;
};

} // namespace nags 