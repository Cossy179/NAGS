#pragma once

#include <cstdint>
#include <array>
#include <immintrin.h>  // SIMD intrinsics

namespace nags {

/**
 * Bitboard implementation for ultra-fast chess move generation and board representation
 * Optimized with SIMD operations where available
 */
class BitBoard {
public:
    // Piece type definitions
    enum PieceType {
        EMPTY = 0,
        PAWN = 1,
        KNIGHT = 2,
        BISHOP = 3,
        ROOK = 4,
        QUEEN = 5,
        KING = 6
    };
    
    // Color definitions
    enum Color {
        WHITE = 0,
        BLACK = 1
    };
    
    // Square indices (0-63)
    enum Square {
        A1=0, B1, C1, D1, E1, F1, G1, H1,
        A2, B2, C2, D2, E2, F2, G2, H2,
        A3, B3, C3, D3, E3, F3, G3, H3,
        A4, B4, C4, D4, E4, F4, G4, H4,
        A5, B5, C5, D5, E5, F5, G5, H5,
        A6, B6, C6, D6, E6, F6, G6, H6,
        A7, B7, C7, D7, E7, F7, G7, H7,
        A8, B8, C8, D8, E8, F8, G8, H8,
        NO_SQUARE = 64
    };
    
    // Direction offsets
    enum Direction {
        NORTH = 8,
        EAST = 1,
        SOUTH = -8,
        WEST = -1,
        NORTH_EAST = 9,
        SOUTH_EAST = -7,
        SOUTH_WEST = -9,
        NORTH_WEST = 7
    };
    
    // Default constructor - initialize to starting position
    BitBoard();
    
    // Constructor with FEN string
    BitBoard(const std::string& fen);
    
    // Copy constructor
    BitBoard(const BitBoard& other) = default;
    
    // Move constructor
    BitBoard(BitBoard&& other) noexcept = default;
    
    // Assignment operator
    BitBoard& operator=(const BitBoard& other) = default;
    
    // Move assignment operator
    BitBoard& operator=(BitBoard&& other) noexcept = default;
    
    // Destructor
    ~BitBoard() = default;
    
    // Get piece at square
    PieceType getPiece(Square square) const;
    
    // Get piece color at square
    Color getPieceColor(Square square) const;
    
    // Make a move
    bool makeMove(uint16_t move);
    
    // Undo a move
    void undoMove();
    
    // Make a null move (pass the turn)
    void makeNullMove();
    
    // Undo a null move
    void undoNullMove();
    
    // Check if square is attacked by a given color
    bool isSquareAttacked(Square square, Color attacker) const;
    
    // Check if king is in check
    bool inCheck(Color side) const;
    
    // Generate all legal moves
    std::vector<uint16_t> generateLegalMoves();
    
    // Generate only capture moves
    std::vector<uint16_t> generateCaptures();
    
    // Check if move is legal
    bool isMoveLegal(uint16_t move) const;
    
    // Get the side to move
    Color getSideToMove() const { return side_to_move_; }
    
    // Get the position hash
    uint64_t getPositionHash() const { return hash_; }
    
    // Get the halfmove clock
    int getHalfmoveClock() const { return halfmove_clock_; }
    
    // Get the fullmove number
    int getFullmoveNumber() const { return fullmove_number_; }
    
    // Check if we're in endgame
    bool isEndgame() const;
    
    // Get FEN string of current position
    std::string getFEN() const;
    
    // Set position from FEN string
    void setFromFEN(const std::string& fen);
    
    // Get the piece count
    int getPieceCount() const;
    
    // Get piece value based on piece type and position
    int getPieceValue(Square square) const;
    
    // Reset to starting position
    void resetToStartPos();
    
    // Check if the position is a draw by repetition
    bool isDrawByRepetition() const;
    
    // Check if the position is a draw by 50-move rule
    bool isDrawByFiftyMoveRule() const;
    
    // Check if the position is a draw by insufficient material
    bool isDrawByInsufficientMaterial() const;

private:
    // Piece bitboards (indexed by [color][piece_type])
    std::array<std::array<uint64_t, 7>, 2> pieces_{};
    
    // Combined bitboards for all pieces of each color
    std::array<uint64_t, 2> color_bb_{};
    
    // Combined bitboard for all pieces
    uint64_t occupied_bb_{};
    
    // Side to move
    Color side_to_move_{WHITE};
    
    // Castling rights
    bool castle_rights_[2][2]{}; // [color][kingside/queenside]
    
    // En passant square
    Square ep_square_{NO_SQUARE};
    
    // Halfmove clock (for 50-move rule)
    int halfmove_clock_{0};
    
    // Fullmove number
    int fullmove_number_{1};
    
    // Position hash (Zobrist hashing)
    uint64_t hash_{0};
    
    // Move history for undo
    struct UndoInfo {
        uint16_t move;
        bool castle_rights[2][2];
        Square ep_square;
        int halfmove_clock;
        uint64_t hash;
        PieceType captured_piece;
    };
    std::vector<UndoInfo> history_;
    
    // Repetition history (position hashes)
    std::vector<uint64_t> repetition_history_;
    
    // Helper methods
    
    // Add a piece to the board
    void addPiece(Color color, PieceType piece, Square square);
    
    // Remove a piece from the board
    void removePiece(Color color, PieceType piece, Square square);
    
    // Move a piece on the board
    void movePiece(Color color, PieceType piece, Square from, Square to);
    
    // Generate Zobrist hash
    uint64_t generateZobristHash() const;
    
    // Update Zobrist hash when adding/removing a piece
    void updateHashPiece(Color color, PieceType piece, Square square);
    
    // Update Zobrist hash when changing side to move
    void updateHashSideToMove();
    
    // Update Zobrist hash when changing castling rights
    void updateHashCastling(Color color, bool kingside, bool old_right, bool new_right);
    
    // Update Zobrist hash when changing en passant square
    void updateHashEnPassant(Square old_square, Square new_square);
    
    // Precomputed attack tables
    static const std::array<uint64_t, 64> knight_attacks_;
    static const std::array<uint64_t, 64> king_attacks_;
    static const std::array<std::array<uint64_t, 64>, 2> pawn_attacks_;
    
    // Precomputed line and diagonal attack masks
    static const std::array<uint64_t, 64> rank_masks_;
    static const std::array<uint64_t, 64> file_masks_;
    static const std::array<uint64_t, 64> diagonal_masks_;
    static const std::array<uint64_t, 64> anti_diagonal_masks_;
    
    // Magic bitboard tables for sliding piece move generation
    static const std::array<uint64_t, 64> rook_magics_;
    static const std::array<uint64_t, 64> bishop_magics_;
    static const std::array<int, 64> rook_shift_;
    static const std::array<int, 64> bishop_shift_;
    static const std::array<uint64_t, 64> rook_masks_;
    static const std::array<uint64_t, 64> bishop_masks_;
    static const std::array<std::array<uint64_t, 4096>, 64> rook_attacks_;
    static const std::array<std::array<uint64_t, 512>, 64> bishop_attacks_;
    
    // Magic bitboard implementation
    uint64_t getRookAttacks(Square square, uint64_t occupied) const;
    uint64_t getBishopAttacks(Square square, uint64_t occupied) const;
    uint64_t getQueenAttacks(Square square, uint64_t occupied) const;
    
    // Precomputed Zobrist hash keys
    static const std::array<std::array<std::array<uint64_t, 64>, 7>, 2> zobrist_piece_keys_;
    static const std::array<uint64_t, 16> zobrist_castling_keys_;
    static const std::array<uint64_t, 65> zobrist_ep_keys_;
    static const uint64_t zobrist_side_key_;
    
    // SIMD-optimized methods
    
    // Count bits in a bitboard (population count)
    static int popCount(uint64_t bb) {
        #ifdef __POPCNT__
            return _mm_popcnt_u64(bb);
        #else
            // Fallback implementation
            int count = 0;
            while (bb) {
                count++;
                bb &= bb - 1; // Clear the least significant bit
            }
            return count;
        #endif
    }
    
    // Get index of least significant bit
    static Square getLSB(uint64_t bb) {
        #ifdef __BMI__
            return static_cast<Square>(_tzcnt_u64(bb));
        #else
            // Fallback implementation
            if (bb == 0) return NO_SQUARE;
            return static_cast<Square>(__builtin_ctzll(bb));
        #endif
    }
    
    // Get index of most significant bit
    static Square getMSB(uint64_t bb) {
        #ifdef __BMI__
            return static_cast<Square>(63 - _lzcnt_u64(bb));
        #else
            // Fallback implementation
            if (bb == 0) return NO_SQUARE;
            return static_cast<Square>(63 - __builtin_clzll(bb));
        #endif
    }
    
    // PEXT-based magic-less move generation
    #ifdef __BMI2__
    uint64_t getRookAttacksBMI2(Square square, uint64_t occupied) const {
        uint64_t mask = rook_masks_[square];
        uint64_t index = _pext_u64(occupied, mask);
        return rook_attacks_[square][index];
    }
    
    uint64_t getBishopAttacksBMI2(Square square, uint64_t occupied) const {
        uint64_t mask = bishop_masks_[square];
        uint64_t index = _pext_u64(occupied, mask);
        return bishop_attacks_[square][index];
    }
    #endif
};

// Move encoding (16 bits)
// bits 0-5:   from square (0-63)
// bits 6-11:  to square (0-63)
// bits 12-14: promotion piece (0-7, 0=none, 1=pawn, 2=knight, ...)
// bit 15:     special flag (castling, en passant)
inline uint16_t makeMove(BitBoard::Square from, BitBoard::Square to, 
                        BitBoard::PieceType promotion = BitBoard::EMPTY, bool special = false) {
    return static_cast<uint16_t>(
        from | (to << 6) | (promotion << 12) | (special ? (1 << 15) : 0)
    );
}

inline BitBoard::Square getMoveFrom(uint16_t move) {
    return static_cast<BitBoard::Square>(move & 0x3F);
}

inline BitBoard::Square getMoveTo(uint16_t move) {
    return static_cast<BitBoard::Square>((move >> 6) & 0x3F);
}

inline BitBoard::PieceType getMovePromotion(uint16_t move) {
    return static_cast<BitBoard::PieceType>((move >> 12) & 0x7);
}

inline bool getMoveSpecial(uint16_t move) {
    return (move & (1 << 15)) != 0;
}

} // namespace nags 