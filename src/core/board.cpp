#include "board.h"
#include <sstream>
#include <algorithm>
#include <cctype>

namespace nags {

// Square utility methods
std::string Square::toAlgebraic() const {
    return std::string(1, 'a' + file) + std::to_string(rank + 1);
}

Square Square::fromAlgebraic(const std::string& algebraic) {
    if (algebraic.length() < 2) return Square{-1, -1};
    return Square{algebraic[0] - 'a', algebraic[1] - '1'};
}

int Square::toIndex() const {
    return rank * 8 + file;
}

Square Square::fromIndex(int index) {
    return Square{index % 8, index / 8};
}

std::string Square::toAlgebraic(int index) {    if (index < 0 || index >= 64) return "??";    return fromIndex(index).toAlgebraic();}int Square::fromAlgebraic(const std::string& algebraic) {    Square sq = fromAlgebraic(algebraic);    return sq.toIndex();}

Board::Board() {
    resetToStartPosition();
}

Board::~Board() = default;

void Board::resetToStartPosition() {
    setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Board::setFromFEN(const std::string& fen) {
    // Initialize board state
    for (int i = 0; i < 64; ++i) {
        pieces_[i] = Piece{PieceType::NONE, Color::WHITE};
    }
    
    std::istringstream ss(fen);
    std::string board, side, castling, enPassant;
    int halfmove, fullmove;
    
    ss >> board >> side >> castling >> enPassant >> halfmove >> fullmove;
    
    // Parse board position
    int rank = 7, file = 0;
    for (char c : board) {
        if (c == '/') {
            rank--;
            file = 0;
        } else if (std::isdigit(c)) {
            file += c - '0';
        } else {
            Color color = std::isupper(c) ? Color::WHITE : Color::BLACK;
            PieceType type;
            switch (std::tolower(c)) {
                case 'p': type = PieceType::PAWN; break;
                case 'n': type = PieceType::KNIGHT; break;
                case 'b': type = PieceType::BISHOP; break;
                case 'r': type = PieceType::ROOK; break;
                case 'q': type = PieceType::QUEEN; break;
                case 'k': type = PieceType::KING; break;
                default: type = PieceType::NONE;
            }
            pieces_[rank * 8 + file] = Piece{type, color};
            file++;
        }
    }
    
    // Set side to move
    sideToMove_ = (side == "w") ? Color::WHITE : Color::BLACK;
    
    // Set castling rights
    castlingRights_ = 0;
    if (castling.find('K') != std::string::npos) castlingRights_ |= 1;
    if (castling.find('Q') != std::string::npos) castlingRights_ |= 2;
    if (castling.find('k') != std::string::npos) castlingRights_ |= 4;
    if (castling.find('q') != std::string::npos) castlingRights_ |= 8;
    
        // Set en passant square    if (enPassant != "-") {        enPassantSquare_ = Square::fromAlgebraic(enPassant);    } else {        enPassantSquare_ = 64; // Invalid square    }
    
    // Set move counters
    halfMoveClock_ = halfmove;
    fullMoveNumber_ = fullmove;
    
    // Calculate position hash
    calculateHash();
}

std::string Board::toFEN() const {
    std::ostringstream fen;
    
    // Board position
    for (int rank = 7; rank >= 0; --rank) {
        int emptyCount = 0;
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            if (pieces_[sq].type == PieceType::NONE) {
                emptyCount++;
            } else {
                if (emptyCount > 0) {
                    fen << emptyCount;
                    emptyCount = 0;
                }
                char piece = pieceToChar(pieces_[sq]);
                fen << piece;
            }
        }
        if (emptyCount > 0) {
            fen << emptyCount;
        }
        if (rank > 0) {
            fen << '/';
        }
    }
    
    // Side to move
    fen << ' ' << (sideToMove_ == Color::WHITE ? 'w' : 'b');
    
    // Castling rights
    fen << ' ';
    if (castlingRights_ == 0) {
        fen << '-';
    } else {
        if (castlingRights_ & 1) fen << 'K';
        if (castlingRights_ & 2) fen << 'Q';
        if (castlingRights_ & 4) fen << 'k';
        if (castlingRights_ & 8) fen << 'q';
    }
    
    // En passant
    fen << ' ';
    if (enPassantSquare_ < 64) {
        fen << Square::toAlgebraic(enPassantSquare_);
    } else {
        fen << '-';
    }
    
    // Move counters
    fen << ' ' << halfMoveClock_ << ' ' << fullMoveNumber_;
    
    return fen.str();
}

std::vector<Move> Board::generateLegalMoves() const {
    std::vector<Move> moves;
    moves.reserve(256);
    
    // Generate pseudo-legal moves for all pieces
    for (int sq = 0; sq < 64; ++sq) {
        if (pieces_[sq].type != PieceType::NONE && pieces_[sq].color == sideToMove_) {
            generatePieceMoves(sq, moves);
        }
    }
    
    // Filter out illegal moves (that leave king in check)
    std::vector<Move> legalMoves;
    for (const auto& move : moves) {
        Board temp = *this;
        if (temp.makeMove(move)) {
            legalMoves.push_back(move);
        }
    }
    
    return legalMoves;
}

bool Board::makeMove(const Move& move) {
    // Extract move components
    int from = Square::fromAlgebraic(move.from);
    int to = Square::fromAlgebraic(move.to);
    
    // Save state for move generation
    Piece movingPiece = pieces_[from];
    Piece capturedPiece = pieces_[to];
    
    // Make the move
    pieces_[to] = movingPiece;
    pieces_[from] = Piece{PieceType::NONE, Color::WHITE};
    
    // Handle special moves
    if (movingPiece.type == PieceType::PAWN) {
        // En passant capture
        if (to == enPassantSquare_) {
            int captureSquare = (sideToMove_ == Color::WHITE) ? to - 8 : to + 8;
            pieces_[captureSquare] = Piece{PieceType::NONE, Color::WHITE};
        }
        
        // Promotion
        if (move.promotion.has_value()) {
            pieces_[to].type = move.promotion.value();
        }
    } else if (movingPiece.type == PieceType::KING) {
        // Castling
        int delta = to - from;
        if (std::abs(delta) == 2) {
            // Move rook
            int rookFrom = (delta > 0) ? to + 1 : to - 2;
            int rookTo = (delta > 0) ? to - 1 : to + 1;
            pieces_[rookTo] = pieces_[rookFrom];
            pieces_[rookFrom] = Piece{PieceType::NONE, Color::WHITE};
        }
        
        // Update castling rights
        if (sideToMove_ == Color::WHITE) {
            castlingRights_ &= ~3; // Remove white castling rights
        } else {
            castlingRights_ &= ~12; // Remove black castling rights
        }
    }
    
    // Update castling rights if rook moves
    if (movingPiece.type == PieceType::ROOK) {
        if (from == 0) castlingRights_ &= ~2;
        else if (from == 7) castlingRights_ &= ~1;
        else if (from == 56) castlingRights_ &= ~8;
        else if (from == 63) castlingRights_ &= ~4;
    }
    
    // Update en passant square
    enPassantSquare_ = 64;
    if (movingPiece.type == PieceType::PAWN && std::abs(to - from) == 16) {
        enPassantSquare_ = (from + to) / 2;
    }
    
    // Update move counters
    if (movingPiece.type == PieceType::PAWN || capturedPiece.type != PieceType::NONE) {
        halfMoveClock_ = 0;
    } else {
        halfMoveClock_++;
    }
    
    if (sideToMove_ == Color::BLACK) {
        fullMoveNumber_++;
    }
    
    // Switch side to move
    sideToMove_ = (sideToMove_ == Color::WHITE) ? Color::BLACK : Color::WHITE;
    
    // Check if move is legal (doesn't leave king in check)
    if (isInCheck(movingPiece.color)) {
        // Undo the move
        *this = Board(); // Reset to avoid complex undo logic
        return false;
    }
    
    // Update position hash
    calculateHash();
    
    return true;
}

bool Board::isCheck() const {
    return isInCheck(sideToMove_);
}

bool Board::isInCheck(Color color) const {
    // Find king position
    int kingSquare = -1;
    for (int sq = 0; sq < 64; ++sq) {
        if (pieces_[sq].type == PieceType::KING && pieces_[sq].color == color) {
            kingSquare = sq;
            break;
        }
    }
    
    if (kingSquare == -1) return false; // No king found
    
    // Check if any opponent piece attacks the king
    Color opponent = (color == Color::WHITE) ? Color::BLACK : Color::WHITE;
    for (int sq = 0; sq < 64; ++sq) {
        if (pieces_[sq].type != PieceType::NONE && pieces_[sq].color == opponent) {
            if (isAttacking(sq, kingSquare)) {
                return true;
            }
        }
    }
    
    return false;
}

std::string Board::toString() const {
    std::ostringstream ss;
    ss << "  a b c d e f g h\n";
    for (int rank = 7; rank >= 0; --rank) {
        ss << (rank + 1) << ' ';
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            if (pieces_[sq].type == PieceType::NONE) {
                ss << ". ";
            } else {
                ss << pieceToChar(pieces_[sq]) << ' ';
            }
        }
        ss << (rank + 1) << '\n';
    }
    ss << "  a b c d e f g h\n";
    ss << "Side to move: " << (sideToMove_ == Color::WHITE ? "White" : "Black") << '\n';
    ss << "Castling: ";
    if (castlingRights_ == 0) ss << "-";
    else {
        if (castlingRights_ & 1) ss << "K";
        if (castlingRights_ & 2) ss << "Q";
        if (castlingRights_ & 4) ss << "k";
        if (castlingRights_ & 8) ss << "q";
    }
    ss << '\n';
    if (enPassantSquare_ < 64) {
        ss << "En passant: " << Square::toAlgebraic(enPassantSquare_) << '\n';
    }
    ss << "Half moves: " << halfMoveClock_ << '\n';
    ss << "Full moves: " << fullMoveNumber_ << '\n';
    ss << "Position hash: " << std::hex << positionHash_ << std::dec << '\n';
    
    return ss.str();
}

void Board::generatePieceMoves(int square, std::vector<Move>& moves) const {
    // Simplified move generation - would be optimized with bitboards in production
    const Piece& piece = pieces_[square];
    
    switch (piece.type) {
        case PieceType::PAWN:
            generatePawnMoves(square, moves);
            break;
        case PieceType::KNIGHT:
            generateKnightMoves(square, moves);
            break;
        case PieceType::BISHOP:
            generateSlidingMoves(square, moves, true, false);
            break;
        case PieceType::ROOK:
            generateSlidingMoves(square, moves, false, true);
            break;
        case PieceType::QUEEN:
            generateSlidingMoves(square, moves, true, true);
            break;
        case PieceType::KING:
            generateKingMoves(square, moves);
            break;
        default:
            break;
    }
}

char Board::pieceToChar(const Piece& piece) const {
    char c;
    switch (piece.type) {
        case PieceType::PAWN: c = 'p'; break;
        case PieceType::KNIGHT: c = 'n'; break;
        case PieceType::BISHOP: c = 'b'; break;
        case PieceType::ROOK: c = 'r'; break;
        case PieceType::QUEEN: c = 'q'; break;
        case PieceType::KING: c = 'k'; break;
        default: return '.';
    }
    return piece.color == Color::WHITE ? std::toupper(c) : c;
}

void Board::calculateHash() {
    // Simple hash calculation - would use Zobrist hashing in production
    positionHash_ = 0;
    for (int sq = 0; sq < 64; ++sq) {
        if (pieces_[sq].type != PieceType::NONE) {
            positionHash_ ^= (uint64_t(pieces_[sq].type) << (sq * 4));
            positionHash_ ^= (uint64_t(pieces_[sq].color) << (sq * 4 + 3));
        }
    }
    positionHash_ ^= (uint64_t(sideToMove_) << 60);
    positionHash_ ^= (uint64_t(castlingRights_) << 56);
    positionHash_ ^= (uint64_t(enPassantSquare_) << 48);
}

// Move generation helpers (simplified implementations)
void Board::generatePawnMoves(int square, std::vector<Move>& moves) const {
    // Simplified pawn move generation
    int rank = square / 8;
    int file = square % 8;
    int direction = (pieces_[square].color == Color::WHITE) ? 8 : -8;
    int startRank = (pieces_[square].color == Color::WHITE) ? 1 : 6;
    int promotionRank = (pieces_[square].color == Color::WHITE) ? 6 : 1;
    
    // Forward moves
    int to = square + direction;
    if (to >= 0 && to < 64 && pieces_[to].type == PieceType::NONE) {
        if (rank == promotionRank) {
            // Promotion
            moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to), PieceType::QUEEN});
            moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to), PieceType::ROOK});
            moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to), PieceType::BISHOP});
            moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to), PieceType::KNIGHT});
        } else {
            moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to)});
        }
        
        // Double move from start
        if (rank == startRank) {
            to = square + 2 * direction;
            if (pieces_[to].type == PieceType::NONE) {
                moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to)});
            }
        }
    }
    
    // Captures
    for (int df : {-1, 1}) {
        if (file + df >= 0 && file + df < 8) {
            to = square + direction + df;
            if (to >= 0 && to < 64 && 
                (pieces_[to].type != PieceType::NONE && pieces_[to].color != pieces_[square].color) ||
                to == enPassantSquare_) {
                if (rank == promotionRank) {
                    moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to), PieceType::QUEEN});
                    moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to), PieceType::ROOK});
                    moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to), PieceType::BISHOP});
                    moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to), PieceType::KNIGHT});
                } else {
                    moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to)});
                }
            }
        }
    }
}

void Board::generateKnightMoves(int square, std::vector<Move>& moves) const {
    static const int deltas[] = {-17, -15, -10, -6, 6, 10, 15, 17};
    int rank = square / 8;
    int file = square % 8;
    
    for (int delta : deltas) {
        int to = square + delta;
        int toRank = to / 8;
        int toFile = to % 8;
        
        if (to >= 0 && to < 64 && std::abs(toRank - rank) <= 2 && std::abs(toFile - file) <= 2 &&
            (pieces_[to].type == PieceType::NONE || pieces_[to].color != pieces_[square].color)) {
            moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to)});
        }
    }
}

void Board::generateSlidingMoves(int square, std::vector<Move>& moves, bool diagonal, bool straight) const {
    static const int directions[] = {-9, -7, 7, 9, -8, -1, 1, 8};
    int startIdx = diagonal ? 0 : 4;
    int endIdx = straight ? 8 : 4;
    
    for (int i = startIdx; i < endIdx; ++i) {
        int delta = directions[i];
        int to = square;
        
        while (true) {
            to += delta;
            
            // Check bounds
            if (to < 0 || to >= 64) break;
            
            // Check wraparound
            int fromFile = (to - delta) % 8;
            int toFile = to % 8;
            if (std::abs(delta) == 1 && std::abs(toFile - fromFile) > 1) break;
            
            if (pieces_[to].type == PieceType::NONE) {
                moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to)});
            } else {
                if (pieces_[to].color != pieces_[square].color) {
                    moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to)});
                }
                break;
            }
        }
    }
}

void Board::generateKingMoves(int square, std::vector<Move>& moves) const {
    static const int deltas[] = {-9, -8, -7, -1, 1, 7, 8, 9};
    
    for (int delta : deltas) {
        int to = square + delta;
        int fromFile = square % 8;
        int toFile = to % 8;
        
        if (to >= 0 && to < 64 && std::abs(toFile - fromFile) <= 1 &&
            (pieces_[to].type == PieceType::NONE || pieces_[to].color != pieces_[square].color)) {
            moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(to)});
        }
    }
    
    // Castling
    if (!isInCheck(pieces_[square].color)) {
        // King-side castling
        if ((pieces_[square].color == Color::WHITE && (castlingRights_ & 1)) ||
            (pieces_[square].color == Color::BLACK && (castlingRights_ & 4))) {
            if (pieces_[square + 1].type == PieceType::NONE && 
                pieces_[square + 2].type == PieceType::NONE) {
                moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(square + 2)});
            }
        }
        
        // Queen-side castling
        if ((pieces_[square].color == Color::WHITE && (castlingRights_ & 2)) ||
            (pieces_[square].color == Color::BLACK && (castlingRights_ & 8))) {
            if (pieces_[square - 1].type == PieceType::NONE && 
                pieces_[square - 2].type == PieceType::NONE &&
                pieces_[square - 3].type == PieceType::NONE) {
                moves.push_back({Square::toAlgebraic(square), Square::toAlgebraic(square - 2)});
            }
        }
    }
}

bool Board::isAttacking(int from, int to) const {
    // Simplified attack detection
    const Piece& piece = pieces_[from];
    int delta = to - from;
    int rankDiff = std::abs((to / 8) - (from / 8));
    int fileDiff = std::abs((to % 8) - (from % 8));
    
    switch (piece.type) {
        case PieceType::PAWN:
            {
                int direction = (piece.color == Color::WHITE) ? 8 : -8;
                return (delta == direction - 1 || delta == direction + 1) && fileDiff == 1;
            }
        case PieceType::KNIGHT:
            return (rankDiff == 2 && fileDiff == 1) || (rankDiff == 1 && fileDiff == 2);
        case PieceType::BISHOP:
            return rankDiff == fileDiff && isClearPath(from, to);
        case PieceType::ROOK:
            return (rankDiff == 0 || fileDiff == 0) && isClearPath(from, to);
        case PieceType::QUEEN:
            return (rankDiff == fileDiff || rankDiff == 0 || fileDiff == 0) && isClearPath(from, to);
        case PieceType::KING:
            return rankDiff <= 1 && fileDiff <= 1;
        default:
            return false;
    }
}

bool Board::isClearPath(int from, int to) const {
    int delta = 0;
    if (from / 8 == to / 8) delta = (to > from) ? 1 : -1;
    else if (from % 8 == to % 8) delta = (to > from) ? 8 : -8;
    else if ((to - from) % 9 == 0) delta = (to > from) ? 9 : -9;
    else if ((to - from) % 7 == 0) delta = (to > from) ? 7 : -7;
    else return false;
    
    int current = from + delta;
    while (current != to) {
        if (pieces_[current].type != PieceType::NONE) return false;
        current += delta;
    }
    
    return true;
}

} // namespace nags 