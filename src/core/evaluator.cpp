#include "evaluator.h"
#include "board.h"

namespace nags {

Evaluator::Evaluator() 
    : is_initialized_(false) {
}

Evaluator::~Evaluator() = default;

bool Evaluator::initialize(const std::string& weightsPath) {
    // Initialize evaluator with neural network weights
    weights_path_ = weightsPath;
    is_initialized_ = true;
    return true;
}

EvalResult Evaluator::evaluate(const Board* board) {
    EvalResult result;
    
    // Sophisticated evaluation combining multiple factors
    // This is a placeholder - in production would use NNUE or similar
    
    // Material evaluation
    int materialScore = evaluateMaterial(board);
    
    // Positional evaluation
    int positionalScore = evaluatePosition(board);
    
    // King safety evaluation
    int kingSafetyScore = evaluateKingSafety(board);
    
    // Pawn structure evaluation
    int pawnStructureScore = evaluatePawnStructure(board);
    
    // Mobility evaluation
    int mobilityScore = evaluateMobility(board);
    
    // Combine scores with sophisticated weighting
    result.score = materialScore + positionalScore + kingSafetyScore + 
                   pawnStructureScore + mobilityScore;
    
    // Adjust for side to move
    if (board->getSideToMove() == Color::BLACK) {
        result.score = -result.score;
    }
    
    // Add uncertainty based on position complexity
    result.uncertainty = calculateUncertainty(board);
    
    return result;
}

int Evaluator::evaluateMaterial(const Board* board) const {
    static const int pieceValues[] = {
        100,   // Pawn
        320,   // Knight
        330,   // Bishop
        500,   // Rook
        900,   // Queen
        20000  // King (should never be captured)
    };
    
    int score = 0;
    for (int sq = 0; sq < 64; ++sq) {
        Piece piece = board->getPieceAt(sq);
        if (piece.type != PieceType::NONE) {
            int value = pieceValues[static_cast<int>(piece.type) - 1];
            if (piece.color == Color::WHITE) {
                score += value;
            } else {
                score -= value;
            }
        }
    }
    
    return score;
}

int Evaluator::evaluatePosition(const Board* board) const {
    // Piece-square tables (simplified)
    static const int pawnTable[64] = {
         0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
         5,  5, 10, 25, 25, 10,  5,  5,
         0,  0,  0, 20, 20,  0,  0,  0,
         5, -5,-10,  0,  0,-10, -5,  5,
         5, 10, 10,-20,-20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0
    };
    
    static const int knightTable[64] = {
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50
    };
    
    int score = 0;
    for (int sq = 0; sq < 64; ++sq) {
        Piece piece = board->getPieceAt(sq);
        if (piece.type != PieceType::NONE) {
            int tableScore = 0;
            int rank = sq / 8;
            int file = sq % 8;
            int index = (piece.color == Color::WHITE) ? sq : (7 - rank) * 8 + file;
            
            switch (piece.type) {
                case PieceType::PAWN:
                    tableScore = pawnTable[index];
                    break;
                case PieceType::KNIGHT:
                    tableScore = knightTable[index];
                    break;
                // Add more piece-square tables
                default:
                    break;
            }
            
            if (piece.color == Color::WHITE) {
                score += tableScore;
            } else {
                score -= tableScore;
            }
        }
    }
    
    return score;
}

int Evaluator::evaluateKingSafety(const Board* board) const {
    // Simplified king safety evaluation
    int score = 0;
    
    // Find kings
    for (int sq = 0; sq < 64; ++sq) {
        Piece piece = board->getPieceAt(sq);
        if (piece.type == PieceType::KING) {
            int safety = 0;
            
            // Check pawn shield
            int rank = sq / 8;
            int file = sq % 8;
            int direction = (piece.color == Color::WHITE) ? 8 : -8;
            
            for (int f = std::max(0, file - 1); f <= std::min(7, file + 1); ++f) {
                int shieldSq = sq + direction + (f - file);
                if (shieldSq >= 0 && shieldSq < 64) {
                    Piece shieldPiece = board->getPieceAt(shieldSq);
                    if (shieldPiece.type == PieceType::PAWN && shieldPiece.color == piece.color) {
                        safety += 10;
                    }
                }
            }
            
            if (piece.color == Color::WHITE) {
                score += safety;
            } else {
                score -= safety;
            }
        }
    }
    
    return score;
}

int Evaluator::evaluatePawnStructure(const Board* board) const {
    int score = 0;
    
    // Check for doubled, isolated, and passed pawns
    bool whitePawns[8] = {false};
    bool blackPawns[8] = {false};
    
    for (int sq = 0; sq < 64; ++sq) {
        Piece piece = board->getPieceAt(sq);
        if (piece.type == PieceType::PAWN) {
            int file = sq % 8;
            if (piece.color == Color::WHITE) {
                if (whitePawns[file]) {
                    score -= 10; // Doubled pawn penalty
                }
                whitePawns[file] = true;
            } else {
                if (blackPawns[file]) {
                    score += 10; // Doubled pawn penalty for black
                }
                blackPawns[file] = true;
            }
        }
    }
    
    // Check for isolated pawns
    for (int f = 0; f < 8; ++f) {
        bool leftNeighbor = (f > 0) && whitePawns[f - 1];
        bool rightNeighbor = (f < 7) && whitePawns[f + 1];
        
        if (whitePawns[f] && !leftNeighbor && !rightNeighbor) {
            score -= 20; // Isolated pawn penalty
        }
        
        leftNeighbor = (f > 0) && blackPawns[f - 1];
        rightNeighbor = (f < 7) && blackPawns[f + 1];
        
        if (blackPawns[f] && !leftNeighbor && !rightNeighbor) {
            score += 20; // Isolated pawn penalty for black
        }
    }
    
    return score;
}

int Evaluator::evaluateMobility(const Board* board) const {
    // Count legal moves as a proxy for mobility
    std::vector<Move> moves = board->generateLegalMoves();
    int mobility = moves.size();
    
    // Temporarily switch sides to count opponent mobility
    Board tempBoard = *board;
    tempBoard.setSideToMove((board->getSideToMove() == Color::WHITE) ? Color::BLACK : Color::WHITE);
    std::vector<Move> opponentMoves = tempBoard.generateLegalMoves();
    int opponentMobility = opponentMoves.size();
    
    return (mobility - opponentMobility) * 10;
}

float Evaluator::calculateUncertainty(const Board* board) const {
    // Calculate uncertainty based on position complexity
    float uncertainty = 0.1f;
    
    // More pieces = more uncertainty
    int pieceCount = 0;
    for (int sq = 0; sq < 64; ++sq) {
        if (board->getPieceAt(sq).type != PieceType::NONE) {
            pieceCount++;
        }
    }
    
    uncertainty += (pieceCount - 16) * 0.01f;
    
    // King safety affects uncertainty
    if (board->isCheck()) {
        uncertainty += 0.2f;
    }
    
    return std::max(0.0f, std::min(1.0f, uncertainty));
}

} // namespace nags 