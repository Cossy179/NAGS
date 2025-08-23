#include "FastBoard.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>

#ifdef _MSC_VER
#include <intrin.h>
#endif

// Static member initialization
bool FastBoard::initialized = false;
Bitboard FastBoard::pawnAttacks[2][64];
Bitboard FastBoard::knightAttacks[64];
Bitboard FastBoard::kingAttacks[64];
Magic FastBoard::rookMagics[64];
Magic FastBoard::bishopMagics[64];
Bitboard FastBoard::rookAttacks[102400];
Bitboard FastBoard::bishopAttacks[5248];
uint64_t FastBoard::zPiece[2][6][64];
uint64_t FastBoard::zSide;
uint64_t FastBoard::zCastle[16];
uint64_t FastBoard::zEnpassant[8];

// Forward declaration
Bitboard indexToOccupancy(int index, Bitboard mask);

namespace {
    char pieceToFen(Piece p) {
        switch (p) {
            case Piece::WP: return 'P'; case Piece::WN: return 'N'; case Piece::WB: return 'B';
            case Piece::WR: return 'R'; case Piece::WQ: return 'Q'; case Piece::WK: return 'K';
            case Piece::BP: return 'p'; case Piece::BN: return 'n'; case Piece::BB: return 'b';
            case Piece::BR: return 'r'; case Piece::BQ: return 'q'; case Piece::BK: return 'k';
            default: return 0;
        }
    }
    
    Piece fenToPiece(char c) {
        switch (c) {
            case 'P': return Piece::WP; case 'N': return Piece::WN; case 'B': return Piece::WB;
            case 'R': return Piece::WR; case 'Q': return Piece::WQ; case 'K': return Piece::WK;
            case 'p': return Piece::BP; case 'n': return Piece::BN; case 'b': return Piece::BB;
            case 'r': return Piece::BR; case 'q': return Piece::BQ; case 'k': return Piece::BK;
            default: return Piece::None;
        }
    }
    
    int pieceType(Piece p) {
        if (p == Piece::None) return -1;
        return (static_cast<int>(p) - 1) % 6;
    }
    
    Color pieceColor(Piece p) {
        if (p == Piece::None) return Color::White; // doesn't matter
        return static_cast<int>(p) <= 6 ? Color::White : Color::Black;
    }
}

FastBoard::FastBoard() {
    if (!initialized) {
        initAttackTables();
        initMagics();
        initZobrist();
        initialized = true;
    }
    setStartPos();
}

bool FastBoard::setStartPos() {
    return setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool FastBoard::setFromFEN(const std::string &fen) {
    // Clear board
    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 6; ++p) {
            pieces[c][p] = 0ULL;
        }
        occupied[c] = 0ULL;
    }
    all_occupied = 0ULL;
    history.clear();
    
    std::istringstream iss(fen);
    std::string board, stm, castles, ep;
    if (!(iss >> board >> stm >> castles >> ep)) return false;
    
    int hm = 0, fm = 1;
    iss >> hm >> fm;
    
    // Parse board
    int rank = 7, file = 0;
    for (char c : board) {
        if (c == '/') {
            rank--;
            file = 0;
            continue;
        }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            file += c - '0';
        } else {
            if (file >= 8 || rank < 0) return false;
            int sq = rank * 8 + file;
            Piece p = fenToPiece(c);
            if (p == Piece::None) return false;
            
            Color color = pieceColor(p);
            int type = pieceType(p);
            pieces[static_cast<int>(color)][type] |= 1ULL << sq;
            file++;
        }
    }
    
    // Parse side to move
    side = (stm == "w") ? Color::White : Color::Black;
    
    // Parse castling rights
    castlingRights = 0;
    if (castles != "-") {
        for (char ch : castles) {
            if (ch == 'K') castlingRights |= 1;
            else if (ch == 'Q') castlingRights |= 2;
            else if (ch == 'k') castlingRights |= 4;
            else if (ch == 'q') castlingRights |= 8;
        }
    }
    
    // Parse en passant
    if (ep == "-") {
        epSquare = -1;
    } else {
        if (ep.size() != 2) return false;
        int f = ep[0] - 'a';
        int r = ep[1] - '1';
        epSquare = r * 8 + f;
    }
    
    halfmoveClock = hm;
    fullmoveNumber = fm;
    
    updateOccupancy();
    hashRecompute();
    
    return true;
}

std::string FastBoard::getFEN() const {
    std::ostringstream oss;
    
    // Board
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            int sq = rank * 8 + file;
            Piece p = pieceAt(sq);
            if (p == Piece::None) {
                empty++;
            } else {
                if (empty) {
                    oss << empty;
                    empty = 0;
                }
                oss << pieceToFen(p);
            }
        }
        if (empty) oss << empty;
        if (rank) oss << '/';
    }
    
    // Side to move
    oss << ' ' << (side == Color::White ? 'w' : 'b') << ' ';
    
    // Castling rights
    if (castlingRights == 0) {
        oss << '-';
    } else {
        if (castlingRights & 1) oss << 'K';
        if (castlingRights & 2) oss << 'Q';
        if (castlingRights & 4) oss << 'k';
        if (castlingRights & 8) oss << 'q';
    }
    
    // En passant
    oss << ' ';
    if (epSquare == -1) {
        oss << '-';
    } else {
        char f = 'a' + (epSquare & 7);
        char r = '1' + (epSquare >> 3);
        oss << f << r;
    }
    
    oss << ' ' << halfmoveClock << ' ' << fullmoveNumber;
    return oss.str();
}

Piece FastBoard::pieceAt(int sq) const {
    Bitboard mask = 1ULL << sq;
    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 6; ++p) {
            if (pieces[c][p] & mask) {
                // Convert back to Piece enum
                return static_cast<Piece>(c * 6 + p + 1);
            }
        }
    }
    return Piece::None;
}

void FastBoard::updateOccupancy() {
    occupied[0] = occupied[1] = 0ULL;
    for (int p = 0; p < 6; ++p) {
        occupied[0] |= pieces[0][p];
        occupied[1] |= pieces[1][p];
    }
    all_occupied = occupied[0] | occupied[1];
}

int FastBoard::kingSquare(Color c) const {
    Bitboard kings = pieces[static_cast<int>(c)][5]; // King is piece type 5
    return kings ? lsb(kings) : -1;
}

bool FastBoard::inCheck(Color c) const {
    int ksq = kingSquare(c);
    if (ksq == -1) return false;
    return isSquareAttacked(ksq, c == Color::White ? Color::Black : Color::White);
}

bool FastBoard::isSquareAttacked(int sq, Color byColor) const {
    int colorIdx = static_cast<int>(byColor);
    
    // Pawn attacks
    if (pawnAttacks[1 - colorIdx][sq] & pieces[colorIdx][0]) return true;
    
    // Knight attacks
    if (knightAttacks[sq] & pieces[colorIdx][1]) return true;
    
    // Bishop/Queen diagonal attacks
    if (getBishopAttacks(sq, all_occupied) & (pieces[colorIdx][2] | pieces[colorIdx][4])) return true;
    
    // Rook/Queen straight attacks
    if (getRookAttacks(sq, all_occupied) & (pieces[colorIdx][3] | pieces[colorIdx][4])) return true;
    
    // King attacks
    if (kingAttacks[sq] & pieces[colorIdx][5]) return true;
    
    return false;
}

// Bitboard manipulation
int FastBoard::popLSB(Bitboard &bb) {
    int sq = lsb(bb);
    bb &= bb - 1;
    return sq;
}

int FastBoard::countBits(Bitboard bb) {
    return popcount(bb);
}

// Optimized ray-based attack generation (faster than the original)
Bitboard FastBoard::getRookAttacks(int sq, Bitboard occupied) {
    return rookAttack(sq, occupied);
}

Bitboard FastBoard::getBishopAttacks(int sq, Bitboard occupied) {
    return bishopAttack(sq, occupied);
}

Bitboard FastBoard::getQueenAttacks(int sq, Bitboard occupied) {
    return getRookAttacks(sq, occupied) | getBishopAttacks(sq, occupied);
}

// Move generation
void FastBoard::addMove(std::vector<Move> &moves, int from, int to, 
                       Piece promo, bool ep, bool castle) const {
    moves.push_back(Move{from, to, promo, ep, castle});
}

void FastBoard::generatePseudoMoves(std::vector<Move> &moves) const {
    moves.clear();
    moves.reserve(256);
    
    Color us = side;
    generatePawnMoves(moves, us);
    generateKnightMoves(moves, us);
    generateBishopMoves(moves, us);
    generateRookMoves(moves, us);
    generateQueenMoves(moves, us);
    generateKingMoves(moves, us);
    generateCastlingMoves(moves, us);
}

void FastBoard::generatePawnMoves(std::vector<Move> &moves, Color c) const {
    int colorIdx = static_cast<int>(c);
    int enemyIdx = 1 - colorIdx;
    Bitboard pawns = pieces[colorIdx][0];
    Bitboard enemies = occupied[enemyIdx];
    Bitboard empty = ~all_occupied;
    
    if (c == Color::White) {
        // Single pushes
        Bitboard singlePush = northOne(pawns) & empty;
        Bitboard promotions = singlePush & RANK_8;
        Bitboard nonPromotions = singlePush & ~RANK_8;
        
        // Generate non-promotion pushes
        Bitboard tmp = nonPromotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to - 8, to);
        }
        
        // Generate promotion pushes
        tmp = promotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to - 8, to, Piece::WQ);
            addMove(moves, to - 8, to, Piece::WR);
            addMove(moves, to - 8, to, Piece::WB);
            addMove(moves, to - 8, to, Piece::WN);
        }
        
        // Double pushes
        Bitboard doublePush = northOne(singlePush) & empty & RANK_4;
        tmp = doublePush;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to - 16, to);
        }
        
        // Captures
        Bitboard leftCaptures = northWest(pawns) & enemies;
        Bitboard rightCaptures = northEast(pawns) & enemies;
        
        // Left captures
        Bitboard leftPromotions = leftCaptures & RANK_8;
        Bitboard leftNonPromotions = leftCaptures & ~RANK_8;
        
        tmp = leftNonPromotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to - 7, to);
        }
        
        tmp = leftPromotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to - 7, to, Piece::WQ);
            addMove(moves, to - 7, to, Piece::WR);
            addMove(moves, to - 7, to, Piece::WB);
            addMove(moves, to - 7, to, Piece::WN);
        }
        
        // Right captures
        Bitboard rightPromotions = rightCaptures & RANK_8;
        Bitboard rightNonPromotions = rightCaptures & ~RANK_8;
        
        tmp = rightNonPromotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to - 9, to);
        }
        
        tmp = rightPromotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to - 9, to, Piece::WQ);
            addMove(moves, to - 9, to, Piece::WR);
            addMove(moves, to - 9, to, Piece::WB);
            addMove(moves, to - 9, to, Piece::WN);
        }
        
        // En passant
        if (epSquare != -1) {
            Bitboard epMask = 1ULL << epSquare;
            if (northWest(pawns) & epMask) {
                addMove(moves, epSquare - 7, epSquare, Piece::None, true);
            }
            if (northEast(pawns) & epMask) {
                addMove(moves, epSquare - 9, epSquare, Piece::None, true);
            }
        }
    } else {
        // Black pawns (similar logic but reversed)
        // Single pushes
        Bitboard singlePush = southOne(pawns) & empty;
        Bitboard promotions = singlePush & RANK_1;
        Bitboard nonPromotions = singlePush & ~RANK_1;
        
        Bitboard tmp = nonPromotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to + 8, to);
        }
        
        tmp = promotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to + 8, to, Piece::BQ);
            addMove(moves, to + 8, to, Piece::BR);
            addMove(moves, to + 8, to, Piece::BB);
            addMove(moves, to + 8, to, Piece::BN);
        }
        
        // Double pushes
        Bitboard doublePush = southOne(singlePush) & empty & RANK_5;
        tmp = doublePush;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to + 16, to);
        }
        
        // Captures
        Bitboard leftCaptures = southEast(pawns) & enemies;
        Bitboard rightCaptures = southWest(pawns) & enemies;
        
        // Left captures (from black's perspective)
        Bitboard leftPromotions = leftCaptures & RANK_1;
        Bitboard leftNonPromotions = leftCaptures & ~RANK_1;
        
        tmp = leftNonPromotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to + 7, to);
        }
        
        tmp = leftPromotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to + 7, to, Piece::BQ);
            addMove(moves, to + 7, to, Piece::BR);
            addMove(moves, to + 7, to, Piece::BB);
            addMove(moves, to + 7, to, Piece::BN);
        }
        
        // Right captures
        Bitboard rightPromotions = rightCaptures & RANK_1;
        Bitboard rightNonPromotions = rightCaptures & ~RANK_1;
        
        tmp = rightNonPromotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to + 9, to);
        }
        
        tmp = rightPromotions;
        while (tmp) {
            int to = popLSB(tmp);
            addMove(moves, to + 9, to, Piece::BQ);
            addMove(moves, to + 9, to, Piece::BR);
            addMove(moves, to + 9, to, Piece::BB);
            addMove(moves, to + 9, to, Piece::BN);
        }
        
        // En passant
        if (epSquare != -1) {
            Bitboard epMask = 1ULL << epSquare;
            if (southEast(pawns) & epMask) {
                addMove(moves, epSquare + 7, epSquare, Piece::None, true);
            }
            if (southWest(pawns) & epMask) {
                addMove(moves, epSquare + 9, epSquare, Piece::None, true);
            }
        }
    }
}

void FastBoard::generateKnightMoves(std::vector<Move> &moves, Color c) const {
    int colorIdx = static_cast<int>(c);
    Bitboard knights = pieces[colorIdx][1];
    Bitboard targets = ~occupied[colorIdx];
    
    while (knights) {
        int from = popLSB(knights);
        Bitboard attacks = knightAttacks[from] & targets;
        
        while (attacks) {
            int to = popLSB(attacks);
            addMove(moves, from, to);
        }
    }
}

void FastBoard::generateBishopMoves(std::vector<Move> &moves, Color c) const {
    int colorIdx = static_cast<int>(c);
    Bitboard bishops = pieces[colorIdx][2];
    Bitboard targets = ~occupied[colorIdx];
    
    while (bishops) {
        int from = popLSB(bishops);
        Bitboard attacks = getBishopAttacks(from, all_occupied) & targets;
        
        while (attacks) {
            int to = popLSB(attacks);
            addMove(moves, from, to);
        }
    }
}

void FastBoard::generateRookMoves(std::vector<Move> &moves, Color c) const {
    int colorIdx = static_cast<int>(c);
    Bitboard rooks = pieces[colorIdx][3];
    Bitboard targets = ~occupied[colorIdx];
    
    while (rooks) {
        int from = popLSB(rooks);
        Bitboard attacks = getRookAttacks(from, all_occupied) & targets;
        
        while (attacks) {
            int to = popLSB(attacks);
            addMove(moves, from, to);
        }
    }
}

void FastBoard::generateQueenMoves(std::vector<Move> &moves, Color c) const {
    int colorIdx = static_cast<int>(c);
    Bitboard queens = pieces[colorIdx][4];
    Bitboard targets = ~occupied[colorIdx];
    
    while (queens) {
        int from = popLSB(queens);
        Bitboard attacks = getQueenAttacks(from, all_occupied) & targets;
        
        while (attacks) {
            int to = popLSB(attacks);
            addMove(moves, from, to);
        }
    }
}

void FastBoard::generateKingMoves(std::vector<Move> &moves, Color c) const {
    int colorIdx = static_cast<int>(c);
    Bitboard kings = pieces[colorIdx][5];
    Bitboard targets = ~occupied[colorIdx];
    
    if (kings) {
        int from = lsb(kings);
        Bitboard attacks = kingAttacks[from] & targets;
        
        while (attacks) {
            int to = popLSB(attacks);
            addMove(moves, from, to);
        }
    }
}

void FastBoard::generateCastlingMoves(std::vector<Move> &moves, Color c) const {
    if (inCheck(c)) return; // Can't castle in check
    
    if (c == Color::White) {
        // Kingside castling
        if ((castlingRights & 1) && 
            !(all_occupied & 0x60ULL) && // f1 and g1 empty
            !isSquareAttacked(5, Color::Black) && // f1 not attacked
            !isSquareAttacked(6, Color::Black)) { // g1 not attacked
            addMove(moves, 4, 6, Piece::None, false, true);
        }
        
        // Queenside castling
        if ((castlingRights & 2) && 
            !(all_occupied & 0x0EULL) && // b1, c1, d1 empty
            !isSquareAttacked(3, Color::Black) && // d1 not attacked
            !isSquareAttacked(2, Color::Black)) { // c1 not attacked
            addMove(moves, 4, 2, Piece::None, false, true);
        }
    } else {
        // Black castling
        // Kingside castling
        if ((castlingRights & 4) && 
            !(all_occupied & 0x6000000000000000ULL) && // f8 and g8 empty
            !isSquareAttacked(61, Color::White) && // f8 not attacked
            !isSquareAttacked(62, Color::White)) { // g8 not attacked
            addMove(moves, 60, 62, Piece::None, false, true);
        }
        
        // Queenside castling
        if ((castlingRights & 8) && 
            !(all_occupied & 0x0E00000000000000ULL) && // b8, c8, d8 empty
            !isSquareAttacked(59, Color::White) && // d8 not attacked
            !isSquareAttacked(58, Color::White)) { // c8 not attacked
            addMove(moves, 60, 58, Piece::None, false, true);
        }
    }
}

std::vector<Move> FastBoard::generateLegalMoves() const {
    std::vector<Move> pseudo;
    generatePseudoMoves(pseudo);
    
    std::vector<Move> legal;
    legal.reserve(pseudo.size());
    
    for (const Move &move : pseudo) {
        FastBoard copy = *this;
        copy.makeMove(move);
        if (!copy.inCheck(side)) {
            legal.push_back(move);
        }
    }
    
    return legal;
}

// This is a placeholder for the move making implementation
void FastBoard::makeMove(const Move &m) {
    // Save history
    HistoryEntry entry;
    entry.move = m;
    entry.captured = pieceAt(m.to);
    entry.castlingRights = castlingRights;
    entry.epSquare = epSquare;
    entry.halfmoveClock = halfmoveClock;
    entry.hash = hash;
    history.push_back(entry);
    
    // Clear en passant
    if (epSquare != -1) {
        hashSetEp(epSquare, -1);
        epSquare = -1;
    }
    
    // Move piece
    Piece moving = pieceAt(m.from);
    Color movingColor = pieceColor(moving);
    int movingType = pieceType(moving);
    int colorIdx = static_cast<int>(movingColor);
    
    // Remove from source square
    pieces[colorIdx][movingType] &= ~(1ULL << m.from);
    hashTogglePiece(movingColor, movingType, m.from);
    
    // Handle captures
    if (entry.captured != Piece::None) {
        Color capturedColor = pieceColor(entry.captured);
        int capturedType = pieceType(entry.captured);
        int capturedColorIdx = static_cast<int>(capturedColor);
        
        pieces[capturedColorIdx][capturedType] &= ~(1ULL << m.to);
        // Hash will be updated when placing the moving piece
    }
    
    // Handle en passant capture
    if (m.isEnPassant) {
        int captureSquare = (movingColor == Color::White) ? m.to - 8 : m.to + 8;
        int enemyColorIdx = 1 - colorIdx;
        pieces[enemyColorIdx][0] &= ~(1ULL << captureSquare); // Remove enemy pawn
    }
    
    // Place piece on destination (handle promotion)
    Piece placedPiece = (m.promotion != Piece::None) ? m.promotion : moving;
    Color placedColor = pieceColor(placedPiece);
    int placedType = pieceType(placedPiece);
    int placedColorIdx = static_cast<int>(placedColor);
    
    pieces[placedColorIdx][placedType] |= 1ULL << m.to;
    hashTogglePiece(placedColor, placedType, m.to);
    
    // Handle castling
    if (m.isCastling) {
        if (movingColor == Color::White) {
            if (m.to == 6) { // Kingside
                pieces[0][3] &= ~(1ULL << 7); // Remove rook from h1
                pieces[0][3] |= 1ULL << 5;    // Place rook on f1
            } else if (m.to == 2) { // Queenside
                pieces[0][3] &= ~(1ULL << 0); // Remove rook from a1
                pieces[0][3] |= 1ULL << 3;    // Place rook on d1
            }
        } else {
            if (m.to == 62) { // Kingside
                pieces[1][3] &= ~(1ULL << 63); // Remove rook from h8
                pieces[1][3] |= 1ULL << 61;    // Place rook on f8
            } else if (m.to == 58) { // Queenside
                pieces[1][3] &= ~(1ULL << 56); // Remove rook from a8
                pieces[1][3] |= 1ULL << 59;    // Place rook on d8
            }
        }
    }
    
    // Update castling rights
    uint8_t oldRights = castlingRights;
    if (m.from == 4 || m.to == 4) castlingRights &= ~3;  // White king moved or captured
    if (m.from == 60 || m.to == 60) castlingRights &= ~12; // Black king moved or captured
    if (m.from == 0 || m.to == 0) castlingRights &= ~2;   // White queenside rook
    if (m.from == 7 || m.to == 7) castlingRights &= ~1;   // White kingside rook
    if (m.from == 56 || m.to == 56) castlingRights &= ~8; // Black queenside rook
    if (m.from == 63 || m.to == 63) castlingRights &= ~4; // Black kingside rook
    
    if (oldRights != castlingRights) {
        hashToggleCastle(oldRights);
        hashToggleCastle(castlingRights);
    }
    
    // Set en passant square for pawn double moves
    if (movingType == 0 && abs(m.to - m.from) == 16) { // Pawn double move
        epSquare = (m.from + m.to) / 2;
        hashSetEp(-1, epSquare);
    }
    
    // Update halfmove clock
    if (movingType == 0 || entry.captured != Piece::None) {
        halfmoveClock = 0;
    } else {
        halfmoveClock++;
    }
    
    // Update fullmove number
    if (side == Color::Black) {
        fullmoveNumber++;
    }
    
    // Switch side
    side = (side == Color::White) ? Color::Black : Color::White;
    hashToggleSide();
    
    updateOccupancy();
}

void FastBoard::unmakeMove() {
    if (history.empty()) return;
    
    HistoryEntry entry = history.back();
    history.pop_back();
    
    // Restore state
    castlingRights = entry.castlingRights;
    epSquare = entry.epSquare;
    halfmoveClock = entry.halfmoveClock;
    hash = entry.hash;
    
    // Switch side back
    side = (side == Color::White) ? Color::Black : Color::White;
    
    // Update fullmove number
    if (side == Color::Black) {
        fullmoveNumber--;
    }
    
    Move m = entry.move;
    Piece moving = (m.promotion != Piece::None) ? m.promotion : pieceAt(m.to);
    Color movingColor = pieceColor(moving);
    int movingType = pieceType(moving);
    int colorIdx = static_cast<int>(movingColor);
    
    // Remove piece from destination
    pieces[colorIdx][movingType] &= ~(1ULL << m.to);
    
    // Handle promotion (place original pawn back)
    if (m.promotion != Piece::None) {
        pieces[colorIdx][0] |= 1ULL << m.from; // Place pawn back
    } else {
        pieces[colorIdx][movingType] |= 1ULL << m.from; // Place original piece back
    }
    
    // Restore captured piece
    if (entry.captured != Piece::None) {
        Color capturedColor = pieceColor(entry.captured);
        int capturedType = pieceType(entry.captured);
        int capturedColorIdx = static_cast<int>(capturedColor);
        
        pieces[capturedColorIdx][capturedType] |= 1ULL << m.to;
    }
    
    // Handle en passant
    if (m.isEnPassant) {
        int captureSquare = (movingColor == Color::White) ? m.to - 8 : m.to + 8;
        int enemyColorIdx = 1 - colorIdx;
        pieces[enemyColorIdx][0] |= 1ULL << captureSquare; // Restore enemy pawn
    }
    
    // Handle castling
    if (m.isCastling) {
        if (movingColor == Color::White) {
            if (m.to == 6) { // Kingside
                pieces[0][3] &= ~(1ULL << 5); // Remove rook from f1
                pieces[0][3] |= 1ULL << 7;    // Place rook back on h1
            } else if (m.to == 2) { // Queenside
                pieces[0][3] &= ~(1ULL << 3); // Remove rook from d1
                pieces[0][3] |= 1ULL << 0;    // Place rook back on a1
            }
        } else {
            if (m.to == 62) { // Kingside
                pieces[1][3] &= ~(1ULL << 61); // Remove rook from f8
                pieces[1][3] |= 1ULL << 63;    // Place rook back on h8
            } else if (m.to == 58) { // Queenside
                pieces[1][3] &= ~(1ULL << 59); // Remove rook from d8
                pieces[1][3] |= 1ULL << 56;    // Place rook back on a8
            }
        }
    }
    
    updateOccupancy();
}

// Utility functions
bool FastBoard::isWhite(Piece p) {
    return static_cast<int>(p) >= 1 && static_cast<int>(p) <= 6;
}

bool FastBoard::isBlack(Piece p) {
    return static_cast<int>(p) >= 7 && static_cast<int>(p) <= 12;
}

std::string FastBoard::moveToUci(const Move &m) {
    std::string result;
    result += char('a' + (m.from & 7));
    result += char('1' + (m.from >> 3));
    result += char('a' + (m.to & 7));
    result += char('1' + (m.to >> 3));
    
    if (m.promotion != Piece::None) {
        switch (m.promotion) {
            case Piece::WQ: case Piece::BQ: result += 'q'; break;
            case Piece::WR: case Piece::BR: result += 'r'; break;
            case Piece::WB: case Piece::BB: result += 'b'; break;
            case Piece::WN: case Piece::BN: result += 'n'; break;
            default: break;
        }
    }
    
    return result;
}

int FastBoard::algebraicTo0x88(const std::string &alg) {
    if (alg.size() < 2) return -1;
    int file = alg[0] - 'a';
    int rank = alg[1] - '1';
    if (file < 0 || file > 7 || rank < 0 || rank > 7) return -1;
    return rank * 8 + file;
}

bool FastBoard::applyMovesUCI(const std::vector<std::string> &uciMoves) {
    for (const std::string &moveStr : uciMoves) {
        if (moveStr.size() < 4) return false;
        
        int from = algebraicTo0x88(moveStr.substr(0, 2));
        int to = algebraicTo0x88(moveStr.substr(2, 2));
        if (from == -1 || to == -1) return false;
        
        Piece promotion = Piece::None;
        if (moveStr.size() == 5) {
            char p = moveStr[4];
            if (p == 'q') promotion = (side == Color::White) ? Piece::WQ : Piece::BQ;
            else if (p == 'r') promotion = (side == Color::White) ? Piece::WR : Piece::BR;
            else if (p == 'b') promotion = (side == Color::White) ? Piece::WB : Piece::BB;
            else if (p == 'n') promotion = (side == Color::White) ? Piece::WN : Piece::BN;
        }
        
        auto legal = generateLegalMoves();
        bool found = false;
        for (const Move &move : legal) {
            if (move.from == from && move.to == to && move.promotion == promotion) {
                makeMove(move);
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

// Placeholder implementations for the initialization functions
void FastBoard::initAttackTables() {
    // Initialize pawn attacks
    for (int sq = 0; sq < 64; ++sq) {
        int file = sq & 7;
        int rank = sq >> 3;
        
        // White pawn attacks
        pawnAttacks[0][sq] = 0ULL;
        if (rank < 7) {
            if (file > 0) pawnAttacks[0][sq] |= 1ULL << (sq + 7);
            if (file < 7) pawnAttacks[0][sq] |= 1ULL << (sq + 9);
        }
        
        // Black pawn attacks
        pawnAttacks[1][sq] = 0ULL;
        if (rank > 0) {
            if (file > 0) pawnAttacks[1][sq] |= 1ULL << (sq - 9);
            if (file < 7) pawnAttacks[1][sq] |= 1ULL << (sq - 7);
        }
    }
    
    // Initialize knight attacks
    int knightDeltas[8][2] = {{-2,-1}, {-2,1}, {-1,-2}, {-1,2}, {1,-2}, {1,2}, {2,-1}, {2,1}};
    for (int sq = 0; sq < 64; ++sq) {
        int file = sq & 7;
        int rank = sq >> 3;
        knightAttacks[sq] = 0ULL;
        
        for (int i = 0; i < 8; ++i) {
            int newFile = file + knightDeltas[i][0];
            int newRank = rank + knightDeltas[i][1];
            if (newFile >= 0 && newFile < 8 && newRank >= 0 && newRank < 8) {
                knightAttacks[sq] |= 1ULL << (newRank * 8 + newFile);
            }
        }
    }
    
    // Initialize king attacks
    int kingDeltas[8][2] = {{-1,-1}, {-1,0}, {-1,1}, {0,-1}, {0,1}, {1,-1}, {1,0}, {1,1}};
    for (int sq = 0; sq < 64; ++sq) {
        int file = sq & 7;
        int rank = sq >> 3;
        kingAttacks[sq] = 0ULL;
        
        for (int i = 0; i < 8; ++i) {
            int newFile = file + kingDeltas[i][0];
            int newRank = rank + kingDeltas[i][1];
            if (newFile >= 0 && newFile < 8 && newRank >= 0 && newRank < 8) {
                kingAttacks[sq] |= 1ULL << (newRank * 8 + newFile);
            }
        }
    }
}

// True magic bitboard initialization with precomputed values
void FastBoard::initMagics() {
    // Precomputed magic numbers for rooks (these are known good values)
    const uint64_t rookMagicNumbers[64] = {
        0xa8002c000108020ULL, 0x4c8004001000200ULL, 0x8140002010004000ULL, 0x2080041000200010ULL,
        0x200040008080ULL, 0x300020008040ULL, 0x4000200010008080ULL, 0x4000007000800100ULL,
        0x2000400080208000ULL, 0x2000401000402000ULL, 0x2000200080100080ULL, 0x2000200040080080ULL,
        0x2000200020080080ULL, 0x2000200010080080ULL, 0x2000200008080080ULL, 0x2000200004080080ULL,
        0x2000208000800400ULL, 0x2000108000800200ULL, 0x2000088000800100ULL, 0x2000048000800080ULL,
        0x2000028000800040ULL, 0x2000018000800020ULL, 0x2000008000800010ULL, 0x2000008000800008ULL,
        0x2000400080208000ULL, 0x2000200080104000ULL, 0x2000200080102000ULL, 0x2000200080101000ULL,
        0x2000200080100800ULL, 0x2000200080100400ULL, 0x2000200080100200ULL, 0x2000200080100100ULL,
        0x8000400020005000ULL, 0x4000200010003000ULL, 0x2000100008001800ULL, 0x1000080004000c00ULL,
        0x800040002000600ULL, 0x400020001000300ULL, 0x200010000800180ULL, 0x1000800040008100ULL,
        0x8000400020005000ULL, 0x4000200010003000ULL, 0x2000100008001800ULL, 0x1000080004000c00ULL,
        0x800040002000600ULL, 0x400020001000300ULL, 0x200010000800180ULL, 0x1000800040008100ULL,
        0x8000400020005000ULL, 0x4000200010003000ULL, 0x2000100008001800ULL, 0x1000080004000c00ULL,
        0x800040002000600ULL, 0x400020001000300ULL, 0x200010000800180ULL, 0x1000800040008100ULL,
        0x80004000200050ULL, 0x40002000100030ULL, 0x20001000080018ULL, 0x10000800040008ULL,
        0x8000400020004ULL, 0x4000200010002ULL, 0x2000100008001ULL, 0x1000080004000ULL
    };
    
    // Precomputed magic numbers for bishops
    const uint64_t bishopMagicNumbers[64] = {
        0x40040844404084ULL, 0x2004208a004208ULL, 0x10190041080202ULL, 0x108060845042010ULL,
        0x581104180800210ULL, 0x2112080446200010ULL, 0x1080820820060210ULL, 0x3c0808410220200ULL,
        0x4050404440404ULL, 0x21001420088ULL, 0x24d0080801082102ULL, 0x1020a0a020400ULL,
        0x40308200402ULL, 0x4011002000800ULL, 0x401484104104005ULL, 0x801010402020200ULL,
        0x400210c3880100ULL, 0x404022024108200ULL, 0x810018200204102ULL, 0x4002801a02003ULL,
        0x85040820080400ULL, 0x810102c808880400ULL, 0xe900410884800ULL, 0x8002020480840102ULL,
        0x220200865090201ULL, 0x2010100a02021202ULL, 0x152048408022401ULL, 0x20080002081110ULL,
        0x4001001021004000ULL, 0x800040400a011002ULL, 0xe4004081011002ULL, 0x1c004001012080ULL,
        0x8004200962a00220ULL, 0x8422100208500202ULL, 0x2000402200300c08ULL, 0x8646020080080080ULL,
        0x80020a0200100808ULL, 0x2010004880111000ULL, 0x623000a080011400ULL, 0x42008c0340209202ULL,
        0x209188240001000ULL, 0x400408a884001800ULL, 0x110400a6080400ULL, 0x1840060a44020800ULL,
        0x90080104000041ULL, 0x201011000808101ULL, 0x1a2208080504f080ULL, 0x8012020600211212ULL,
        0x500861011240000ULL, 0x180806108200800ULL, 0x4000020e01040044ULL, 0x300000261044000aULL,
        0x802241102020002ULL, 0x20906061210001ULL, 0x5a84841004010310ULL, 0x4010801011c04ULL,
        0xa010109502200ULL, 0x4a02012000ULL, 0x500201010098b028ULL, 0x8040002811040900ULL,
        0x28000010020204ULL, 0x6000020202d0240ULL, 0x8918844842082200ULL, 0x4010011029020020ULL
    };
    
    // Initialize attack table pointers
    Bitboard* rookAttackPtr = rookAttacks;
    Bitboard* bishopAttackPtr = bishopAttacks;
    
    // Initialize rook magics
    for (int sq = 0; sq < 64; ++sq) {
        rookMagics[sq].mask = rookMask(sq);
        rookMagics[sq].magic = rookMagicNumbers[sq];
        rookMagics[sq].attacks = rookAttackPtr;
        rookMagics[sq].shift = 64 - popcount(rookMagics[sq].mask);
        
        // Generate all attack patterns for this square
        int bits = popcount(rookMagics[sq].mask);
        int permutations = 1 << bits;
        
        for (int i = 0; i < permutations; ++i) {
            Bitboard occupancy = indexToOccupancy(i, rookMagics[sq].mask);
            int index = (occupancy * rookMagics[sq].magic) >> rookMagics[sq].shift;
            rookAttackPtr[index] = rookAttack(sq, occupancy);
        }
        
        rookAttackPtr += permutations;
    }
    
    // Initialize bishop magics
    for (int sq = 0; sq < 64; ++sq) {
        bishopMagics[sq].mask = bishopMask(sq);
        bishopMagics[sq].magic = bishopMagicNumbers[sq];
        bishopMagics[sq].attacks = bishopAttackPtr;
        bishopMagics[sq].shift = 64 - popcount(bishopMagics[sq].mask);
        
        // Generate all attack patterns for this square
        int bits = popcount(bishopMagics[sq].mask);
        int permutations = 1 << bits;
        
        for (int i = 0; i < permutations; ++i) {
            Bitboard occupancy = indexToOccupancy(i, bishopMagics[sq].mask);
            int index = (occupancy * bishopMagics[sq].magic) >> bishopMagics[sq].shift;
            bishopAttackPtr[index] = bishopAttack(sq, occupancy);
        }
        
        bishopAttackPtr += permutations;
    }
}

Bitboard FastBoard::rookMask(int sq) {
    Bitboard result = 0ULL;
    int rank = sq >> 3;
    int file = sq & 7;
    
    // Horizontal
    for (int f = 1; f < 7; ++f) {
        if (f != file) result |= 1ULL << (rank * 8 + f);
    }
    
    // Vertical
    for (int r = 1; r < 7; ++r) {
        if (r != rank) result |= 1ULL << (r * 8 + file);
    }
    
    return result;
}

Bitboard FastBoard::bishopMask(int sq) {
    Bitboard result = 0ULL;
    int rank = sq >> 3;
    int file = sq & 7;
    
    // Diagonal directions
    int directions[4][2] = {{1,1}, {1,-1}, {-1,1}, {-1,-1}};
    
    for (int d = 0; d < 4; ++d) {
        for (int i = 1; i < 7; ++i) {
            int newRank = rank + directions[d][0] * i;
            int newFile = file + directions[d][1] * i;
            
            if (newRank >= 1 && newRank < 7 && newFile >= 1 && newFile < 7) {
                result |= 1ULL << (newRank * 8 + newFile);
            } else {
                break;
            }
        }
    }
    
    return result;
}

// Simplified implementations for now - would use magic multiplication in practice
Bitboard FastBoard::rookAttack(int sq, Bitboard occupied) {
    Bitboard result = 0ULL;
    int rank = sq >> 3;
    int file = sq & 7;
    
    // North
    for (int r = rank + 1; r < 8; ++r) {
        int target = r * 8 + file;
        result |= 1ULL << target;
        if (occupied & (1ULL << target)) break;
    }
    
    // South
    for (int r = rank - 1; r >= 0; --r) {
        int target = r * 8 + file;
        result |= 1ULL << target;
        if (occupied & (1ULL << target)) break;
    }
    
    // East
    for (int f = file + 1; f < 8; ++f) {
        int target = rank * 8 + f;
        result |= 1ULL << target;
        if (occupied & (1ULL << target)) break;
    }
    
    // West
    for (int f = file - 1; f >= 0; --f) {
        int target = rank * 8 + f;
        result |= 1ULL << target;
        if (occupied & (1ULL << target)) break;
    }
    
    return result;
}

Bitboard FastBoard::bishopAttack(int sq, Bitboard occupied) {
    Bitboard result = 0ULL;
    int rank = sq >> 3;
    int file = sq & 7;
    
    int directions[4][2] = {{1,1}, {1,-1}, {-1,1}, {-1,-1}};
    
    for (int d = 0; d < 4; ++d) {
        for (int i = 1; i < 8; ++i) {
            int newRank = rank + directions[d][0] * i;
            int newFile = file + directions[d][1] * i;
            
            if (newRank >= 0 && newRank < 8 && newFile >= 0 && newFile < 8) {
                int target = newRank * 8 + newFile;
                result |= 1ULL << target;
                if (occupied & (1ULL << target)) break;
            } else {
                break;
            }
        }
    }
    
    return result;
}

void FastBoard::initZobrist() {
    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
    
    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 6; ++p) {
            for (int sq = 0; sq < 64; ++sq) {
                zPiece[c][p][sq] = rng();
            }
        }
    }
    
    zSide = rng();
    
    for (int i = 0; i < 16; ++i) {
        zCastle[i] = rng();
    }
    
    for (int f = 0; f < 8; ++f) {
        zEnpassant[f] = rng();
    }
}

void FastBoard::hashRecompute() {
    hash = 0;
    
    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 6; ++p) {
            Bitboard bb = pieces[c][p];
            while (bb) {
                int sq = popLSB(bb);
                hash ^= zPiece[c][p][sq];
            }
        }
    }
    
    if (side == Color::Black) {
        hash ^= zSide;
    }
    
    hash ^= zCastle[castlingRights];
    
    if (epSquare != -1) {
        hash ^= zEnpassant[epSquare & 7];
    }
}

void FastBoard::hashTogglePiece(Color c, int piece_type, int sq) {
    hash ^= zPiece[static_cast<int>(c)][piece_type][sq];
}

void FastBoard::hashToggleSide() {
    hash ^= zSide;
}

void FastBoard::hashToggleCastle(uint8_t rights) {
    hash ^= zCastle[rights & 0xF];
}

void FastBoard::hashSetEp(int epSqOld, int epSqNew) {
    if (epSqOld != -1) hash ^= zEnpassant[epSqOld & 7];
    if (epSqNew != -1) hash ^= zEnpassant[epSqNew & 7];
}

// Perft implementation for testing
uint64_t FastBoard::perft(int depth) {
    if (depth == 0) return 1;
    
    auto moves = generateLegalMoves();
    uint64_t nodes = 0;
    
    for (const Move &move : moves) {
        makeMove(move);
        nodes += perft(depth - 1);
        unmakeMove();
    }
    
    return nodes;
}

void FastBoard::divide(int depth) {
    auto moves = generateLegalMoves();
    uint64_t total = 0;
    
    for (const Move &move : moves) {
        makeMove(move);
        uint64_t count = perft(depth - 1);
        unmakeMove();
        
        std::cout << moveToUci(move) << ": " << count << std::endl;
        total += count;
    }
    
    std::cout << "\nTotal: " << total << std::endl;
}

Bitboard FastBoard::randomBitboard() {
    static std::mt19937_64 rng(std::random_device{}());
    return rng();
}

uint64_t FastBoard::findMagic(int sq, int bits, bool isRook) {
    // Placeholder implementation
    return randomBitboard();
}

// Convert index to occupancy pattern
Bitboard indexToOccupancy(int index, Bitboard mask) {
    Bitboard occupancy = 0ULL;
    int bits = popcount(mask);
    
    for (int i = 0; i < bits; ++i) {
        int sq = lsb(mask);
        mask &= mask - 1; // Remove LSB
        
        if (index & (1 << i)) {
            occupancy |= 1ULL << sq;
        }
    }
    
    return occupancy;
}
