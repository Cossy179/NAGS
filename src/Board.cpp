#include "Board.h"

#include <algorithm>
#include <cassert>
#include <cctype>
#include <cstring>
#include <intrin.h>
#include <random>
#include <sstream>

namespace {
inline int bsf64(uint64_t bb) {
    unsigned long idx;
    _BitScanForward64(&idx, bb);
    return static_cast<int>(idx);
}
inline int bsr64(uint64_t bb) {
    unsigned long idx;
    _BitScanReverse64(&idx, bb);
    return static_cast<int>(idx);
}

char pieceToFen(Piece p) {
    switch (p) {
        case Piece::WP: return 'P';
        case Piece::WN: return 'N';
        case Piece::WB: return 'B';
        case Piece::WR: return 'R';
        case Piece::WQ: return 'Q';
        case Piece::WK: return 'K';
        case Piece::BP: return 'p';
        case Piece::BN: return 'n';
        case Piece::BB: return 'b';
        case Piece::BR: return 'r';
        case Piece::BQ: return 'q';
        case Piece::BK: return 'k';
        default: return 0;
    }
}

Piece fenToPiece(char c) {
    switch (c) {
        case 'P': return Piece::WP;
        case 'N': return Piece::WN;
        case 'B': return Piece::WB;
        case 'R': return Piece::WR;
        case 'Q': return Piece::WQ;
        case 'K': return Piece::WK;
        case 'p': return Piece::BP;
        case 'n': return Piece::BN;
        case 'b': return Piece::BB;
        case 'r': return Piece::BR;
        case 'q': return Piece::BQ;
        case 'k': return Piece::BK;
        default: return Piece::None;
    }
}

constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_H = 0x8080808080808080ULL;
constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;
}

Bitboard Board::knightAttacks[64];
Bitboard Board::kingAttacks[64];
Bitboard Board::pawnAttacks[2][64];
bool Board::attackTablesInit = false;
bool Board::zobristInit = false;
uint64_t Board::zPiece[12][64];
uint64_t Board::zSide;
uint64_t Board::zCastle[16];
uint64_t Board::zEnpassant[8];

// Ray tables for sliding pieces
static Bitboard rayN[64], rayS[64], rayE[64], rayW[64], rayNE[64], rayNW[64], raySE[64], raySW[64];

static void initRays() {
    for (int sq = 0; sq < 64; ++sq) {
        int f = sq & 7, r = sq >> 3;
        Bitboard b = 0;
        // North
        b = 0; for (int rr = r + 1; rr <= 7; ++rr) b |= 1ULL << (rr * 8 + f); rayN[sq] = b;
        // South
        b = 0; for (int rr = r - 1; rr >= 0; --rr) b |= 1ULL << (rr * 8 + f); rayS[sq] = b;
        // East
        b = 0; for (int ff = f + 1; ff <= 7; ++ff) b |= 1ULL << (r * 8 + ff); rayE[sq] = b;
        // West
        b = 0; for (int ff = f - 1; ff >= 0; --ff) b |= 1ULL << (r * 8 + ff); rayW[sq] = b;
        // NE
        b = 0; for (int ff = f + 1, rr = r + 1; ff <= 7 && rr <= 7; ++ff, ++rr) b |= 1ULL << (rr * 8 + ff); rayNE[sq] = b;
        // NW
        b = 0; for (int ff = f - 1, rr = r + 1; ff >= 0 && rr <= 7; --ff, ++rr) b |= 1ULL << (rr * 8 + ff); rayNW[sq] = b;
        // SE
        b = 0; for (int ff = f + 1, rr = r - 1; ff <= 7 && rr >= 0; ++ff, --rr) b |= 1ULL << (rr * 8 + ff); raySE[sq] = b;
        // SW
        b = 0; for (int ff = f - 1, rr = r - 1; ff >= 0 && rr >= 0; --ff, --rr) b |= 1ULL << (rr * 8 + ff); raySW[sq] = b;
    }
}

void Board::initAttackTables() {
    if (attackTablesInit) return;
    attackTablesInit = true;
    initRays();
    for (int sq = 0; sq < 64; ++sq) {
        int f = sq & 7, r = sq >> 3;
        Bitboard km = 0, nm = 0;
        for (int df = -1; df <= 1; ++df) for (int dr = -1; dr <= 1; ++dr) {
            if (df == 0 && dr == 0) continue;
            int nf = f + df, nr = r + dr;
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) km |= 1ULL << (nr * 8 + nf);
        }
        static const int kdf[8] = {1,2,2,1,-1,-2,-2,-1};
        static const int kdr[8] = {2,1,-1,-2,-2,-1,1,2};
        for (int i = 0; i < 8; ++i) {
            int nf = f + kdf[i], nr = r + kdr[i];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) nm |= 1ULL << (nr * 8 + nf);
        }
        knightAttacks[sq] = nm;
        kingAttacks[sq] = km;
        // Pawns
        Bitboard w = 0, bl = 0;
        if (f + 1 < 8 && r + 1 < 8) w |= 1ULL << ((r + 1) * 8 + (f + 1));
        if (f - 1 >= 0 && r + 1 < 8) w |= 1ULL << ((r + 1) * 8 + (f - 1));
        pawnAttacks[0][sq] = w;
        if (f + 1 < 8 && r - 1 >= 0) bl |= 1ULL << ((r - 1) * 8 + (f + 1));
        if (f - 1 >= 0 && r - 1 >= 0) bl |= 1ULL << ((r - 1) * 8 + (f - 1));
        pawnAttacks[1][sq] = bl;
    }
}

int Board::popLSB(Bitboard &bb) {
    int sq = bsf64(bb);
    bb &= bb - 1;
    return sq;
}

int Board::bitScanForward(Bitboard bb) { return bsf64(bb); }
int Board::countBits(Bitboard bb) { return static_cast<int>(__popcnt64(bb)); }

Bitboard Board::slidingAttackRook(int sq, Bitboard occ) {
    Bitboard attacks = 0ULL;
    Bitboard r;
    // North
    r = rayN[sq]; Bitboard blockers = r & occ; if (blockers) { int b = bitScanForward(blockers); r &= ~rayN[b]; } attacks |= r;
    // South
    r = rayS[sq]; blockers = r & occ; if (blockers) { int b = bsr64(blockers); r &= ~rayS[b]; } attacks |= r;
    // East
    r = rayE[sq]; blockers = r & occ; if (blockers) { int b = bitScanForward(blockers); r &= ~rayE[b]; } attacks |= r;
    // West
    r = rayW[sq]; blockers = r & occ; if (blockers) { int b = bsr64(blockers); r &= ~rayW[b]; } attacks |= r;
    return attacks;
}

Bitboard Board::slidingAttackBishop(int sq, Bitboard occ) {
    Bitboard attacks = 0ULL;
    Bitboard r;
    // NE
    r = rayNE[sq]; Bitboard blockers = r & occ; if (blockers) { int b = bitScanForward(blockers); r &= ~rayNE[b]; } attacks |= r;
    // NW
    r = rayNW[sq]; blockers = r & occ; if (blockers) { int b = bitScanForward(blockers); r &= ~rayNW[b]; } attacks |= r;
    // SE
    r = raySE[sq]; blockers = r & occ; if (blockers) { int b = bsr64(blockers); r &= ~raySE[b]; } attacks |= r;
    // SW
    r = raySW[sq]; blockers = r & occ; if (blockers) { int b = bsr64(blockers); r &= ~raySW[b]; } attacks |= r;
    return attacks;
}

Board::Board() {
    initAttackTables();
    initZobrist();
    setStartPos();
}

void Board::setEmpty() {
    for (auto &bb : pieces) bb = 0ULL;
    occWhite = occBlack = occAll = 0ULL;
    side = Color::White;
    castlingRights = 0;
    epSquare = -1;
    halfmoveClock = 0;
    fullmoveNumber = 1;
    history.clear();
    hash = 0;
}

bool Board::isWhite(Piece p) { return p >= Piece::WP && p <= Piece::WK; }
bool Board::isBlack(Piece p) { return p >= Piece::BP && p <= Piece::BK; }

int Board::pieceIndex(Piece p) {
    switch (p) {
        case Piece::WP: return 0; case Piece::WN: return 1; case Piece::WB: return 2; case Piece::WR: return 3; case Piece::WQ: return 4; case Piece::WK: return 5;
        case Piece::BP: return 6; case Piece::BN: return 7; case Piece::BB: return 8; case Piece::BR: return 9; case Piece::BQ: return 10; case Piece::BK: return 11;
        default: return -1;
    }
}

void Board::updateOccupancy() {
    occWhite = pieces[0] | pieces[1] | pieces[2] | pieces[3] | pieces[4] | pieces[5];
    occBlack = pieces[6] | pieces[7] | pieces[8] | pieces[9] | pieces[10] | pieces[11];
    occAll = occWhite | occBlack;
}

Piece Board::pieceAt(int sq) const {
    Bitboard mask = 1ULL << sq;
    for (int i = 0; i < 12; ++i) if (pieces[i] & mask) {
        return static_cast<Piece>(i + 1);
    }
    return Piece::None;
}

Piece Board::makePiece(Color c, Piece pKind) {
    if (pKind == Piece::None) return Piece::None;
    int base = (pKind == Piece::WP || pKind == Piece::BP) ? 0 :
               (pKind == Piece::WN || pKind == Piece::BN) ? 1 :
               (pKind == Piece::WB || pKind == Piece::BB) ? 2 :
               (pKind == Piece::WR || pKind == Piece::BR) ? 3 :
               (pKind == Piece::WQ || pKind == Piece::BQ) ? 4 : 5;
    static Piece whites[6] = { Piece::WP, Piece::WN, Piece::WB, Piece::WR, Piece::WQ, Piece::WK };
    static Piece blacks[6] = { Piece::BP, Piece::BN, Piece::BB, Piece::BR, Piece::BQ, Piece::BK };
    return c == Color::White ? whites[base] : blacks[base];
}

bool Board::setStartPos() { return setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

std::string Board::getFEN() const {
    std::ostringstream oss;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            int sq = r * 8 + f;
            Piece p = pieceAt(sq);
            if (p == Piece::None) {
                empty++;
            } else {
                if (empty) { oss << empty; empty = 0; }
                oss << pieceToFen(p);
            }
        }
        if (empty) oss << empty;
        if (r) oss << '/';
    }
    oss << ' ' << (side == Color::White ? 'w' : 'b') << ' ';
    if (castlingRights == 0) oss << '-';
    else {
        if (castlingRights & 1) oss << 'K';
        if (castlingRights & 2) oss << 'Q';
        if (castlingRights & 4) oss << 'k';
        if (castlingRights & 8) oss << 'q';
    }
    oss << ' ';
    if (epSquare == -1) oss << '-';
    else {
        char file = 'a' + (epSquare & 7);
        char rank = '1' + (epSquare >> 3);
        oss << file << rank;
    }
    oss << ' ' << halfmoveClock << ' ' << fullmoveNumber;
    return oss.str();
}

bool Board::setFromFEN(const std::string &fen) {
    setEmpty();
    std::istringstream iss(fen);
    std::string board, stm, castles, ep;
    if (!(iss >> board >> stm >> castles >> ep)) return false;
    int hm = 0, fm = 1;
    iss >> hm >> fm;
    int r = 7, f = 0;
    for (char c : board) {
        if (c == '/') { r--; f = 0; continue; }
        if (std::isdigit(static_cast<unsigned char>(c))) {
            f += c - '0';
        } else {
            if (f >= 8 || r < 0) return false;
            int sq = r * 8 + f;
            Piece p = fenToPiece(c);
            int idx = pieceIndex(p);
            if (idx < 0) return false;
            pieces[idx] |= 1ULL << sq;
            f++;
        }
    }
    updateOccupancy();
    side = (stm == "w") ? Color::White : Color::Black;
    castlingRights = 0;
    if (castles != "-") {
        for (char ch : castles) {
            if (ch == 'K') castlingRights |= 1;
            else if (ch == 'Q') castlingRights |= 2;
            else if (ch == 'k') castlingRights |= 4;
            else if (ch == 'q') castlingRights |= 8;
        }
    }
    if (ep == "-") epSquare = -1; else {
        if (ep.size() != 2) return false;
        int file = ep[0] - 'a';
        int rank = ep[1] - '1';
        epSquare = rank * 8 + file;
    }
    halfmoveClock = hm;
    fullmoveNumber = fm;
    hashRecompute();
    return true;
}

int Board::kingSquare(Color c) const {
    int idx = (c == Color::White) ? pieceIndex(Piece::WK) : pieceIndex(Piece::BK);
    Bitboard bb = pieces[idx];
    if (bb == 0) return -1;
    return bitScanForward(bb);
}

bool Board::isSquareAttacked(int sq, Color byColor) const {
    if (byColor == Color::White) {
        if (pawnAttacks[0][sq] & pieces[pieceIndex(Piece::WP)]) return true;
        if (knightAttacks[sq] & pieces[pieceIndex(Piece::WN)]) return true;
        if (kingAttacks[sq] & pieces[pieceIndex(Piece::WK)]) return true;
        Bitboard bs = slidingAttackBishop(sq, occAll) & (pieces[pieceIndex(Piece::WB)] | pieces[pieceIndex(Piece::WQ)]);
        if (bs) return true;
        Bitboard rs = slidingAttackRook(sq, occAll) & (pieces[pieceIndex(Piece::WR)] | pieces[pieceIndex(Piece::WQ)]);
        if (rs) return true;
    } else {
        if (pawnAttacks[1][sq] & pieces[pieceIndex(Piece::BP)]) return true;
        if (knightAttacks[sq] & pieces[pieceIndex(Piece::BN)]) return true;
        if (kingAttacks[sq] & pieces[pieceIndex(Piece::BK)]) return true;
        Bitboard bs = slidingAttackBishop(sq, occAll) & (pieces[pieceIndex(Piece::BB)] | pieces[pieceIndex(Piece::BQ)]);
        if (bs) return true;
        Bitboard rs = slidingAttackRook(sq, occAll) & (pieces[pieceIndex(Piece::BR)] | pieces[pieceIndex(Piece::BQ)]);
        if (rs) return true;
    }
    return false;
}

bool Board::inCheck(Color c) const {
    int ksq = kingSquare(c);
    if (ksq == -1) return false;
    Color them = (c == Color::White) ? Color::Black : Color::White;
    return isSquareAttacked(ksq, them);
}

void Board::addMove(std::vector<Move> &moves, int from, int to, Piece promo, bool ep, bool castle) const {
    moves.push_back(Move{from, to, promo, ep, castle});
}

void Board::generatePseudoMoves(std::vector<Move> &moves) const {
    Color c = side;
    Bitboard us = (c == Color::White) ? occWhite : occBlack;
    Bitboard them = (c == Color::White) ? occBlack : occWhite;
    (void)them;

    // Pawns
    generatePawnMoves(moves, c);

    // Knights
    Bitboard knights = (c == Color::White) ? pieces[pieceIndex(Piece::WN)] : pieces[pieceIndex(Piece::BN)];
    Bitboard notUs = ~us;
    Bitboard tmp = knights;
    while (tmp) {
        int from = popLSB(tmp);
        Bitboard movesMask = knightAttacks[from] & notUs;
        Bitboard mm = movesMask;
        while (mm) { int to = popLSB(mm); addMove(moves, from, to); }
    }

    // Sliding: bishops, rooks, queens
    auto genSliding = [&](Bitboard bb, bool rookLike, bool bishopLike) {
        Bitboard t = bb;
        while (t) {
            int from = popLSB(t);
            Bitboard attacks = 0ULL;
            if (rookLike) attacks |= slidingAttackRook(from, occAll);
            if (bishopLike) attacks |= slidingAttackBishop(from, occAll);
            Bitboard movesMask = attacks & ~us;
            Bitboard mm = movesMask; while (mm) { int to = popLSB(mm); addMove(moves, from, to); }
        }
    };
    if (c == Color::White) {
        genSliding(pieces[pieceIndex(Piece::WB)], false, true);
        genSliding(pieces[pieceIndex(Piece::WR)], true, false);
        genSliding(pieces[pieceIndex(Piece::WQ)], true, true);
    } else {
        genSliding(pieces[pieceIndex(Piece::BB)], false, true);
        genSliding(pieces[pieceIndex(Piece::BR)], true, false);
        genSliding(pieces[pieceIndex(Piece::BQ)], true, true);
    }

    // King
    int ksq = kingSquare(c);
    if (ksq != -1) {
        Bitboard km = kingAttacks[ksq] & ~us;
        Bitboard mm2 = km; while (mm2) { int to = popLSB(mm2); addMove(moves, ksq, to); }
        // Castling
        if (c == Color::White) {
            // King-side: e1->g1 (4->6), empty f1(5), g1(6), rook at h1(7)
            if ((castlingRights & 1) && !(occAll & ((1ULL<<5)|(1ULL<<6))) && !isSquareAttacked(4, Color::Black) && !isSquareAttacked(5, Color::Black) && !isSquareAttacked(6, Color::Black) && (pieces[pieceIndex(Piece::WR)] & (1ULL<<7))) {
                addMove(moves, 4, 6, Piece::None, false, true);
            }
            // Queen-side: e1->c1 (4->2), empty d1(3), c1(2), b1(1), rook at a1(0)
            if ((castlingRights & 2) && !(occAll & ((1ULL<<3)|(1ULL<<2)|(1ULL<<1))) && !isSquareAttacked(4, Color::Black) && !isSquareAttacked(3, Color::Black) && !isSquareAttacked(2, Color::Black) && (pieces[pieceIndex(Piece::WR)] & (1ULL<<0))) {
                addMove(moves, 4, 2, Piece::None, false, true);
            }
        } else {
            // King-side: e8->g8 (60->62)
            if ((castlingRights & 4) && !(occAll & ((1ULL<<61)|(1ULL<<62))) && !isSquareAttacked(60, Color::White) && !isSquareAttacked(61, Color::White) && !isSquareAttacked(62, Color::White) && (pieces[pieceIndex(Piece::BR)] & (1ULL<<63))) {
                addMove(moves, 60, 62, Piece::None, false, true);
            }
            // Queen-side: e8->c8 (60->58)
            if ((castlingRights & 8) && !(occAll & ((1ULL<<59)|(1ULL<<58)|(1ULL<<57))) && !isSquareAttacked(60, Color::White) && !isSquareAttacked(59, Color::White) && !isSquareAttacked(58, Color::White) && (pieces[pieceIndex(Piece::BR)] & (1ULL<<56))) {
                addMove(moves, 60, 58, Piece::None, false, true);
            }
        }
    }
}

void Board::generatePawnMoves(std::vector<Move> &moves, Color c) const {
    if (c == Color::White) {
        Bitboard pawns = pieces[pieceIndex(Piece::WP)];
        Bitboard single = (pawns << 8) & ~occAll;
        Bitboard promoSingle = single & RANK_8;
        Bitboard nonPromoSingle = single & ~RANK_8;
        Bitboard rank2 = RANK_2 & pawns;
        Bitboard dblPush = ((rank2 << 8) & ~occAll) << 8 & ~occAll;
        Bitboard tmp = nonPromoSingle; while (tmp) { int to = popLSB(tmp); addMove(moves, to - 8, to); }
        tmp = promoSingle; while (tmp) { int to = popLSB(tmp); addMove(moves, to - 8, to, Piece::WQ); addMove(moves, to - 8, to, Piece::WR); addMove(moves, to - 8, to, Piece::WB); addMove(moves, to - 8, to, Piece::WN); }
        tmp = dblPush; while (tmp) { int to = popLSB(tmp); addMove(moves, to - 16, to); }
        Bitboard capL = (pawns << 7) & ~FILE_H & occBlack;
        Bitboard capR = (pawns << 9) & ~FILE_A & occBlack;
        Bitboard promoCapL = capL & RANK_8; Bitboard promoCapR = capR & RANK_8;
        Bitboard nonPromoCapL = capL & ~RANK_8; Bitboard nonPromoCapR = capR & ~RANK_8;
        tmp = nonPromoCapL; while (tmp) { int to = popLSB(tmp); addMove(moves, to - 7, to); }
        tmp = nonPromoCapR; while (tmp) { int to = popLSB(tmp); addMove(moves, to - 9, to); }
        tmp = promoCapL; while (tmp) { int to = popLSB(tmp); int from = to - 7; addMove(moves, from, to, Piece::WQ); addMove(moves, from, to, Piece::WR); addMove(moves, from, to, Piece::WB); addMove(moves, from, to, Piece::WN); }
        tmp = promoCapR; while (tmp) { int to = popLSB(tmp); int from = to - 9; addMove(moves, from, to, Piece::WQ); addMove(moves, from, to, Piece::WR); addMove(moves, from, to, Piece::WB); addMove(moves, from, to, Piece::WN); }
        if (epSquare != -1) {
            Bitboard epMask = 1ULL << epSquare;
            Bitboard epL = (pawns << 7) & ~FILE_H & epMask; if (epL) { int to = epSquare; addMove(moves, to - 7, to, Piece::None, true, false); }
            Bitboard epR = (pawns << 9) & ~FILE_A & epMask; if (epR) { int to = epSquare; addMove(moves, to - 9, to, Piece::None, true, false); }
        }
    } else {
        Bitboard pawns = pieces[pieceIndex(Piece::BP)];
        Bitboard single = (pawns >> 8) & ~occAll;
        Bitboard promoSingle = single & RANK_1;
        Bitboard nonPromoSingle = single & ~RANK_1;
        Bitboard rank7 = RANK_7 & pawns;
        Bitboard dblPush = ((rank7 >> 8) & ~occAll) >> 8 & ~occAll;
        Bitboard tmp = nonPromoSingle; while (tmp) { int to = popLSB(tmp); addMove(moves, to + 8, to); }
        tmp = promoSingle; while (tmp) { int to = popLSB(tmp); addMove(moves, to + 8, to, Piece::BQ); addMove(moves, to + 8, to, Piece::BR); addMove(moves, to + 8, to, Piece::BB); addMove(moves, to + 8, to, Piece::BN); }
        tmp = dblPush; while (tmp) { int to = popLSB(tmp); addMove(moves, to + 16, to); }
        Bitboard capL = (pawns >> 9) & ~FILE_H & occWhite;
        Bitboard capR = (pawns >> 7) & ~FILE_A & occWhite;
        Bitboard promoCapL = capL & RANK_1; Bitboard promoCapR = capR & RANK_1;
        Bitboard nonPromoCapL = capL & ~RANK_1; Bitboard nonPromoCapR = capR & ~RANK_1;
        tmp = nonPromoCapL; while (tmp) { int to = popLSB(tmp); addMove(moves, to + 9, to); }
        tmp = nonPromoCapR; while (tmp) { int to = popLSB(tmp); addMove(moves, to + 7, to); }
        tmp = promoCapL; while (tmp) { int to = popLSB(tmp); int from = to + 9; addMove(moves, from, to, Piece::BQ); addMove(moves, from, to, Piece::BR); addMove(moves, from, to, Piece::BB); addMove(moves, from, to, Piece::BN); }
        tmp = promoCapR; while (tmp) { int to = popLSB(tmp); int from = to + 7; addMove(moves, from, to, Piece::BQ); addMove(moves, from, to, Piece::BR); addMove(moves, from, to, Piece::BB); addMove(moves, from, to, Piece::BN); }
        if (epSquare != -1) {
            Bitboard epMask = 1ULL << epSquare;
            Bitboard epL = (pawns >> 9) & ~FILE_H & epMask; if (epL) { int to = epSquare; addMove(moves, to + 9, to, Piece::None, true, false); }
            Bitboard epR = (pawns >> 7) & ~FILE_A & epMask; if (epR) { int to = epSquare; addMove(moves, to + 7, to, Piece::None, true, false); }
        }
    }
}

std::vector<Move> Board::generateLegalMoves() const {
    std::vector<Move> pseudo;
    pseudo.reserve(128);
    generatePseudoMoves(pseudo);
    std::vector<Move> legal;
    Board tmp = *this;
    Color us = side;
    Color them = (us == Color::White) ? Color::Black : Color::White;
    for (const Move &m : pseudo) {
        tmp.makeMove(m);
        int ksq = tmp.kingSquare(us);
        if (ksq != -1 && !tmp.isSquareAttacked(ksq, them)) legal.push_back(m);
        tmp.unmakeMove();
    }
    return legal;
}

void Board::makeMove(const Move &m) {
    HistoryEntry he{};
    he.move = m;
    he.castlingRights = castlingRights;
    he.epSquare = epSquare;
    he.halfmoveClock = halfmoveClock;
    he.side = side;
    he.occWhite = occWhite; he.occBlack = occBlack; he.occAll = occAll;
    he.pieces = pieces;

    Piece moving = pieceAt(m.from);
    he.captured = Piece::None;
    he.prevHash = hash;

    // Reset EP by default (update hash)
    if (epSquare != -1) hashSetEp(epSquare, -1);
    epSquare = -1;

    // Captures
    if (m.isEnPassant) {
        int capSq = (side == Color::White) ? (m.to - 8) : (m.to + 8);
        he.captured = (side == Color::White) ? Piece::BP : Piece::WP;
        int capIdx = pieceIndex(he.captured);
        pieces[capIdx] &= ~(1ULL << capSq);
    } else {
        Piece cap = pieceAt(m.to);
        if (cap != Piece::None) {
            he.captured = cap;
            pieces[pieceIndex(cap)] &= ~(1ULL << m.to);
        }
    }

    // Move piece
    int mi = pieceIndex(moving);
    pieces[mi] &= ~(1ULL << m.from);
    hashTogglePiece(moving, m.from);
    Piece placed = moving;
    if (m.promotion != Piece::None) placed = m.promotion;
    pieces[pieceIndex(placed)] |= (1ULL << m.to);
    hashTogglePiece(placed, m.to);

    // Castling rook move
    if (m.isCastling) {
        if (moving == Piece::WK && m.to == 6) { // white king side
            pieces[pieceIndex(Piece::WR)] &= ~(1ULL << 7);
            pieces[pieceIndex(Piece::WR)] |= (1ULL << 5);
        } else if (moving == Piece::WK && m.to == 2) { // white queen side
            pieces[pieceIndex(Piece::WR)] &= ~(1ULL << 0);
            pieces[pieceIndex(Piece::WR)] |= (1ULL << 3);
        } else if (moving == Piece::BK && m.to == 62) { // black king side
            pieces[pieceIndex(Piece::BR)] &= ~(1ULL << 63);
            pieces[pieceIndex(Piece::BR)] |= (1ULL << 61);
        } else if (moving == Piece::BK && m.to == 58) { // black queen side
            pieces[pieceIndex(Piece::BR)] &= ~(1ULL << 56);
            pieces[pieceIndex(Piece::BR)] |= (1ULL << 59);
        }
    }

    // Update castling rights
    auto disableCastlingBySquare = [&](int sq) {
        uint8_t before = castlingRights;
        if (sq == 4) { castlingRights &= ~(1|2); }
        if (sq == 7) { castlingRights &= ~1; }
        if (sq == 0) { castlingRights &= ~2; }
        if (sq == 60) { castlingRights &= ~(4|8); }
        if (sq == 63) { castlingRights &= ~4; }
        if (sq == 56) { castlingRights &= ~8; }
        if (before != castlingRights) { hashToggleCastle(before); hashToggleCastle(castlingRights); }
    };
    disableCastlingBySquare(m.from);
    disableCastlingBySquare(m.to);

    // Set EP square if double pawn move
    if (moving == Piece::WP && (m.to - m.from) == 16) { epSquare = m.from + 8; hashSetEp(-1, epSquare); }
    else if (moving == Piece::BP && (m.from - m.to) == 16) { epSquare = m.from - 8; hashSetEp(-1, epSquare); }

    // Halfmove clock
    if (moving == Piece::WP || moving == Piece::BP || he.captured != Piece::None) halfmoveClock = 0; else halfmoveClock++;

    updateOccupancy();

    if (side == Color::Black) fullmoveNumber++;
    side = (side == Color::White) ? Color::Black : Color::White;
    hashToggleSide();

    history.push_back(he);
}

void Board::unmakeMove() {
    if (history.empty()) return;
    HistoryEntry he = history.back();
    history.pop_back();
    pieces = he.pieces;
    occWhite = he.occWhite; occBlack = he.occBlack; occAll = he.occAll;
    side = he.side;
    castlingRights = he.castlingRights;
    epSquare = he.epSquare;
    halfmoveClock = he.halfmoveClock;
    hash = he.prevHash;
    if (side == Color::Black) fullmoveNumber--; // we incremented when side was Black before the move
}

static std::string squareToAlg(int sq) {
    char f = 'a' + (sq & 7);
    char r = '1' + (sq >> 3);
    return std::string{f, r};
}

std::string Board::moveToUci(const Move &m) {
    std::string s;
    s += squareToAlg(m.from);
    s += squareToAlg(m.to);
    if (m.promotion != Piece::None) {
        char c = 'q';
        switch (m.promotion) {
            case Piece::WQ: case Piece::BQ: c = 'q'; break;
            case Piece::WR: case Piece::BR: c = 'r'; break;
            case Piece::WB: case Piece::BB: c = 'b'; break;
            case Piece::WN: case Piece::BN: c = 'n'; break;
            default: break;
        }
        s += c;
    }
    return s;
}

int Board::algebraicTo0x88(const std::string &alg) {
    if (alg.size() != 2) return -1;
    int file = alg[0] - 'a';
    int rank = alg[1] - '1';
    if (file < 0 || file > 7 || rank < 0 || rank > 7) return -1;
    return rank * 8 + file;
}

bool Board::applyMovesUCI(const std::vector<std::string> &uciMoves) {
    for (const std::string &mstr : uciMoves) {
        if (mstr.size() < 4) return false;
        int from = algebraicTo0x88(mstr.substr(0,2));
        int to = algebraicTo0x88(mstr.substr(2,2));
        if (from == -1 || to == -1) return false;
        Piece promo = Piece::None;
        if (mstr.size() == 5) {
            char pc = std::tolower(static_cast<unsigned char>(mstr[4]));
            if (pc == 'q') promo = (side == Color::White) ? Piece::WQ : Piece::BQ;
            else if (pc == 'r') promo = (side == Color::White) ? Piece::WR : Piece::BR;
            else if (pc == 'b') promo = (side == Color::White) ? Piece::WB : Piece::BB;
            else if (pc == 'n') promo = (side == Color::White) ? Piece::WN : Piece::BN;
        }
        auto legal = generateLegalMoves();
        bool found = false;
        for (const auto &mv : legal) {
            if (mv.from == from && mv.to == to) {
                if ((promo == Piece::None && mv.promotion == Piece::None) || (promo != Piece::None && mv.promotion == promo)) {
                    makeMove(mv);
                    found = true;
                    break;
                }
            }
        }
        if (!found) return false;
    }
    return true;
}

// Zobrist implementation
void Board::initZobrist() {
    if (zobristInit) return;
    zobristInit = true;
    std::mt19937_64 rng(0x9E3779B97F4A7C15ull);
    auto rnd = [&]() { return rng(); };
    for (int p = 0; p < 12; ++p) for (int sq = 0; sq < 64; ++sq) zPiece[p][sq] = rnd();
    zSide = rnd();
    for (int i = 0; i < 16; ++i) zCastle[i] = rnd();
    for (int f = 0; f < 8; ++f) zEnpassant[f] = rnd();
}

void Board::hashRecompute() {
    hash = 0;
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = pieceAt(sq);
        if (p != Piece::None) hashTogglePiece(p, sq);
    }
    hashToggleCastle(castlingRights);
    if (epSquare != -1) hashSetEp(-1, epSquare);
    if (side == Color::Black) hashToggleSide();
}

