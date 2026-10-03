#include "Board.h"

#include "BitOps.h"

#include <algorithm>
#include <cctype>
#include <random>
#include <sstream>

namespace {

char pieceToFen(Piece p) {
    static const char chars[] = " PNBRQKpnbrqk";
    return chars[static_cast<int>(p)];
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

constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_H = 0x8080808080808080ULL;
constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_2 = 0x000000000000FF00ULL;
constexpr Bitboard RANK_7 = 0x00FF000000000000ULL;
constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;

constexpr int WP_ = 0, WN_ = 1, WB_ = 2, WR_ = 3, WQ_ = 4, WK_ = 5;
constexpr int BP_ = 6, BN_ = 7, BB_ = 8, BR_ = 9, BQ_ = 10, BK_ = 11;

// Ray tables for sliding pieces
Bitboard rayN[64], rayS[64], rayE[64], rayW[64], rayNE[64], rayNW[64], raySE[64], raySW[64];

} // namespace

Bitboard Board::knightAttacks[64];
Bitboard Board::kingAttacks[64];
Bitboard Board::pawnAttacks[2][64];
uint64_t Board::zPiece[12][64];
uint64_t Board::zSide;
uint64_t Board::zCastle[16];
uint64_t Board::zEnpassant[8];

void Board::initTables() {
    for (int sq = 0; sq < 64; ++sq) {
        int f = sq & 7, r = sq >> 3;
        Bitboard b;
        b = 0; for (int rr = r + 1; rr <= 7; ++rr) b |= 1ULL << (rr * 8 + f); rayN[sq] = b;
        b = 0; for (int rr = r - 1; rr >= 0; --rr) b |= 1ULL << (rr * 8 + f); rayS[sq] = b;
        b = 0; for (int ff = f + 1; ff <= 7; ++ff) b |= 1ULL << (r * 8 + ff); rayE[sq] = b;
        b = 0; for (int ff = f - 1; ff >= 0; --ff) b |= 1ULL << (r * 8 + ff); rayW[sq] = b;
        b = 0; for (int ff = f + 1, rr = r + 1; ff <= 7 && rr <= 7; ++ff, ++rr) b |= 1ULL << (rr * 8 + ff); rayNE[sq] = b;
        b = 0; for (int ff = f - 1, rr = r + 1; ff >= 0 && rr <= 7; --ff, ++rr) b |= 1ULL << (rr * 8 + ff); rayNW[sq] = b;
        b = 0; for (int ff = f + 1, rr = r - 1; ff <= 7 && rr >= 0; ++ff, --rr) b |= 1ULL << (rr * 8 + ff); raySE[sq] = b;
        b = 0; for (int ff = f - 1, rr = r - 1; ff >= 0 && rr >= 0; --ff, --rr) b |= 1ULL << (rr * 8 + ff); raySW[sq] = b;

        Bitboard km = 0, nm = 0;
        for (int df = -1; df <= 1; ++df) for (int dr = -1; dr <= 1; ++dr) {
            if (df == 0 && dr == 0) continue;
            int nf = f + df, nr = r + dr;
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) km |= 1ULL << (nr * 8 + nf);
        }
        static const int kdf[8] = {1, 2, 2, 1, -1, -2, -2, -1};
        static const int kdr[8] = {2, 1, -1, -2, -2, -1, 1, 2};
        for (int i = 0; i < 8; ++i) {
            int nf = f + kdf[i], nr = r + kdr[i];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) nm |= 1ULL << (nr * 8 + nf);
        }
        knightAttacks[sq] = nm;
        kingAttacks[sq] = km;

        Bitboard w = 0, bl = 0;
        if (f + 1 < 8 && r + 1 < 8) w |= 1ULL << ((r + 1) * 8 + (f + 1));
        if (f - 1 >= 0 && r + 1 < 8) w |= 1ULL << ((r + 1) * 8 + (f - 1));
        if (f + 1 < 8 && r - 1 >= 0) bl |= 1ULL << ((r - 1) * 8 + (f + 1));
        if (f - 1 >= 0 && r - 1 >= 0) bl |= 1ULL << ((r - 1) * 8 + (f - 1));
        pawnAttacks[0][sq] = w;
        pawnAttacks[1][sq] = bl;
    }

    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
    for (auto &row : zPiece) for (auto &z : row) z = rng();
    zSide = rng();
    for (auto &z : zCastle) z = rng();
    for (auto &z : zEnpassant) z = rng();
}

Bitboard Board::slidingAttackRook(int sq, Bitboard occ) {
    Bitboard attacks = 0, r, blockers;
    r = rayN[sq]; blockers = r & occ; if (blockers) r &= ~rayN[lsb(blockers)]; attacks |= r;
    r = rayS[sq]; blockers = r & occ; if (blockers) r &= ~rayS[msb(blockers)]; attacks |= r;
    r = rayE[sq]; blockers = r & occ; if (blockers) r &= ~rayE[lsb(blockers)]; attacks |= r;
    r = rayW[sq]; blockers = r & occ; if (blockers) r &= ~rayW[msb(blockers)]; attacks |= r;
    return attacks;
}

Bitboard Board::slidingAttackBishop(int sq, Bitboard occ) {
    Bitboard attacks = 0, r, blockers;
    r = rayNE[sq]; blockers = r & occ; if (blockers) r &= ~rayNE[lsb(blockers)]; attacks |= r;
    r = rayNW[sq]; blockers = r & occ; if (blockers) r &= ~rayNW[lsb(blockers)]; attacks |= r;
    r = raySE[sq]; blockers = r & occ; if (blockers) r &= ~raySE[msb(blockers)]; attacks |= r;
    r = raySW[sq]; blockers = r & occ; if (blockers) r &= ~raySW[msb(blockers)]; attacks |= r;
    return attacks;
}

Board::Board() {
    static const bool tablesReady = (initTables(), true); // thread-safe one-time init
    (void)tablesReady;
    setStartPos();
}

Board::Board(const Board &o, NoHistory)
    : pieces(o.pieces), occWhite(o.occWhite), occBlack(o.occBlack), occAll(o.occAll), side(o.side),
      castlingRights(o.castlingRights), epSquare(o.epSquare), halfmoveClock(o.halfmoveClock),
      fullmoveNumber(o.fullmoveNumber), hash(o.hash) {
    history.reserve(1);
}

void Board::updateOccupancy() {
    occWhite = pieces[WP_] | pieces[WN_] | pieces[WB_] | pieces[WR_] | pieces[WQ_] | pieces[WK_];
    occBlack = pieces[BP_] | pieces[BN_] | pieces[BB_] | pieces[BR_] | pieces[BQ_] | pieces[BK_];
    occAll = occWhite | occBlack;
}

Piece Board::pieceAt(int sq) const {
    Bitboard mask = 1ULL << sq;
    if (!(occAll & mask)) return Piece::None;
    for (int i = 0; i < 12; ++i)
        if (pieces[i] & mask) return static_cast<Piece>(i + 1);
    return Piece::None;
}

bool Board::setStartPos() { return setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"); }

std::string Board::getFEN() const {
    std::ostringstream oss;
    for (int r = 7; r >= 0; --r) {
        int empty = 0;
        for (int f = 0; f < 8; ++f) {
            Piece p = pieceAt(r * 8 + f);
            if (p == Piece::None) { ++empty; continue; }
            if (empty) { oss << empty; empty = 0; }
            oss << pieceToFen(p);
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
    oss << ' ' << (epSquare == -1 ? std::string("-") : squareName(epSquare));
    oss << ' ' << halfmoveClock << ' ' << fullmoveNumber;
    return oss.str();
}

bool Board::setFromFEN(const std::string &fen) {
    std::istringstream iss(fen);
    std::string placement, stm, castles, ep;
    if (!(iss >> placement >> stm >> castles >> ep)) return false;
    int hm = 0, fm = 1;
    if (!(iss >> hm)) hm = 0;
    if (!(iss >> fm)) fm = 1;
    if (hm < 0 || fm < 1) return false;

    Board b(*this, NoHistory{});
    b.pieces.fill(0);
    int r = 7, f = 0;
    for (char c : placement) {
        if (c == '/') {
            if (f != 8 || r == 0) return false;
            --r; f = 0;
            continue;
        }
        if (c >= '1' && c <= '8') {
            f += c - '0';
            if (f > 8) return false;
            continue;
        }
        Piece p = fenToPiece(c);
        if (p == Piece::None || f >= 8) return false;
        b.pieces[pieceIndex(p)] |= 1ULL << (r * 8 + f);
        ++f;
    }
    if (r != 0 || f != 8) return false;
    if (popcount(b.pieces[WK_]) != 1 || popcount(b.pieces[BK_]) != 1) return false;
    if ((b.pieces[WP_] | b.pieces[BP_]) & (RANK_1 | RANK_8)) return false;
    b.updateOccupancy();

    if (stm == "w") b.side = Color::White;
    else if (stm == "b") b.side = Color::Black;
    else return false;

    b.castlingRights = 0;
    if (castles != "-") {
        for (char ch : castles) {
            if (ch == 'K') b.castlingRights |= 1;
            else if (ch == 'Q') b.castlingRights |= 2;
            else if (ch == 'k') b.castlingRights |= 4;
            else if (ch == 'q') b.castlingRights |= 8;
            else return false;
        }
    }
    b.sanitizeCastlingRights();

    b.epSquare = -1;
    if (ep != "-") {
        int sq = parseSquare(ep);
        if (sq < 0 || ep.size() != 2) return false;
        if (rankOf(sq) != (b.side == Color::White ? 5 : 2)) return false;
        b.epSquare = sq;
    }
    // The side that just moved must not be in check.
    if (b.inCheck(opposite(b.side))) return false;

    b.halfmoveClock = hm;
    b.fullmoveNumber = fm;
    b.hashRecompute();
    *this = std::move(b);
    history.clear();
    return true;
}

// Drop castling rights whose king or rook is not on its home square, so that
// castling can never conjure a rook out of thin air.
void Board::sanitizeCastlingRights() {
    if (!(pieces[WK_] & (1ULL << 4))) castlingRights &= ~3;
    if (!(pieces[WR_] & (1ULL << 7))) castlingRights &= ~1;
    if (!(pieces[WR_] & (1ULL << 0))) castlingRights &= ~2;
    if (!(pieces[BK_] & (1ULL << 60))) castlingRights &= ~12;
    if (!(pieces[BR_] & (1ULL << 63))) castlingRights &= ~4;
    if (!(pieces[BR_] & (1ULL << 56))) castlingRights &= ~8;
}

int Board::kingSquare(Color c) const {
    Bitboard bb = pieces[c == Color::White ? WK_ : BK_];
    return bb ? lsb(bb) : -1;
}

bool Board::isSquareAttacked(int sq, Color byColor) const {
    // A white pawn attacks sq if it stands where a black pawn on sq would
    // attack, and vice versa, so the pawn table is indexed by the *other* colour.
    if (byColor == Color::White) {
        if (pawnAttacks[1][sq] & pieces[WP_]) return true;
        if (knightAttacks[sq] & pieces[WN_]) return true;
        if (kingAttacks[sq] & pieces[WK_]) return true;
        if (slidingAttackBishop(sq, occAll) & (pieces[WB_] | pieces[WQ_])) return true;
        if (slidingAttackRook(sq, occAll) & (pieces[WR_] | pieces[WQ_])) return true;
    } else {
        if (pawnAttacks[0][sq] & pieces[BP_]) return true;
        if (knightAttacks[sq] & pieces[BN_]) return true;
        if (kingAttacks[sq] & pieces[BK_]) return true;
        if (slidingAttackBishop(sq, occAll) & (pieces[BB_] | pieces[BQ_])) return true;
        if (slidingAttackRook(sq, occAll) & (pieces[BR_] | pieces[BQ_])) return true;
    }
    return false;
}

bool Board::inCheck(Color c) const {
    int ksq = kingSquare(c);
    if (ksq == -1) return false;
    return isSquareAttacked(ksq, opposite(c));
}

void Board::generatePseudoMoves(std::vector<Move> &moves) const {
    Color c = side;
    Bitboard us = (c == Color::White) ? occWhite : occBlack;
    int base = (c == Color::White) ? 0 : 6;

    generatePawnMoves(moves, c);

    Bitboard knights = pieces[base + 1];
    while (knights) {
        int from = popLsb(knights);
        Bitboard mm = knightAttacks[from] & ~us;
        while (mm) addMove(moves, from, popLsb(mm));
    }

    auto genSliding = [&](Bitboard bb, bool rookLike, bool bishopLike) {
        while (bb) {
            int from = popLsb(bb);
            Bitboard attacks = 0;
            if (rookLike) attacks |= slidingAttackRook(from, occAll);
            if (bishopLike) attacks |= slidingAttackBishop(from, occAll);
            Bitboard mm = attacks & ~us;
            while (mm) addMove(moves, from, popLsb(mm));
        }
    };
    genSliding(pieces[base + 2], false, true);
    genSliding(pieces[base + 3], true, false);
    genSliding(pieces[base + 4], true, true);

    int ksq = kingSquare(c);
    if (ksq == -1) return;
    Bitboard km = kingAttacks[ksq] & ~us;
    while (km) addMove(moves, ksq, popLsb(km));

    Color them = opposite(c);
    if (c == Color::White) {
        if ((castlingRights & 1) && !(occAll & ((1ULL << 5) | (1ULL << 6))) && (pieces[WR_] & (1ULL << 7)) &&
            !isSquareAttacked(4, them) && !isSquareAttacked(5, them) && !isSquareAttacked(6, them))
            addMove(moves, 4, 6, Piece::None, false, true);
        if ((castlingRights & 2) && !(occAll & ((1ULL << 3) | (1ULL << 2) | (1ULL << 1))) && (pieces[WR_] & 1ULL) &&
            !isSquareAttacked(4, them) && !isSquareAttacked(3, them) && !isSquareAttacked(2, them))
            addMove(moves, 4, 2, Piece::None, false, true);
    } else {
        if ((castlingRights & 4) && !(occAll & ((1ULL << 61) | (1ULL << 62))) && (pieces[BR_] & (1ULL << 63)) &&
            !isSquareAttacked(60, them) && !isSquareAttacked(61, them) && !isSquareAttacked(62, them))
            addMove(moves, 60, 62, Piece::None, false, true);
        if ((castlingRights & 8) && !(occAll & ((1ULL << 59) | (1ULL << 58) | (1ULL << 57))) && (pieces[BR_] & (1ULL << 56)) &&
            !isSquareAttacked(60, them) && !isSquareAttacked(59, them) && !isSquareAttacked(58, them))
            addMove(moves, 60, 58, Piece::None, false, true);
    }
}

void Board::generatePawnMoves(std::vector<Move> &moves, Color c) const {
    auto addPromos = [&](int from, int to) {
        addMove(moves, from, to, makePiece(c, QUEEN));
        addMove(moves, from, to, makePiece(c, ROOK));
        addMove(moves, from, to, makePiece(c, BISHOP));
        addMove(moves, from, to, makePiece(c, KNIGHT));
    };
    Bitboard tmp;
    if (c == Color::White) {
        Bitboard pawns = pieces[WP_];
        Bitboard single = (pawns << 8) & ~occAll;
        Bitboard dbl = ((single & (RANK_2 << 8)) << 8) & ~occAll;
        tmp = single & ~RANK_8; while (tmp) { int to = popLsb(tmp); addMove(moves, to - 8, to); }
        tmp = single & RANK_8;  while (tmp) { int to = popLsb(tmp); addPromos(to - 8, to); }
        tmp = dbl;              while (tmp) { int to = popLsb(tmp); addMove(moves, to - 16, to); }
        Bitboard capL = (pawns << 7) & ~FILE_H & occBlack;
        Bitboard capR = (pawns << 9) & ~FILE_A & occBlack;
        tmp = capL & ~RANK_8; while (tmp) { int to = popLsb(tmp); addMove(moves, to - 7, to); }
        tmp = capR & ~RANK_8; while (tmp) { int to = popLsb(tmp); addMove(moves, to - 9, to); }
        tmp = capL & RANK_8;  while (tmp) { int to = popLsb(tmp); addPromos(to - 7, to); }
        tmp = capR & RANK_8;  while (tmp) { int to = popLsb(tmp); addPromos(to - 9, to); }
        if (epSquare != -1) {
            Bitboard epMask = 1ULL << epSquare;
            if ((pawns << 7) & ~FILE_H & epMask) addMove(moves, epSquare - 7, epSquare, Piece::None, true);
            if ((pawns << 9) & ~FILE_A & epMask) addMove(moves, epSquare - 9, epSquare, Piece::None, true);
        }
    } else {
        Bitboard pawns = pieces[BP_];
        Bitboard single = (pawns >> 8) & ~occAll;
        Bitboard dbl = ((single & (RANK_7 >> 8)) >> 8) & ~occAll;
        tmp = single & ~RANK_1; while (tmp) { int to = popLsb(tmp); addMove(moves, to + 8, to); }
        tmp = single & RANK_1;  while (tmp) { int to = popLsb(tmp); addPromos(to + 8, to); }
        tmp = dbl;              while (tmp) { int to = popLsb(tmp); addMove(moves, to + 16, to); }
        Bitboard capL = (pawns >> 9) & ~FILE_H & occWhite;
        Bitboard capR = (pawns >> 7) & ~FILE_A & occWhite;
        tmp = capL & ~RANK_1; while (tmp) { int to = popLsb(tmp); addMove(moves, to + 9, to); }
        tmp = capR & ~RANK_1; while (tmp) { int to = popLsb(tmp); addMove(moves, to + 7, to); }
        tmp = capL & RANK_1;  while (tmp) { int to = popLsb(tmp); addPromos(to + 9, to); }
        tmp = capR & RANK_1;  while (tmp) { int to = popLsb(tmp); addPromos(to + 7, to); }
        if (epSquare != -1) {
            Bitboard epMask = 1ULL << epSquare;
            if ((pawns >> 9) & ~FILE_H & epMask) addMove(moves, epSquare + 9, epSquare, Piece::None, true);
            if ((pawns >> 7) & ~FILE_A & epMask) addMove(moves, epSquare + 7, epSquare, Piece::None, true);
        }
    }
}

void Board::generatePseudoLegalMoves(std::vector<Move> &out, bool noisyOnly) const {
    out.clear();
    out.reserve(128);
    generatePseudoMoves(out);
    if (noisyOnly) {
        out.erase(std::remove_if(out.begin(), out.end(),
                                 [&](const Move &m) {
                                     return m.promotion == Piece::None && !m.isEnPassant && !(occAll & (1ULL << m.to));
                                 }),
                  out.end());
    }
}

std::vector<Move> Board::generateLegalMoves(bool noisyOnly) const {
    std::vector<Move> pseudo;
    pseudo.reserve(128);
    generatePseudoMoves(pseudo);
    std::vector<Move> legal;
    legal.reserve(pseudo.size());
    Board tmp(*this, NoHistory{}); // copying the game history here would cost O(game length) per node
    Color us = side;
    for (const Move &m : pseudo) {
        if (noisyOnly && m.promotion == Piece::None && !m.isEnPassant && !(occAll & (1ULL << m.to))) continue;
        tmp.makeMove(m);
        if (!tmp.inCheck(us)) legal.push_back(m);
        tmp.unmakeMove();
    }
    return legal;
}

void Board::makeMove(const Move &m) {
    HistoryEntry he;
    he.move = m;
    he.castlingRights = castlingRights;
    he.epSquare = epSquare;
    he.halfmoveClock = halfmoveClock;
    he.fullmoveNumber = fullmoveNumber;
    he.side = side;
    he.pieces = pieces;
    he.prevHash = hash;
    he.captured = Piece::None;

    Piece moving = pieceAt(m.from);

    if (epSquare != -1) hash ^= zEnpassant[epSquare & 7];
    epSquare = -1;

    if (m.isEnPassant) {
        int capSq = (side == Color::White) ? (m.to - 8) : (m.to + 8);
        he.captured = (side == Color::White) ? Piece::BP : Piece::WP;
        pieces[pieceIndex(he.captured)] &= ~(1ULL << capSq);
        hashTogglePiece(he.captured, capSq);
    } else {
        Piece cap = pieceAt(m.to);
        if (cap != Piece::None) {
            he.captured = cap;
            pieces[pieceIndex(cap)] &= ~(1ULL << m.to);
            hashTogglePiece(cap, m.to);
        }
    }

    pieces[pieceIndex(moving)] &= ~(1ULL << m.from);
    hashTogglePiece(moving, m.from);
    Piece placed = (m.promotion != Piece::None) ? m.promotion : moving;
    pieces[pieceIndex(placed)] |= (1ULL << m.to);
    hashTogglePiece(placed, m.to);

    if (m.isCastling) {
        int rookFrom = -1, rookTo = -1;
        Piece rook = (side == Color::White) ? Piece::WR : Piece::BR;
        if (m.to == 6) { rookFrom = 7; rookTo = 5; }
        else if (m.to == 2) { rookFrom = 0; rookTo = 3; }
        else if (m.to == 62) { rookFrom = 63; rookTo = 61; }
        else if (m.to == 58) { rookFrom = 56; rookTo = 59; }
        if (rookFrom >= 0) {
            pieces[pieceIndex(rook)] &= ~(1ULL << rookFrom);
            pieces[pieceIndex(rook)] |= (1ULL << rookTo);
            hashTogglePiece(rook, rookFrom);
            hashTogglePiece(rook, rookTo);
        }
    }

    uint8_t before = castlingRights;
    auto clearBySquare = [&](int sq) {
        switch (sq) {
            case 4: castlingRights &= ~(1 | 2); break;
            case 7: castlingRights &= ~1; break;
            case 0: castlingRights &= ~2; break;
            case 60: castlingRights &= ~(4 | 8); break;
            case 63: castlingRights &= ~4; break;
            case 56: castlingRights &= ~8; break;
            default: break;
        }
    };
    clearBySquare(m.from);
    clearBySquare(m.to);
    if (before != castlingRights) hash ^= zCastle[before] ^ zCastle[castlingRights];

    if (moving == Piece::WP && m.to - m.from == 16) epSquare = m.from + 8;
    else if (moving == Piece::BP && m.from - m.to == 16) epSquare = m.from - 8;
    if (epSquare != -1) hash ^= zEnpassant[epSquare & 7];

    if (moving == Piece::WP || moving == Piece::BP || he.captured != Piece::None) halfmoveClock = 0;
    else ++halfmoveClock;

    updateOccupancy();
    if (side == Color::Black) ++fullmoveNumber;
    side = opposite(side);
    hash ^= zSide;

    history.push_back(he);
}

void Board::unmakeMove() {
    if (history.empty()) return;
    const HistoryEntry &he = history.back();
    pieces = he.pieces;
    side = he.side;
    castlingRights = he.castlingRights;
    epSquare = he.epSquare;
    halfmoveClock = he.halfmoveClock;
    fullmoveNumber = he.fullmoveNumber;
    hash = he.prevHash;
    history.pop_back();
    updateOccupancy();
}

bool Board::applyMovesUCI(const std::vector<std::string> &uciMoves) {
    for (const std::string &mstr : uciMoves) {
        if (mstr.size() < 4 || mstr.size() > 5) return false;
        int from = parseSquare(mstr.substr(0, 2));
        int to = parseSquare(mstr.substr(2, 2));
        if (from == -1 || to == -1) return false;
        int promoType = -1;
        if (mstr.size() == 5) {
            switch (std::tolower(static_cast<unsigned char>(mstr[4]))) {
                case 'q': promoType = QUEEN; break;
                case 'r': promoType = ROOK; break;
                case 'b': promoType = BISHOP; break;
                case 'n': promoType = KNIGHT; break;
                default: return false;
            }
        }
        bool found = false;
        for (const Move &mv : generateLegalMoves()) {
            if (mv.from == from && mv.to == to && pieceTypeOf(mv.promotion) == promoType) {
                makeMove(mv);
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

bool Board::isRepetition() const {
    // history[i].prevHash is the hash of the position i plies into the game;
    // the current position is at ply history.size(). Only positions with the
    // same side to move and no irreversible move in between can repeat.
    int n = static_cast<int>(history.size());
    int stop = n - halfmoveClock;
    if (stop < 0) stop = 0;
    for (int i = n - 2; i >= stop; i -= 2)
        if (history[i].prevHash == hash) return true;
    return false;
}

bool Board::isInsufficientMaterial() const {
    if (pieces[WP_] | pieces[BP_] | pieces[WR_] | pieces[BR_] | pieces[WQ_] | pieces[BQ_]) return false;
    int minors = popcount(pieces[WN_] | pieces[BN_] | pieces[WB_] | pieces[BB_]);
    return minors <= 1;
}

bool Board::isDraw() const {
    if (halfmoveClock >= 100) {
        // Checkmate takes precedence over the fifty-move rule.
        if (!inCheck() || !generateLegalMoves().empty()) return true;
        return false;
    }
    return isRepetition() || isInsufficientMaterial();
}

uint64_t Board::perft(int depth) {
    if (depth <= 0) return 1;
    auto moves = generateLegalMoves();
    if (depth == 1) return moves.size();
    uint64_t nodes = 0;
    for (const Move &m : moves) {
        makeMove(m);
        nodes += perft(depth - 1);
        unmakeMove();
    }
    return nodes;
}

void Board::hashRecompute() {
    hash = 0;
    for (int i = 0; i < 12; ++i) {
        Bitboard bb = pieces[i];
        while (bb) hash ^= zPiece[i][popLsb(bb)];
    }
    hash ^= zCastle[castlingRights & 0xF];
    if (epSquare != -1) hash ^= zEnpassant[epSquare & 7];
    if (side == Color::Black) hash ^= zSide;
}
