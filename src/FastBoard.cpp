#include "FastBoard.h"

#include "BitOps.h"
#include "Eval.h"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <random>
#include <sstream>

namespace {

constexpr Bitboard FILE_A = 0x0101010101010101ULL;
constexpr Bitboard FILE_H = 0x8080808080808080ULL;
constexpr Bitboard RANK_1 = 0x00000000000000FFULL;
constexpr Bitboard RANK_4 = 0x00000000FF000000ULL;
constexpr Bitboard RANK_5 = 0x000000FF00000000ULL;
constexpr Bitboard RANK_8 = 0xFF00000000000000ULL;

inline Bitboard northOne(Bitboard bb) { return bb << 8; }
inline Bitboard southOne(Bitboard bb) { return bb >> 8; }
inline Bitboard northEast(Bitboard bb) { return (bb << 9) & ~FILE_A; }
inline Bitboard northWest(Bitboard bb) { return (bb << 7) & ~FILE_H; }
inline Bitboard southEast(Bitboard bb) { return (bb >> 7) & ~FILE_A; }
inline Bitboard southWest(Bitboard bb) { return (bb >> 9) & ~FILE_H; }

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

// Slow reference attack generation, used only to build the magic tables.
Bitboard slidingAttack(int sq, Bitboard occupied, bool rook) {
    static const int rookDirs[4][2] = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
    static const int bishopDirs[4][2] = {{1, 1}, {1, -1}, {-1, 1}, {-1, -1}};
    const int (*dirs)[2] = rook ? rookDirs : bishopDirs;
    Bitboard result = 0;
    for (int d = 0; d < 4; ++d) {
        int r = rankOf(sq) + dirs[d][0], f = fileOf(sq) + dirs[d][1];
        while (r >= 0 && r < 8 && f >= 0 && f < 8) {
            Bitboard b = 1ULL << (r * 8 + f);
            result |= b;
            if (occupied & b) break;
            r += dirs[d][0];
            f += dirs[d][1];
        }
    }
    return result;
}

// Relevant-occupancy mask: the attack rays without their final edge square.
Bitboard relevantMask(int sq, bool rook) {
    Bitboard edges = ((RANK_1 | RANK_8) & ~(RANK_1 << (8 * rankOf(sq)))) |
                     ((FILE_A | FILE_H) & ~(FILE_A << fileOf(sq)));
    return slidingAttack(sq, 0, rook) & ~edges;
}

} // namespace

Bitboard FastBoard::pawnAttacks[2][64];
Bitboard FastBoard::knightAttacks[64];
Bitboard FastBoard::kingAttacks[64];
Magic FastBoard::rookMagics[64];
Magic FastBoard::bishopMagics[64];
Bitboard FastBoard::rookTable[102400];
Bitboard FastBoard::bishopTable[5248];
uint64_t FastBoard::zPiece[12][64];
int FastBoard::psqValue[12][64];
int FastBoard::phaseValue[12];
uint64_t FastBoard::zSide;
uint64_t FastBoard::zCastle[16];
uint64_t FastBoard::zEnpassant[8];

// Finds a magic multiplier for every square by trial and error (fixed seed, so
// the result is deterministic) and fills the shared attack table.
void FastBoard::initMagics(Magic *magics, Bitboard *table, bool isRook) {
    std::mt19937_64 rng(isRook ? 0x5DEECE66DULL : 0x2545F4914F6CDD1DULL);
    std::vector<Bitboard> occupancy(4096), reference(4096);
    std::vector<int> epoch(4096, 0);
    int attempt = 0;
    Bitboard *ptr = table;
    for (int sq = 0; sq < 64; ++sq) {
        Magic &m = magics[sq];
        m.mask = relevantMask(sq, isRook);
        int bits = popcount(m.mask);
        int size = 1 << bits;
        m.shift = 64 - bits;
        m.attacks = ptr;

        // Enumerate all subsets of the mask (carry-rippler trick).
        Bitboard subset = 0;
        int n = 0;
        do {
            occupancy[n] = subset;
            reference[n] = slidingAttack(sq, subset, isRook);
            ++n;
            subset = (subset - m.mask) & m.mask;
        } while (subset);

        for (;;) {
            Bitboard magic = rng() & rng() & rng();
            if (popcount((m.mask * magic) >> 56) < 6) continue;
            ++attempt;
            bool ok = true;
            for (int i = 0; i < n && ok; ++i) {
                size_t idx = static_cast<size_t>((occupancy[i] * magic) >> m.shift);
                if (epoch[idx] != attempt) {
                    epoch[idx] = attempt;
                    ptr[idx] = reference[i];
                } else if (ptr[idx] != reference[i]) {
                    ok = false;
                }
            }
            if (ok) {
                m.magic = magic;
                break;
            }
        }
        ptr += size;
    }
}

void FastBoard::initTables() {
    for (int sq = 0; sq < 64; ++sq) {
        int file = sq & 7, rank = sq >> 3;
        pawnAttacks[0][sq] = pawnAttacks[1][sq] = 0;
        if (rank < 7) {
            if (file > 0) pawnAttacks[0][sq] |= 1ULL << (sq + 7);
            if (file < 7) pawnAttacks[0][sq] |= 1ULL << (sq + 9);
        }
        if (rank > 0) {
            if (file > 0) pawnAttacks[1][sq] |= 1ULL << (sq - 9);
            if (file < 7) pawnAttacks[1][sq] |= 1ULL << (sq - 7);
        }
        static const int knightDeltas[8][2] = {{-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}};
        static const int kingDeltas[8][2] = {{-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};
        knightAttacks[sq] = kingAttacks[sq] = 0;
        for (int i = 0; i < 8; ++i) {
            int nf = file + knightDeltas[i][0], nr = rank + knightDeltas[i][1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) knightAttacks[sq] |= 1ULL << (nr * 8 + nf);
            nf = file + kingDeltas[i][0]; nr = rank + kingDeltas[i][1];
            if (nf >= 0 && nf < 8 && nr >= 0 && nr < 8) kingAttacks[sq] |= 1ULL << (nr * 8 + nf);
        }
    }
    initMagics(rookMagics, rookTable, true);
    initMagics(bishopMagics, bishopTable, false);

    std::mt19937_64 rng(0x9E3779B97F4A7C15ULL);
    for (auto &row : zPiece) for (auto &z : row) z = rng();
    zSide = rng();
    for (auto &z : zCastle) z = rng();
    for (auto &z : zEnpassant) z = rng();

    for (int p = 0; p < 12; ++p) {
        Piece piece = static_cast<Piece>(p + 1);
        Color c = colorOf(piece);
        int t = pieceTypeOf(piece), sign = c == Color::White ? 1 : -1;
        phaseValue[p] = t == KNIGHT || t == BISHOP ? 1 : t == ROOK ? 2 : t == QUEEN ? 4 : 0;
        for (int sq = 0; sq < 64; ++sq)
            psqValue[p][sq] = t == KING ? 0 : sign * (eval::PIECE_VALUE[t] + eval::PST[t][eval::pstIndex(c, sq)]);
    }
}

Bitboard FastBoard::getRookAttacks(int sq, Bitboard occupied) {
    const Magic &m = rookMagics[sq];
    return m.attacks[((occupied & m.mask) * m.magic) >> m.shift];
}

Bitboard FastBoard::getBishopAttacks(int sq, Bitboard occupied) {
    const Magic &m = bishopMagics[sq];
    return m.attacks[((occupied & m.mask) * m.magic) >> m.shift];
}

FastBoard::FastBoard() {
    static const bool tablesReady = (initTables(), true); // thread-safe one-time init
    (void)tablesReady;
    setStartPos();
}

FastBoard::FastBoard(const FastBoard &o, NoHistory)
    : all_occupied(o.all_occupied), side(o.side), castlingRights(o.castlingRights), epSquare(o.epSquare),
      halfmoveClock(o.halfmoveClock), fullmoveNumber(o.fullmoveNumber), pliesFromNull(o.pliesFromNull),
      hash(o.hash), psq(o.psq), phase(o.phase) {
    for (int c = 0; c < 2; ++c) {
        occupied[c] = o.occupied[c];
        for (int p = 0; p < 6; ++p) pieces[c][p] = o.pieces[c][p];
    }
    for (int sq = 0; sq < 64; ++sq) mailbox[sq] = o.mailbox[sq];
    history.reserve(1);
}

void FastBoard::putPiece(Piece p, int sq) {
    int c = colorIndex(colorOf(p)), t = pieceTypeOf(p);
    Bitboard b = 1ULL << sq;
    pieces[c][t] |= b;
    occupied[c] |= b;
    all_occupied |= b;
    mailbox[sq] = p;
    hash ^= zPiece[static_cast<int>(p) - 1][sq];
    psq += psqValue[static_cast<int>(p) - 1][sq];
    phase += phaseValue[static_cast<int>(p) - 1];
}

void FastBoard::removePiece(int sq) {
    Piece p = mailbox[sq];
    int c = colorIndex(colorOf(p)), t = pieceTypeOf(p);
    Bitboard b = ~(1ULL << sq);
    pieces[c][t] &= b;
    occupied[c] &= b;
    all_occupied &= b;
    mailbox[sq] = Piece::None;
    hash ^= zPiece[static_cast<int>(p) - 1][sq];
    psq -= psqValue[static_cast<int>(p) - 1][sq];
    phase -= phaseValue[static_cast<int>(p) - 1];
}

void FastBoard::movePiece(int from, int to) {
    Piece p = mailbox[from];
    removePiece(from);
    putPiece(p, to);
}

bool FastBoard::setStartPos() {
    return setFromFEN("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

bool FastBoard::setFromFEN(const std::string &fen) {
    std::istringstream iss(fen);
    std::string placement, stm, castles, ep;
    if (!(iss >> placement >> stm >> castles >> ep)) return false;
    int hm = 0, fm = 1;
    if (!(iss >> hm)) hm = 0;
    if (!(iss >> fm)) fm = 1;
    if (hm < 0 || fm < 1) return false;

    FastBoard b(*this, NoHistory{});
    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 6; ++p) b.pieces[c][p] = 0;
        b.occupied[c] = 0;
    }
    b.all_occupied = 0;
    b.psq = 0;
    b.phase = 0;
    for (auto &sq : b.mailbox) sq = Piece::None;

    int rank = 7, file = 0;
    for (char c : placement) {
        if (c == '/') {
            if (file != 8 || rank == 0) return false;
            --rank; file = 0;
            continue;
        }
        if (c >= '1' && c <= '8') {
            file += c - '0';
            if (file > 8) return false;
            continue;
        }
        Piece p = fenToPiece(c);
        if (p == Piece::None || file >= 8) return false;
        b.putPiece(p, rank * 8 + file);
        ++file;
    }
    if (rank != 0 || file != 8) return false;
    if (popcount(b.pieces[0][KING]) != 1 || popcount(b.pieces[1][KING]) != 1) return false;
    if ((b.pieces[0][PAWN] | b.pieces[1][PAWN]) & (RANK_1 | RANK_8)) return false;

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
    if (b.inCheck(opposite(b.side))) return false;

    b.halfmoveClock = hm;
    b.pliesFromNull = 0;
    b.fullmoveNumber = fm;
    b.hashRecompute();
    *this = std::move(b);
    history.clear();
    return true;
}

void FastBoard::sanitizeCastlingRights() {
    if (!(pieces[0][KING] & (1ULL << 4))) castlingRights &= ~3;
    if (!(pieces[0][ROOK] & (1ULL << 7))) castlingRights &= ~1;
    if (!(pieces[0][ROOK] & (1ULL << 0))) castlingRights &= ~2;
    if (!(pieces[1][KING] & (1ULL << 60))) castlingRights &= ~12;
    if (!(pieces[1][ROOK] & (1ULL << 63))) castlingRights &= ~4;
    if (!(pieces[1][ROOK] & (1ULL << 56))) castlingRights &= ~8;
}

std::string FastBoard::getFEN() const {
    std::ostringstream oss;
    for (int rank = 7; rank >= 0; --rank) {
        int empty = 0;
        for (int file = 0; file < 8; ++file) {
            Piece p = mailbox[rank * 8 + file];
            if (p == Piece::None) { ++empty; continue; }
            if (empty) { oss << empty; empty = 0; }
            oss << pieceToFen(p);
        }
        if (empty) oss << empty;
        if (rank) oss << '/';
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

int FastBoard::kingSquare(Color c) const {
    Bitboard kings = pieces[colorIndex(c)][KING];
    return kings ? lsb(kings) : -1;
}

bool FastBoard::inCheck(Color c) const {
    int ksq = kingSquare(c);
    if (ksq == -1) return false;
    return isSquareAttacked(ksq, opposite(c));
}

bool FastBoard::isSquareAttacked(int sq, Color byColor) const {
    int ci = colorIndex(byColor);
    if (pawnAttacks[1 - ci][sq] & pieces[ci][PAWN]) return true;
    if (knightAttacks[sq] & pieces[ci][KNIGHT]) return true;
    if (getBishopAttacks(sq, all_occupied) & (pieces[ci][BISHOP] | pieces[ci][QUEEN])) return true;
    if (getRookAttacks(sq, all_occupied) & (pieces[ci][ROOK] | pieces[ci][QUEEN])) return true;
    if (kingAttacks[sq] & pieces[ci][KING]) return true;
    return false;
}

void FastBoard::generatePseudoMoves(MoveList &moves) const {
    moves.clear();
    generatePawnMoves(moves, side);
    generatePieceMoves(moves, side);
    generateCastlingMoves(moves, side);
}

void FastBoard::generatePawnMoves(MoveList &moves, Color c) const {
    int ci = colorIndex(c);
    Bitboard pawns = pieces[ci][PAWN];
    Bitboard enemies = occupied[1 - ci];
    Bitboard empty = ~all_occupied;
    auto addPromos = [&](int from, int to) {
        addMove(moves, from, to, makePiece(c, QUEEN));
        addMove(moves, from, to, makePiece(c, ROOK));
        addMove(moves, from, to, makePiece(c, BISHOP));
        addMove(moves, from, to, makePiece(c, KNIGHT));
    };
    Bitboard tmp;
    if (c == Color::White) {
        Bitboard single = northOne(pawns) & empty;
        Bitboard dbl = northOne(single) & empty & RANK_4;
        tmp = single & ~RANK_8; while (tmp) { int to = popLsb(tmp); addMove(moves, to - 8, to); }
        tmp = single & RANK_8;  while (tmp) { int to = popLsb(tmp); addPromos(to - 8, to); }
        tmp = dbl;              while (tmp) { int to = popLsb(tmp); addMove(moves, to - 16, to); }
        Bitboard capL = northWest(pawns) & enemies, capR = northEast(pawns) & enemies;
        tmp = capL & ~RANK_8; while (tmp) { int to = popLsb(tmp); addMove(moves, to - 7, to); }
        tmp = capR & ~RANK_8; while (tmp) { int to = popLsb(tmp); addMove(moves, to - 9, to); }
        tmp = capL & RANK_8;  while (tmp) { int to = popLsb(tmp); addPromos(to - 7, to); }
        tmp = capR & RANK_8;  while (tmp) { int to = popLsb(tmp); addPromos(to - 9, to); }
        if (epSquare != -1) {
            Bitboard epMask = 1ULL << epSquare;
            if (northWest(pawns) & epMask) addMove(moves, epSquare - 7, epSquare, Piece::None, true);
            if (northEast(pawns) & epMask) addMove(moves, epSquare - 9, epSquare, Piece::None, true);
        }
    } else {
        Bitboard single = southOne(pawns) & empty;
        Bitboard dbl = southOne(single) & empty & RANK_5;
        tmp = single & ~RANK_1; while (tmp) { int to = popLsb(tmp); addMove(moves, to + 8, to); }
        tmp = single & RANK_1;  while (tmp) { int to = popLsb(tmp); addPromos(to + 8, to); }
        tmp = dbl;              while (tmp) { int to = popLsb(tmp); addMove(moves, to + 16, to); }
        Bitboard capL = southEast(pawns) & enemies, capR = southWest(pawns) & enemies;
        tmp = capL & ~RANK_1; while (tmp) { int to = popLsb(tmp); addMove(moves, to + 7, to); }
        tmp = capR & ~RANK_1; while (tmp) { int to = popLsb(tmp); addMove(moves, to + 9, to); }
        tmp = capL & RANK_1;  while (tmp) { int to = popLsb(tmp); addPromos(to + 7, to); }
        tmp = capR & RANK_1;  while (tmp) { int to = popLsb(tmp); addPromos(to + 9, to); }
        if (epSquare != -1) {
            Bitboard epMask = 1ULL << epSquare;
            if (southEast(pawns) & epMask) addMove(moves, epSquare + 7, epSquare, Piece::None, true);
            if (southWest(pawns) & epMask) addMove(moves, epSquare + 9, epSquare, Piece::None, true);
        }
    }
}

void FastBoard::generatePieceMoves(MoveList &moves, Color c) const {
    int ci = colorIndex(c);
    Bitboard targets = ~occupied[ci];
    for (int type = KNIGHT; type <= KING; ++type) {
        Bitboard bb = pieces[ci][type];
        while (bb) {
            int from = popLsb(bb);
            Bitboard attacks;
            switch (type) {
                case KNIGHT: attacks = knightAttacks[from]; break;
                case BISHOP: attacks = getBishopAttacks(from, all_occupied); break;
                case ROOK: attacks = getRookAttacks(from, all_occupied); break;
                case QUEEN: attacks = getQueenAttacks(from, all_occupied); break;
                default: attacks = kingAttacks[from]; break;
            }
            attacks &= targets;
            while (attacks) addMove(moves, from, popLsb(attacks));
        }
    }
}

void FastBoard::generateCastlingMoves(MoveList &moves, Color c) const {
    if (!castlingRights || inCheck(c)) return;
    Color them = opposite(c);
    if (c == Color::White) {
        if ((castlingRights & 1) && !(all_occupied & 0x60ULL) && (pieces[0][ROOK] & (1ULL << 7)) &&
            !isSquareAttacked(5, them) && !isSquareAttacked(6, them))
            addMove(moves, 4, 6, Piece::None, false, true);
        if ((castlingRights & 2) && !(all_occupied & 0x0EULL) && (pieces[0][ROOK] & 1ULL) &&
            !isSquareAttacked(3, them) && !isSquareAttacked(2, them))
            addMove(moves, 4, 2, Piece::None, false, true);
    } else {
        if ((castlingRights & 4) && !(all_occupied & 0x6000000000000000ULL) && (pieces[1][ROOK] & (1ULL << 63)) &&
            !isSquareAttacked(61, them) && !isSquareAttacked(62, them))
            addMove(moves, 60, 62, Piece::None, false, true);
        if ((castlingRights & 8) && !(all_occupied & 0x0E00000000000000ULL) && (pieces[1][ROOK] & (1ULL << 56)) &&
            !isSquareAttacked(59, them) && !isSquareAttacked(58, them))
            addMove(moves, 60, 58, Piece::None, false, true);
    }
}

void FastBoard::generatePseudoLegalMoves(MoveList &out, bool noisyOnly) const {
    generatePseudoMoves(out);
    if (noisyOnly) {
        int kept = 0;
        for (int i = 0; i < out.size(); ++i) {
            const Move &m = out[i];
            if (m.promotion != Piece::None || m.isEnPassant || mailbox[m.to] != Piece::None) out[kept++] = m;
        }
        out.resize(kept);
    }
}

std::vector<Move> FastBoard::generateLegalMoves(bool noisyOnly) const {
    MoveList pseudo;
    generatePseudoLegalMoves(pseudo, noisyOnly);
    std::vector<Move> legal;
    legal.reserve(pseudo.size());
    FastBoard tmp(*this, NoHistory{}); // one copy without the game history, then make/unmake
    for (int i = 0; i < pseudo.size(); ++i) {
        tmp.makeMove(pseudo[i]);
        if (!tmp.inCheck(side)) legal.push_back(pseudo[i]);
        tmp.unmakeMove();
    }
    return legal;
}

void FastBoard::makeMove(const Move &m) {
    HistoryEntry entry;
    entry.move = m;
    entry.moved = mailbox[m.from];
    entry.captured = m.isEnPassant ? makePiece(opposite(side), PAWN) : mailbox[m.to];
    entry.castlingRights = castlingRights;
    entry.epSquare = epSquare;
    entry.halfmoveClock = halfmoveClock;
    entry.fullmoveNumber = fullmoveNumber;
    entry.hash = hash;
    entry.pliesFromNull = pliesFromNull;
    history.push_back(entry);

    if (epSquare != -1) {
        hash ^= zEnpassant[epSquare & 7];
        epSquare = -1;
    }

    Piece moving = entry.moved;
    if (m.isEnPassant) {
        removePiece(side == Color::White ? m.to - 8 : m.to + 8);
    } else if (entry.captured != Piece::None) {
        removePiece(m.to);
    }

    if (m.promotion != Piece::None) {
        removePiece(m.from);
        putPiece(m.promotion, m.to);
    } else {
        movePiece(m.from, m.to);
    }

    if (m.isCastling) {
        if (m.to == 6) movePiece(7, 5);
        else if (m.to == 2) movePiece(0, 3);
        else if (m.to == 62) movePiece(63, 61);
        else if (m.to == 58) movePiece(56, 59);
    }

    uint8_t oldRights = castlingRights;
    if (m.from == 4 || m.to == 4) castlingRights &= ~3;
    if (m.from == 60 || m.to == 60) castlingRights &= ~12;
    if (m.from == 0 || m.to == 0) castlingRights &= ~2;
    if (m.from == 7 || m.to == 7) castlingRights &= ~1;
    if (m.from == 56 || m.to == 56) castlingRights &= ~8;
    if (m.from == 63 || m.to == 63) castlingRights &= ~4;
    if (oldRights != castlingRights) hash ^= zCastle[oldRights] ^ zCastle[castlingRights];

    int movingType = pieceTypeOf(moving);
    if (movingType == PAWN && (m.to - m.from == 16 || m.from - m.to == 16)) {
        epSquare = (m.from + m.to) / 2;
        hash ^= zEnpassant[epSquare & 7];
    }

    if (movingType == PAWN || entry.captured != Piece::None) halfmoveClock = 0;
    else ++halfmoveClock;

    if (side == Color::Black) ++fullmoveNumber;
    side = opposite(side);
    hash ^= zSide;
    ++pliesFromNull;
}

void FastBoard::makeNullMove() {
    HistoryEntry entry;
    entry.move = Move{};
    entry.moved = Piece::None;
    entry.captured = Piece::None;
    entry.castlingRights = castlingRights;
    entry.epSquare = epSquare;
    entry.halfmoveClock = halfmoveClock;
    entry.fullmoveNumber = fullmoveNumber;
    entry.hash = hash;
    entry.pliesFromNull = pliesFromNull;
    history.push_back(entry);
    if (epSquare != -1) {
        hash ^= zEnpassant[epSquare & 7];
        epSquare = -1;
    }
    ++halfmoveClock;
    if (side == Color::Black) ++fullmoveNumber;
    side = opposite(side);
    hash ^= zSide;
    pliesFromNull = 0;
}

void FastBoard::unmakeNullMove() {
    if (history.empty()) return;
    const HistoryEntry &entry = history.back();
    side = opposite(side);
    epSquare = entry.epSquare;
    halfmoveClock = entry.halfmoveClock;
    fullmoveNumber = entry.fullmoveNumber;
    hash = entry.hash;
    pliesFromNull = entry.pliesFromNull;
    history.pop_back();
}

void FastBoard::unmakeMove() {
    if (history.empty()) return;
    HistoryEntry entry = history.back();
    history.pop_back();
    const Move &m = entry.move;

    side = opposite(side);

    if (m.isCastling) {
        if (m.to == 6) movePiece(5, 7);
        else if (m.to == 2) movePiece(3, 0);
        else if (m.to == 62) movePiece(61, 63);
        else if (m.to == 58) movePiece(59, 56);
    }

    removePiece(m.to);
    putPiece(entry.moved, m.from);

    if (m.isEnPassant) putPiece(entry.captured, side == Color::White ? m.to - 8 : m.to + 8);
    else if (entry.captured != Piece::None) putPiece(entry.captured, m.to);

    castlingRights = entry.castlingRights;
    epSquare = entry.epSquare;
    halfmoveClock = entry.halfmoveClock;
    fullmoveNumber = entry.fullmoveNumber;
    hash = entry.hash;
    pliesFromNull = entry.pliesFromNull;
}

bool FastBoard::applyMovesUCI(const std::vector<std::string> &uciMoves) {
    for (const std::string &moveStr : uciMoves) {
        if (moveStr.size() < 4 || moveStr.size() > 5) return false;
        int from = parseSquare(moveStr.substr(0, 2));
        int to = parseSquare(moveStr.substr(2, 2));
        if (from == -1 || to == -1) return false;
        int promoType = -1;
        if (moveStr.size() == 5) {
            switch (std::tolower(static_cast<unsigned char>(moveStr[4]))) {
                case 'q': promoType = QUEEN; break;
                case 'r': promoType = ROOK; break;
                case 'b': promoType = BISHOP; break;
                case 'n': promoType = KNIGHT; break;
                default: return false;
            }
        }
        bool found = false;
        for (const Move &move : generateLegalMoves()) {
            if (move.from == from && move.to == to && pieceTypeOf(move.promotion) == promoType) {
                makeMove(move);
                found = true;
                break;
            }
        }
        if (!found) return false;
    }
    return true;
}

bool FastBoard::isRepetition() const {
    int n = static_cast<int>(history.size());
    int stop = n - std::min(halfmoveClock, pliesFromNull);
    if (stop < 0) stop = 0;
    // history[i].hash is the position i plies into the game; the current one is ply n.
    for (int i = n - 2; i >= stop; i -= 2)
        if (history[i].hash == hash) return true;
    return false;
}

bool FastBoard::isInsufficientMaterial() const {
    for (int c = 0; c < 2; ++c)
        if (pieces[c][PAWN] | pieces[c][ROOK] | pieces[c][QUEEN]) return false;
    int minors = popcount(pieces[0][KNIGHT] | pieces[1][KNIGHT] | pieces[0][BISHOP] | pieces[1][BISHOP]);
    return minors <= 1;
}

bool FastBoard::isDraw() const {
    if (halfmoveClock >= 100) return !inCheck() || !generateLegalMoves().empty();
    return isRepetition() || isInsufficientMaterial();
}

void FastBoard::hashRecompute() {
    hash = 0;
    for (int sq = 0; sq < 64; ++sq)
        if (mailbox[sq] != Piece::None) hash ^= zPiece[static_cast<int>(mailbox[sq]) - 1][sq];
    if (side == Color::Black) hash ^= zSide;
    hash ^= zCastle[castlingRights & 0xF];
    if (epSquare != -1) hash ^= zEnpassant[epSquare & 7];
}

uint64_t FastBoard::perft(int depth) {
    if (depth <= 0) return 1;
    auto moves = generateLegalMoves();
    if (depth == 1) return moves.size();
    uint64_t nodes = 0;
    for (const Move &move : moves) {
        makeMove(move);
        nodes += perft(depth - 1);
        unmakeMove();
    }
    return nodes;
}

void FastBoard::divide(int depth) {
    uint64_t total = 0;
    for (const Move &move : generateLegalMoves()) {
        makeMove(move);
        uint64_t count = perft(depth - 1);
        unmakeMove();
        std::cout << moveToUci(move) << ": " << count << "\n";
        total += count;
    }
    std::cout << "\nTotal: " << total << std::endl;
}
