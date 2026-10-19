#include "RelGraph.h"

#include "BitOps.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <numeric>

namespace {

// Feature columns and relations, as relational_graph.FEATURE_COLUMNS / RELATIONS.
enum Col {
    IS_SQUARE, IS_PIECE, IS_GLOBAL, TYPE_PAWN, // + piece type (P N B R Q K)
    OWN = 9, OPPONENT, FILE_COL, RANK_COL, ATT_OWN, ATT_OPP, EN_PASSANT, PINNED, MOBILITY, LEGAL_FROM, PASSED,
    CASTLING_PIECE, IN_CHECK, WHITE_TO_MOVE, CASTLE_OWN_K, CASTLE_OWN_Q, CASTLE_OPP_K, CASTLE_OPP_Q, HALFMOVE,
    REPETITIONS, LEGAL_COUNT, MATERIAL_OWN, MATERIAL_OPP
};
enum Rel {
    PIECE_ON, OCCUPIED_BY, ATTACKS, ATTACKED_BY, DEFENDS, DEFENDED_BY, CONTROLS, CONTROLLED_BY, LEGAL_MOVE,
    LEGAL_MOVE_FROM, PAWN_SAME_FILE, PAWN_SUPPORTS, PAWN_SUPPORTED_BY, PASSED_PATH, PASSED_PATH_OF, KING_ZONE,
    KING_ZONE_OF, PINNED_TO, PINS, GLOBAL_TO_PIECE, PIECE_TO_GLOBAL
};
constexpr int kMaterial[6] = {1, 3, 3, 5, 9, 0};

struct Tables {
    Bitboard knight[64], king[64], pawn[2][64], between[64][64], passed[2][64];
    Tables() {
        for (int sq = 0; sq < 64; ++sq) {
            const int f = sq & 7, r = sq >> 3;
            knight[sq] = king[sq] = pawn[0][sq] = pawn[1][sq] = 0;
            static const int kn[8][2] = {{1, 2}, {2, 1}, {2, -1}, {1, -2}, {-1, -2}, {-2, -1}, {-2, 1}, {-1, 2}};
            for (const auto &d : kn)
                if (f + d[0] >= 0 && f + d[0] < 8 && r + d[1] >= 0 && r + d[1] < 8)
                    knight[sq] |= 1ULL << ((r + d[1]) * 8 + f + d[0]);
            for (int df = -1; df <= 1; ++df)
                for (int dr = -1; dr <= 1; ++dr)
                    if ((df || dr) && f + df >= 0 && f + df < 8 && r + dr >= 0 && r + dr < 8)
                        king[sq] |= 1ULL << ((r + dr) * 8 + f + df);
            for (int df : {-1, 1}) {
                if (f + df < 0 || f + df > 7) continue;
                if (r < 7) pawn[0][sq] |= 1ULL << ((r + 1) * 8 + f + df);
                if (r > 0) pawn[1][sq] |= 1ULL << ((r - 1) * 8 + f + df);
            }
            // An enemy pawn on these squares stops a pawn being passed (relational_graph._passed).
            passed[0][sq] = passed[1][sq] = 0;
            for (int ff = f - 1; ff <= f + 1; ++ff) {
                if (ff < 0 || ff > 7) continue;
                for (int rr = r + 1; rr < 8; ++rr) passed[0][sq] |= 1ULL << (rr * 8 + ff);
                for (int rr = 0; rr < r; ++rr) passed[1][sq] |= 1ULL << (rr * 8 + ff);
            }
        }
        for (int a = 0; a < 64; ++a)
            for (int b = 0; b < 64; ++b) {
                between[a][b] = 0;
                const int df = (b & 7) - (a & 7), dr = (b >> 3) - (a >> 3);
                if (a == b || (df && dr && std::abs(df) != std::abs(dr))) continue;
                const int step = (dr > 0 ? 8 : dr < 0 ? -8 : 0) + (df > 0 ? 1 : df < 0 ? -1 : 0);
                for (int s = a + step; s != b; s += step) between[a][b] |= 1ULL << s;
            }
    }
};
const Tables &tables() {
    static const Tables t;
    return t;
}

// python-chess's attacks_mask: what the piece on `sq` attacks.
Bitboard attacksOf(Piece p, int sq, Bitboard occ) {
    const Tables &t = tables();
    switch (pieceTypeOf(p)) {
        case PAWN: return t.pawn[colorIndex(colorOf(p))][sq];
        case KNIGHT: return t.knight[sq];
        case BISHOP: return FastBoard::bishopAttacks(sq, occ);
        case ROOK: return FastBoard::rookAttacks(sq, occ);
        case QUEEN: return FastBoard::bishopAttacks(sq, occ) | FastBoard::rookAttacks(sq, occ);
        case KING: return t.king[sq];
        default: return 0;
    }
}

// policy_encoding.PROMOTION_PAIRS: index of each (from, to) promotion pair, -1 for none.
struct PromotionPairs {
    int index[64][64];
    PromotionPairs() {
        for (auto &row : index)
            for (int &v : row) v = -1;
        int n = 0;
        for (int side = 0; side < 2; ++side) { // White 7 -> 8, then Black 2 -> 1
            const int from = side ? 1 : 6, to = side ? 0 : 7;
            for (int f = 0; f < 8; ++f)
                for (int df = -1; df <= 1; ++df)
                    if (f + df >= 0 && f + df < 8) index[from * 8 + f][to * 8 + f + df] = n++;
        }
    }
};

} // namespace

int policyIndexV3(const Move &m) {
    static const PromotionPairs pairs;
    const int type = m.promotion == Piece::None ? -1 : pieceTypeOf(m.promotion);
    if (type == KNIGHT || type == BISHOP || type == ROOK) {
        const int pair = pairs.index[m.from][m.to];
        if (pair >= 0) return 64 * 64 + pair * 3 + (type == KNIGHT ? 0 : type == BISHOP ? 1 : 2);
    }
    return m.from * 64 + m.to;
}

void buildRelationalGraph(const FastBoard &board, RelGraph &g) {
    // All the legal moves: they shape the graph (legal_moves_from, legal_move_count, legal_move edges).
    const std::vector<Move> legal = board.generateLegalMoves();
    const Tables &t = tables();
    const Color stm = board.sideToMove();
    const Bitboard occ = board.occupancy();
    Bitboard byColor[2] = {0, 0};
    for (int c = 0; c < 2; ++c)
        for (int type = 0; type < 6; ++type) byColor[c] |= board.pieceBB(static_cast<Color>(c), type);
    int squares[32], nPieces = 0;
    int nodeOf[64];
    for (Bitboard b = occ; b; b &= b - 1) {
        const int sq = lsb(b);
        nodeOf[sq] = 64 + nPieces;
        squares[nPieces++] = sq;
    }
    const int gnode = 64 + nPieces;
    g.nodes = gnode + 1;
    g.x.assign(static_cast<size_t>(g.nodes) * RelGraph::kFeatures, 0.0f);
    auto set = [&](int node, int col, double v) { g.x[static_cast<size_t>(node) * RelGraph::kFeatures + col] = static_cast<float>(v); };

    // Attack counts per square for each colour.
    int att[2][64] = {};
    Bitboard mask[32];
    for (int i = 0; i < nPieces; ++i) {
        const Piece p = board.pieceAt(squares[i]);
        mask[i] = attacksOf(p, squares[i], occ);
        for (Bitboard b = mask[i]; b; b &= b - 1) ++att[colorIndex(colorOf(p))][lsb(b)];
    }
    const int us = colorIndex(stm), them = us ^ 1;
    auto relRank = [&](int sq) { return (stm == Color::White ? (sq >> 3) : 7 - (sq >> 3)) / 7.0; };

    // Legal moves of the side to move.
    int legalFrom[64] = {};
    std::vector<int> pairs; // from * 64 + to, each once
    pairs.reserve(legal.size());
    int epTarget = -1;
    g.legal.clear();
    for (const Move &m : legal) {
        g.legal.push_back(static_cast<uint16_t>(policyIndexV3(m)));
        ++legalFrom[m.from];
        pairs.push_back(m.from * 64 + m.to);
        if (m.isEnPassant) epTarget = m.to;
    }
    std::sort(pairs.begin(), pairs.end());
    pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

    const int kingSq[2] = {lsb(board.pieceBB(Color::White, KING)), lsb(board.pieceBB(Color::Black, KING))};
    const bool inCheck = board.inCheck();
    // Castling rights as python-chess's clean_castling_rights keeps them: a right needs its rook in the
    // corner and the king on its home square (a FEN may claim one without them).
    uint8_t rights = board.getCastlingRights(); // K Q k q = bits 0..3
    const Bitboard wr = board.pieceBB(Color::White, ROOK), br = board.pieceBB(Color::Black, ROOK);
    if (!(wr >> 7 & 1)) rights &= ~1;
    if (!(wr & 1)) rights &= ~2;
    if (!(br >> 63 & 1)) rights &= ~4;
    if (!(br >> 56 & 1)) rights &= ~8;
    if (kingSq[0] != 4) rights &= ~3;
    if (kingSq[1] != 60) rights &= ~12;
    const bool hasRights[2] = {(rights & 3) != 0, (rights & 12) != 0};
    const Bitboard castleRooks = ((rights & 1) ? 1ULL << 7 : 0) | ((rights & 2) ? 1ULL << 0 : 0) |
                                 ((rights & 4) ? 1ULL << 63 : 0) | ((rights & 8) ? 1ULL << 56 : 0);

    // Absolutely pinned pieces of each colour: the single piece between a king and an enemy slider on its line.
    Bitboard pinned = 0;
    for (int c = 0; c < 2; ++c) {
        const int k = kingSq[c];
        const Color enemy = static_cast<Color>(c ^ 1);
        const Bitboard rq = board.pieceBB(enemy, ROOK) | board.pieceBB(enemy, QUEEN);
        const Bitboard bq = board.pieceBB(enemy, BISHOP) | board.pieceBB(enemy, QUEEN);
        Bitboard snipers = (FastBoard::rookAttacks(k, 0) & rq) | (FastBoard::bishopAttacks(k, 0) & bq);
        for (; snipers; snipers &= snipers - 1) {
            const Bitboard blockers = t.between[k][lsb(snipers)] & occ;
            if (blockers && !(blockers & (blockers - 1))) pinned |= blockers & byColor[c];
        }
    }

    // Edges in relational_graph.build_graph's insertion order.
    std::vector<uint8_t> es, ed, er;
    es.reserve(1024);
    ed.reserve(1024);
    er.reserve(1024);
    auto edge = [&](int a, int b, int r) {
        es.push_back(static_cast<uint8_t>(a));
        ed.push_back(static_cast<uint8_t>(b));
        er.push_back(static_cast<uint8_t>(r));
    };
    int material[2] = {0, 0};
    for (int i = 0; i < nPieces; ++i) {
        const int sq = squares[i], n = 64 + i;
        const Piece p = board.pieceAt(sq);
        const int c = colorIndex(colorOf(p)), type = pieceTypeOf(p);
        const bool own = c == us;
        material[c] += kMaterial[type];
        set(n, IS_PIECE, 1.0);
        set(n, TYPE_PAWN + type, 1.0);
        set(n, own ? OWN : OPPONENT, 1.0);
        set(n, FILE_COL, (sq & 7) / 7.0);
        set(n, RANK_COL, relRank(sq));
        set(n, ATT_OWN, att[us][sq] / 8.0);
        set(n, ATT_OPP, att[them][sq] / 8.0);
        set(sq, TYPE_PAWN + type, 1.0);
        set(sq, own ? OWN : OPPONENT, 1.0);
        set(n, MOBILITY, popcount(mask[i] & ~byColor[c]) / 27.0);
        if (own) set(n, LEGAL_FROM, legalFrom[sq] / 27.0);
        if (type == ROOK && (castleRooks >> sq & 1)) set(n, CASTLING_PIECE, 1.0);
        if (type == KING) {
            if (hasRights[c]) set(n, CASTLING_PIECE, 1.0);
            if (inCheck && own) set(n, IN_CHECK, 1.0);
        }
        if (type == PAWN && !(board.pieceBB(static_cast<Color>(c ^ 1), PAWN) & t.passed[c][sq])) set(n, PASSED, 1.0);
        if (pinned >> sq & 1) set(n, PINNED, 1.0);

        edge(n, sq, PIECE_ON);
        edge(sq, n, OCCUPIED_BY);
        for (Bitboard b = mask[i]; b; b &= b - 1) {
            const int to = lsb(b);
            if (!(occ >> to & 1)) {
                edge(n, to, CONTROLS);
                edge(to, n, CONTROLLED_BY);
            } else if (byColor[c] >> to & 1) {
                edge(n, nodeOf[to], DEFENDS);
                edge(nodeOf[to], n, DEFENDED_BY);
            } else {
                edge(n, nodeOf[to], ATTACKS);
                edge(nodeOf[to], n, ATTACKED_BY);
            }
        }
        edge(gnode, n, GLOBAL_TO_PIECE);
        edge(n, gnode, PIECE_TO_GLOBAL);
    }
    for (int pr : pairs) {
        const int from = pr / 64, to = pr % 64;
        edge(nodeOf[from], to, LEGAL_MOVE);
        edge(to, nodeOf[from], LEGAL_MOVE_FROM);
    }
    // Pawn structure.
    const Bitboard allPawns = board.pieceBB(Color::White, PAWN) | board.pieceBB(Color::Black, PAWN);
    for (int i = 0; i < nPieces; ++i) {
        const int sq = squares[i], n = 64 + i;
        const Piece p = board.pieceAt(sq);
        if (pieceTypeOf(p) != PAWN) continue;
        const int c = colorIndex(colorOf(p));
        for (int j = 0; j < nPieces; ++j)
            if (j != i && pieceTypeOf(board.pieceAt(squares[j])) == PAWN && (squares[j] & 7) == (sq & 7))
                edge(n, 64 + j, PAWN_SAME_FILE);
        for (Bitboard b = mask[i] & allPawns & byColor[c]; b; b &= b - 1) {
            const int m = nodeOf[lsb(b)];
            edge(n, m, PAWN_SUPPORTS);
            edge(m, n, PAWN_SUPPORTED_BY);
        }
        if (!(board.pieceBB(static_cast<Color>(c ^ 1), PAWN) & t.passed[c][sq])) {
            const int step = c == 0 ? 8 : -8;
            for (int to = sq + step; to >= 0 && to < 64; to += step) {
                edge(n, to, PASSED_PATH);
                edge(to, n, PASSED_PATH_OF);
            }
        }
    }
    // King zones, White's first.
    for (int c = 0; c < 2; ++c) {
        const int kn = nodeOf[kingSq[c]];
        for (Bitboard b = t.king[kingSq[c]]; b; b &= b - 1) {
            edge(kn, lsb(b), KING_ZONE);
            edge(lsb(b), kn, KING_ZONE_OF);
        }
    }
    // Pinned piece <-> the enemy slider pinning it to its king.
    for (Bitboard b = pinned; b; b &= b - 1) {
        const int sq = lsb(b), n = nodeOf[sq];
        const int c = colorIndex(colorOf(board.pieceAt(sq)));
        const int k = kingSq[c];
        for (Bitboard a = board.attackersTo(sq, occ) & byColor[c ^ 1]; a; a &= a - 1) {
            const int s = lsb(a), type = pieceTypeOf(board.pieceAt(s));
            if ((type == BISHOP || type == ROOK || type == QUEEN) && (t.between[k][s] >> sq & 1)) {
                edge(n, nodeOf[s], PINNED_TO);
                edge(nodeOf[s], n, PINS);
            }
        }
    }

    // Square and global features.
    for (int sq = 0; sq < 64; ++sq) {
        set(sq, IS_SQUARE, 1.0);
        set(sq, FILE_COL, (sq & 7) / 7.0);
        set(sq, RANK_COL, relRank(sq));
        set(sq, ATT_OWN, att[us][sq] / 8.0);
        set(sq, ATT_OPP, att[them][sq] / 8.0);
    }
    if (epTarget >= 0) set(epTarget, EN_PASSANT, 1.0);
    const bool white = stm == Color::White;
    const int ownK = white ? 1 : 4, ownQ = white ? 2 : 8, oppK = white ? 4 : 1, oppQ = white ? 8 : 2;
    set(gnode, IS_GLOBAL, 1.0);
    set(gnode, IN_CHECK, inCheck ? 1.0 : 0.0);
    set(gnode, WHITE_TO_MOVE, white ? 1.0 : 0.0);
    set(gnode, CASTLE_OWN_K, (rights & ownK) ? 1.0 : 0.0);
    set(gnode, CASTLE_OWN_Q, (rights & ownQ) ? 1.0 : 0.0);
    set(gnode, CASTLE_OPP_K, (rights & oppK) ? 1.0 : 0.0);
    set(gnode, CASTLE_OPP_Q, (rights & oppQ) ? 1.0 : 0.0);
    set(gnode, HALFMOVE, std::min(board.getHalfmoveClock(), 100) / 100.0);
    set(gnode, REPETITIONS, 0.0); // a FEN carries no history
    set(gnode, LEGAL_COUNT, static_cast<double>(legal.size()) / 64.0);
    set(gnode, MATERIAL_OWN, material[us] / 39.0);
    set(gnode, MATERIAL_OPP, material[them] / 39.0);

    // Edges sorted by destination, keeping the insertion order among equals.
    std::vector<int> order(es.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return ed[a] < ed[b]; });
    g.src.resize(order.size());
    g.dst.resize(order.size());
    g.relation.resize(order.size());
    for (size_t i = 0; i < order.size(); ++i) {
        g.src[i] = es[order[i]];
        g.dst[i] = ed[order[i]];
        g.relation[i] = er[order[i]];
    }
}

std::string encodeGraph(const RelGraph &g) {
    std::string raw;
    auto put8 = [&](unsigned v) { raw.push_back(static_cast<char>(v & 0xFF)); };
    auto put16 = [&](unsigned v) { put8(v); put8(v >> 8); };
    size_t nonzero = 0;
    for (float v : g.x) nonzero += v != 0.0f;
    put8(static_cast<unsigned>(g.nodes));
    put16(static_cast<unsigned>(g.src.size()));
    put16(static_cast<unsigned>(nonzero));
    put8(static_cast<unsigned>(g.legal.size()));
    for (size_t i = 0; i < g.x.size(); ++i)
        if (g.x[i] != 0.0f) put16(static_cast<unsigned>(i));
    for (float v : g.x)
        if (v != 0.0f) {
            uint32_t bits;
            std::memcpy(&bits, &v, 4);
            put16(bits & 0xFFFF);
            put16(bits >> 16);
        }
    for (size_t i = 0; i < g.src.size(); ++i) {
        put8(g.src[i]);
        put8(g.dst[i]);
        put8(g.relation[i]);
    }
    for (uint16_t idx : g.legal) put16(idx);
    static const char *kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((raw.size() + 2) / 3 * 4);
    for (size_t i = 0; i < raw.size(); i += 3) {
        const uint32_t n = static_cast<uint8_t>(raw[i]) << 16 |
                           (i + 1 < raw.size() ? static_cast<uint8_t>(raw[i + 1]) << 8 : 0) |
                           (i + 2 < raw.size() ? static_cast<uint8_t>(raw[i + 2]) : 0);
        out.push_back(kB64[n >> 18 & 63]);
        out.push_back(kB64[n >> 12 & 63]);
        out.push_back(i + 1 < raw.size() ? kB64[n >> 6 & 63] : '=');
        out.push_back(i + 2 < raw.size() ? kB64[n & 63] : '=');
    }
    return out;
}
