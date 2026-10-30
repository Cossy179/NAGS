// NNUE tests with a random network: incrementally updated accumulators must
// equal recomputed ones, colour-mirrored positions must evaluate the same,
// and the search must work with the network.

#include "FastBoard.h"
#include "Nnue.h"
#include "Search.h"
#include "TT.h"

#include <atomic>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

static int failures = 0;

#define CHECK(cond, ...)                                     \
    do {                                                     \
        if (!(cond)) {                                       \
            ++failures;                                      \
            std::printf("FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                        \
            std::printf("\n");                               \
        }                                                    \
    } while (0)

// Writes a network file with small random weights: version 1 (one feature
// set, clipped ReLU) or version 3 with 4 king buckets, mirroring, SCReLU and
// 8 output buckets.
static std::string writeRandomNet(bool kingBuckets) {
    std::string path = kingBuckets ? "nnue_test_random_kb.nnue" : "nnue_test_random.nnue";
    std::ofstream out(path, std::ios::binary);
    out.write("NAGSNNUE", 8);
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) out.put(static_cast<char>((v >> (8 * i)) & 0xFF)); };
    auto i16 = [&](int v) { uint16_t u = static_cast<uint16_t>(v); out.put(static_cast<char>(u & 0xFF)); out.put(static_cast<char>(u >> 8)); };
    const int sets = kingBuckets ? 4 : 1, outputs = kingBuckets ? 8 : 1;
    if (kingBuckets) {
        u32(3);
        u32(nnue::kHidden);
        u32(outputs);
        u32(sets);
        u32(3); // SCReLU, mirroring
        // Bucket by rank of the (mirrored) king: 1st, 2nd, 3rd-4th, 5th-8th.
        for (int sq = 0; sq < 64; ++sq) out.put(static_cast<char>(sq < 8 ? 0 : sq < 16 ? 1 : sq < 32 ? 2 : 3));
    } else {
        u32(1);
        u32(nnue::kHidden);
    }
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> w(-60, 60), bias(0, 200), ow(-40, 40);
    for (int i = 0; i < sets * nnue::kFeatures * nnue::kHidden; ++i) i16(w(rng));
    for (int i = 0; i < nnue::kHidden; ++i) i16(bias(rng));
    for (int i = 0; i < outputs * 2 * nnue::kHidden; ++i) i16(ow(rng));
    for (int k = 0; k < outputs; ++k) u32(static_cast<uint32_t>(1234 + k));
    return path;
}

// A version 4 network with random weights, kept in file order for a plain
// reference evaluation: 2 king buckets (by rank), 3 output buckets.
struct StackNet {
    int flags, l1, l2, inputs;
    std::vector<int> w1, b1, w2, b2, w3, b3; // [bucket][out][in] ...
};

static std::string writeStackNet(StackNet &n, unsigned seed) {
    const int buckets = 3, sets = 2, H = nnue::kHidden;
    n.inputs = (n.flags & 4) ? H : 2 * H;
    std::string path = "nnue_test_stack.nnue";
    std::ofstream out(path, std::ios::binary);
    out.write("NAGSNNUE", 8);
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) out.put(static_cast<char>((v >> (8 * i)) & 0xFF)); };
    auto i16 = [&](int v) { uint16_t u = static_cast<uint16_t>(v); out.put(static_cast<char>(u & 0xFF)); out.put(static_cast<char>(u >> 8)); };
    u32(4);
    u32(H);
    u32(buckets);
    u32(sets);
    u32(static_cast<uint32_t>(n.flags));
    for (int sq = 0; sq < 64; ++sq) out.put(static_cast<char>(sq < 16 ? 0 : 1));
    u32(static_cast<uint32_t>(n.l1));
    u32(static_cast<uint32_t>(n.l2));
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> fw(-60, 60), fb(0, 200), w(-127, 127);
    std::uniform_int_distribution<int> b1(-255 * 64 / 4, 255 * 64), b2(-255 * 64 / 2, 255 * 64 / 2);
    for (int i = 0; i < sets * nnue::kFeatures * H; ++i) i16(fw(rng));
    for (int i = 0; i < H; ++i) i16(fb(rng));
    for (int k = 0; k < buckets; ++k) {
        auto put16 = [&](std::vector<int> &v, int count, std::uniform_int_distribution<int> &d) {
            for (int i = 0; i < count; ++i) v.push_back(d(rng)), i16(v.back());
        };
        auto put32 = [&](std::vector<int> &v, int count, std::uniform_int_distribution<int> &d) {
            for (int i = 0; i < count; ++i) v.push_back(d(rng)), u32(static_cast<uint32_t>(v.back()));
        };
        put16(n.w1, n.l1 * n.inputs, w);
        put32(n.b1, n.l1, b1);
        put16(n.w2, n.l2 * n.l1, w);
        put32(n.b2, n.l2, b2);
        put16(n.w3, n.l2, w);
        put32(n.b3, 1, b2);
    }
    return path;
}

// The version 4 evaluation written out plainly from the file's layout, with
// C++ division (truncation) wherever the format divides.
static int stackReference(const StackNet &n, const nnue::Accumulator &acc, Color stm, uint64_t occupied) {
    const int H = nnue::kHidden, k = nnue::bucketOf(popcount(occupied), 3);
    std::vector<int> x;
    for (const int16_t *a : {acc.v[colorIndex(stm)], acc.v[colorIndex(stm) ^ 1]}) {
        auto c = [&](int i) { return a[i] < 0 ? 0 : a[i] > 255 ? 255 : a[i]; };
        if (n.flags & 4)
            for (int i = 0; i < H / 2; ++i) x.push_back(c(i) * c(H / 2 + i) / 255);
        else
            for (int i = 0; i < H; ++i) x.push_back((n.flags & 1) ? c(i) * c(i) / 255 : c(i));
    }
    auto layer = [](const std::vector<int> &in, const int *w, const int *b, int outs) {
        std::vector<int> h;
        for (int j = 0; j < outs; ++j) {
            int64_t s = b[j];
            for (size_t i = 0; i < in.size(); ++i) s += static_cast<int64_t>(in[i]) * w[j * in.size() + i];
            const int64_t v = s / 64;
            h.push_back(static_cast<int>(v < 0 ? 0 : v > 255 ? 255 : v));
        }
        return h;
    };
    const auto h1 = layer(x, n.w1.data() + static_cast<size_t>(k) * n.l1 * n.inputs, n.b1.data() + k * n.l1, n.l1);
    const auto h2 = layer(h1, n.w2.data() + static_cast<size_t>(k) * n.l2 * n.l1, n.b2.data() + k * n.l2, n.l2);
    int64_t out = n.b3[k];
    for (int j = 0; j < n.l2; ++j) out += static_cast<int64_t>(h2[j]) * n.w3[k * n.l2 + j];
    return static_cast<int>(out * 400 / (255 * 64));
}

// The engine's version 4 evaluation (AVX2 or portable, whichever this
// machine runs) equals the plain reference through random games.
static void checkStackNetwork(int flags, int l1, int l2, unsigned seed) {
    StackNet n{flags, l1, l2, 0, {}, {}, {}, {}, {}, {}};
    std::string error;
    const std::string path = writeStackNet(n, seed);
    CHECK(nnue::load(path, error), "loading the version 4 network failed: %s", error.c_str());
    std::remove(path.c_str());
    std::mt19937 rng(seed);
    int checked = 0, distinct = 0, last = 0;
    for (const char *start : {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                              "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1"}) {
        for (int game = 0; game < 6; ++game) {
            FastBoard b;
            b.setFromFEN(start);
            for (int ply = 0; ply < 80; ++ply) {
                auto moves = b.generateLegalMoves();
                if (moves.empty()) break;
                b.makeMove(moves[rng() % moves.size()]);
                const int got = nnue::evaluate(b.accumulator(), b.sideToMove(), b.occupancy());
                const int want = stackReference(n, b.accumulator(), b.sideToMove(), b.occupancy());
                CHECK(got == want, "version 4 (flags %d, %d-%d): %d, reference %d at %s", flags, l1, l2, got, want,
                      b.getFEN().c_str());
                distinct += got != last;
                last = got;
                ++checked;
            }
        }
    }
    CHECK(checked > 500 && distinct > checked / 4, "version 4 test too weak: %d positions, %d changes", checked, distinct);
}

static bool sameAccumulator(const FastBoard &b) {
    FastBoard copy = b;
    copy.refreshAccumulator();
    return std::memcmp(&copy.accumulator(), &b.accumulator(), sizeof(nnue::Accumulator)) == 0;
}

static std::string mirrorFen(const std::string &fen) {
    std::istringstream iss(fen);
    std::string placement, stm, castles, ep, hm, fm;
    iss >> placement >> stm >> castles >> ep >> hm >> fm;
    std::vector<std::string> ranks(1);
    for (char c : placement) {
        if (c == '/') ranks.emplace_back();
        else ranks.back() += std::isalpha(static_cast<unsigned char>(c)) ? static_cast<char>(std::isupper(static_cast<unsigned char>(c)) ? std::tolower(c) : std::toupper(c)) : c;
    }
    std::string out;
    for (int i = static_cast<int>(ranks.size()) - 1; i >= 0; --i) out += ranks[i] + (i ? "/" : "");
    return out + (stm == "w" ? " b - - " : " w - - ") + hm + " " + fm;
}

// The same position mirrored left-right (only valid without castling rights).
static std::string flipFiles(const std::string &fen) {
    std::istringstream iss(fen);
    std::string placement, rest;
    iss >> placement;
    std::getline(iss, rest);
    std::string out, rank;
    auto flush = [&] {
        std::string expanded;
        for (char c : rank) expanded += std::isdigit(static_cast<unsigned char>(c)) ? std::string(c - '0', '1') : std::string(1, c);
        std::string rev(expanded.rbegin(), expanded.rend()), packed;
        int empty = 0;
        for (char c : rev) {
            if (c == '1') { ++empty; continue; }
            if (empty) packed += std::to_string(empty), empty = 0;
            packed += c;
        }
        if (empty) packed += std::to_string(empty);
        out += packed;
        rank.clear();
    };
    for (char c : placement) {
        if (c == '/') { flush(); out += '/'; }
        else rank += c;
    }
    flush();
    return out + rest;
}

static void checkNetwork(bool kingBuckets) {
    std::string error;
    CHECK(nnue::load(writeRandomNet(kingBuckets), error), "loading the random network failed: %s", error.c_str());
    CHECK(nnue::network() != nullptr, "network not active after load");
    CHECK(nnue::network()->kingBuckets == (kingBuckets ? 4 : 1), "wrong king bucket count");

    // Random games: after every move, null move and unmake, the incremental
    // accumulators equal recomputed ones.
    const char *starts[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    };
    std::mt19937 rng(7);
    int checked = 0;
    for (const char *start : starts) {
        for (int game = 0; game < 10; ++game) {
            FastBoard b;
            b.setFromFEN(start);
            CHECK(sameAccumulator(b), "accumulator wrong after setFromFEN");
            int made = 0;
            for (int ply = 0; ply < 120; ++ply) {
                auto moves = b.generateLegalMoves();
                if (moves.empty()) break;
                b.makeMove(moves[rng() % moves.size()]);
                ++made;
                CHECK(sameAccumulator(b), "accumulator wrong after a move at %s", b.getFEN().c_str());
                if (!b.inCheck() && rng() % 8 == 0) {
                    b.makeNullMove();
                    CHECK(sameAccumulator(b), "accumulator wrong after a null move");
                    b.unmakeNullMove();
                }
                ++checked;
            }
            for (; made > 0; --made) {
                b.unmakeMove();
                CHECK(sameAccumulator(b), "accumulator wrong after unmake at %s", b.getFEN().c_str());
            }
        }
    }
    CHECK(checked > 1000, "only %d positions checked", checked);

    // Mirrored positions (colours swapped, board flipped) give identical
    // feature sets, so any perspective network evaluates them the same.
    const char *fens[] = {
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w - - 2 3",
        "8/8/1k6/8/8/8/6K1/7Q b - - 0 1",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    };
    for (const char *fen : fens) {
        FastBoard a, m;
        a.setFromFEN(fen);
        CHECK(m.setFromFEN(mirrorFen(fen)), "bad mirror of %s", fen);
        CHECK(eval::evaluate(a) == eval::evaluate(m), "%s: %d vs mirrored %d", fen, eval::evaluate(a), eval::evaluate(m));
    }

    // With mirroring, positions mirrored left-right evaluate the same.
    if (kingBuckets) {
        const char *noCastling[] = {
            "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/2N2N2/PPPP1PPP/R1BQ1RK1 w - - 6 5",
            "8/8/1k6/8/8/8/6K1/7Q b - - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        };
        for (const char *fen : noCastling) {
            FastBoard a, m;
            a.setFromFEN(fen);
            CHECK(m.setFromFEN(flipFiles(fen)), "bad flip of %s", fen);
            CHECK(eval::evaluate(a) == eval::evaluate(m), "%s: %d vs flipped %d", fen, eval::evaluate(a), eval::evaluate(m));
        }
    }
    std::remove((kingBuckets ? "nnue_test_random_kb.nnue" : "nnue_test_random.nnue"));
}

extern const unsigned long kNagsEmbeddedNetSize;

int main() {
    // A network embedded in the build must load: one of another hidden size
    // than NAGS_NNUE_HIDDEN would silently leave every engine and
    // nags_datagen on the hand-written evaluation.
    if (kNagsEmbeddedNetSize > 0) CHECK(nnue::embedded() != nullptr, "the embedded network does not load in this build");
    checkNetwork(false);
    checkNetwork(true);

    // Version 4 (experimental): every activation, mirroring, padded and
    // unpadded layer sizes. A version 3 network loaded afterwards evaluates
    // exactly as before.
    {
        std::string error;
        const std::string v3 = writeRandomNet(true);
        CHECK(nnue::load(v3, error), "loading the random network failed: %s", error.c_str());
        std::vector<int> before;
        FastBoard p;
        for (const char *fen : {"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3", "8/8/1k6/8/8/8/6K1/7Q b - - 0 1"}) {
            p.setFromFEN(fen);
            before.push_back(eval::evaluate(p));
        }
        checkStackNetwork(0, 16, 32, 1);
        checkStackNetwork(1 | 2, 8, 16, 2);
        checkStackNetwork(4, 24, 20, 3);
        checkStackNetwork(4 | 2, 32, 1, 4);
        CHECK(nnue::load(v3, error), "reloading the random network failed: %s", error.c_str());
        std::vector<int> after;
        for (const char *fen : {"r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3", "8/8/1k6/8/8/8/6K1/7Q b - - 0 1"}) {
            p.setFromFEN(fen);
            after.push_back(eval::evaluate(p));
        }
        CHECK(before == after, "a version 3 network evaluates differently after a version 4 one was loaded");
        nnue::useEmbedded();
    }

    // DrawScaling (off by default): scales only where the favoured side has no pawns and at most a minor
    // piece more (and two knights against a bare king), whichever side is to move.
    if (nnue::network()) {
        struct Case { const char *fen; int num, den; }; // expected: scaled = raw * num / den (den 0: 0)
        const Case cases[] = {
            {"8/6k1/2b5/8/8/8/3RK3/8 w - - 0 1", 4, 64},      // rook against bishop
            {"8/6k1/2b5/8/8/8/3RK3/8 b - - 0 1", 4, 64},
            {"8/6k1/2n5/8/8/8/3RK3/8 w - - 0 1", 4, 64},      // rook against knight
            {"8/6k1/8/8/8/8/2NNK3/8 w - - 0 1", 0, 0},        // two knights against the king
            {"8/6k1/8/8/8/8/2B1K3/8 w - - 0 1", 0, 0},        // a lone minor piece (insufficient anyway)
            {"8/6k1/2r5/8/8/8/2RBK3/8 w - - 0 1", 14, 64},    // rook and bishop against rook
            {"8/6k1/8/8/8/8/2BNK3/8 w - - 0 1", 1, 1},        // bishop and knight: a win, unchanged
            {"8/6k1/2r5/8/8/8/3QK3/8 w - - 0 1", 1, 1},       // queen against rook: unchanged
            {"8/6k1/2b5/8/8/8/3RKP2/8 w - - 0 1", 1, 1},      // the rook's side has a pawn: unchanged
        };
        FastBoard p;
        for (const Case &c : cases) {
            CHECK(p.setFromFEN(c.fen), "bad FEN %s", c.fen);
            eval::drawScaling = false;
            const int raw = eval::evaluate(p);
            eval::drawScaling = true;
            const int scaled = eval::evaluate(p);
            eval::drawScaling = false;
            // The rule applies to the side the network favours, here White, the side with more material.
            CHECK((p.sideToMove() == Color::White) == (raw > 0), "%s: the network favours Black (%d)", c.fen, raw);
            const int want = c.den == 0 ? 0 : raw * c.num / c.den;
            CHECK(scaled == want, "DrawScaling %s: raw %d, scaled %d, expected %d", c.fen, raw, scaled, want);
        }
    }

    // The search runs on the network (mate detection does not depend on it).
    TranspositionTable tt(16);
    Searcher<FastBoard> s(&tt);
    FastBoard b;
    b.setFromFEN("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3");
    SearchLimits limits;
    limits.depth = 4;
    std::atomic<bool> stop{false};
    SearchResult r = s.search(b, limits, stop, nullptr);
    CHECK(moveToUciString(r.bestMove) == "f3f7" && r.score == MATE_SCORE - 1, "NNUE search missed mate in 1 (%s)", moveToUciString(r.bestMove).c_str());

    // A network whose output is far beyond any real evaluation (nothing in
    // the format bounds it) is clamped below the tablebase and mate scores,
    // so the search never mistakes it for a proven result.
    {
        const char *path = "nnue_test_huge.nnue";
        std::ofstream out(path, std::ios::binary);
        out.write("NAGSNNUE", 8);
        auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) out.put(static_cast<char>((v >> (8 * i)) & 0xFF)); };
        u32(1);
        u32(nnue::kHidden);
        for (int i = 0; i < (nnue::kFeatures + 1 + 2) * nnue::kHidden; ++i) out.put(0), out.put(0); // weights, biases
        u32(static_cast<uint32_t>(2000000000)); // output bias: about +49 million centipawns
        out.close();
        std::string error;
        CHECK(nnue::load(path, error), "loading the huge network failed: %s", error.c_str());
        FastBoard h;
        CHECK(eval::evaluate(h) == eval::kEvalLimit, "evaluation not clamped: %d", eval::evaluate(h));
        TranspositionTable tt2(4);
        Searcher<FastBoard> s2(&tt2);
        SearchLimits l2;
        l2.depth = 5;
        SearchResult r2 = s2.search(h, l2, stop, nullptr);
        CHECK(std::abs(r2.score) < TB_BOUND, "a clamped evaluation became a proven score: %d", r2.score);
        nnue::useEmbedded();
        std::remove(path);
    }

    // Switching off restores the classical evaluation.
    nnue::setEnabled(false);
    CHECK(nnue::network() == nullptr, "NNUE still active after disabling");
    FastBoard c;
    CHECK(eval::evaluate(c) == 0, "classical start position evaluation should be 0");

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all nnue tests passed\n");
    return 0;
}
