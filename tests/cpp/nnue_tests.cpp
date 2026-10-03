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

// Writes a network file with small random weights.
static std::string writeRandomNet() {
    std::string path = "nnue_test_random.nnue";
    std::ofstream out(path, std::ios::binary);
    out.write("NAGSNNUE", 8);
    auto u32 = [&](uint32_t v) { for (int i = 0; i < 4; ++i) out.put(static_cast<char>((v >> (8 * i)) & 0xFF)); };
    auto i16 = [&](int v) { uint16_t u = static_cast<uint16_t>(v); out.put(static_cast<char>(u & 0xFF)); out.put(static_cast<char>(u >> 8)); };
    u32(1);
    u32(nnue::kHidden);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int> w(-60, 60), bias(0, 200), ow(-40, 40);
    for (int i = 0; i < nnue::kFeatures * nnue::kHidden; ++i) i16(w(rng));
    for (int i = 0; i < nnue::kHidden; ++i) i16(bias(rng));
    for (int i = 0; i < 2 * nnue::kHidden; ++i) i16(ow(rng));
    u32(static_cast<uint32_t>(1234));
    return path;
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

int main() {
    std::string error;
    CHECK(nnue::load(writeRandomNet(), error), "loading the random network failed: %s", error.c_str());
    CHECK(nnue::network() != nullptr, "network not active after load");

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

    // Switching off restores the classical evaluation.
    nnue::setEnabled(false);
    CHECK(nnue::network() == nullptr, "NNUE still active after disabling");
    FastBoard c;
    CHECK(eval::evaluate(c) == 0, "classical start position evaluation should be 0");
    std::remove("nnue_test_random.nnue");

    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all nnue tests passed\n");
    return 0;
}
