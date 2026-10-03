// Syzygy tablebase tests, using the 3-piece tables in tests/data/syzygy.
// Expected results were checked with python-chess.

#include "FastBoard.h"
#include "Search.h"
#include "Syzygy.h"
#include "TT.h"

#include <atomic>
#include <cstdio>
#include <string>

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

static bool wdlOf(const std::string &fen, syzygy::Wdl &wdl) {
    FastBoard b;
    if (!b.setFromFEN(fen)) {
        std::printf("bad test FEN %s\n", fen.c_str());
        ++failures;
        return false;
    }
    return syzygy::probeWdl(syzygy::position(b), wdl);
}

static void testProbe() {
    struct Case {
        const char *fen;
        syzygy::Wdl wdl;
    };
    const Case cases[] = {
        {"8/8/1k6/8/8/8/6K1/7Q w - - 0 1", syzygy::Wdl::Win},
        {"8/8/1k6/8/8/8/6K1/7Q b - - 0 1", syzygy::Wdl::Loss},
        {"8/8/8/4k3/8/8/8/R3K3 b - - 0 1", syzygy::Wdl::Loss},
        {"1k6/8/8/8/8/8/1P6/1K6 w - - 0 1", syzygy::Wdl::Win},
        {"8/8/8/8/3k4/8/4P3/4K3 w - - 0 1", syzygy::Wdl::Draw}, // the black king is in time
        {"8/2k5/8/8/8/8/P7/K7 w - - 0 1", syzygy::Wdl::Draw},   // rook pawn
        {"8/8/8/8/8/2k5/2p5/2K5 b - - 0 1", syzygy::Wdl::Win},
        {"8/8/8/3k4/8/8/8/2B1K3 w - - 0 1", syzygy::Wdl::Draw},
    };
    for (const auto &c : cases) {
        syzygy::Wdl wdl;
        bool ok = wdlOf(c.fen, wdl);
        CHECK(ok && wdl == c.wdl, "%s: WDL %d, expected %d", c.fen, ok ? static_cast<int>(wdl) : -1, static_cast<int>(c.wdl));
    }
    syzygy::Wdl wdl;
    CHECK(!wdlOf("8/8/1k6/8/8/8/6K1/7Q w - - 5 10", wdl), "WDL probed with a nonzero half-move clock");
    CHECK(!wdlOf("8/8/1k6/8/8/8/6K1/6RQ w - - 0 1", wdl), "WDL probed a 4-piece position without its table");

    // Root: the move keeps the win even with the fifty-move counter running.
    FastBoard b;
    b.setFromFEN("8/8/8/4k3/8/8/8/R3K3 w - - 40 60");
    syzygy::RootResult r;
    CHECK(syzygy::probeRoot(syzygy::position(b), r) && r.wdl == syzygy::Wdl::Win, "KRvK root probe failed");
    bool legal = false;
    for (const Move &m : b.generateLegalMoves()) legal |= m.from == r.from && m.to == r.to;
    CHECK(legal, "root probe returned an illegal move");
    // With 99 half-moves gone the win cannot be forced in time.
    b.setFromFEN("8/8/8/4k3/8/8/8/R3K3 w - - 99 60");
    CHECK(syzygy::probeRoot(syzygy::position(b), r) && r.wdl != syzygy::Wdl::Win, "fifty-move rule ignored at the root");
}

static void testSearch() {
    TranspositionTable tt(16);
    Searcher<FastBoard> s(&tt);
    std::atomic<bool> stop{false};
    SearchLimits limits;
    limits.depth = 4;
    uint64_t tbHits = 0;
    auto onInfo = [&](const SearchInfo &info) { tbHits = info.tbHits; };

    // Four pieces (no table): Rxd2 reaches a won KRvK, found by the search's
    // WDL probe one ply in.
    FastBoard b;
    b.setFromFEN("8/8/8/8/8/5k2/3r4/K2R4 w - - 0 1");
    SearchResult r = s.search(b, limits, stop, onInfo);
    CHECK(moveToUciString(r.bestMove) == "d1d2" && r.score == TB_WIN_SCORE - 1,
          "expected Rxd2 with a tablebase win, got %s %d", moveToUciString(r.bestMove).c_str(), r.score);
    CHECK(tbHits > 0, "no tablebase hits reported");

    // Three pieces: the root probe picks the move.
    b.setFromFEN("8/8/8/8/3k4/8/4P3/4K3 w - - 0 1");
    r = s.search(b, limits, stop, onInfo);
    CHECK(!r.bestMove.isNull() && r.score == 0, "KPvK draw scored %d", r.score);

    // Mate and tablebase scores keep their distance through the TT.
    CHECK(scoreFromTT(scoreToTT(TB_WIN_SCORE - 7, 7), 3) == TB_WIN_SCORE - 3, "TB score not re-based to a new ply");
    CHECK(TB_WIN_SCORE < MATE_BOUND && TB_BOUND > 10000, "TB score range overlaps mates or evaluations");
}

int main() {
    int largest = syzygy::init(NAGS_SYZYGY_DIR);
    CHECK(largest == 3 && syzygy::cardinality() == 3, "expected 3-piece tables in %s, got %d", NAGS_SYZYGY_DIR, largest);
    if (largest == 3) {
        testProbe();
        testSearch();
        syzygy::setProbeLimit(2);
        syzygy::Wdl wdl;
        CHECK(syzygy::cardinality() == 2, "probe limit ignored");
        syzygy::setProbeLimit(7);
        CHECK(syzygy::init("") == 0 && syzygy::cardinality() == 0, "tables not unloaded");
        CHECK(!wdlOf("8/8/1k6/8/8/8/6K1/7Q w - - 0 1", wdl), "probe succeeded after unloading");
    }
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all syzygy tests passed\n");
    return 0;
}
