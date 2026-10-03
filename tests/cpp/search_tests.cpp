// Evaluation, transposition table, time management and search tests.

#include "Board.h"
#include "FastBoard.h"
#include "Search.h"
#include "TT.h"
#include "Uci.h"

#include <atomic>
#include <cstdio>
#include <random>
#include <sstream>
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

// Colour-flipped mirror image of a FEN (ranks reversed, case swapped).
static std::string mirrorFen(const std::string &fen) {
    std::istringstream iss(fen);
    std::string placement, stm, castles, ep, hm, fm;
    iss >> placement >> stm >> castles >> ep >> hm >> fm;
    std::vector<std::string> ranks;
    std::string cur;
    for (char c : placement) {
        if (c == '/') { ranks.push_back(cur); cur.clear(); }
        else cur += c;
    }
    ranks.push_back(cur);
    std::string out;
    for (int i = static_cast<int>(ranks.size()) - 1; i >= 0; --i) {
        for (char c : ranks[i]) out += std::isalpha(static_cast<unsigned char>(c)) ? (std::isupper(static_cast<unsigned char>(c)) ? static_cast<char>(std::tolower(c)) : static_cast<char>(std::toupper(c))) : c;
        if (i) out += '/';
    }
    std::string cs;
    for (char c : castles) cs += c == '-' ? '-' : (std::isupper(static_cast<unsigned char>(c)) ? static_cast<char>(std::tolower(c)) : static_cast<char>(std::toupper(c)));
    std::string mirroredCs;
    for (char want : std::string("KQkq")) if (cs.find(want) != std::string::npos) mirroredCs += want;
    if (mirroredCs.empty()) mirroredCs = "-";
    std::string mep = ep == "-" ? "-" : std::string{ep[0], static_cast<char>('1' + ('8' - ep[1]))};
    return out + " " + (stm == "w" ? "b" : "w") + " " + mirroredCs + " " + mep + " " + hm + " " + fm;
}

template <class B>
static int evalFen(const std::string &fen) {
    B b;
    if (!b.setFromFEN(fen)) {
        std::printf("bad test FEN %s\n", fen.c_str());
        ++failures;
        return 0;
    }
    return eval::evaluate(b);
}

static void testEvalOrientation() {
    // A white pawn is worth more the further it has advanced.
    int e2 = evalFen<Board>("7k/8/8/8/8/8/4P3/K7 w - - 0 1");
    int e4 = evalFen<Board>("7k/8/8/8/4P3/8/8/K7 w - - 0 1");
    int e7 = evalFen<Board>("7k/4P3/8/8/8/8/8/K7 w - - 0 1");
    CHECK(e7 > e4 && e4 > e2, "pawn PST upside down: e2=%d e4=%d e7=%d", e2, e4, e7);
    // A castled king is better than one wandering in the middlegame.
    const char *castled = "rnbq1rk1/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 w - - 0 1";
    const char *wandering = "rnbq1rk1/pppppppp/8/8/4K3/8/PPPPPPPP/RNBQ1R2 w - - 0 1";
    CHECK(evalFen<Board>(castled) > evalFen<Board>(wandering), "king safety PST upside down");
}

static void testEvalSymmetry() {
    std::mt19937 rng(7);
    int checked = 0;
    for (int game = 0; game < 30; ++game) {
        FastBoard b;
        for (int ply = 0; ply < 80; ++ply) {
            auto moves = b.generateLegalMoves();
            if (moves.empty()) break;
            b.makeMove(moves[rng() % moves.size()]);
            std::string fen = b.getFEN();
            std::string mirror = mirrorFen(fen);
            int a = evalFen<FastBoard>(fen), m = evalFen<FastBoard>(mirror);
            int ab = evalFen<Board>(fen);
            CHECK(a == m, "eval not colour-symmetric: %s -> %d, mirror %s -> %d", fen.c_str(), a, mirror.c_str(), m);
            CHECK(a == ab, "Board and FastBoard evaluate differently at %s", fen.c_str());
            ++checked;
        }
    }
    CHECK(checked > 500, "symmetry test only checked %d positions", checked);
}

static void testTT() {
    TranspositionTable tt(1);
    CHECK(tt.entryCount() * TranspositionTable::entryBytes() <= 1024 * 1024, "1 MB table uses more than 1 MB");
    TranspositionTable tt3(3);
    CHECK(tt3.entryCount() * TranspositionTable::entryBytes() <= 3 * 1024 * 1024, "3 MB table uses more than 3 MB");

    Move m{12, 28, Piece::None, false, false};
    tt.store(0xABCDEF1234567ULL, 7, Bound::Lower, 123, m);
    TTHit hit;
    CHECK(tt.probe(0xABCDEF1234567ULL, hit), "stored entry not found");
    CHECK(hit.depth == 7 && hit.bound == Bound::Lower && hit.score == 123 && sameMove(hit.move, m), "entry round trip failed");
    CHECK(!tt.probe(0xABCDEF1234568ULL, hit), "probe matched a different key");

    Move promo{52, 60, Piece::BN, false, false};
    tt.store(42, 3, Bound::Exact, -5, promo);
    CHECK(tt.probe(42, hit) && sameMove(hit.move, promo), "promotion kind lost in TT");
    tt.store(42, 1, Bound::Upper, 9, Move{});
    CHECK(tt.probe(42, hit) && sameMove(hit.move, promo), "stored best move dropped when a moveless entry replaced it");

    // Mate scores are stored relative to the node.
    int mateIn3FromRoot = MATE_SCORE - 5;
    int stored = scoreToTT(mateIn3FromRoot, 2);
    CHECK(scoreFromTT(stored, 2) == mateIn3FromRoot, "mate score round trip");
    CHECK(scoreFromTT(stored, 4) == mateIn3FromRoot - 2, "mate score not re-based to a new ply");

    tt.clear();
    CHECK(tt.hashfull() == 0, "hashfull not zero after clear");
}

static void testLimits() {
    uci::GoParams go;
    std::string err;
    CHECK(uci::parseGo({"go", "wtime", "60000", "btime", "1000", "winc", "0", "binc", "0"}, go, err), "parseGo failed: %s", err.c_str());
    SearchLimits w = uci::computeLimits(go, true, 50), bl = uci::computeLimits(go, false, 50);
    CHECK(w.hardMs > 0 && w.hardMs <= 30000 && w.softMs <= w.hardMs, "white limits %lld/%lld", (long long)w.softMs, (long long)w.hardMs);
    CHECK(bl.hardMs > 0 && bl.hardMs <= 950, "black limits exceed clock: %lld", (long long)bl.hardMs);
    CHECK(uci::parseGo({"go", "movetime", "500"}, go, err), "parseGo movetime");
    SearchLimits mt = uci::computeLimits(go, true, 50);
    CHECK(mt.hardMs == 450 && mt.softMs == 450, "movetime limits %lld/%lld", (long long)mt.softMs, (long long)mt.hardMs);
    CHECK(uci::parseGo({"go", "depth", "5"}, go, err) && go.depth == 5 && !go.infinite, "parseGo depth");
    CHECK(uci::parseGo({"go"}, go, err) && go.infinite, "bare go should be infinite");
    CHECK(!uci::parseGo({"go", "wtime", "abc"}, go, err), "accepted non-numeric wtime");
    CHECK(uci::parseGo({"go", "wtime", "10", "btime", "10", "movestogo", "1"}, go, err), "parseGo tiny clock");
    SearchLimits tiny = uci::computeLimits(go, true, 50);
    CHECK(tiny.hardMs >= 1 && tiny.hardMs <= 10, "tiny clock limits %lld", (long long)tiny.hardMs);
    CHECK(uci::parseGo({"go", "wtime", "60000"}, go, err), "parseGo wtime only");
    SearchLimits other = uci::computeLimits(go, false, 50);
    CHECK(other.hardMs > 0, "black with only wtime given must still get a time limit");
}

template <class B>
static SearchResult searchFen(const std::string &fen, int depth, TranspositionTable *tt, int threads = 1) {
    B b;
    if (!b.setFromFEN(fen)) {
        std::printf("bad test FEN %s\n", fen.c_str());
        ++failures;
        return {};
    }
    Searcher<B> s(tt);
    s.setThreads(threads);
    SearchLimits limits;
    limits.depth = depth;
    std::atomic<bool> stop{false};
    return s.search(b, limits, stop, nullptr);
}

template <class B>
static void testSearch(TranspositionTable *tt, const char *label) {
    SearchResult r = searchFen<B>("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 2 3", 4, tt);
    CHECK(moveToUciString(r.bestMove) == "f3f7" && r.score == MATE_SCORE - 1, "%s: missed mate in 1 (%s, %d)", label, moveToUciString(r.bestMove).c_str(), r.score);

    // Mate in 2 (verified with python-chess), so exactly 3 plies to mate.
    r = searchFen<B>("1k6/8/2K5/8/8/8/8/7R w - - 0 1", 6, tt);
    CHECK(r.score == MATE_SCORE - 3, "%s: mate in 2 not found (score %d, move %s)", label, r.score, moveToUciString(r.bestMove).c_str());
    CHECK(formatScore(r.score) == "mate 2", "%s: mate in 2 formatted as '%s'", label, formatScore(r.score).c_str());

    // Win the hanging queen.
    r = searchFen<B>("4k3/8/8/3q4/8/8/8/3QK3 w - - 0 1", 4, tt);
    CHECK(moveToUciString(r.bestMove) == "d1d5", "%s: did not capture hanging queen (%s)", label, moveToUciString(r.bestMove).c_str());

    // A rook down, White forces a perpetual (Qe8+ Kh7 Qh5+ Kg8 ..., Black's
    // replies are forced), so repetition detection must score it as a draw.
    r = searchFen<B>("6k1/r5p1/5p2/8/8/8/1q3PPP/4Q1K1 w - - 0 1", 8, tt);
    CHECK(r.score == 0 && moveToUciString(r.bestMove) == "e1e8", "%s: expected perpetual Qe8+ scored 0, got %d (%s)", label, r.score, moveToUciString(r.bestMove).c_str());

    // Stalemate is a draw, not a win: with K+Q vs K, don't stalemate.
    r = searchFen<B>("k7/8/1K6/8/8/8/8/2Q5 w - - 0 1", 4, tt);
    CHECK(r.score >= MATE_BOUND, "%s: K+Q vs K should see mate (score %d, %s)", label, r.score, moveToUciString(r.bestMove).c_str());

    // No legal moves: no best move.
    r = searchFen<B>("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 3, tt);
    CHECK(r.bestMove.isNull(), "%s: stalemated side returned a move", label);
}

static void testThreadsAndStop() {
    TranspositionTable tt(16);
    SearchResult r = searchFen<FastBoard>("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 7, &tt, 4);
    CHECK(!r.bestMove.isNull() && r.depth == 7, "4-thread search did not complete depth 7");

    // An immediate stop still yields a legal move.
    FastBoard b;
    Searcher<FastBoard> s(&tt);
    SearchLimits limits;
    std::atomic<bool> stop{true};
    r = s.search(b, limits, stop, nullptr);
    bool legal = false;
    for (const Move &m : b.generateLegalMoves()) legal |= sameMove(m, r.bestMove);
    CHECK(legal, "stopped search returned an illegal move %s", moveToUciString(r.bestMove).c_str());
}

int main() {
    testEvalOrientation();
    testEvalSymmetry();
    testTT();
    testLimits();
    TranspositionTable tt(16);
    testSearch<Board>(nullptr, "Board/noTT");
    testSearch<FastBoard>(&tt, "FastBoard/TT");
    testThreadsAndStop();
    if (failures) {
        std::printf("%d failure(s)\n", failures);
        return 1;
    }
    std::printf("all search tests passed\n");
    return 0;
}
