// Board correctness tests, compiled once against Board and once against
// FastBoard (TEST_FAST_BOARD). Run via ctest.
//
// Perft references were cross-checked with python-chess.

#ifdef TEST_FAST_BOARD
#include "FastBoard.h"
using TestBoard = FastBoard;
static const char *kBoardName = "FastBoard";
#else
#include "Board.h"
using TestBoard = Board;
static const char *kBoardName = "Board";
#endif

#include <cstdio>
#include <random>
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

struct PerftCase {
    const char *name;
    const char *fen;
    int depth;
    unsigned long long nodes;
};

static const PerftCase kPerft[] = {
    {"startpos", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 4, 197281ULL},
    {"kiwipete", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 3, 97862ULL},
    {"pos3", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674624ULL},
    {"pos4", "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", 3, 9467ULL},
    {"pos4mirror", "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1", 3, 9467ULL},
    {"pos5", "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62379ULL},
    {"pos6", "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P3/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 81467ULL},
    {"ep_pin", "8/8/8/K2pP2q/8/8/8/7k w - d6 0 1", 3, 776ULL},
    {"promo", "n1n5/PPPk4/8/8/8/8/4Kppp/5N1N b - - 0 1", 3, 9483ULL},
};

static void testPerft() {
    for (const auto &c : kPerft) {
        TestBoard b;
        CHECK(b.setFromFEN(c.fen), "%s: FEN rejected", c.name);
        unsigned long long n = b.perft(c.depth);
        CHECK(n == c.nodes, "%s perft(%d) = %llu, expected %llu", c.name, c.depth, n, c.nodes);
        CHECK(b.getFEN() == std::string(c.fen), "%s: board not restored after perft: %s", c.name, b.getFEN().c_str());
    }
}

// Plays seeded random games and checks, at every ply, that the incrementally
// updated hash equals the hash of the same position loaded from its FEN, and
// that unmaking every move restores the original position exactly.
static void testHashConsistency() {
    const char *starts[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    };
    std::mt19937 rng(12345);
    int checked = 0;
    for (const char *start : starts) {
        for (int game = 0; game < 20; ++game) {
            TestBoard b;
            b.setFromFEN(start);
            std::vector<std::string> fens{b.getFEN()};
            std::vector<uint64_t> hashes{b.zobrist()};
            for (int ply = 0; ply < 150; ++ply) {
                auto moves = b.generateLegalMoves();
                if (moves.empty() || b.getHalfmoveClock() >= 100) break;
                b.makeMove(moves[rng() % moves.size()]);
                TestBoard fresh;
                CHECK(fresh.setFromFEN(b.getFEN()), "FEN round trip rejected: %s", b.getFEN().c_str());
                CHECK(fresh.zobrist() == b.zobrist(), "hash mismatch after %d plies at %s", ply + 1, b.getFEN().c_str());
                fens.push_back(b.getFEN());
                hashes.push_back(b.zobrist());
                ++checked;
            }
            for (size_t i = fens.size() - 1; i > 0; --i) {
                b.unmakeMove();
                CHECK(b.getFEN() == fens[i - 1], "unmake did not restore %s (got %s)", fens[i - 1].c_str(), b.getFEN().c_str());
                CHECK(b.zobrist() == hashes[i - 1], "unmake did not restore hash at ply %zu", i - 1);
            }
        }
    }
    CHECK(checked > 1000, "hash test only checked %d positions", checked);
}

static void testFenValidation() {
    TestBoard b;
    const std::string start = b.getFEN();
    const char *bad[] = {
        "",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1",          // 7 ranks
        "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", // 9 files
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1", // bad side
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq e9 0 1", // bad ep
        "rnbq1bnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQ - 0 1",   // no black king
        "4k3/8/8/8/8/8/8/4K2R w KQkqX - 0 1",                       // bad castling char
        "4k3/8/8/8/8/8/8/R3K2R b KQ - 0 1",                         // fine except...
    };
    for (int i = 0; i < 7; ++i) {
        CHECK(!b.setFromFEN(bad[i]), "accepted invalid FEN '%s'", bad[i]);
        CHECK(b.getFEN() == start, "invalid FEN '%s' modified the board", bad[i]);
    }
    CHECK(b.setFromFEN(bad[7]), "rejected valid FEN '%s'", bad[7]);
    // Side not to move in check is illegal.
    CHECK(!b.setFromFEN("4k3/8/8/8/8/8/8/4R1K1 w - - 0 1"), "accepted position where side not to move is in check");
    // Castling rights without the rook are dropped rather than trusted.
    CHECK(b.setFromFEN("4k3/8/8/8/8/8/8/4K3 w KQ - 0 1"), "rejected FEN with stale castling rights");
    for (const Move &m : b.generateLegalMoves()) CHECK(!m.isCastling, "castling generated without a rook");
}

static void testDraws() {
    TestBoard b;
    b.setStartPos();
    CHECK(!b.isDraw(), "start position reported as draw");
    CHECK(b.applyMovesUCI({"g1f3", "g8f6", "f3g1", "f6g8"}), "knight shuffle rejected");
    CHECK(b.isRepetition(), "repetition of the start position not detected");
    b.setStartPos();
    CHECK(b.applyMovesUCI({"e2e4", "g8f6", "g1f3", "f6g8", "f3g1"}), "moves rejected");
    CHECK(!b.isRepetition(), "false repetition after a pawn move");
    CHECK(b.setFromFEN("8/8/4k3/8/8/3NK3/8/8 w - - 0 1") && b.isInsufficientMaterial(), "KN vs K not a draw");
    CHECK(b.setFromFEN("8/8/4k3/8/8/3RK3/8/8 w - - 0 1") && !b.isInsufficientMaterial(), "KR vs K treated as draw");
    CHECK(b.setFromFEN("8/8/4k3/8/8/3RK3/8/8 w - - 100 80") && b.isDraw(), "fifty-move rule not detected");
    // Checkmate on the 100th half-move is still checkmate.
    CHECK(b.setFromFEN("7k/6Q1/6K1/8/8/8/8/8 b - - 100 80") && !b.isDraw(), "mate overridden by fifty-move rule");
}

static void testApplyMoves() {
    TestBoard b;
    CHECK(b.applyMovesUCI({"e2e4", "e7e5"}), "legal moves rejected");
    CHECK(!b.applyMovesUCI({"e1e3"}), "illegal move accepted");
    CHECK(!b.applyMovesUCI({"e2e4x"}), "malformed move accepted");
    b.setFromFEN("8/P6k/8/8/8/8/8/K7 w - - 0 1");
    CHECK(b.applyMovesUCI({"a7a8n"}), "under-promotion rejected");
    CHECK(b.pieceAt(56) == Piece::WN, "under-promotion produced the wrong piece");
}

static void testNullMove() {
    TestBoard b;
    b.setFromFEN("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2");
    const std::string fen = b.getFEN();
    const uint64_t hash = b.zobrist();
    b.makeNullMove();
    CHECK(b.sideToMove() == Color::Black && b.lastMoveWasNull(), "null move did not pass the turn");
    TestBoard flipped;
    flipped.setFromFEN("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 1 2");
    CHECK(b.zobrist() == flipped.zobrist(), "null-move hash differs from the side-flipped position");
    b.unmakeNullMove();
    CHECK(b.getFEN() == fen && b.zobrist() == hash, "unmakeNullMove did not restore the position");

    // A repetition is not looked for across a null move.
    b.setStartPos();
    b.applyMovesUCI({"g1f3", "g8f6"});
    b.makeNullMove();
    b.makeNullMove();
    CHECK(!b.isRepetition(), "repetition detected across a null move");
}

int main() {
    testPerft();
    testNullMove();
    testHashConsistency();
    testFenValidation();
    testDraws();
    testApplyMoves();
    if (failures) {
        std::printf("%s: %d failure(s)\n", kBoardName, failures);
        return 1;
    }
    std::printf("%s: all board tests passed\n", kBoardName);
    return 0;
}
