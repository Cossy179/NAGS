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

#include "Eval.h"
#include "Legality.h"

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
        const std::string before = b.getFEN(); // ep_pin's d6 is dropped: no legal capture there
        unsigned long long n = b.perft(c.depth);
        CHECK(n == c.nodes, "%s perft(%d) = %llu, expected %llu", c.name, c.depth, n, c.nodes);
        CHECK(b.getFEN() == before, "%s: board not restored after perft: %s", c.name, b.getFEN().c_str());
    }
}

#ifdef TEST_FAST_BOARD
static_assert(eval::HasIncrementalEval<FastBoard>::value, "FastBoard should evaluate incrementally");
#endif

// Incrementally updated evaluation terms (FastBoard) must equal the ones
// computed from scratch.
template <class B>
static bool incrementalEvalOk(const B &b) {
    if constexpr (eval::HasIncrementalEval<B>::value) {
        int score, phase;
        eval::materialAndPhase(b, score, phase);
        return score == b.psqScore() && phase == b.gamePhase();
    } else {
        (void)b;
        return true;
    }
}

// Plays seeded random games and checks, at every ply, that the incrementally
// updated hash (and evaluation terms) equal those of the same position loaded
// from its FEN, and that unmaking every move restores the original position
// exactly.
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
                CHECK(incrementalEvalOk(b), "incremental evaluation terms wrong after %d plies at %s", ply + 1, b.getFEN().c_str());
                fens.push_back(b.getFEN());
                hashes.push_back(b.zobrist());
                ++checked;
            }
            for (size_t i = fens.size() - 1; i > 0; --i) {
                b.unmakeMove();
                CHECK(b.getFEN() == fens[i - 1], "unmake did not restore %s (got %s)", fens[i - 1].c_str(), b.getFEN().c_str());
                CHECK(b.zobrist() == hashes[i - 1], "unmake did not restore hash at ply %zu", i - 1);
                CHECK(incrementalEvalOk(b), "unmake did not restore the evaluation terms at ply %zu", i - 1);
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

    // More material than the starting position allows (it would overflow the
    // move list) is rejected.
    b.setStartPos();
    CHECK(!b.setFromFEN("BQQQQQBk/Q5RB/Q6Q/Q6Q/Q6Q/Q6Q/Q6Q/KQQQQQQQ w - - 0 1"), "accepted 40 queens");
    CHECK(!b.setFromFEN("4k3/8/8/8/8/8/PPPPPPPP/N2NK1NN w - - 0 1"), "accepted 3 extra knights with 8 pawns");
    CHECK(b.setFromFEN("4k3/8/8/8/8/8/PPPPPPP1/N2NK1N1 w - - 0 1"), "rejected 1 extra knight with 7 pawns");
    CHECK(b.setFromFEN("4k3/8/8/8/8/8/PPPPPP2/QQQ1K3 w - - 0 1"), "rejected 2 extra queens with 6 pawns");
    // An en-passant square no double push can have produced is ignored: the
    // capture would remove a pawn that is not there.
    CHECK(b.setFromFEN("4k3/8/8/3P4/8/8/8/4K3 w - e6 0 1") && b.getFEN() == "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",
          "kept an en-passant square without a pawn: %s", b.getFEN().c_str());
    CHECK(b.setFromFEN("4k3/8/4n3/3Pp3/8/8/8/4K3 w - e6 0 1") && b.getFEN() == "4k3/8/4n3/3Pp3/8/8/8/4K3 w - - 0 1",
          "kept an occupied en-passant square: %s", b.getFEN().c_str());
    for (const Move &m : b.generateLegalMoves()) CHECK(!m.isEnPassant, "en passant onto an occupied square");
    CHECK(b.setFromFEN("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3") &&
              b.getFEN() == "rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3",
          "dropped a valid en-passant square: %s", b.getFEN().c_str());
    // An en-passant square no pawn can use is dropped, as after the move itself.
    CHECK(b.setFromFEN("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1") &&
              b.getFEN() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
          "kept an unusable en-passant square: %s", b.getFEN().c_str());
    b.setStartPos();
    CHECK(b.applyMovesUCI({"e2e4"}) && b.getFEN() == "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
          "unusable en-passant square after e2e4: %s", b.getFEN().c_str());
    CHECK(b.setFromFEN("rnbqkbnr/ppp1pppp/8/8/3p4/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") && b.applyMovesUCI({"e2e4"}) &&
              b.getFEN() == "rnbqkbnr/ppp1pppp/8/8/3pP3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
          "usable en-passant square lost after e2e4: %s", b.getFEN().c_str());
    // Move counters: "0 0" (written by some tools) is accepted.
    CHECK(b.setFromFEN("4k3/8/8/8/8/8/8/R3K3 w Q - 0 0") && b.getFEN() == "4k3/8/8/8/8/8/8/R3K3 w Q - 0 1",
          "fullmove number 0 not accepted: %s", b.getFEN().c_str());
}

static void testDraws() {
    TestBoard b;
    b.setStartPos();
    CHECK(!b.isDraw(), "start position reported as draw");
    CHECK(b.applyMovesUCI({"g1f3", "g8f6", "f3g1", "f6g8"}), "knight shuffle rejected");
    CHECK(b.isRepetition(), "repetition of the start position not detected");
    // The laws' rule (RepetitionRule Laws): a repetition inside the search tree
    // is a draw, one of a position from before the root (or of the root) only
    // at its third occurrence; with 0 plies from the root, the claimable threefold.
    CHECK(!b.isRepetition(0) && !b.isRepetition(4) && !b.isDraw(0), "second occurrence of a pre-root position counted as a draw");
    CHECK(b.isRepetition(5) && b.isDraw(5), "repetition inside the search tree not detected");
    CHECK(b.applyMovesUCI({"g1f3", "g8f6", "f3g1", "f6g8"}) && b.isRepetition(0) && b.isDraw(0), "third occurrence not detected");
    // After pawn moves, a knight shuffle repeats the position reached by them.
    b.setStartPos();
    CHECK(b.applyMovesUCI({"e2e4", "e7e5", "g1f3", "g8f6", "f3g1", "f6g8"}) && b.isRepetition(),
          "repetition after pawn moves not detected");
    // No black pawn can take on e3, so the position after e2e4 has no
    // en-passant right and recurs after the knight shuffle...
    b.setStartPos();
    CHECK(b.applyMovesUCI({"e2e4", "g8f6", "g1f3", "f6g8"}) && !b.isRepetition(), "false repetition");
    CHECK(b.applyMovesUCI({"f3g1"}) && b.isRepetition(), "repetition missed: unusable en-passant square counted");
    // ...but with a pawn on d4 it has one, and the later position differs.
    CHECK(b.setFromFEN("rnbqkb1r/ppp1pppp/5n2/8/3p4/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1") &&
              b.applyMovesUCI({"e2e4", "f6g8", "g1f3", "g8f6", "f3g1"}) && !b.isRepetition(),
          "false repetition: a usable en-passant right ignored");
    CHECK(b.setFromFEN("8/8/4k3/8/8/3NK3/8/8 w - - 0 1") && b.isInsufficientMaterial(), "KN vs K not a draw");
    CHECK(b.setFromFEN("8/8/4k3/8/8/3RK3/8/8 w - - 0 1") && !b.isInsufficientMaterial(), "KR vs K treated as draw");
    CHECK(b.setFromFEN("8/8/4k3/8/8/3RK3/8/8 w - - 100 80") && b.isDraw(), "fifty-move rule not detected");
    // Checkmate on the 100th half-move is still checkmate.
    CHECK(b.setFromFEN("7k/6Q1/6K1/8/8/8/8/8 b - - 100 80") && !b.isDraw(), "mate overridden by fifty-move rule");
}

// An en-passant right exists only if a capture there is legal (python-chess's
// has_legal_en_passant, and the laws' "same position"): otherwise the square
// is dropped, from FENs and after double pushes, and the position hashes and
// repeats as the one without it.
static void testEnPassantLegality() {
    struct Case {
        const char *withEp, *without;
        bool legal;
        const char *what;
    };
    const Case cases[] = {
        {"4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 2", "4r2k/8/8/3pP3/8/8/8/4K3 w - - 0 2", false, "pawn pinned on its file"},
        {"7k/8/8/K2pP2q/8/8/8/8 w - d6 0 2", "7k/8/8/K2pP2q/8/8/8/8 w - - 0 2", false, "both pawns between king and queen"},
        {"7k/8/8/3pP3/8/8/B7/K7 w - d6 0 2", "7k/8/8/3pP3/8/8/B7/K7 w - - 0 2", true, "an ordinary capture"},
        {"8/8/8/2kpP3/8/8/8/4K3 w - d6 0 2", "8/8/8/2kpP3/8/8/8/4K3 w - - 0 2", true, "next to the other king"},
        {"7k/8/8/3pP3/8/1b6/8/5K2 w - d6 0 2", "7k/8/8/3pP3/8/1b6/8/5K2 w - - 0 2", true, "a bishop elsewhere"},
        // The pushed pawn gives check and taking it en passant is the answer.
        {"8/8/8/5k2/3pP3/8/8/7K b - e3 0 1", "8/8/8/5k2/3pP3/8/8/7K b - - 0 1", true, "taking the checking pawn"},
        // The capturing pawn is pinned on the diagonal it captures along.
        {"8/k7/8/8/3pP2K/8/8/6B1 b - e3 0 1", "8/k7/8/8/3pP2K/8/8/6B1 b - - 0 1", true, "pinned along the capture"},
        {"8/8/8/8/3pP3/8/8/k2K2R1 b - e3 0 1", "8/8/8/8/3pP3/8/8/k2K2R1 b - - 0 1", true, "a rook on the other rank"},
        {"8/8/8/8/k2pP2R/8/8/3K4 b - e3 0 1", "8/8/8/8/k2pP2R/8/8/3K4 b - - 0 1", false, "black pawns between king and rook"},
    };
    for (const Case &c : cases) {
        TestBoard a, b;
        CHECK(a.setFromFEN(c.withEp) && b.setFromFEN(c.without), "%s: FEN rejected", c.what);
        const bool kept = a.getFEN() != b.getFEN();
        CHECK(kept == c.legal, "%s: en-passant square %s (%s)", c.what, kept ? "kept" : "dropped", a.getFEN().c_str());
        CHECK((a.zobrist() == b.zobrist()) == !c.legal, "%s: hash %s the position without the right", c.what,
              c.legal ? "equals" : "differs from");
    }
    // Through a game: d7-d5 cannot be taken (pin), so after king moves the
    // position is the one after d5 again.
    TestBoard g;
    CHECK(g.setFromFEN("4r2k/3p4/8/4P3/8/8/8/4K3 b - - 0 1") && g.applyMovesUCI({"d7d5"}) &&
              g.getFEN() == "4r2k/8/8/3pP3/8/8/8/4K3 w - - 0 2",
          "unusable en-passant square kept after d7-d5: %s", g.getFEN().c_str());
    CHECK(g.applyMovesUCI({"e1d1", "h8g8", "d1e1", "g8h8"}) && g.isRepetition(), "repetition through d7-d5 missed");
    // With a legal capture the right stays, and the later position differs.
    CHECK(g.setFromFEN("7k/3p4/8/4P3/8/8/8/4K3 b - - 0 1") && g.applyMovesUCI({"d7d5"}) &&
              g.getFEN() == "7k/8/8/3pP3/8/8/8/4K3 w - d6 0 2",
          "usable en-passant square dropped: %s", g.getFEN().c_str());
    CHECK(g.applyMovesUCI({"e1d1", "h8g8", "d1e1", "g8h8"}) && !g.isRepetition(), "false repetition with a usable right");
}

// Dead positions without pawns, rooks or queens: at most one minor piece, or
// only bishops all on squares of one colour (python-chess agrees on all).
static void testInsufficientMaterial() {
    struct Case {
        const char *fen;
        bool dead;
    };
    const Case cases[] = {
        {"8/8/4k3/8/8/3NK3/8/8 w - - 0 1", true},     // KN vs K
        {"8/7k/8/8/B7/8/8/3bK3 w - - 0 1", true},     // KB vs KB, both on light squares
        {"8/8/3k4/5B2/B7/8/8/K7 b - - 0 1", true},    // KBB (one colour) vs K
        {"8/2k5/2b5/8/8/3K4/8/1B1B4 w - - 0 1", true}, // three bishops, one colour
        {"8/7k/8/8/B7/8/8/2b1K3 w - - 0 1", false},   // KB vs KB, opposite colours: mate possible
        {"8/8/4k3/8/8/2BNK3/8/8 w - - 0 1", false},   // KBN vs K
        {"8/8/4k3/8/8/2NNK3/8/8 w - - 0 1", false},   // KNN vs K (a mate exists)
        {"8/8/4k3/8/8/3BK3/8/2b5 w - - 0 1", false},  // bishops on both colours
        {"8/8/4k3/8/8/3RK3/8/8 w - - 0 1", false},
    };
    for (const Case &c : cases) {
        TestBoard b;
        CHECK(b.setFromFEN(c.fen) && b.isInsufficientMaterial() == c.dead, "%s: %s", c.fen,
              c.dead ? "dead position not seen" : "playable position called dead");
    }
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
    // A usable en-passant square (the e5 pawn can take on d6), which the null move must clear.
    b.setFromFEN("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3");
    const std::string fen = b.getFEN();
    CHECK(fen.find(" d6 ") != std::string::npos, "test position lost its en-passant square: %s", fen.c_str());
    const uint64_t hash = b.zobrist();
    b.makeNullMove();
    CHECK(b.sideToMove() == Color::Black && b.lastMoveWasNull(), "null move did not pass the turn");
    TestBoard flipped;
    flipped.setFromFEN("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 1 3");
    CHECK(b.zobrist() == flipped.zobrist(), "null-move hash differs from the side-flipped position");
    CHECK(b.getFEN() == flipped.getFEN(), "null move left the en-passant square set: %s", b.getFEN().c_str());
    b.unmakeNullMove();
    CHECK(b.getFEN() == fen && b.zobrist() == hash, "unmakeNullMove did not restore the position");

    // A repetition is not looked for across a null move.
    b.setStartPos();
    b.applyMovesUCI({"g1f3", "g8f6"});
    b.makeNullMove();
    b.makeNullMove();
    CHECK(!b.isRepetition(), "repetition detected across a null move");
}

static bool identical(const Move &a, const Move &b) {
    return a.from == b.from && a.to == b.to && a.promotion == b.promotion && a.isEnPassant == b.isEnPassant &&
           a.isCastling == b.isCastling;
}

// The staged move picker relies on these: generating one kind of move gives
// exactly the matching moves of the full list, in the same order, and
// toPseudoLegal accepts exactly the generated moves (from/to/promotion kind)
// and reproduces them with their flags.
static void testGenerationModes() {
    const char *starts[] = {
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/1P6/8/8/8/8/6p1/4K3 b - - 0 1",
    };
    std::mt19937 rng(99);
    int checked = 0;
    for (const char *start : starts) {
        for (int game = 0; game < 8; ++game) {
            TestBoard b;
            b.setFromFEN(start);
            for (int ply = 0; ply < 80; ++ply) {
                MoveList all, viaMode, noisy, quiet;
                b.generatePseudoLegalMoves(all);
                b.generateMoves(viaMode, GenMode::All);
                b.generateMoves(noisy, GenMode::Noisy);
                b.generateMoves(quiet, GenMode::Quiet);
                CHECK(viaMode.size() == all.size(), "GenMode::All differs at %s", b.getFEN().c_str());
                int n = 0, q = 0;
                for (int i = 0; i < all.size(); ++i) {
                    if (i < viaMode.size()) CHECK(identical(all[i], viaMode[i]), "GenMode::All order differs at %s", b.getFEN().c_str());
                    const bool isNoisy = eval::isNoisy(b, all[i]);
                    MoveList &part = isNoisy ? noisy : quiet;
                    int &k = isNoisy ? n : q;
                    CHECK(k < part.size() && identical(part[k], all[i]), "%s move %s missing or out of order at %s",
                          isNoisy ? "noisy" : "quiet", moveToUciString(all[i]).c_str(), b.getFEN().c_str());
                    ++k;
                }
                CHECK(n == noisy.size() && q == quiet.size(), "extra moves in a subset at %s", b.getFEN().c_str());
                MoveList noisyOnly;
                b.generatePseudoLegalMoves(noisyOnly, true);
                CHECK(noisyOnly.size() == noisy.size(), "noisyOnly differs at %s", b.getFEN().c_str());
                for (int i = 0; i < noisyOnly.size() && i < noisy.size(); ++i)
                    CHECK(identical(noisyOnly[i], noisy[i]), "noisyOnly order differs at %s", b.getFEN().c_str());
                // Every from/to/promotion-kind combination.
                for (int from = 0; from < 64; ++from)
                    for (int to = 0; to < 64; ++to)
                        for (int promo = -1; promo <= QUEEN; ++promo) {
                            if (promo == PAWN) continue;
                            Move probe{from, to, promo < 0 ? Piece::None : makePiece(Color::White, promo), false, false};
                            const Move *gen = nullptr;
                            for (int i = 0; i < all.size(); ++i)
                                if (sameMove(all[i], probe)) gen = &all[i];
                            Move out;
                            const bool ok = b.toPseudoLegal(probe, out);
                            CHECK(ok == (gen != nullptr), "toPseudoLegal(%s) = %d at %s", moveToUciString(probe).c_str(), ok,
                                  b.getFEN().c_str());
                            if (ok && gen) CHECK(identical(out, *gen), "toPseudoLegal(%s) flags differ at %s",
                                                 moveToUciString(probe).c_str(), b.getFEN().c_str());
                        }
                ++checked;
                auto legal = b.generateLegalMoves();
                if (legal.empty()) break;
                b.makeMove(legal[rng() % legal.size()]);
            }
        }
    }
    CHECK(checked > 1000, "generation modes checked on only %d positions", checked);
}

// hasLegalMoveOutsideCheck (Legality.h, quiescence's stalemate test) against
// the generator: positions python-chess classified, then random sparse
// positions, where kings are often boxed in and pieces pinned.
static void testHasLegalMove() {
    struct Case {
        const char *fen;
        bool any;
    };
    const Case cases[] = {
        {"Rn4k1/2BN3R/8/7Q/K7/8/8/8 b - - 0 1", false}, // stalemate: the knight is pinned along the rank
        {"7k/5Kn1/6P1/8/8/8/QB6/8 b - - 0 1", false},   // stalemate: the knight is pinned on the diagonal
        {"7k/5K2/6Q1/8/8/8/8/8 b - - 0 1", false},      // stalemate, king alone
        {"8/6Q1/K6R/P7/8/8/7q/7k b - - 0 1", true},     // only the pinned queen moves, along the pin
        {"1r5k/8/4p3/3pP3/8/2n5/8/K7 w - d6 0 1", true}, // only en passant
        {"7k/5Kn1/6P1/8/8/8/Q7/8 b - - 0 1", true},     // the knight is free
        {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", true},
    };
    for (const Case &c : cases) {
        TestBoard b;
        CHECK(b.setFromFEN(c.fen), "FEN rejected: %s", c.fen);
        CHECK(!b.inCheck(), "in check: %s", c.fen);
        CHECK(!b.generateLegalMoves().empty() == c.any, "generator disagrees with python-chess: %s", c.fen);
        CHECK(hasLegalMoveOutsideCheck(b) == c.any, "hasLegalMoveOutsideCheck wrong: %s", c.fen);
    }
    std::mt19937 rng(2718);
    const char *kinds = "PNBRQpnbrq";
    int checked = 0, none = 0;
    for (int i = 0; i < 300000; ++i) {
        char grid[64];
        for (char &x : grid) x = '.';
        auto empty = [&](bool pawn) {
            for (;;) {
                const int sq = static_cast<int>(rng() % 64);
                if (grid[sq] == '.' && (!pawn || (sq >= 8 && sq < 56))) return sq;
            }
        };
        grid[empty(false)] = 'K';
        grid[empty(false)] = 'k';
        for (int n = 1 + static_cast<int>(rng() % 9); n > 0; --n) {
            const char p = kinds[rng() % 10];
            grid[empty(p == 'P' || p == 'p')] = p;
        }
        std::string fen;
        for (int r = 7; r >= 0; --r) {
            int gap = 0;
            for (int f = 0; f < 8; ++f) {
                const char x = grid[r * 8 + f];
                if (x == '.') {
                    ++gap;
                    continue;
                }
                if (gap) fen += static_cast<char>('0' + gap);
                gap = 0;
                fen += x;
            }
            if (gap) fen += static_cast<char>('0' + gap);
            if (r) fen += '/';
        }
        fen += rng() % 2 ? " w - - 0 1" : " b - - 0 1";
        TestBoard b;
        if (!b.setFromFEN(fen) || b.inCheck()) continue;
        const bool any = !b.generateLegalMoves().empty();
        ++checked;
        none += !any;
        CHECK(hasLegalMoveOutsideCheck(b) == any, "hasLegalMoveOutsideCheck wrong: %s", fen.c_str());
    }
    CHECK(checked > 50000 && none > 20, "too few random positions (%d) or stalemates (%d)", checked, none);
}

// The laws' rule against a direct count of earlier occurrences, over random
// games steered into repetitions and with occasional null moves: `fromRoot`
// plies below a search root, an earlier occurrence inside the tree draws,
// pre-root ones only from two on; a root further back than the game equals
// the twofold rule. (Distances 0-3 behave alike: a position recurs with the
// same side to move at least 4 plies later.)
static void testRepetitionRuleRandom() {
    TestBoard b;
    std::mt19937 rng(4242);
    const char *starts[] = {"rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                            "4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "r3k3/8/2n5/8/8/5B2/8/4K2R w Kq - 0 1",
                            "1q2k3/8/8/8/8/8/8/1Q2K3 w - - 0 1", "4k3/3n4/8/2b5/8/4N3/8/2R1K3 b - - 0 1"};
    int thirds = 0, seconds = 0;
    for (int game = 0; game < 300; ++game) {
        CHECK(b.setFromFEN(starts[game % 5]), "start position rejected");
        std::vector<uint64_t> keys{b.zobrist()}; // positions since the last irreversible move, oldest first
        for (int ply = 0; ply < 100; ++ply) {
            auto moves = b.generateLegalMoves();
            if (moves.empty()) break;
            std::vector<Move> quiet;
            for (const Move &m : moves)
                if (b.pieceAt(m.to) == Piece::None && pieceTypeOf(b.pieceAt(m.from)) != PAWN && !m.isCastling)
                    quiet.push_back(m);
            if (!b.inCheck() && rng() % 23 == 0) {
                // A null move ends the window the rules look back over: the
                // positions before it no longer count.
                b.makeNullMove();
                keys.assign(1, b.zobrist());
                continue;
            }
            // Half the time, repeat a position when a move can (so third occurrences come up).
            std::vector<Move> repeating;
            if (rng() % 2)
                for (const Move &m : moves) {
                    b.makeMove(m);
                    if (b.isRepetition()) repeating.push_back(m);
                    b.unmakeMove();
                }
            const auto &pool = !repeating.empty() ? repeating : !quiet.empty() && rng() % 12 ? quiet : moves;
            b.makeMove(pool[rng() % pool.size()]);
            if (b.getHalfmoveClock() == 0) keys.clear();
            keys.push_back(b.zobrist());
            const int last = static_cast<int>(keys.size()) - 1;
            for (int fromRoot : {0, 1, 2, 3, 4, 6, 9, 10000}) {
                int pre = 0;
                bool inTree = false;
                for (int i = 0; i < last; ++i) {
                    if (keys[i] != keys[last]) continue;
                    if (i > last - fromRoot) inTree = true; // inside the search tree
                    else ++pre;                             // the root or before it
                }
                const bool expected = inTree || pre >= 2;
                CHECK(b.isRepetition(fromRoot) == expected, "game %d ply %d, %d plies from the root: %d, expected %d (%s)",
                      game, ply, fromRoot, b.isRepetition(fromRoot), expected, b.getFEN().c_str());
                if (fromRoot == 0) {
                    thirds += pre >= 2;
                    seconds += pre == 1;
                }
            }
            CHECK(b.isRepetition(10000) == b.isRepetition(), "game %d ply %d: a root before the game differs from twofold",
                  game, ply);
            if (b.isDraw(0)) break; // the claimable draw ends the game
        }
    }
    CHECK(thirds > 50 && seconds > 50, "laws' rule barely exercised: %d third occurrences, %d second ones", thirds, seconds);
}

int main() {
    testPerft();
    testNullMove();
    testHashConsistency();
    testFenValidation();
    testDraws();
    testEnPassantLegality();
    testInsufficientMaterial();
    testRepetitionRuleRandom();
    testApplyMoves();
    testGenerationModes();
    testHasLegalMove();
    if (failures) {
        std::printf("%s: %d failure(s)\n", kBoardName, failures);
        return 1;
    }
    std::printf("%s: all board tests passed\n", kBoardName);
    return 0;
}
