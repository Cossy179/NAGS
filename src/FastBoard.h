#pragma once

// Bitboard board with magic-bitboard sliding attacks and a mailbox for O(1)
// piece lookup. Used by `nags_fast` and `nags_enhanced`. Same interface as
// Board so the search/evaluation templates work with either.

#include "ChessTypes.h"
#include "Nnue.h"

#include <cstdint>
#include <string>
#include <vector>

struct Magic {
    Bitboard mask;
    Bitboard magic;
    Bitboard *attacks;
    int shift;
};

class FastBoard {
public:
    FastBoard();

    // Loads a FEN. On failure the board is left unchanged and false is returned.
    bool setFromFEN(const std::string &fen);
    std::string getFEN() const;
    bool setStartPos();
    // Applies UCI moves in order; stops at (and returns false for) the first illegal one.
    bool applyMovesUCI(const std::vector<std::string> &uciMoves);

    // Legal moves for the side to move; noisyOnly keeps captures and promotions.
    std::vector<Move> generateLegalMoves(bool noisyOnly = false) const;
    void generatePseudoMoves(MoveList &moves) const;
    // Pseudo-legal moves (they may leave the own king in check), in the same
    // order generateLegalMoves filters them.
    void generatePseudoLegalMoves(MoveList &out, bool noisyOnly = false) const;

    bool inCheck() const { return inCheck(side); }
    bool inCheck(Color c) const;
    Color sideToMove() const { return side; }
    Piece pieceAt(int sq) const { return mailbox[sq]; }
    Bitboard pieceBB(Color c, int type) const { return pieces[colorIndex(c)][type]; }
    Bitboard occupancy() const { return all_occupied; }
    uint64_t zobrist() const { return hash; }
    int getHalfmoveClock() const { return halfmoveClock; }
    // Kept up to date incrementally for eval::evaluate: material plus
    // piece-square values of every piece except the kings (White minus
    // Black), and the uncapped game phase (minor 1, rook 2, queen 4).
    int psqScore() const { return psq; }
    int gamePhase() const { return phase; }
    // NNUE accumulators of the current position, valid while a network is
    // active. makeMove only records which pieces changed; the accumulators
    // are computed from the nearest computed position below when first
    // asked for (many positions are never evaluated: illegal moves, TT
    // cutoffs, PV nodes), and unmakeMove just drops them.
    const nnue::Accumulator &accumulator() const;
    // Recomputes them from scratch (after the network changed) and forgets
    // the saved ones.
    void refreshAccumulator();
    uint8_t getCastlingRights() const { return castlingRights; } // KQkq = bits 0..3
    int getEpSquare() const { return epSquare; } // -1 if none
    int getFullmoveNumber() const { return fullmoveNumber; }

    void makeMove(const Move &m);
    void unmakeMove();
    // Passes the turn (for null-move pruning). Undo with unmakeNullMove().
    void makeNullMove();
    void unmakeNullMove();
    bool lastMoveWasNull() const { return !history.empty() && history.back().move.isNull(); }
    Move lastMove() const { return history.empty() ? Move{} : history.back().move; }
    // Pieces of both colours attacking `sq` given the occupancy `occupied`.
    Bitboard attackersTo(int sq, Bitboard occupied) const;
    static Bitboard rookAttacks(int sq, Bitboard occupied) { return getRookAttacks(sq, occupied); }
    static Bitboard bishopAttacks(int sq, Bitboard occupied) { return getBishopAttacks(sq, occupied); }

    // Fifty-move rule, repetition (one earlier occurrence) or insufficient material.
    bool isDraw() const;
    bool isRepetition() const;
    bool isInsufficientMaterial() const;

    static bool isValidSquare(int sq) { return sq >= 0 && sq < 64; }
    static int fileOf(int sq) { return sq & 7; }
    static int rankOf(int sq) { return sq >> 3; }
    static std::string moveToUci(const Move &m) { return moveToUciString(m); }
    static int algebraicToSquare(const std::string &alg) { return parseSquare(alg); }
    static bool isWhite(Piece p) { return isWhitePiece(p); }
    static bool isBlack(Piece p) { return isBlackPiece(p); }

    static Bitboard getRookAttacks(int sq, Bitboard occupied);
    static Bitboard getBishopAttacks(int sq, Bitboard occupied);
    static Bitboard getQueenAttacks(int sq, Bitboard occupied) {
        return getRookAttacks(sq, occupied) | getBishopAttacks(sq, occupied);
    }

    uint64_t perft(int depth);
    void divide(int depth);

private:
    struct NoHistory {};
    FastBoard(const FastBoard &other, NoHistory);

    Bitboard pieces[2][6] = {}; // [color][PAWN..KING]
    Bitboard occupied[2] = {};
    Bitboard all_occupied = 0;
    Piece mailbox[64] = {};

    Color side = Color::White;
    uint8_t castlingRights = 0; // KQkq bits
    int epSquare = -1;
    int halfmoveClock = 0;
    int fullmoveNumber = 1;
    int pliesFromNull = 0; // repetitions are not looked for across a null move
    uint64_t hash = 0;
    int psq = 0;
    int phase = 0;
    struct AccEntry {
        nnue::Accumulator acc;
        nnue::DirtyPieces dirty;          // pieces changed by the move that led to this position
        uint8_t kingSq[2] = {};           // both kings' squares in this position
        bool computed[2] = {false, false}; // acc.v[perspective] is valid
    };
    // back(): the current position. Mutable: accumulator() fills it in lazily.
    mutable std::vector<AccEntry> accStack = std::vector<AccEntry>(1);

    // Accumulator refresh cache, one entry per perspective and king state:
    // the accumulator of the pieces last seen with that king state, so a
    // refresh after the king changes bucket only applies the difference.
    // Not copied with the board (a copy starts with an empty cache).
    struct RefreshEntry {
        alignas(64) int16_t acc[nnue::kHidden];
        Bitboard pieces[2][6];
    };
    struct RefreshCache {
        std::vector<RefreshEntry> entries;
        RefreshCache() = default;
        RefreshCache(const RefreshCache &) {}
        RefreshCache &operator=(const RefreshCache &) {
            entries.clear();
            return *this;
        }
    };
    mutable RefreshCache refreshCache;
    void refreshPerspective(int perspective, int kstate) const;
    void setKingSquares(AccEntry &e) const;

    struct HistoryEntry {
        Move move;
        Piece moved;
        Piece captured;
        uint8_t castlingRights;
        int epSquare;
        int halfmoveClock;
        int fullmoveNumber;
        int pliesFromNull;
        uint64_t hash;
    };
    std::vector<HistoryEntry> history;

    static void initTables();
    static Bitboard pawnAttacks[2][64];
    static Bitboard knightAttacks[64];
    static Bitboard kingAttacks[64];
    static Magic rookMagics[64];
    static Magic bishopMagics[64];
    static Bitboard rookTable[102400];
    static Bitboard bishopTable[5248];
    static void initMagics(Magic *magics, Bitboard *table, bool isRook);

    void putPiece(Piece p, int sq);
    void removePiece(int sq);
    void movePiece(int from, int to);

    void generatePawnMoves(MoveList &moves, Color c) const;
    void generatePieceMoves(MoveList &moves, Color c) const;
    void generateCastlingMoves(MoveList &moves, Color c) const;
    bool isSquareAttacked(int sq, Color byColor) const;
    int kingSquare(Color c) const;
    void sanitizeCastlingRights();
    static void addMove(MoveList &moves, int from, int to,
                        Piece promo = Piece::None, bool ep = false, bool castle = false) {
        moves.push_back(Move{from, to, promo, ep, castle});
    }

    static uint64_t zPiece[12][64];
    static int psqValue[12][64]; // contribution of a piece on a square to psq
    static int phaseValue[12];
    static uint64_t zSide;
    static uint64_t zCastle[16];
    static uint64_t zEnpassant[8];
    void hashRecompute();
};
