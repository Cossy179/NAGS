// nags_datagen: self-play training data for the NNUE evaluation.
//
// Each thread plays games with its own nags_enhanced search (FastBoard,
// private transposition table): a few random opening moves, then every move
// searched to a fixed node budget. Quiet positions are written as
//
//   <fen> | <score> | <result>
//
// with the search score in centipawns and the game result (1.0 / 0.5 / 0.0),
// both from White's point of view. With an output file ending in ".bin" each
// position is a 32-byte record instead (about half the size, and much faster
// to load), all little endian:
//
//   uint64 occupancy        bit i = square i (a1 = 0, h8 = 63)
//   uint8  pieces[16]       4-bit piece codes (P N B R Q K p n b r q k =
//                           0..11) of the occupied squares in ascending
//                           order, low nibble first
//   int16  score            centipawns, White's point of view
//   uint8  result           0 = Black won, 1 = draw, 2 = White won
//   uint8  side to move     0 = White, 1 = Black
//   uint8  reserved[4]      0
//
//   nags_datagen --out data/selfplay.txt --games 10000 --threads 3 --nodes 5000
//
// Start positions (--book FILE): each game starts from a position of the
// file (FEN or EPD lines; the first four fields are read, so EPD operations
// are ignored), drawn by the game's random stream, before the random plies.
// A book of unfamiliar openings or of imbalanced material steers the data
// there (raise --max-opening-score for lopsided ones). Without --book the
// games start from the initial position, exactly as before.
//
// Shards (--first-game N): the games are numbered N .. N + games - 1 and each
// one draws its opening from its own random stream, seeded by (seed, game
// number), so a game is the same whichever thread plays it and whatever was
// played before (only the order games are written in depends on the
// threads). <out>.games records every finished game:
//
//   # nags_datagen games v2 seed=... nodes=... ... network-fnv=... search=...
//   <game> <records> <bytes> <result> <checksum>
//
// The first line holds the settings, the network (FNV-1a 64 of its bytes)
// and the search (the node count of the engine's bench, the fingerprint of
// docs/TESTING.md), so shards made by another build or network are told
// apart. The checksum is FNV-1a 64 of the game's bytes. A game's bytes are
// written before its index line. Run the same command again after a stop and
// it resumes: an unfinished last index line is dropped, the last games' bytes
// are checked against their checksums (those that do not match, and those
// the output is too short for, are played again: the same games), the
// output is cut back to the games kept, and only the missing games are
// played. An output that already holds data but has no index is never
// touched. Several machines can share one seed with disjoint game ranges and
// never write the same game. Without --first-game the games are played as
// always (one random stream per thread), which keeps the output of a given
// command unchanged.

#include "Bench.h"
#include "FastBoard.h"
#include "Search.h"
#include "TT.h"
#include "Uci.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern const unsigned char kNagsEmbeddedNet[];
extern const unsigned long kNagsEmbeddedNetSize;

namespace {

struct Options {
    std::string out;
    bool binary = false; // 32-byte records (the output name ends in ".bin")
    long long games = 1000;
    int threads = 1;
    uint64_t nodes = 5000;
    int randomPlies = 8;      // random moves before the search takes over
    int maxOpeningScore = 400; // skip openings the search already judges lopsided
    int hashMB = 16;
    uint64_t seed = 1;
    // When a repetition ends a game: at the first one (the default, the rule
    // the search plays by) or, with "--repetition third", at the third
    // occurrence, as the laws give it (the search still plays by its own rule).
    bool thirdRepetition = false;
    long long firstGame = -1; // >= 0: shard mode (numbered games, own random streams, resumable)
    // Deeper labels (--relabel-nodes N --relabel-select ...): a kept position chosen by the rule is searched
    // again with N nodes by a second searcher, whose score becomes its label; the game itself is unchanged.
    uint64_t relabelNodes = 0;
    bool relabelRandom = false;     // "random:P": a fraction P of the positions, by a hash of the position
    int relabelThreshold = 0;       // "disagree:T": |static evaluation - search score| >= T centipawns
    double relabelFraction = 0.0;
    std::string bookPath;           // --book: start positions, one FEN or EPD line each
    std::vector<std::string> book;
    uint64_t bookFnv = 0;           // FNV-1a 64 of the file's bytes, for the shard header
};

struct Record {
    std::string fen;                  // text output
    std::array<unsigned char, 32> bin; // binary output (without the result)
    int score;                        // White's point of view
};

// The binary record of a position, except for the result byte.
std::array<unsigned char, 32> packPosition(const FastBoard &b, int score) {
    std::array<unsigned char, 32> r{};
    const uint64_t occ = b.occupancy();
    for (int i = 0; i < 8; ++i) r[i] = static_cast<unsigned char>((occ >> (8 * i)) & 0xFF);
    int n = 0;
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = b.pieceAt(sq);
        if (p == Piece::None) continue;
        int code = pieceTypeOf(p) + (colorOf(p) == Color::White ? 0 : 6);
        r[8 + n / 2] = static_cast<unsigned char>(r[8 + n / 2] | (code << (4 * (n % 2))));
        ++n;
    }
    const uint16_t s = static_cast<uint16_t>(static_cast<int16_t>(std::clamp(score, -32000, 32000)));
    r[24] = static_cast<unsigned char>(s & 0xFF);
    r[25] = static_cast<unsigned char>(s >> 8);
    r[27] = b.sideToMove() == Color::White ? 0 : 1;
    return r;
}

std::mutex outMutex;
std::atomic<long long> gamesStarted{0}, gamesDone{0}, positionsWritten{0};

void usage() {
    std::cerr << "usage: nags_datagen --out FILE [--games N] [--threads N] [--nodes N] [--random-plies N]\n"
                 "                    [--max-opening-score CP] [--hash MB] [--seed N] [--repetition first|third]\n"
                 "                    [--first-game N] [--relabel-nodes N --relabel-select disagree:CP|random:P]\n"
                 "                    [--book FILE]\n";
}

uint64_t fnv1a(const unsigned char *data, size_t n, uint64_t h = 0xcbf29ce484222325ULL);

// Reads --book: every non-empty line's first four fields (placement, side to move, castling, en passant) must
// be a position the board accepts; the halfmove and fullmove counters start at 0 and 1.
bool loadBook(Options &o) {
    std::ifstream in(o.bookPath, std::ios::binary);
    if (!in) {
        std::cerr << "--book: cannot open " << o.bookPath << "\n";
        return false;
    }
    const std::string bytes((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    o.bookFnv = fnv1a(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size());
    std::istringstream lines(bytes);
    std::string line;
    for (int n = 1; std::getline(lines, line); ++n) {
        std::istringstream fields(line);
        std::string f[4];
        if (!(fields >> f[0])) continue; // an empty line
        if (!(fields >> f[1] >> f[2] >> f[3])) {
            std::cerr << "--book " << o.bookPath << " line " << n << ": not a FEN or EPD position\n";
            return false;
        }
        const std::string fen = f[0] + " " + f[1] + " " + f[2] + " " + f[3] + " 0 1";
        FastBoard b;
        if (!b.setFromFEN(fen)) {
            std::cerr << "--book " << o.bookPath << " line " << n << ": not a legal position\n";
            return false;
        }
        o.book.push_back(fen);
    }
    if (o.book.empty()) {
        std::cerr << "--book " << o.bookPath << ": no positions\n";
        return false;
    }
    return true;
}

bool parseArgs(int argc, char **argv, Options &o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (i + 1 >= argc) { usage(); return false; }
        std::string v = argv[++i];
        long long n = 0;
        if (a == "--out") {
            o.out = v;
            o.binary = v.size() >= 4 && v.compare(v.size() - 4, 4, ".bin") == 0;
            continue;
        }
        if (a == "--relabel-select") {
            const size_t colon = v.find(':');
            const std::string kind = v.substr(0, colon), arg = colon == std::string::npos ? "" : v.substr(colon + 1);
            long long t = 0;
            if (kind == "disagree" && uci::parseInt(arg, t) && t >= 0) {
                o.relabelRandom = false;
                o.relabelThreshold = static_cast<int>(t);
            } else if (kind == "random" && !arg.empty()) {
                o.relabelRandom = true;
                o.relabelFraction = std::atof(arg.c_str());
                if (o.relabelFraction <= 0.0 || o.relabelFraction > 1.0) {
                    std::cerr << "--relabel-select random:P needs 0 < P <= 1\n";
                    return false;
                }
            } else {
                std::cerr << "--relabel-select: disagree:CP or random:P\n";
                return false;
            }
            continue;
        }
        if (a == "--book") {
            o.bookPath = v;
            continue;
        }
        if (a == "--repetition") {
            if (v != "first" && v != "third") { std::cerr << "--repetition: first or third\n"; return false; }
            o.thirdRepetition = v == "third";
            continue;
        }
        if (!uci::parseInt(v, n) || n < 0) { std::cerr << "invalid value for " << a << "\n"; return false; }
        if (a == "--games") o.games = n;
        else if (a == "--threads") o.threads = static_cast<int>(std::max(1LL, n));
        else if (a == "--nodes") o.nodes = static_cast<uint64_t>(std::max(1LL, n));
        else if (a == "--random-plies") o.randomPlies = static_cast<int>(n);
        else if (a == "--max-opening-score") o.maxOpeningScore = static_cast<int>(n);
        else if (a == "--hash") o.hashMB = static_cast<int>(std::max(1LL, n));
        else if (a == "--seed") o.seed = static_cast<uint64_t>(n);
        else if (a == "--first-game") o.firstGame = n;
        else if (a == "--relabel-nodes") o.relabelNodes = static_cast<uint64_t>(n);
        else { usage(); return false; }
    }
    if (o.out.empty()) { usage(); return false; }
    return o.bookPath.empty() || loadBook(o);
}

bool isQuiet(const FastBoard &b, const Move &m) { return !eval::isNoisy(b, m); }

// The game is drawn (fifty-move rule, repetition by --repetition, insufficient material).
bool isGameDraw(const Options &o, const FastBoard &b) { return o.thirdRepetition ? b.isDraw(0) : b.isDraw(); }

// The second searcher of --relabel-nodes, with its own table: the game's searcher (and so the game) is
// untouched by the deeper searches.
struct Relabeler {
    explicit Relabeler(const Options &o) : tt(static_cast<size_t>(o.hashMB)), searcher(&tt) {}
    TranspositionTable tt;
    Searcher<FastBoard> searcher;
};

std::atomic<long long> relabelled{0}, relabelDropped{0}, relabelChange{0};
// What this run's searches cost, summed over the threads: nodes and microseconds of the game's moves and of
// the deeper relabelling searches apart, so relabelling choices can be compared by the compute they take.
std::atomic<long long> gameNodes{0}, gameMicros{0}, relabelNodes{0}, relabelMicros{0};

long long microsSince(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - t).count();
}

// Whether --relabel-select picks this kept position (`score`: the game search's, side to move). The random
// rule hashes the position and the seed, so the choice does not depend on the order games are played in.
bool relabelChosen(const Options &o, const FastBoard &b, int score) {
    if (!o.relabelRandom) return std::abs(eval::evaluate(b) - score) >= o.relabelThreshold;
    uint64_t h = b.zobrist() ^ (o.seed * 0x9E3779B97F4A7C15ULL);
    h = (h ^ (h >> 30)) * 0xBF58476D1CE4E5B9ULL;
    h = (h ^ (h >> 27)) * 0x94D049BB133111EBULL;
    h ^= h >> 31;
    return static_cast<double>(h >> 11) * (1.0 / 9007199254740992.0) < o.relabelFraction;
}

// Plays one game; returns false if the opening was rejected. Appends the
// positions to keep (without results) to `records` and sets `result`.
bool playGame(const Options &o, std::mt19937_64 &rng, Searcher<FastBoard> &searcher, TranspositionTable &tt,
              std::vector<Record> &records, double &result, Relabeler *relabeler = nullptr) {
    FastBoard board;
    if (!o.book.empty()) board.setFromFEN(o.book[rng() % o.book.size()]);
    for (int ply = 0; ply < o.randomPlies; ++ply) {
        auto moves = board.generateLegalMoves();
        if (moves.empty()) return false;
        board.makeMove(moves[rng() % moves.size()]);
    }
    if (board.generateLegalMoves().empty() || isGameDraw(o, board)) return false;

    tt.clear();
    searcher.clearHistory();
    if (relabeler) {
        relabeler->tt.clear();
        relabeler->searcher.clearHistory();
    }
    std::atomic<bool> stop{false};
    {
        SearchLimits check;
        check.depth = 6;
        SearchResult r = searcher.search(board, check, stop, nullptr);
        if (std::abs(r.score) > o.maxOpeningScore) return false;
    }

    records.clear();
    int winStreak = 0, drawStreak = 0;
    for (int ply = 0;; ++ply) {
        auto legal = board.generateLegalMoves();
        if (legal.empty()) {
            // Checkmate: the side to move lost. Stalemate: draw.
            result = board.inCheck() ? (board.sideToMove() == Color::White ? 0.0 : 1.0) : 0.5;
            return true;
        }
        if (isGameDraw(o, board) || ply >= 400) {
            result = 0.5;
            return true;
        }
        SearchLimits limits;
        limits.nodes = o.nodes;
        const auto started = std::chrono::steady_clock::now();
        SearchResult r = searcher.search(board, limits, stop, nullptr);
        gameMicros += microsSince(started);
        gameNodes += static_cast<long long>(r.nodes);
        int whiteScore = board.sideToMove() == Color::White ? r.score : -r.score;

        // Adjudication: a decisive score for 6 plies in a row, or a near-zero
        // score for 20 plies late in the game.
        winStreak = std::abs(r.score) >= 2000 ? winStreak + 1 : 0;
        drawStreak = ply >= 80 && std::abs(r.score) <= 10 ? drawStreak + 1 : 0;
        if (winStreak >= 6) {
            result = whiteScore > 0 ? 1.0 : 0.0;
            return true;
        }
        if (drawStreak >= 20) {
            result = 0.5;
            return true;
        }

        // Training positions: not in check, a quiet best move, and no mate
        // or tablebase score (those say little about the evaluation).
        bool keep = !board.inCheck() && isQuiet(board, r.bestMove) && std::abs(r.score) < TB_BOUND;
        int label = whiteScore;
        if (keep && relabeler && relabelChosen(o, board, r.score)) {
            SearchLimits deeper;
            deeper.nodes = o.relabelNodes;
            const auto started = std::chrono::steady_clock::now();
            const SearchResult d = relabeler->searcher.search(board, deeper, stop, nullptr);
            relabelMicros += microsSince(started);
            relabelNodes += static_cast<long long>(d.nodes);
            if (std::abs(d.score) >= TB_BOUND) { // a mate or tablebase score: not kept, as in the game search
                keep = false;
                ++relabelDropped;
            } else {
                label = board.sideToMove() == Color::White ? d.score : -d.score;
                relabelChange += std::abs(label - whiteScore);
                ++relabelled;
            }
        }
        if (keep)
            records.push_back(o.binary ? Record{{}, packPosition(board, label), label}
                                       : Record{board.getFEN(), {}, label});
        board.makeMove(r.bestMove);
    }
}

// Shard mode: game `game`'s own random stream (std::seed_seq and mt19937_64
// are specified exactly by the standard, so every compiler gives the same).
std::mt19937_64 gameRng(uint64_t seed, long long game) {
    const uint64_t g = static_cast<uint64_t>(game);
    std::seed_seq s{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32), static_cast<uint32_t>(g),
                    static_cast<uint32_t>(g >> 32), 0x4E414753u};
    return std::mt19937_64(s);
}

// The records of one finished game, as written to the output.
std::string serialise(const Options &o, std::vector<Record> &records, double result) {
    std::ostringstream buf;
    if (o.binary) {
        const unsigned char res = result == 1.0 ? 2 : result == 0.0 ? 0 : 1;
        for (Record &rec : records) {
            rec.bin[26] = res;
            buf.write(reinterpret_cast<const char *>(rec.bin.data()), static_cast<std::streamsize>(rec.bin.size()));
        }
    } else {
        const char *res = result == 1.0 ? "1.0" : result == 0.0 ? "0.0" : "0.5";
        for (const Record &rec : records) buf << rec.fen << " | " << rec.score << " | " << res << '\n';
    }
    return buf.str();
}

// FNV-1a 64 of `n` bytes, continuing from `h`.
uint64_t fnv1a(const unsigned char *data, size_t n, uint64_t h) {
    for (size_t i = 0; i < n; ++i) h = (h ^ data[i]) * 0x100000001b3ULL;
    return h;
}

std::string hex64(uint64_t v) {
    char buf[17];
    std::snprintf(buf, sizeof buf, "%016llx", static_cast<unsigned long long>(v));
    return buf;
}

// The node count of nags_enhanced's bench (the same positions, depth 10, a
// 64 MB table cleared first, one thread): the search's fingerprint.
uint64_t searchFingerprint() {
    TranspositionTable tt(64);
    Searcher<FastBoard> searcher(&tt);
    tt.clear();
    searcher.clearHistory();
    std::atomic<bool> stop{false};
    uint64_t total = 0;
    for (const std::string &fen : bench::positions()) {
        FastBoard b;
        b.setFromFEN(fen);
        SearchLimits limits;
        limits.depth = 10;
        total += searcher.search(b, limits, stop, nullptr).nodes;
    }
    return total;
}

// What a shard's games depend on: its first line, which a resume must find
// unchanged (computed once: the search fingerprint takes about a second).
const std::string &shardHeader(const Options &o) {
    static const std::string header = [&] {
        std::ostringstream h;
        h << "# nags_datagen games v2 seed=" << o.seed << " nodes=" << o.nodes << " random-plies=" << o.randomPlies
          << " max-opening-score=" << o.maxOpeningScore << " hash=" << o.hashMB
          << " repetition=" << (o.thirdRepetition ? "third" : "first") << " format=" << (o.binary ? "bin" : "text");
        if (!nnue::network()) h << " network=hand-written";
        else if (nnue::networkName() == "embedded")
            h << " network=embedded network-fnv=" << hex64(fnv1a(kNagsEmbeddedNet, kNagsEmbeddedNetSize));
        else h << " network=" << nnue::networkName();
        h << " search=" << searchFingerprint();
        if (!o.book.empty()) h << " book=" << hex64(o.bookFnv) << ":" << o.book.size();
        if (o.relabelNodes)
            h << " relabel=" << o.relabelNodes << "@"
              << (o.relabelRandom ? "random:" + std::to_string(o.relabelFraction)
                                  : "disagree:" + std::to_string(o.relabelThreshold));
        return h.str();
    }();
    return header;
}

std::vector<long long> shardTodo; // shard mode: the games still to play
std::atomic<size_t> shardNext{0};

void shardWorker(const Options &o, std::ofstream &out, std::ofstream &index) {
    TranspositionTable tt(static_cast<size_t>(o.hashMB));
    Searcher<FastBoard> searcher(&tt);
    std::unique_ptr<Relabeler> relabeler = o.relabelNodes ? std::make_unique<Relabeler>(o) : nullptr;
    std::vector<Record> records;
    for (size_t k; (k = shardNext.fetch_add(1)) < shardTodo.size();) {
        const long long game = shardTodo[k];
        std::mt19937_64 rng = gameRng(o.seed, game);
        double result = 0.5;
        while (!playGame(o, rng, searcher, tt, records, result, relabeler.get())) {
        }
        const std::string bytes = serialise(o, records, result);
        {
            std::lock_guard<std::mutex> lock(outMutex);
            out << bytes;
            out.flush();
            index << game << ' ' << records.size() << ' ' << bytes.size() << ' ' << result << ' '
                  << hex64(fnv1a(reinterpret_cast<const unsigned char *>(bytes.data()), bytes.size())) << '\n';
            index.flush();
        }
        positionsWritten += static_cast<long long>(records.size());
        ++gamesDone;
    }
}

struct IndexEntry {
    long long game = 0;
    long long records = 0;
    uintmax_t bytes = 0;
    std::string result, checksum;
};

// One complete index line; false if it is not one.
bool parseIndexLine(const Options &o, const std::string &line, IndexEntry &e) {
    std::istringstream in(line);
    std::string extra;
    if (!(in >> e.game >> e.records >> e.bytes >> e.result >> e.checksum) || (in >> extra)) return false;
    const bool resultOk = e.result == "0" || e.result == "0.5" || e.result == "1";
    const bool sumOk = e.checksum.size() == 16 && e.checksum.find_first_not_of("0123456789abcdef") == std::string::npos;
    return e.game >= 0 && e.records >= 0 && resultOk && sumOk &&
           (!o.binary || e.bytes == 32 * static_cast<uintmax_t>(e.records));
}

// Shard mode: reads <out>.games, keeps the games whose bytes are in <out>
// (the last ones checked against their checksums), cuts <out> and the index
// back to them, and fills shardTodo with the rest. False (with a message) if
// the index belongs to other settings or is damaged before its last line, or
// if <out> holds data without an index: nothing is changed then.
bool prepareShard(const Options &o, const std::string &indexPath, long long &resumedGames) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const uintmax_t have = fs::exists(o.out, ec) ? fs::file_size(o.out, ec) : 0;
    std::string text;
    if (std::ifstream in{indexPath, std::ios::binary}) {
        std::ostringstream all;
        all << in.rdbuf();
        text = all.str();
    }
    const size_t headerEnd = text.find('\n');
    if (headerEnd == std::string::npos) { // no index, or only an unfinished first line
        if (have > 0) {
            std::cerr << o.out << " holds " << have << " bytes but has no index (" << indexPath
                      << "): not touching it (use another --out, or move it away)\n";
            return false;
        }
        std::ofstream(indexPath, std::ios::trunc | std::ios::binary) << shardHeader(o) << '\n';
        for (long long g = o.firstGame; g < o.firstGame + o.games; ++g) shardTodo.push_back(g);
        resumedGames = 0;
        return true;
    }
    // A line may end in "\r\n" (an index edited on Windows): the '\r' is not part of it.
    const auto line = [&](size_t from, size_t to) {
        return text.substr(from, to > from && text[to - 1] == '\r' ? to - 1 - from : to - from);
    };
    const std::string header = line(0, headerEnd);
    if (header != shardHeader(o)) {
        std::cerr << indexPath << " was written with other settings, network or search:\n  " << header << "\nnow\n  "
                  << shardHeader(o) << "\n(use another --out, or the same settings and build)\n";
        return false;
    }
    std::vector<IndexEntry> entries;
    std::set<long long> seen;
    size_t pos = headerEnd + 1;
    for (size_t end; (end = text.find('\n', pos)) != std::string::npos; pos = end + 1) {
        IndexEntry e;
        if (!parseIndexLine(o, line(pos, end), e) || !seen.insert(e.game).second) {
            std::cerr << indexPath << ": damaged line " << entries.size() + 2 << " ('" << line(pos, end)
                      << "'): not touching the shard\n";
            return false;
        }
        entries.push_back(e);
    }
    const bool unfinishedLine = pos < text.size(); // cut off while it was written: that game is played again
    // Keep the games the output holds in full, then check the last ones' bytes.
    std::vector<uintmax_t> offset(entries.size() + 1, 0);
    for (size_t i = 0; i < entries.size(); ++i) offset[i + 1] = offset[i] + entries[i].bytes;
    size_t keep = entries.size();
    while (keep > 0 && offset[keep] > have) --keep;
    constexpr size_t kChecked = 64; // a stop or a crash only ever damages the end
    if (keep > 0) {
        std::ifstream data(o.out, std::ios::binary);
        const size_t first = keep > kChecked ? keep - kChecked : 0;
        for (size_t i = first; i < keep; ++i) {
            std::vector<unsigned char> buf(entries[i].bytes);
            data.seekg(static_cast<std::streamoff>(offset[i]));
            data.read(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(buf.size()));
            if (!data || hex64(fnv1a(buf.data(), buf.size())) != entries[i].checksum) {
                keep = i;
                break;
            }
        }
    }
    const size_t dropped = entries.size() - keep;
    if (dropped > 0)
        std::cerr << "nags_datagen: " << dropped << " indexed game(s) at the end of " << o.out
                  << " are missing or do not match their checksums: played again\n";
    if (have > offset[keep]) {
        fs::resize_file(o.out, offset[keep], ec); // a game cut off before its index line: played again
        if (ec) {
            std::cerr << "cannot truncate " << o.out << ": " << ec.message() << "\n";
            return false;
        }
    }
    if (dropped > 0 || unfinishedLine) { // rewrite the index as the games kept, then swap it in
        const std::string tmp = indexPath + ".tmp";
        {
            std::ofstream out(tmp, std::ios::trunc | std::ios::binary);
            out << text.substr(0, headerEnd + 1);
            size_t at = headerEnd + 1;
            for (size_t i = 0; i < keep; ++i) {
                const size_t end = text.find('\n', at);
                out << text.substr(at, end + 1 - at);
                at = end + 1;
            }
            if (!out.flush()) {
                std::cerr << "cannot write " << tmp << "\n";
                return false;
            }
        }
        fs::rename(tmp, indexPath, ec);
        if (ec) {
            std::cerr << "cannot replace " << indexPath << ": " << ec.message() << "\n";
            return false;
        }
    }
    std::set<long long> done;
    for (size_t i = 0; i < keep; ++i) done.insert(entries[i].game);
    for (long long g = o.firstGame; g < o.firstGame + o.games; ++g)
        if (!done.count(g)) shardTodo.push_back(g);
    resumedGames = o.games - static_cast<long long>(shardTodo.size());
    return true;
}

void worker(const Options &o, int id, std::ofstream &out) {
    std::mt19937_64 rng(o.seed * 1000003ULL + static_cast<uint64_t>(id));
    TranspositionTable tt(static_cast<size_t>(o.hashMB));
    Searcher<FastBoard> searcher(&tt);
    std::unique_ptr<Relabeler> relabeler = o.relabelNodes ? std::make_unique<Relabeler>(o) : nullptr;
    std::vector<Record> records;
    while (gamesStarted.fetch_add(1) < o.games) {
        double result = 0.5;
        while (!playGame(o, rng, searcher, tt, records, result, relabeler.get())) {
        }
        const std::string bytes = serialise(o, records, result);
        {
            std::lock_guard<std::mutex> lock(outMutex);
            out << bytes;
            out.flush();
        }
        positionsWritten += static_cast<long long>(records.size());
        ++gamesDone;
    }
}

} // namespace

int main(int argc, char **argv) {
    Options o;
    if (!parseArgs(argc, argv, o)) return 2;
    const bool shard = o.firstGame >= 0;
    const std::string indexPath = o.out + ".games";
    long long resumed = 0;
    if (shard && !prepareShard(o, indexPath, resumed)) return 2;
    // Shards are written as bytes (no newline translation), so the index's sizes and checksums are the file's.
    std::ofstream out(o.out, o.binary || shard ? std::ios::app | std::ios::binary : std::ios::app);
    if (!out) {
        std::cerr << "cannot open " << o.out << "\n";
        return 1;
    }
    std::ofstream index;
    if (shard) {
        index.open(indexPath, std::ios::app | std::ios::binary);
        if (!index) {
            std::cerr << "cannot open " << indexPath << "\n";
            return 1;
        }
        std::fprintf(stderr, "nags_datagen: games %lld-%lld, %lld already written, %zu to play\n", o.firstGame,
                     o.firstGame + o.games - 1, resumed, shardTodo.size());
    }
    // A network that does not fit this build (another NAGS_NNUE_HIDDEN)
    // silently means the hand-written evaluation: say which one plays.
    std::fprintf(stderr, "nags_datagen: evaluation %s\n",
                 nnue::network() ? ("NNUE " + nnue::networkName()).c_str()
                                 : "hand-written (no usable NNUE network embedded)");
    auto start = std::chrono::steady_clock::now();
    std::vector<std::thread> threads;
    for (int i = 0; i < o.threads; ++i) {
        if (shard) threads.emplace_back(shardWorker, std::cref(o), std::ref(out), std::ref(index));
        else threads.emplace_back(worker, std::cref(o), i, std::ref(out));
    }
    std::atomic<bool> finished{false};
    std::thread progress([&] {
        while (!finished) {
            for (int i = 0; i < 100 && !finished; ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            std::fprintf(stderr, "games %lld/%lld  positions %lld  %.0f positions/s\n", gamesDone.load() + resumed, o.games,
                         positionsWritten.load(), s > 0 ? positionsWritten.load() / s : 0.0);
        }
    });
    for (auto &t : threads) t.join();
    finished = true;
    progress.join();
    if (o.relabelNodes) {
        const long long kept = positionsWritten.load();
        std::fprintf(stderr, "nags_datagen: relabelled %lld of %lld positions written (%.1f%%) at %llu nodes, mean "
                     "change %.1f cp; %lld chosen positions dropped (mate or tablebase score)\n", relabelled.load(),
                     kept, kept ? 100.0 * static_cast<double>(relabelled.load()) / static_cast<double>(kept) : 0.0,
                     static_cast<unsigned long long>(o.relabelNodes),
                     relabelled.load() ? static_cast<double>(relabelChange.load()) / static_cast<double>(relabelled.load())
                                       : 0.0,
                     relabelDropped.load());
    }
    const double gameSeconds = static_cast<double>(gameMicros.load()) / 1e6;
    std::fprintf(stderr, "nags_datagen: searches in this run: game moves %lld nodes, %.0f thread-seconds", gameNodes.load(),
                 gameSeconds);
    if (o.relabelNodes) {
        const double relabelSeconds = static_cast<double>(relabelMicros.load()) / 1e6;
        const double allNodes = static_cast<double>(gameNodes.load() + relabelNodes.load());
        std::fprintf(stderr, "; relabelling %lld nodes (%.1f%% of all), %.0f thread-seconds (%.1f%% of all)",
                     relabelNodes.load(), allNodes > 0 ? 100.0 * static_cast<double>(relabelNodes.load()) / allNodes : 0.0,
                     relabelSeconds,
                     gameSeconds + relabelSeconds > 0 ? 100.0 * relabelSeconds / (gameSeconds + relabelSeconds) : 0.0);
    }
    std::fprintf(stderr, "\n");
    return 0;
}
