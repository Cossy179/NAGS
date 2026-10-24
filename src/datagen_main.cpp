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
// Shards (--first-game N): the games are numbered N .. N + games - 1 and each
// one draws its opening from its own random stream, seeded by (seed, game
// number), so a game is the same whichever thread plays it and whatever was
// played before (only the order games are written in depends on the
// threads). <out>.games records every finished game:
//
//   # nags_datagen games seed=... nodes=... ...   (the settings)
//   <game> <records> <bytes> <result>
//
// Run the same command again after a stop and it resumes: a game cut off
// mid-write is truncated and only the games not in the list are played.
// Several machines can share one seed with disjoint game ranges and never
// write the same game. Without --first-game the games are played as always
// (one random stream per thread), which keeps the output of a given command
// unchanged.

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
#include <mutex>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

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
                 "                    [--first-game N]\n";
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
        else { usage(); return false; }
    }
    if (o.out.empty()) { usage(); return false; }
    return true;
}

bool isQuiet(const FastBoard &b, const Move &m) { return !eval::isNoisy(b, m); }

// The game is drawn (fifty-move rule, repetition by --repetition, insufficient material).
bool isGameDraw(const Options &o, const FastBoard &b) { return o.thirdRepetition ? b.isDraw(0) : b.isDraw(); }

// Plays one game; returns false if the opening was rejected. Appends the
// positions to keep (without results) to `records` and sets `result`.
bool playGame(const Options &o, std::mt19937_64 &rng, Searcher<FastBoard> &searcher, TranspositionTable &tt,
              std::vector<Record> &records, double &result) {
    FastBoard board;
    for (int ply = 0; ply < o.randomPlies; ++ply) {
        auto moves = board.generateLegalMoves();
        if (moves.empty()) return false;
        board.makeMove(moves[rng() % moves.size()]);
    }
    if (board.generateLegalMoves().empty() || isGameDraw(o, board)) return false;

    tt.clear();
    searcher.clearHistory();
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
        SearchResult r = searcher.search(board, limits, stop, nullptr);
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
        if (!board.inCheck() && isQuiet(board, r.bestMove) && std::abs(r.score) < TB_BOUND)
            records.push_back(o.binary ? Record{{}, packPosition(board, whiteScore), whiteScore}
                                       : Record{board.getFEN(), {}, whiteScore});
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

// What a shard's games depend on besides the binary: its first line, which a
// resume must find unchanged.
std::string shardHeader(const Options &o) {
    std::ostringstream h;
    h << "# nags_datagen games seed=" << o.seed << " nodes=" << o.nodes << " random-plies=" << o.randomPlies
      << " max-opening-score=" << o.maxOpeningScore << " hash=" << o.hashMB
      << " repetition=" << (o.thirdRepetition ? "third" : "first") << " format=" << (o.binary ? "bin" : "text")
      << " network=" << (nnue::network() ? nnue::networkName() : "hand-written");
    return h.str();
}

std::vector<long long> shardTodo; // shard mode: the games still to play
std::atomic<size_t> shardNext{0};

void shardWorker(const Options &o, std::ofstream &out, std::ofstream &index) {
    TranspositionTable tt(static_cast<size_t>(o.hashMB));
    Searcher<FastBoard> searcher(&tt);
    std::vector<Record> records;
    for (size_t k; (k = shardNext.fetch_add(1)) < shardTodo.size();) {
        const long long game = shardTodo[k];
        std::mt19937_64 rng = gameRng(o.seed, game);
        double result = 0.5;
        while (!playGame(o, rng, searcher, tt, records, result)) {
        }
        const std::string bytes = serialise(o, records, result);
        {
            std::lock_guard<std::mutex> lock(outMutex);
            out << bytes;
            out.flush();
            index << game << ' ' << records.size() << ' ' << bytes.size() << ' ' << result << '\n';
            index.flush();
        }
        positionsWritten += static_cast<long long>(records.size());
        ++gamesDone;
    }
}

// Shard mode: reads <out>.games, truncates <out> to the games it lists and
// fills shardTodo with the rest. False (with a message) if the index does not
// belong to these settings or does not match the output.
bool prepareShard(const Options &o, const std::string &indexPath, long long &resumedGames) {
    namespace fs = std::filesystem;
    std::set<long long> done;
    uintmax_t bytes = 0;
    std::ifstream in(indexPath);
    if (in) {
        std::string line;
        if (!std::getline(in, line) || line != shardHeader(o)) {
            std::cerr << indexPath << " was written with other settings:\n  " << line << "\nnow\n  " << shardHeader(o)
                      << "\n(use another --out, or the same settings)\n";
            return false;
        }
        long long game = 0, recs = 0;
        uintmax_t size = 0;
        double res = 0;
        while (in >> game >> recs >> size >> res) {
            done.insert(game);
            bytes += size;
        }
    }
    std::error_code ec;
    const uintmax_t have = fs::exists(o.out, ec) ? fs::file_size(o.out, ec) : 0;
    if (have < bytes) {
        std::cerr << o.out << " holds " << have << " bytes, less than the " << bytes << " its index lists\n";
        return false;
    }
    if (have > bytes) fs::resize_file(o.out, bytes, ec); // a game cut off before its index line: played again
    if (ec) {
        std::cerr << "cannot truncate " << o.out << ": " << ec.message() << "\n";
        return false;
    }
    for (long long g = o.firstGame; g < o.firstGame + o.games; ++g)
        if (!done.count(g)) shardTodo.push_back(g);
    resumedGames = o.games - static_cast<long long>(shardTodo.size());
    return true;
}

void worker(const Options &o, int id, std::ofstream &out) {
    std::mt19937_64 rng(o.seed * 1000003ULL + static_cast<uint64_t>(id));
    TranspositionTable tt(static_cast<size_t>(o.hashMB));
    Searcher<FastBoard> searcher(&tt);
    std::vector<Record> records;
    while (gamesStarted.fetch_add(1) < o.games) {
        double result = 0.5;
        while (!playGame(o, rng, searcher, tt, records, result)) {
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
    std::ofstream out(o.out, o.binary ? std::ios::app | std::ios::binary : std::ios::app);
    if (!out) {
        std::cerr << "cannot open " << o.out << "\n";
        return 1;
    }
    std::ofstream index;
    if (shard) {
        const bool fresh = !std::filesystem::exists(indexPath);
        index.open(indexPath, std::ios::app);
        if (!index) {
            std::cerr << "cannot open " << indexPath << "\n";
            return 1;
        }
        if (fresh) index << shardHeader(o) << '\n' << std::flush;
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
    return 0;
}
