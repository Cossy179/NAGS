#pragma once

#include <cstdint>
#include <vector>
#include <atomic>
#include <mutex>

#include "FastBoard.h"

enum class TTFlag : uint8_t { Exact = 0, Lower = 1, Upper = 2 };

// Transposition table entry with padding to avoid false sharing
struct alignas(64) TTEntry {
    uint64_t key;
    int32_t score;
    int16_t depth;
    uint8_t flag;
    uint32_t movePacked;
    
    // Padding to cache line size
    char padding[64 - sizeof(uint64_t) - sizeof(int32_t) - sizeof(int16_t) - sizeof(uint8_t) - sizeof(uint32_t)];

    TTEntry() : key(0), score(0), depth(-1), flag(0), movePacked(0) {}
    
    TTEntry(uint64_t k, int32_t s, int16_t d, uint8_t f, uint32_t m) 
        : key(k), score(s), depth(d), flag(f), movePacked(m) {}
};

class FastTranspositionTable {
public:
    FastTranspositionTable();
    
    void resizeMB(size_t megabytes);
    void clear();
    
    bool probe(uint64_t key, TTEntry &out) const;
    void store(uint64_t key, int depth, TTFlag flag, int score, const Move *bestMove);
    
    // Statistics
    size_t getHits() const { return hits.load(); }
    size_t getProbes() const { return probes.load(); }
    double getHitRate() const { 
        size_t p = probes.load();
        return p > 0 ? static_cast<double>(hits.load()) / p : 0.0;
    }
    void resetStats() { hits = 0; probes = 0; }
    
    static uint32_t packMove(const Move &m);
    static Move unpackMove(uint32_t p);
    
private:
    std::vector<TTEntry> table;
    size_t mask = 0;
    mutable std::atomic<size_t> hits{0};
    mutable std::atomic<size_t> probes{0};
    
    size_t indexFromKey(uint64_t key) const { return key & mask; }
};

// Thread-safe search statistics
struct SearchStats {
    std::atomic<uint64_t> nodes{0};
    std::atomic<uint64_t> tt_hits{0};
    std::atomic<uint64_t> tt_probes{0};
    std::atomic<uint64_t> beta_cutoffs{0};
    
    void reset() {
        nodes = 0;
        tt_hits = 0;
        tt_probes = 0;
        beta_cutoffs = 0;
    }
};

// Per-thread search data
struct ThreadData {
    FastBoard board;
    std::vector<Move> pv;
    SearchStats stats;
    int thread_id;
    
    ThreadData(int id) : thread_id(id) {}
};

// Search limits and control
struct SearchLimits {
    int depth = 0;
    long long timeMs = 0;
    long long wtime = 0;
    long long btime = 0;
    int movestogo = 0;
    long long movetime = 0;
    bool infinite = false;
};

// Search information shared across threads
struct SearchInfo {
    std::chrono::steady_clock::time_point start_time;
    SearchLimits limits;
    std::atomic<bool> stop_search{false};
    std::atomic<int> completed_depth{0};
    std::atomic<Move> best_move{Move{-1, -1, Piece::None, false, false}};
    std::vector<Move> best_pv;
    std::mutex pv_mutex;
    
    bool should_stop() const {
        if (stop_search.load()) return true;
        
        if (limits.movetime > 0) {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - start_time).count();
            return elapsed >= limits.movetime;
        }
        
        return false;
    }
};
