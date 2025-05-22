#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>
#include <thread>

#include "engine.h"

namespace nags {

// Forward declarations
class Board;
class Evaluator;
class BayesianBandit;

struct MCTSNode {
    Move move;                      // Move that led to this node
    double value_sum;               // Sum of values for this node
    int visit_count;                // Number of visits
    double prior;                   // Prior probability from policy network
    double uncertainty;             // Value uncertainty
    std::vector<MCTSNode> children; // Child nodes
    std::mutex node_mutex;          // Mutex for thread-safety
    
    // PUCT score calculation
    double puct_score(double parent_visit_sqrt, double exploration_constant) const;
    
    // Select best child according to PUCT score
    MCTSNode* select_best_child(double exploration_constant);
    
    // Expand node with policy priors
    void expand(const std::vector<Move>& legal_moves, const std::vector<float>& policy_priors);
    
    // Backup value through the tree
    void backup(double value, int up_to_depth);
};

class Search {
public:
    Search(Board* board, Evaluator* evaluator);
    ~Search();
    
    // Main search function
    SearchResult search(const SearchOptions& options);
    
    // Stop the search
    void stop();
    
    // Clear search history and transposition tables
    void clearHistory();
    
private:
    // Parallel search control
    void launchWorkerThreads(int num_threads);
    void searchWorker(int thread_id);
    void joinWorkerThreads();
    
    // Alpha-beta search with aspiration windows
    int alphaBeta(int depth, int alpha, int beta, std::vector<Move>& pv, bool is_pv_node, int thread_id = 0);
    
    // Quiescence search
    int quiescence(int alpha, int beta, int thread_id = 0);
    
    // Pruning techniques
    bool nullMovePrune(int depth, int beta, int thread_id);
    bool futilityPrune(const Move& move, int depth, int alpha, bool is_pv_node);
    bool lateMoveReduction(const Move& move, int depth, int move_index, bool is_pv_node);
    int reductionAmount(int depth, int move_index, bool is_pv_node);
    
    // MCTS search with virtual loss
    void mcts_search(MCTSNode& root, int num_simulations);
    
    // MCTS simulation
    double mcts_simulate(MCTSNode& node, int depth, int thread_id = 0);
    
    // Multi-variant search
    void searchMultipleVariants(const std::vector<Move>& candidate_moves, int num_variants);
    
    // Hybrid search scheduling
    void schedule_search_resources(const SearchOptions& options);
    
    // Tactical shot detection
    bool detect_tactical_shot(const Move& move, int depth);
    
    // Work stealing for parallel search
    bool stealWork(int thread_id);
    void shareWork(int thread_id);
    
    // Aspiration window search
    int aspirationWindowSearch(int prev_score, int depth, std::vector<Move>& pv);
    
    // Helpers
    bool is_time_up() const;
    void update_best_move(const Move& move, int score, int depth, int thread_id = 0);
    void collect_principal_variation(const MCTSNode& root, std::vector<Move>& pv);
    
    // References to board and evaluator
    Board* board_;
    Evaluator* evaluator_;
    
    // Thread-local board copies for parallel search
    std::vector<std::unique_ptr<Board>> thread_boards_;
    
    // Search control
    std::atomic<bool> stop_flag_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    std::chrono::milliseconds max_time_;
    std::vector<std::thread> worker_threads_;
    std::vector<std::atomic<bool>> thread_idle_;
    std::mutex search_mutex_;
    std::condition_variable cv_;
    
    // Search results
    SearchResult result_;
    std::mutex result_mutex_;
    
    // Transposition table with lock-free access
    struct alignas(64) TTEntry {
        uint64_t key;
        int depth;
        int score;
        Move best_move;
        enum class Flag { EXACT, ALPHA, BETA } flag;
        uint8_t age;
    };
    std::vector<TTEntry> tt_;
    size_t tt_size_;
    uint8_t tt_age_;
    
    // Killer moves and history heuristics
    std::array<std::array<std::array<Move, 2>, 100>, 64> killer_moves_; // [thread][depth][index]
    std::array<std::unordered_map<std::string, int>, 64> history_scores_; // [thread][move]
    std::array<std::unordered_map<std::string, int>, 64> counter_moves_; // [thread][prev_move -> counter]
    
    // Bayesian bandit for search scheduling
    std::unique_ptr<BayesianBandit> bandit_;
    
    // Search statistics
    std::atomic<uint64_t> nodes_searched_;
    std::array<int, 64> selective_depth_;
};

} // namespace nags 