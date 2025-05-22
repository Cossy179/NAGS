#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>
#pragma once

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
namespace nags {
    
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
public:
    Search(Board* board, Evaluator* evaluator);
    ~Search();
    
    // Main search function
    SearchResult search(const SearchOptions& options);
    
    
    // Quiescence search
    
    // Clear search history and transposition tables
    void clearHistory();
    
private:
    // Parallel search control
    void launchWorkerThreads(int num_threads);
    
    void joinWorkerThreads();
    void mcts_search(MCTSNode& root, int num_simulations);
    
    // MCTS simulation
    
    // Quiescence search
    int quiescence(int alpha, int beta, int thread_id = 0);
    
    
    // Hybrid search scheduling
    void schedule_search_resources(const SearchOptions& options);
    
    // Tactical shot detection
    bool detect_tactical_shot(const Move& move, int depth);
    // MCTS search with virtual loss
    void mcts_search(MCTSNode& root, int num_simulations);
    
    // MCTS simulation
    double mcts_simulate(MCTSNode& node, int depth, int thread_id = 0);
    
    // Multi-variant search
    
    // Helpers
    bool is_time_up() const;
    void schedule_search_resources(const SearchOptions& options);
    void collect_principal_variation(const MCTSNode& root, std::vector<Move>& pv);
    
    // References to board and evaluator
    Board* board_;
    Evaluator* evaluator_;
    bool stealWork(int thread_id);
    void shareWork(int thread_id);
    
    
    // Search control
    std::atomic<bool> stop_flag_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    std::chrono::milliseconds max_time_;
    void update_best_move(const Move& move, int score, int depth, int thread_id = 0);
    void collect_principal_variation(const MCTSNode& root, std::vector<Move>& pv);
    
    // References to board and evaluator
    
    // Search results
    SearchResult result_;
    std::mutex result_mutex_;
    
    
    // Search control
        uint64_t key;
        int depth;
        int score;
        Move best_move;
        enum class Flag { EXACT, ALPHA, BETA } flag;
    std::mutex search_mutex_;
    };
    
    // Search results
    SearchResult result_;
    
    // Killer moves and history heuristics
    // Transposition table with lock-free access
    struct alignas(64) TTEntry {
        uint64_t key;
    
    // Bayesian bandit for search scheduling
    std::unique_ptr<BayesianBandit> bandit_;
    
    // Search statistics
    };
    std::vector<TTEntry> tt_;
};

} // namespace nags 