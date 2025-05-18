#pragma once

#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>

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
    // Alpha-beta search
    int alphaBeta(int depth, int alpha, int beta, std::vector<Move>& pv, bool is_pv_node);
    
    // Quiescence search
    int quiescence(int alpha, int beta);
    
    // MCTS search
    void mcts_search(MCTSNode& root, int num_simulations);
    
    // MCTS simulation
    double mcts_simulate(MCTSNode& node, int depth);
    
    // Hybrid search scheduling
    void schedule_search_resources(const SearchOptions& options);
    
    // Tactical shot detection
    bool detect_tactical_shot(const Move& move, int depth);
    
    // Helpers
    bool is_time_up() const;
    void update_best_move(const Move& move, int score, int depth);
    void collect_principal_variation(const MCTSNode& root, std::vector<Move>& pv);
    
    // References to board and evaluator
    Board* board_;
    Evaluator* evaluator_;
    
    // Search control
    std::atomic<bool> stop_flag_;
    std::chrono::time_point<std::chrono::steady_clock> start_time_;
    std::chrono::milliseconds max_time_;
    
    // Search results
    SearchResult result_;
    std::mutex result_mutex_;
    
    // Transposition table
    struct TTEntry {
        uint64_t key;
        int depth;
        int score;
        Move best_move;
        enum class Flag { EXACT, ALPHA, BETA } flag;
    };
    std::unordered_map<uint64_t, TTEntry> tt_;
    
    // Killer moves and history heuristics
    std::array<std::array<Move, 2>, 100> killer_moves_;
    std::unordered_map<std::string, int> history_scores_;
    
    // Bayesian bandit for search scheduling
    std::unique_ptr<BayesianBandit> bandit_;
    
    // Search statistics
    uint64_t nodes_searched_;
    int selective_depth_;
};

} // namespace nags 