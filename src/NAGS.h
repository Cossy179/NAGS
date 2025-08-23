#pragma once

#include "Board.h"
#include "Search.h"
#include "TT.h"
#include "MetaClient.h"

#include <chrono>
#include <random>
#include <vector>
#include <unordered_map>
#include <memory>

// Forward declarations
struct MCTSNode;
struct EvalResult;

// Bandit arm types
enum class SearchArm : int { DFS = 0, MCTS = 1 };

// Evaluation result from neural network
struct EvalResult {
    std::vector<float> policy; // 4096 entries (64x64 from-to)
    float value;
    float uncertainty;
};

// MCTS Node
struct MCTSNode {
    Move move;
    MCTSNode* parent = nullptr;
    std::vector<std::unique_ptr<MCTSNode>> children;
    
    int visits = 0;
    float value_sum = 0.0f;
    float prior = 0.0f;
    bool expanded = false;
    
    float q_value() const { return visits > 0 ? value_sum / visits : 0.0f; }
    float ucb_score(float parent_visits, float c_puct = 1.4f) const {
        if (visits == 0) return prior * std::sqrt(parent_visits) / (1 + visits);
        return q_value() + c_puct * prior * std::sqrt(parent_visits) / (1 + visits);
    }
};

// Bayesian Bandit for arm selection
class BayesianBandit {
public:
    BayesianBandit();
    SearchArm select_arm();
    void update_reward(SearchArm arm, float reward);
    void log_stats() const;
    
private:
    struct ArmStats {
        float alpha = 1.0f; // Beta distribution parameters
        float beta = 1.0f;
        int pulls = 0;
        float total_reward = 0.0f;
    };
    
    ArmStats arms[2]; // DFS, MCTS
    std::mt19937 rng;
    float sample_beta(float alpha, float beta);
};

// Neural Network Interface (stub for RPC)
class NeuralEvaluator {
public:
    NeuralEvaluator();
    EvalResult evaluate(const Board& board);
    
private:
    // In full implementation, this would manage RPC connections
    std::mt19937 rng; // For dummy evaluation
};

// Main NAGS Controller
class NAGSController {
public:
    NAGSController(Board& board);
    
    Move search(const SearchLimits& limits);
    void stop() { should_stop = true; }
    
private:
    Board& board;
    TranspositionTable tt;
    BayesianBandit bandit;
    NeuralEvaluator evaluator;
    std::unique_ptr<MCTSNode> mcts_root;
    MetaClient meta_client;
    
    bool should_stop = false;
    int iteration_count = 0;
    
    // Meta-learning state
    float last_uncertainty = 0.1f;
    int base_dfs_depth = 6;
    int base_mcts_budget = 1000;
    float base_exploration = 1.4f;
    
    // DFS component
    Move dfs_probe(int depth);
    int dfs_search(int depth, int alpha, int beta, int ply);
    int quiescence(int alpha, int beta, int ply);
    
    // MCTS component  
    void mcts_iteration();
    MCTSNode* select_node(MCTSNode* node);
    void expand_node(MCTSNode* node);
    float simulate(MCTSNode* node);
    void backpropagate(MCTSNode* node, float value);
    
    // Hybrid decision making
    Move select_best_move();
    float calculate_reward(SearchArm arm, float value, float uncertainty);
    float tactical_volatility();
    
    // Utilities
    int eval_position();
    void order_moves(std::vector<Move>& moves);
    bool time_up(const SearchLimits& limits, std::chrono::steady_clock::time_point start) const;
};
