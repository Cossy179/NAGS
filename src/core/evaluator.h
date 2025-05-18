#pragma once

#include <vector>
#include <memory>
#include <string>
#include <optional>
#include <future>

namespace nags {

// Forward declarations
class Board;
struct Move;

// Evaluation result with uncertainty
struct EvalResult {
    int score;                  // Centipawn score
    double uncertainty;         // Uncertainty in centipawns
    std::vector<float> policy;  // Policy logits/probabilities for moves
};

// RPC client for neural network inference
class NNClient {
public:
    NNClient(const std::string& server_address);
    ~NNClient();
    
    // Connect to the neural network server
    bool connect();
    
    // Request evaluation for a batch of positions
    std::vector<EvalResult> evaluate_batch(const std::vector<std::string>& fens);
    
    // Asynchronous evaluation
    std::future<EvalResult> evaluate_async(const std::string& fen);
    
private:
    // Implementation details depend on RPC framework (gRPC/ZeroMQ)
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    // Server connection info
    std::string server_address_;
    bool is_connected_;
};

class Evaluator {
public:
    Evaluator();
    ~Evaluator();
    
    // Initialize with model and server settings
    bool initialize(const std::string& server_address);
    
    // Evaluate current position
    EvalResult evaluate(const Board* board);
    
    // Evaluate a specific move without making it
    std::optional<EvalResult> evaluate_move(const Board* board, const Move& move);
    
    // Batch evaluation for multiple positions
    std::vector<EvalResult> evaluate_batch(const std::vector<std::string>& fens);
    
    // Static evaluation (material & basic positional factors)
    int static_eval(const Board* board);
    
private:
    // Neural network client
    std::unique_ptr<NNClient> nn_client_;
    
    // Evaluation cache
    struct CacheEntry {
        uint64_t hash;
        EvalResult result;
    };
    std::vector<CacheEntry> eval_cache_;
    
    // Helper methods
    std::string prepare_fen(const Board* board);
    std::vector<float> prepare_policy_map(const EvalResult& result, const Board* board);
};

} // namespace nags 