#pragma once

#include "../core/board.h"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

namespace nags {

// Forward declarations
class Board;
struct SearchOptions;

// Result of neural network evaluation
struct NNEvaluation {
    float value;                        // Position evaluation (-1 to 1)
    float policy[4096];                 // Move probabilities
    float value_uncertainty;            // Uncertainty in value estimate
    float tactical_score;               // Tactical opportunities score
    float positional_score;             // Positional understanding score
    std::vector<float> feature_values;  // Extracted feature values
    bool is_valid;                      // Whether evaluation is valid
};

// Class for neural network evaluation using GNN
class GNNEvaluator {
public:
    GNNEvaluator();
    ~GNNEvaluator();

    // Initialize the evaluator with model files
    bool initialize(const std::string& modelPath);

    // Evaluate a position
    NNEvaluation evaluate(const Board* board);
    
    // Batch evaluate multiple positions
    std::vector<NNEvaluation> evaluateBatch(const std::vector<const Board*>& boards);
    
    // Get policy for a position (normalized probabilities for all legal moves)
    std::vector<float> getPolicyProbabilities(const Board* board);
    
    // Get value with uncertainty (uses Monte Carlo dropout)
    std::pair<float, float> getValueWithUncertainty(const Board* board, int samples = 10);
    
    // Extract features for meta-learning
    std::vector<float> extractFeatures(const Board* board);
    
    // Get tactical opportunities score
    float getTacticalScore(const Board* board);
    
    // Get positional understanding score
    float getPositionalScore(const Board* board);
    
    // Extract pattern information
    std::unordered_map<std::string, float> detectPatterns(const Board* board);
    
    // Provide search guidance based on neural network insights
    void provideSearchGuidance(const Board* board, SearchOptions& options);
    
    // Save the network state
    bool saveState(const std::string& path);
    
    // Load the network state
    bool loadState(const std::string& path);

private:
    // Implementation details
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    // Whether the evaluator is initialized
    bool is_initialized_;
};

} // namespace nags 