#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace nags {

struct SearchOptions {
    // Basic UCI search options
    int depth = 0;                         // Maximum depth to search
    int nodes = 0;                         // Maximum nodes to search
    bool infinite = false;                 // Analyze until stopped
    std::chrono::milliseconds moveTime{0}; // Time per move (ms)
    std::chrono::milliseconds wtime{0};    // White time remaining (ms)
    std::chrono::milliseconds btime{0};    // Black time remaining (ms)
    std::chrono::milliseconds winc{0};     // White increment (ms)
    std::chrono::milliseconds binc{0};     // Black increment (ms)
    int movesToGo = 0;                     // Moves until next time control
    
    // Advanced search parameters controlled by meta-learner
    double pruningThreshold = 0.05;        // Dynamic pruning threshold
    double explorationFactor = 1.0;        // Controls exploration vs exploitation
    bool usePolicyGuidance = true;         // Use neural network policy for move ordering
    double positionalWeight = 1.0;         // Weight for positional evaluation
    double tacticalWeight = 1.0;           // Weight for tactical evaluation
    double timeAllocationFactor = 1.0;     // Time allocation adjustment
    double uncertaintyTolerance = 0.1;     // Tolerance for value uncertainty
    
    // Hybrid search parameters
    double dfsMctsRatio = 0.5;             // Ratio of DFS vs MCTS budget
    int mctsPlayouts = 10000;              // MCTS playout budget
    int quiescenceDepth = 10;              // Maximum quiescence search depth
    bool useNullMovePruning = true;        // Enable null move pruning
    bool useFutilityPruning = true;        // Enable futility pruning
    bool useLateMoveReduction = true;      // Enable late move reduction
    bool useAspirationWindows = true;      // Enable aspiration windows
    int multiPV = 1;                       // Number of principal variations
    
    // Pattern-based parameters
    std::vector<std::string> activePatterns;// Detected patterns in position
    double patternConfidence = 0.0;        // Confidence in pattern detection
    
    // Opponent modeling parameters
    std::string opponentProfile;           // Identified opponent profile
    double adaptationFactor = 1.0;         // How much to adapt to opponent
    
    // Debug options
    bool debug = false;                    // Enable debug output
    bool logSearch = false;                // Log search details
};

} // namespace nags 