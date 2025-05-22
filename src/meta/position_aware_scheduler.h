#pragma once

#include <vector>
#include <unordered_map>
#include <string>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include "../core/board.h"
#include "../core/search_options.h"

namespace nags {

/**
 * Position-specific features used for adaptive search
 */
struct PositionFeatures {
    float material_balance;          // Material balance (-1.0 to 1.0)
    float piece_mobility;            // Average piece mobility (normalized)
    float king_safety;               // King safety metric (-1.0 to 1.0)
    float pawn_structure;            // Pawn structure quality (-1.0 to 1.0)
    float control_center;            // Center control metric (-1.0 to 1.0)
    float development;               // Piece development metric (0.0 to 1.0)
    float phase;                     // Game phase (0.0=opening, 0.5=middlegame, 1.0=endgame)
    float tactical_complexity;       // Tactical complexity (0.0 to 1.0)
    float position_uncertainty;      // Position evaluation uncertainty (0.0 to 1.0)
    float position_dynamism;         // How dynamic is the position (0.0 to 1.0)
    std::vector<float> piece_threats; // Threats to each piece
    
    // Serialize to string for caching/storage
    std::string serialize() const;
    
    // Deserialize from string
    static PositionFeatures deserialize(const std::string& data);
};

/**
 * Position-aware search scheduler that dynamically adjusts search parameters 
 * based on position characteristics and learning from past searches
 */
class PositionAwareScheduler {
public:
    PositionAwareScheduler();
    ~PositionAwareScheduler() = default;
    
    // Configure search options based on position and time control
    SearchOptions configureSearch(const Board& board, 
                                 int time_left_ms, 
                                 int increment_ms,
                                 int moves_to_go);
    
    // Update scheduler with search results (for learning)
    void updateWithSearchResults(const Board& board,
                                const SearchOptions& options,
                                const SearchResult& result,
                                bool was_best_move);
    
    // Clear search history and learning data
    void clearHistory();
    
    // Save/load learned parameters
    void saveParameters(const std::string& filename);
    bool loadParameters(const std::string& filename);

private:
    // Feature extraction
    PositionFeatures extractFeatures(const Board& board);
    
    // Position classification
    enum class PositionType {
        QUIET,               // Quiet position with no immediate tactics
        TACTICAL,            // Tactical position with many captures/threats
        STRATEGIC,           // Strategic position with long-term considerations
        ENDGAME,             // Endgame position with fewer pieces
        UNCLEAR              // Position is unclear or in transition
    };
    
    PositionType classifyPosition(const PositionFeatures& features);
    
    // Time management strategies
    int calculateTimeForMove(int time_left_ms, 
                           int increment_ms, 
                           int moves_to_go,
                           const PositionFeatures& features);
    
    // Search parameter optimization
    struct SearchParameters {
        int depth;                   // Maximum depth to search
        int selective_depth;         // Selective depth for promising lines
        double null_move_R;          // Null move reduction factor
        double lmr_base;             // Late move reduction base
        double lmr_division;         // Late move reduction division factor
        double futility_margin;      // Futility pruning margin
        double aspiration_window;    // Aspiration window size
        double time_allocation;      // Time allocation factor (0.0-1.0)
        double mcts_exploration;     // MCTS exploration constant
        int threads;                 // Number of threads to use
        int hash_size_mb;            // Transposition table size in MB
        
        // Default parameters
        static SearchParameters defaultParams();
        
        // Parameters for specific position types
        static SearchParameters forPositionType(PositionType type);
        
        // Serialize to string
        std::string serialize() const;
        
        // Deserialize from string
        static SearchParameters deserialize(const std::string& data);
    };
    
    // Thompson sampling implementation for parameter optimization
    class ThompsonSampling {
    public:
        ThompsonSampling();
        
        // Add a parameter trial result
        void addResult(const SearchParameters& params, bool success);
        
        // Sample next parameters to try
        SearchParameters sampleNext(PositionType position_type);
        
        // Get the best parameters found so far
        SearchParameters getBestParameters(PositionType position_type);
        
        // Save/load state
        std::string serializeState() const;
        void deserializeState(const std::string& data);
        
    private:
        // Parameter ranges for sampling
        struct ParamRange {
            double min;
            double max;
            double step;
        };
        
        // Parameter distribution for Bayesian updating
        struct ParamDistribution {
            std::vector<double> values;     // Possible parameter values
            std::vector<double> success;    // Alpha counts (successes)
            std::vector<double> failure;    // Beta counts (failures)
        };
        
        // Distribution for each parameter
        std::unordered_map<std::string, ParamDistribution> distributions_;
        
        // Best parameters found per position type
        std::unordered_map<PositionType, SearchParameters> best_params_;
        
        // Random generator
        std::mt19937 rng_;
        
        // Initialize parameter distributions
        void initializeDistributions();
        
        // Sample from beta distribution
        double sampleBeta(double alpha, double beta);
        
        // Convert position type to string
        static std::string positionTypeToString(PositionType type);
        
        // Convert string to position type
        static PositionType stringToPositionType(const std::string& str);
    };
    
    // Pattern recognition for similar positions
    class PositionPatternRecognizer {
    public:
        PositionPatternRecognizer();
        
        // Add a position pattern with its optimal search parameters
        void addPattern(const PositionFeatures& features, 
                      const SearchParameters& params,
                      bool was_successful);
        
        // Find similar position pattern and get its parameters
        std::pair<bool, SearchParameters> findSimilarPattern(const PositionFeatures& features);
        
        // Save/load patterns
        std::string serializePatterns() const;
        void deserializePatterns(const std::string& data);
        
    private:
        // Position pattern with search parameters
        struct Pattern {
            PositionFeatures features;
            SearchParameters params;
            int success_count;
            int total_count;
            double success_rate;
        };
        
        // Stored patterns
        std::vector<Pattern> patterns_;
        
        // Calculate similarity between position features
        double calculateSimilarity(const PositionFeatures& a, const PositionFeatures& b);
    };
    
    // Dynamic time management
    class DynamicTimeManager {
    public:
        DynamicTimeManager();
        
        // Calculate time for the current move
        int calculateTime(int time_left_ms, 
                        int increment_ms, 
                        int moves_to_go,
                        const PositionFeatures& features,
                        PositionType position_type);
        
        // Update with move result
        void updateWithResult(int time_used_ms, 
                            bool was_best_move, 
                            const PositionFeatures& features);
        
        // Save/load state
        std::string serializeState() const;
        void deserializeState(const std::string& data);
        
    private:
        // Time management strategy parameters
        struct TimeStrategy {
            double base_time_pct;        // Base percentage of available time
            double critical_threshold;   // Threshold for critical positions
            double safety_margin;        // Safety margin for time control
            double complexity_factor;    // Factor for position complexity
            double increment_usage;      // How much of increment to use
        };
        
        // Time strategies for different position types
        std::unordered_map<PositionType, TimeStrategy> strategies_;
        
        // Learning rate for strategy updates
        double learning_rate_;
        
        // History of time usage
        struct TimeUsageRecord {
            PositionFeatures features;
            PositionType type;
            int time_allocated;
            int time_used;
            bool was_best_move;
        };
        std::vector<TimeUsageRecord> time_history_;
        
        // Initialize default strategies
        void initializeStrategies();
        
        // Update strategies based on history
        void updateStrategies();
    };
    
    // Components of the scheduler
    ThompsonSampling thompson_;
    PositionPatternRecognizer pattern_recognizer_;
    DynamicTimeManager time_manager_;
    
    // Cache for position features
    std::unordered_map<uint64_t, PositionFeatures> features_cache_;
    
    // History of searches for learning
    struct SearchHistoryEntry {
        uint64_t position_hash;
        PositionFeatures features;
        PositionType position_type;
        SearchParameters params;
        int time_allocated;
        int time_used;
        bool was_best_move;
    };
    std::vector<SearchHistoryEntry> search_history_;
    
    // Learning from history
    void learnFromHistory();
};

} // namespace nags 