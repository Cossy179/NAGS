#pragma once

#include <string>
#include <vector>
#include <memory>

namespace nags {

// Forward declarations
class Board;
struct SearchOptions;

class MetaLearner {
public:
    MetaLearner();
    ~MetaLearner();
    
    // Initialize from training data
    bool initialize(const std::string& datasetPath);
    
    // Adapt search parameters for current position
    SearchOptions adaptSearchParameters(const Board* board, const SearchOptions& baseOptions);
    
    // Learn from search results
    void learnFromSearchResult(const Board* board, const SearchOptions& options, double eloDelta);
    
    // Save/load model
    bool saveModel(const std::string& path);
    bool loadModel(const std::string& path);
    
private:
    // Implementation details will depend on PyTorch bindings
    class Impl;
    std::unique_ptr<Impl> impl_;
    
    // Training dataset path
    std::string datasetPath_;
    
    // Flag indicating if the model is initialized
    bool isInitialized_;
};

} // namespace nags 