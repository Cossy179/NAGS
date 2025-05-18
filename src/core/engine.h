#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <chrono>

namespace nags {

struct Move {
    std::string from;
    std::string to;
    std::optional<char> promotion;
    
    std::string toUCI() const;
    static Move fromUCI(const std::string& uci);
};

struct SearchResult {
    Move bestMove;
    int score;  // centipawn score
    int depth;
    uint64_t nodesSearched;
    std::vector<Move> principalVariation;
};

struct SearchOptions {
    int depth = 20;
    std::chrono::milliseconds moveTime = std::chrono::milliseconds(0);
    std::chrono::milliseconds wtime = std::chrono::milliseconds(0);
    std::chrono::milliseconds btime = std::chrono::milliseconds(0);
    std::chrono::milliseconds winc = std::chrono::milliseconds(0);
    std::chrono::milliseconds binc = std::chrono::milliseconds(0);
    int movestogo = 0;
    bool infinite = false;
    int nodes = 0;
    bool ponder = false;
};

// Forward declarations
class Board;
class Search;
class Evaluator;
class MetaLearner;

class Engine {
public:
    Engine();
    ~Engine();
    
    // UCI interface methods
    void newGame();
    void setPosition(const std::string& fen, const std::vector<std::string>& moves);
    SearchResult startSearch(const SearchOptions& options);
    void stopSearch();
    std::vector<Move> getLegalMoves() const;
    
    // Utility methods
    std::string getFEN() const;
    std::string getBoardString() const;
    
private:
    std::unique_ptr<Board> board_;
    std::unique_ptr<Search> search_;
    std::unique_ptr<Evaluator> evaluator_;
    std::unique_ptr<MetaLearner> metaLearner_;
    bool isSearching_;
};

} // namespace nags 