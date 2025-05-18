#include "engine.h"
#include "board.h"
#include "search.h"
#include "evaluator.h"
#include "../meta/meta_learner.h"

#include <iostream>
#include <sstream>

namespace nags {

std::string Move::toUCI() const {
    std::string uci = from + to;
    if (promotion.has_value()) {
        uci += promotion.value();
    }
    return uci;
}

Move Move::fromUCI(const std::string& uci) {
    Move move;
    move.from = uci.substr(0, 2);
    move.to = uci.substr(2, 2);
    
    if (uci.length() > 4) {
        move.promotion = uci[4];
    }
    
    return move;
}

Engine::Engine() 
    : isSearching_(false) {
    board_ = std::make_unique<Board>();
    evaluator_ = std::make_unique<Evaluator>();
    search_ = std::make_unique<Search>(board_.get(), evaluator_.get());
    metaLearner_ = std::make_unique<MetaLearner>();
    
    std::cout << "* Engine initialized" << std::endl;
}

Engine::~Engine() {
    if (isSearching_) {
        stopSearch();
    }
}

void Engine::newGame() {
    board_->resetToStartPosition();
    search_->clearHistory();
    std::cout << "* New game started" << std::endl;
}

void Engine::setPosition(const std::string& fen, const std::vector<std::string>& moves) {
    if (fen.empty() || fen == "startpos") {
        board_->resetToStartPosition();
    } else {
        board_->setFromFEN(fen);
    }
    
    for (const auto& moveStr : moves) {
        Move move = Move::fromUCI(moveStr);
        board_->makeMove(move);
    }
}

SearchResult Engine::startSearch(const SearchOptions& options) {
    isSearching_ = true;
    
    // Use meta-learner to adapt search parameters
    auto adaptedParams = metaLearner_->adaptSearchParameters(
        board_.get(), 
        options
    );
    
    // Start search with adapted parameters
    SearchResult result = search_->search(adaptedParams);
    
    isSearching_ = false;
    return result;
}

void Engine::stopSearch() {
    if (isSearching_) {
        search_->stop();
        isSearching_ = false;
    }
}

std::vector<Move> Engine::getLegalMoves() const {
    return board_->generateLegalMoves();
}

std::string Engine::getFEN() const {
    return board_->toFEN();
}

std::string Engine::getBoardString() const {
    return board_->toString();
}

} // namespace nags 