#include "search.h"
#include "board.h"
#include "evaluator.h"
#include "../meta/meta_learner.h"

#include <algorithm>
#include <cmath>
#include <thread>
#include <iostream>

namespace nags {

Search::Search(Board* board, Evaluator* evaluator)
    : board_(board), 
      evaluator_(evaluator),
      stop_flag_(false),
      nodes_searched_(0),
      tt_size_(1024 * 1024 * 64),  // 64MB default
      tt_age_(0) {
    
    // Initialize transposition table
    tt_.resize(tt_size_);
    clearHistory();
    
    // Initialize bandit for search scheduling
    bandit_ = std::make_unique<BayesianBandit>();
}

Search::~Search() {
    // Ensure all worker threads are stopped
    stop();
    joinWorkerThreads();
}

void Search::clearHistory() {
    // Clear transposition table
    for (auto& entry : tt_) {
        entry.key = 0;
        entry.depth = 0;
    }
    
    // Clear killer moves and history heuristics
    for (int t = 0; t < 64; t++) {
        for (int d = 0; d < 100; d++) {
            for (int i = 0; i < 2; i++) {
                killer_moves_[t][d][i] = Move();
            }
        }
        history_scores_[t].clear();
        counter_moves_[t].clear();
    }
    
    // Clear statistics
    nodes_searched_ = 0;
    for (int t = 0; t < 64; t++) {
        selective_depth_[t] = 0;
    }
}

SearchResult Search::search(const SearchOptions& options) {
    // Initialize search
    stop_flag_ = false;
    start_time_ = std::chrono::steady_clock::now();
    max_time_ = std::chrono::milliseconds(options.max_time_ms);
    
    // Clear result
    {
        std::lock_guard<std::mutex> lock(result_mutex_);
        result_ = SearchResult();
    }
    
    // Schedule search resources based on position characteristics
    schedule_search_resources(options);
    
    // Determine number of threads to use
    int num_threads = std::min(options.num_threads, 64);
    
    // Launch worker threads for parallel search
    launchWorkerThreads(num_threads);
    
    // Wait for search to complete
    joinWorkerThreads();
    
    // Increment transposition table age
    tt_age_ = (tt_age_ + 1) % 255;
    
    // Return result
    std::lock_guard<std::mutex> lock(result_mutex_);
    return result_;
}

void Search::launchWorkerThreads(int num_threads) {
    // Initialize thread-local board copies
    thread_boards_.clear();
    for (int i = 0; i < num_threads; i++) {
        thread_boards_.push_back(std::make_unique<Board>(*board_));
    }
    
    // Initialize thread status
    thread_idle_.resize(num_threads);
    for (int i = 0; i < num_threads; i++) {
        thread_idle_[i] = false;
    }
    
    // Launch worker threads
    worker_threads_.clear();
    for (int i = 0; i < num_threads; i++) {
        worker_threads_.push_back(std::thread(&Search::searchWorker, this, i));
    }
}

void Search::joinWorkerThreads() {
    // Signal all threads to stop
    stop_flag_ = true;
    cv_.notify_all();
    
    // Join all threads
    for (auto& thread : worker_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    // Clear threads
    worker_threads_.clear();
}

void Search::searchWorker(int thread_id) {
    // Main thread (thread 0) performs iterative deepening
    if (thread_id == 0) {
        std::vector<Move> pv;
        int prev_score = 0;
        
        // Iterative deepening
        for (int depth = 1; depth <= 100; depth++) {
            if (stop_flag_) break;
            
            // Use aspiration windows for depth > 3
            int score;
            if (depth > 3) {
                score = aspirationWindowSearch(prev_score, depth, pv);
            } else {
                pv.clear();
                score = alphaBeta(depth, -30000, 30000, pv, true, thread_id);
            }
            
            if (stop_flag_) break;
            
            // Update best move
            if (!pv.empty()) {
                update_best_move(pv[0], score, depth);
            }
            
            // Store score for next iteration
            prev_score = score;
        }
    } 
    // Helper threads work on reduced depth
    else {
        while (!stop_flag_) {
            // Try to steal work from other threads
            if (!stealWork(thread_id)) {
                // If no work to steal, help with main search at reduced depth
                std::vector<Move> pv;
                for (int depth = 1; depth <= 100; depth += 2) {  // Skip depths to avoid redundant work
                    if (stop_flag_) break;
                    alphaBeta(depth, -30000, 30000, pv, false, thread_id);
                }
            }
            
            // Check if we should terminate
            if (is_time_up() || stop_flag_) {
                break;
            }
        }
    }
}

int Search::aspirationWindowSearch(int prev_score, int depth, std::vector<Move>& pv) {
    // Initial window size based on depth
    int window = 25 + 5 * depth;
    int alpha = prev_score - window;
    int beta = prev_score + window;
    
    while (true) {
        // Clear PV for this attempt
        pv.clear();
        
        // Search with current window
        int score = alphaBeta(depth, alpha, beta, pv, true, 0);
        
        // Check if time is up
        if (stop_flag_ || is_time_up()) {
            return score;
        }
        
        // If score is within the window, return it
        if (score > alpha && score < beta) {
            return score;
        }
        
        // If score fails low, widen the window on the lower end
        if (score <= alpha) {
            alpha = std::max(-30000, alpha - window);
            window *= 2;
        }
        // If score fails high, widen the window on the upper end
        else if (score >= beta) {
            beta = std::min(30000, beta + window);
            window *= 2;
        }
        
        // If window is already maximum, do a full-width search
        if (alpha <= -29000 && beta >= 29000) {
            pv.clear();
            return alphaBeta(depth, -30000, 30000, pv, true, 0);
        }
    }
}

int Search::alphaBeta(int depth, int alpha, int beta, std::vector<Move>& pv, bool is_pv_node, int thread_id) {
    // Check for stop conditions
    if (is_time_up() || stop_flag_) {
        return 0;
    }
    
    // Update selective depth
    if (depth > selective_depth_[thread_id]) {
        selective_depth_[thread_id] = depth;
    }
    
    // Check transposition table
    uint64_t pos_key = thread_boards_[thread_id]->getPositionKey();
    size_t index = pos_key % tt_size_;
    TTEntry& entry = tt_[index];
    
    if (entry.key == pos_key && entry.depth >= depth) {
        if ((entry.flag == TTEntry::Flag::EXACT) ||
            (entry.flag == TTEntry::Flag::ALPHA && entry.score <= alpha) ||
            (entry.flag == TTEntry::Flag::BETA && entry.score >= beta)) {
            // We can use this entry
            if (!pv.empty() && entry.best_move.from != 0) {
                pv[0] = entry.best_move;
            }
            return entry.score;
        }
    }
    
    // At leaf nodes, use quiescence search
    if (depth <= 0) {
        return quiescence(alpha, beta, thread_id);
    }
    
    // Try null move pruning if not in check and not in PV node
    if (!is_pv_node && depth >= 3 && !thread_boards_[thread_id]->inCheck() && nullMovePrune(depth, beta, thread_id)) {
        return beta;
    }
    
    // Get legal moves
    std::vector<Move> moves;
    thread_boards_[thread_id]->generateLegalMoves(moves);
    
    // Check for checkmate/stalemate
    if (moves.empty()) {
        if (thread_boards_[thread_id]->inCheck()) {
            return -30000 + thread_boards_[thread_id]->getHalfmoveClock(); // Checkmate
        } else {
            return 0; // Stalemate
        }
    }
    
    // Move ordering
    // 1. Hash move
    // 2. Captures ordered by MVV-LVA
    // 3. Killer moves
    // 4. History heuristic
    
    // Check if hash move exists
    Move hash_move;
    if (entry.key == pos_key && entry.best_move.from != 0) {
        hash_move = entry.best_move;
    }
    
    // Score moves for ordering
    std::vector<std::pair<int, Move>> scored_moves;
    for (const Move& move : moves) {
        int score = 0;
        
        // Hash move gets highest priority
        if (move == hash_move) {
            score = 10000000;
        }
        // Captures
        else if (move.is_capture) {
            // MVV-LVA (Most Valuable Victim - Least Valuable Attacker)
            int victim_value = thread_boards_[thread_id]->getPieceValue(move.to);
            int attacker_value = thread_boards_[thread_id]->getPieceValue(move.from);
            score = 1000000 + victim_value * 100 - attacker_value;
        }
        // Killer moves
        else if (move == killer_moves_[thread_id][depth][0]) {
            score = 900000;
        }
        else if (move == killer_moves_[thread_id][depth][1]) {
            score = 800000;
        }
        // History heuristic
        else {
            std::string move_str = move.toString();
            auto it = history_scores_[thread_id].find(move_str);
            if (it != history_scores_[thread_id].end()) {
                score = it->second;
            }
        }
        
        scored_moves.push_back({score, move});
    }
    
    // Sort moves by score (descending)
    std::sort(scored_moves.begin(), scored_moves.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    Move best_move;
    bool found_pv = false;
    int original_alpha = alpha;
    std::vector<Move> new_pv;
    
    // Loop through moves
    for (size_t i = 0; i < scored_moves.size(); i++) {
        const Move& move = scored_moves[i].second;
        
        // Make move
        thread_boards_[thread_id]->makeMove(move);
        
        // Increment nodes count
        nodes_searched_++;
        
        // Detect tactical shots
        bool is_tactical = detect_tactical_shot(move, depth);
        
        // Futility pruning
        if (!is_pv_node && !is_tactical && depth < 3 && !thread_boards_[thread_id]->inCheck() && 
            futilityPrune(move, depth, alpha, is_pv_node)) {
            thread_boards_[thread_id]->undoMove();
            continue;
        }
        
        // Late move reduction
        int score;
        new_pv.clear();
        
        if (!is_pv_node && !is_tactical && i >= 3 && depth >= 3 && lateMoveReduction(move, depth, i, is_pv_node)) {
            int reduction = reductionAmount(depth, i, is_pv_node);
            score = -alphaBeta(depth - 1 - reduction, -beta, -alpha, new_pv, false, thread_id);
            
            // If the reduced search fails high, we need to do a full-depth search
            if (score > alpha) {
                new_pv.clear();
                score = -alphaBeta(depth - 1, -beta, -alpha, new_pv, is_pv_node, thread_id);
            }
        } else {
            // Normal alpha-beta search
            score = -alphaBeta(depth - 1, -beta, -alpha, new_pv, is_pv_node, thread_id);
        }
        
        // Undo move
        thread_boards_[thread_id]->undoMove();
        
        // Check for time up
        if (is_time_up() || stop_flag_) {
            return 0;
        }
        
        // Update alpha/beta
        if (score > alpha) {
            alpha = score;
            best_move = move;
            
            // Update PV
            pv.clear();
            pv.push_back(move);
            pv.insert(pv.end(), new_pv.begin(), new_pv.end());
            
            found_pv = true;
            
            // Beta cutoff
            if (alpha >= beta) {
                // Update killer moves
                if (!move.is_capture) {
                    killer_moves_[thread_id][depth][1] = killer_moves_[thread_id][depth][0];
                    killer_moves_[thread_id][depth][0] = move;
                    
                    // Update history heuristic
                    std::string move_str = move.toString();
                    history_scores_[thread_id][move_str] += depth * depth;
                }
                break;
            }
        }
    }
    
    // Store in transposition table
    TTEntry::Flag flag;
    if (alpha <= original_alpha) {
        flag = TTEntry::Flag::ALPHA;
    } else if (alpha >= beta) {
        flag = TTEntry::Flag::BETA;
    } else {
        flag = TTEntry::Flag::EXACT;
    }
    
    entry.key = pos_key;
    entry.depth = depth;
    entry.score = alpha;
    entry.best_move = best_move;
    entry.flag = flag;
    entry.age = tt_age_;
    
    return alpha;
}

int Search::quiescence(int alpha, int beta, int thread_id) {
    // Check for stop conditions
    if (is_time_up() || stop_flag_) {
        return 0;
    }
    
    // Increment nodes count
    nodes_searched_++;
    
    // Get static evaluation
    int stand_pat = evaluator_->evaluate(*thread_boards_[thread_id]);
    
    // Return if we can't improve
    if (stand_pat >= beta) {
        return beta;
    }
    
    // Update alpha if static eval is better
    if (stand_pat > alpha) {
        alpha = stand_pat;
    }
    
    // Get capture moves only
    std::vector<Move> moves;
    thread_boards_[thread_id]->generateCaptures(moves);
    
    // Score moves for MVV-LVA ordering
    std::vector<std::pair<int, Move>> scored_moves;
    for (const Move& move : moves) {
        int victim_value = thread_boards_[thread_id]->getPieceValue(move.to);
        int attacker_value = thread_boards_[thread_id]->getPieceValue(move.from);
        int score = victim_value * 100 - attacker_value;
        
        // Delta pruning - skip clearly bad captures
        if (stand_pat + victim_value + 200 < alpha) {
            continue;
        }
        
        scored_moves.push_back({score, move});
    }
    
    // Sort moves by score (descending)
    std::sort(scored_moves.begin(), scored_moves.end(), 
              [](const auto& a, const auto& b) { return a.first > b.first; });
    
    // Loop through moves
    for (const auto& [score, move] : scored_moves) {
        // Make move
        thread_boards_[thread_id]->makeMove(move);
        
        // Recursive quiescence search
        int q_score = -quiescence(-beta, -alpha, thread_id);
        
        // Undo move
        thread_boards_[thread_id]->undoMove();
        
        // Check for time up
        if (is_time_up() || stop_flag_) {
            return 0;
        }
        
        // Update alpha
        if (q_score > alpha) {
            alpha = q_score;
            
            // Beta cutoff
            if (alpha >= beta) {
                break;
            }
        }
    }
    
    return alpha;
}

bool Search::nullMovePrune(int depth, int beta, int thread_id) {
    // Don't use null move pruning in endgame
    if (thread_boards_[thread_id]->isEndgame()) {
        return false;
    }
    
    // Make null move
    thread_boards_[thread_id]->makeNullMove();
    
    // Reduced depth search
    std::vector<Move> temp_pv;
    int r = 2 + depth / 4;
    int score = -alphaBeta(depth - 1 - r, -beta, -beta + 1, temp_pv, false, thread_id);
    
    // Undo null move
    thread_boards_[thread_id]->undoNullMove();
    
    // If score exceeds beta, prune this branch
    return score >= beta;
}

bool Search::futilityPrune(const Move& move, int depth, int alpha, bool is_pv_node) {
    // Don't use futility pruning in PV nodes or if in check
    if (is_pv_node || thread_boards_[0]->inCheck()) {
        return false;
    }
    
    // Don't prune captures, promotions, or checks
    if (move.is_capture || move.is_promotion || thread_boards_[0]->moveGivesCheck(move)) {
        return false;
    }
    
    // Futility margin based on depth
    int margin = 100 * depth;
    
    // Static evaluation plus margin
    int futility_value = evaluator_->evaluate(*thread_boards_[0]) + margin;
    
    // If even with the margin we can't exceed alpha, prune this move
    return futility_value <= alpha;
}

bool Search::lateMoveReduction(const Move& move, int depth, int move_index, bool is_pv_node) {
    // Don't use LMR for captures, promotions, checks, or early moves
    return !move.is_capture && !move.is_promotion && !thread_boards_[0]->moveGivesCheck(move) && move_index >= 3;
}

int Search::reductionAmount(int depth, int move_index, bool is_pv_node) {
    // Calculate reduction based on depth and move index
    double r = 0.5 + log(depth) * log(move_index) / 3.0;
    
    // Reduce less in PV nodes
    if (is_pv_node) {
        r = std::max(0.0, r - 0.5);
    }
    
    return static_cast<int>(r);
}

bool Search::stealWork(int thread_id) {
    // Try to find a busy thread to steal work from
    for (size_t i = 0; i < thread_idle_.size(); i++) {
        if (i != thread_id && !thread_idle_[i]) {
            // Mark this thread as idle
            thread_idle_[thread_id] = true;
            
            // Work stealing logic would go here
            // For now, just return false to indicate no work was stolen
            
            // Mark as not idle again
            thread_idle_[thread_id] = false;
            return false;
        }
    }
    
    return false;
}

void Search::shareWork(int thread_id) {
    // Logic to share work with other threads
    // Not implemented in this version
}

void Search::stop() {
    stop_flag_ = true;
}

bool Search::is_time_up() const {
    if (max_time_.count() == 0) {
        return false;
    }
    
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time_);
    
    return elapsed >= max_time_;
}

void Search::update_best_move(const Move& move, int score, int depth, int thread_id) {
    std::lock_guard<std::mutex> lock(result_mutex_);
    
    // Only update if this is a deeper search or from the main thread
    if (depth > result_.depth || (depth == result_.depth && thread_id == 0)) {
        result_.best_move = move;
        result_.score = score;
        result_.depth = depth;
        result_.nodes = nodes_searched_;
        result_.selective_depth = *std::max_element(selective_depth_.begin(), selective_depth_.end());
    }
}

} // namespace nags 