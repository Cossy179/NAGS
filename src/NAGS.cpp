#include "NAGS.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

// Bayesian Bandit Implementation
BayesianBandit::BayesianBandit() : rng(std::random_device{}()) {
    // Initialize with uniform priors
    for (auto& arm : arms) {
        arm.alpha = 1.0f;
        arm.beta = 1.0f;
    }
}

SearchArm BayesianBandit::select_arm() {
    // Thompson sampling: sample from Beta distributions and pick highest
    float dfs_sample = sample_beta(arms[0].alpha, arms[0].beta);
    float mcts_sample = sample_beta(arms[1].alpha, arms[1].beta);
    
    return (dfs_sample > mcts_sample) ? SearchArm::DFS : SearchArm::MCTS;
}

void BayesianBandit::update_reward(SearchArm arm, float reward) {
    int idx = static_cast<int>(arm);
    arms[idx].pulls++;
    arms[idx].total_reward += reward;
    
    // Update Beta distribution parameters
    // Assume reward is in [0,1], treat as Bernoulli success/failure
    if (reward > 0.5f) {
        arms[idx].alpha += 1.0f;
    } else {
        arms[idx].beta += 1.0f;
    }
}

float BayesianBandit::sample_beta(float alpha, float beta) {
    // Simple Beta sampling using Gamma distributions
    std::gamma_distribution<float> gamma_a(alpha, 1.0f);
    std::gamma_distribution<float> gamma_b(beta, 1.0f);
    
    float x = gamma_a(rng);
    float y = gamma_b(rng);
    
    return x / (x + y);
}

void BayesianBandit::log_stats() const {
    std::cout << "info string Bandit Stats - DFS: " << arms[0].pulls 
              << " pulls, avg=" << std::fixed << std::setprecision(3)
              << (arms[0].pulls > 0 ? arms[0].total_reward / arms[0].pulls : 0.0f)
              << " MCTS: " << arms[1].pulls << " pulls, avg="
              << (arms[1].pulls > 0 ? arms[1].total_reward / arms[1].pulls : 0.0f)
              << std::endl;
}

// Neural Evaluator (Stub Implementation)
NeuralEvaluator::NeuralEvaluator() : rng(std::random_device{}()) {}

EvalResult NeuralEvaluator::evaluate(const Board& board) {
    // Dummy evaluation - in real implementation this would call RPC server
    EvalResult result;
    result.policy.resize(4096);
    
    // Generate random policy with slight bias toward center squares
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < 4096; ++i) {
        int from = i / 64;
        int to = i % 64;
        int from_dist = std::min({Board::fileOf(from), 7-Board::fileOf(from), 
                                 Board::rankOf(from), 7-Board::rankOf(from)});
        int to_dist = std::min({Board::fileOf(to), 7-Board::fileOf(to), 
                               Board::rankOf(to), 7-Board::rankOf(to)});
        result.policy[i] = dist(rng) * (1.0f + 0.1f * (from_dist + to_dist));
    }
    
    // Normalize policy
    float sum = 0.0f;
    for (float p : result.policy) sum += p;
    if (sum > 0.0f) {
        for (float& p : result.policy) p /= sum;
    }
    
    // Random value and uncertainty
    std::normal_distribution<float> value_dist(0.0f, 0.3f);
    result.value = std::tanh(value_dist(rng));
    result.uncertainty = std::abs(std::normal_distribution<float>(0.1f, 0.05f)(rng));
    
    return result;
}

// NAGS Controller Implementation
NAGSController::NAGSController(Board& b) : board(b), tt() {
    tt.resizeMB(64); // 64MB hash table
}

Move NAGSController::search(const SearchLimits& limits) {
    should_stop = false;
    iteration_count = 0;
    
    auto start_time = std::chrono::steady_clock::now();
    
    // Query meta-learner for hyperparameter adjustments
    std::string fen = board.getFEN();
    int time_left = static_cast<int>(limits.timeMs > 0 ? limits.timeMs : 30000);
    float tactical_ratio = tactical_volatility();
    
    MetaDeltas deltas = meta_client.predict(fen, time_left, last_uncertainty, tactical_ratio);
    
    // Apply meta-learner adjustments
    int adjusted_dfs_depth = base_dfs_depth + static_cast<int>(deltas.dfs_depth_delta * 3); // +/- 3 depth
    int adjusted_mcts_budget = base_mcts_budget + static_cast<int>(deltas.mcts_budget_delta * 500); // +/- 500 iterations
    float adjusted_exploration = base_exploration + deltas.bandit_exploration_delta * 0.5f; // +/- 0.5
    
    // Clamp values to reasonable ranges
    adjusted_dfs_depth = std::max(2, std::min(12, adjusted_dfs_depth));
    adjusted_mcts_budget = std::max(100, std::min(2000, adjusted_mcts_budget));
    adjusted_exploration = std::max(0.5f, std::min(2.5f, adjusted_exploration));
    
    std::cout << "info string Starting NAGS hybrid search" << std::endl;
    std::cout << "info string Meta-learner adjustments: DFS depth=" << adjusted_dfs_depth 
              << " MCTS budget=" << adjusted_mcts_budget 
              << " exploration=" << std::fixed << std::setprecision(2) << adjusted_exploration << std::endl;
    
    // Initialize MCTS root
    mcts_root = std::make_unique<MCTSNode>();
    mcts_root->move = Move{-1, -1, Piece::None, false, false}; // Root sentinel
    
    while (!should_stop && !time_up(limits, start_time)) {
        iteration_count++;
        
        // Select search arm using bandit
        SearchArm arm = bandit.select_arm();
        
        float reward = 0.0f;
        if (arm == SearchArm::DFS) {
            // DFS probe with adjusted depth
            int depth = limits.depth > 0 ? std::min(limits.depth, adjusted_dfs_depth) : adjusted_dfs_depth;
            Move dfs_move = dfs_probe(depth);
            float volatility = tactical_volatility();
            reward = calculate_reward(arm, 0.5f, volatility); // Use volatility as proxy for tactical success
            
            if (iteration_count % 50 == 0) {
                std::cout << "info string Iter " << iteration_count << " DFS probe, move=" 
                          << Board::moveToUci(dfs_move) << " volatility=" << std::fixed 
                          << std::setprecision(3) << volatility << std::endl;
            }
        } else {
            // MCTS iteration
            mcts_iteration();
            EvalResult eval = evaluator.evaluate(board);
            reward = calculate_reward(arm, eval.value, eval.uncertainty);
            
            if (iteration_count % 50 == 0) {
                std::cout << "info string Iter " << iteration_count << " MCTS, visits=" 
                          << mcts_root->visits << " value=" << std::fixed 
                          << std::setprecision(3) << eval.value << std::endl;
            }
        }
        
        bandit.update_reward(arm, reward);
        
        // Log bandit stats periodically
        if (iteration_count % 100 == 0) {
            bandit.log_stats();
        }
    }
    
    Move best = select_best_move();
    std::cout << "info string NAGS completed " << iteration_count << " iterations" << std::endl;
    bandit.log_stats();
    
    // Send training sample to meta-learner (simplified Elo gain estimation)
    float search_time_sec = std::chrono::duration<float>(std::chrono::steady_clock::now() - start_time).count();
    float estimated_elo_gain = (iteration_count > 500) ? 1.0f : 0.5f; // Reward longer searches
    float elo_gain_per_sec = estimated_elo_gain / std::max(0.1f, search_time_sec);
    
    if (meta_client.is_connected()) {
        meta_client.add_sample(fen, time_left, last_uncertainty, tactical_ratio, deltas, elo_gain_per_sec);
    }
    
    return best;
}

Move NAGSController::dfs_probe(int depth) {
    auto legal_moves = board.generateLegalMoves();
    if (legal_moves.empty()) return Move{-1, -1, Piece::None, false, false};
    
    order_moves(legal_moves);
    
    Move best_move = legal_moves[0];
    int best_score = -100000;
    
    for (const auto& move : legal_moves) {
        board.makeMove(move);
        int score = -dfs_search(depth - 1, -100000, 100000, 1);
        board.unmakeMove();
        
        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        
        if (should_stop) break;
    }
    
    return best_move;
}

int NAGSController::dfs_search(int depth, int alpha, int beta, int ply) {
    if (should_stop) return alpha;
    if (depth <= 0) return quiescence(alpha, beta, ply);
    
    // TT probe
    uint64_t key = board.zobrist();
    TTEntry tte;
    if (tt.probe(key, tte) && tte.depth >= depth) {
        if (tte.flag == static_cast<uint8_t>(TTFlag::Exact)) return tte.score;
        if (tte.flag == static_cast<uint8_t>(TTFlag::Lower) && tte.score > alpha) alpha = tte.score;
        else if (tte.flag == static_cast<uint8_t>(TTFlag::Upper) && tte.score < beta) beta = tte.score;
        if (alpha >= beta) return tte.score;
    }
    
    auto moves = board.generateLegalMoves();
    if (moves.empty()) {
        return board.inCheck() ? -100000 + ply : 0;
    }
    
    order_moves(moves);
    
    int best_score = -100000;
    Move best_move{-1, -1, Piece::None, false, false};
    
    for (const auto& move : moves) {
        board.makeMove(move);
        int score = -dfs_search(depth - 1, -beta, -alpha, ply + 1);
        board.unmakeMove();
        
        if (score > best_score) {
            best_score = score;
            best_move = move;
        }
        
        if (score > alpha) {
            alpha = score;
            if (alpha >= beta) break;
        }
        
        if (should_stop) break;
    }
    
    // TT store
    TTFlag flag = TTFlag::Exact;
    tt.store(key, depth, flag, best_score, best_move.from != -1 ? &best_move : nullptr);
    
    return alpha;
}

int NAGSController::quiescence(int alpha, int beta, int ply) {
    if (should_stop) return alpha;
    
    int stand_pat = eval_position();
    if (stand_pat >= beta) return beta;
    if (stand_pat > alpha) alpha = stand_pat;
    
    auto moves = board.generateLegalMoves();
    std::vector<Move> captures;
    for (const auto& move : moves) {
        if (board.pieceAt(move.to) != Piece::None || move.isEnPassant) {
            captures.push_back(move);
        }
    }
    
    order_moves(captures);
    
    for (const auto& move : captures) {
        board.makeMove(move);
        int score = -quiescence(-beta, -alpha, ply + 1);
        board.unmakeMove();
        
        if (score >= beta) return beta;
        if (score > alpha) alpha = score;
        
        if (should_stop) break;
    }
    
    return alpha;
}

void NAGSController::mcts_iteration() {
    if (!mcts_root) return;
    
    // Selection
    MCTSNode* leaf = select_node(mcts_root.get());
    
    // Expansion
    if (!leaf->expanded && leaf->visits > 0) {
        expand_node(leaf);
        if (!leaf->children.empty()) {
            leaf = leaf->children[0].get(); // Select first child for simulation
        }
    }
    
    // Simulation
    float value = simulate(leaf);
    
    // Backpropagation
    backpropagate(leaf, value);
}

MCTSNode* NAGSController::select_node(MCTSNode* node) {
    while (!node->children.empty()) {
        MCTSNode* best_child = nullptr;
        float best_score = -std::numeric_limits<float>::infinity();
        
        for (auto& child : node->children) {
            float score = child->ucb_score(static_cast<float>(node->visits));
            if (score > best_score) {
                best_score = score;
                best_child = child.get();
            }
        }
        
        if (!best_child) break;
        
        // Make move and continue selection
        board.makeMove(best_child->move);
        node = best_child;
    }
    
    return node;
}

void NAGSController::expand_node(MCTSNode* node) {
    if (node->expanded) return;
    
    auto legal_moves = board.generateLegalMoves();
    EvalResult eval = evaluator.evaluate(board);
    
    // Create children with neural network priors
    for (const auto& move : legal_moves) {
        auto child = std::make_unique<MCTSNode>();
        child->move = move;
        child->parent = node;
        
        // Map move to policy index (from*64 + to)
        int policy_idx = move.from * 64 + move.to;
        if (policy_idx >= 0 && policy_idx < static_cast<int>(eval.policy.size())) {
            child->prior = eval.policy[policy_idx];
        } else {
            child->prior = 1.0f / legal_moves.size(); // Uniform fallback
        }
        
        node->children.push_back(std::move(child));
    }
    
    node->expanded = true;
}

float NAGSController::simulate(MCTSNode* node) {
    // Use neural network evaluation instead of random playout
    EvalResult eval = evaluator.evaluate(board);
    return eval.value;
}

void NAGSController::backpropagate(MCTSNode* node, float value) {
    // Unwind moves made during selection
    std::vector<Move> moves_to_undo;
    MCTSNode* current = node;
    
    while (current && current->parent) {
        moves_to_undo.push_back(current->move);
        current = current->parent;
    }
    
    // Undo moves in reverse order
    for (auto it = moves_to_undo.rbegin(); it != moves_to_undo.rend(); ++it) {
        board.unmakeMove();
    }
    
    // Update statistics
    current = node;
    while (current) {
        current->visits++;
        current->value_sum += value;
        value = -value; // Flip value for opponent
        current = current->parent;
    }
}

Move NAGSController::select_best_move() {
    auto legal_moves = board.generateLegalMoves();
    if (legal_moves.empty()) return Move{-1, -1, Piece::None, false, false};
    
    // Blend DFS and MCTS information
    Move dfs_best = dfs_probe(4); // Quick DFS probe
    
    Move mcts_best{-1, -1, Piece::None, false, false};
    int max_visits = 0;
    
    if (mcts_root && !mcts_root->children.empty()) {
        for (const auto& child : mcts_root->children) {
            if (child->visits > max_visits) {
                max_visits = child->visits;
                mcts_best = child->move;
            }
        }
    }
    
    // Simple blending: prefer MCTS if it has significant visits, otherwise DFS
    if (max_visits > 50) {
        std::cout << "info string Selected MCTS move: " << Board::moveToUci(mcts_best) 
                  << " (visits=" << max_visits << ")" << std::endl;
        return mcts_best;
    } else {
        std::cout << "info string Selected DFS move: " << Board::moveToUci(dfs_best) << std::endl;
        return dfs_best;
    }
}

float NAGSController::calculate_reward(SearchArm arm, float value, float uncertainty) {
    // Higher reward for:
    // - DFS: Low uncertainty (tactical clarity)
    // - MCTS: High absolute value (strong position evaluation)
    
    if (arm == SearchArm::DFS) {
        return std::max(0.0f, 1.0f - uncertainty); // Reward tactical clarity
    } else {
        return 0.5f + 0.5f * std::abs(value); // Reward strong evaluations
    }
}

float NAGSController::tactical_volatility() {
    // Measure how much evaluation changes with tactical moves
    int base_eval = eval_position();
    
    auto legal_moves = board.generateLegalMoves();
    int max_swing = 0;
    
    for (const auto& move : legal_moves) {
        if (board.pieceAt(move.to) != Piece::None) { // Capture
            board.makeMove(move);
            int eval = eval_position();
            max_swing = std::max(max_swing, std::abs(eval - base_eval));
            board.unmakeMove();
        }
    }
    
    return static_cast<float>(max_swing) / 1000.0f; // Normalize
}

int NAGSController::eval_position() {
    // Simple material evaluation
    int score = 0;
    static const int values[] = {100, 320, 330, 500, 900, 0}; // P,N,B,R,Q,K
    
    for (int sq = 0; sq < 64; ++sq) {
        Piece p = board.pieceAt(sq);
        if (p == Piece::None) continue;
        
        int piece_type = static_cast<int>(p) - 1;
        int value = values[piece_type % 6];
        
        if (Board::isWhite(p)) score += value;
        else score -= value;
    }
    
    return board.sideToMove() == Color::White ? score : -score;
}

void NAGSController::order_moves(std::vector<Move>& moves) {
    // Simple move ordering: captures first, then others
    std::stable_sort(moves.begin(), moves.end(), [&](const Move& a, const Move& b) {
        bool a_capture = board.pieceAt(a.to) != Piece::None || a.isEnPassant;
        bool b_capture = board.pieceAt(b.to) != Piece::None || b.isEnPassant;
        if (a_capture != b_capture) return a_capture;
        return false; // Stable for equal elements
    });
}

bool NAGSController::time_up(const SearchLimits& limits, std::chrono::steady_clock::time_point start) const {
    if (limits.depth > 0) return iteration_count > limits.depth * 100; // Depth-based limit
    
    if (limits.timeMs > 0) {
        auto elapsed = std::chrono::steady_clock::now() - start;
        auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
        return elapsed_ms >= limits.timeMs;
    }
    
    return iteration_count > 1000; // Default iteration limit
}
