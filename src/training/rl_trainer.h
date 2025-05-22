#ifndef NAGS_RL_TRAINER_H
#define NAGS_RL_TRAINER_H

#include "../core/board.h"
#include "../core/engine.h"
#include "../meta/meta_learner.h"
#include "../nn/gnn_evaluator.h"

#include <string>
#include <vector>
#include <memory>

namespace nags {

/**
 * Reinforcement Learning trainer class using self-play and PPO algorithm
 */
class RLTrainer {
public:
    RLTrainer();
    ~RLTrainer();

    /**
     * Initialize the trainer with configuration parameters
     * 
     * @param num_selfplay_games Number of self-play games per iteration
     * @param num_iterations Number of training iterations
     * @param learning_rate Learning rate for optimization
     * @param batch_size Batch size for training
     * @return True if initialization successful
     */
    bool initialize(int num_selfplay_games = 1000, int num_iterations = 20, 
                   double learning_rate = 5e-5, int batch_size = 256);

    /**
     * Load pretrained model (from supervised learning) as starting point
     * 
     * @param model_path Path to the pretrained model
     * @return True if loading successful
     */
    bool loadPretrainedModel(const std::string& model_path);

    /**
     * Train the model using self-play reinforcement learning
     * 
     * @param evaluator The GNN evaluator to train
     * @param meta_learner The meta-learner to optimize
     * @return True if training successful
     */
    bool train(GNNEvaluator* evaluator, MetaLearner* meta_learner);

    /**
     * Save the trained model to disk
     * 
     * @param path Path to save the model
     * @return True if save successful
     */
    bool saveModel(const std::string& path);

    /**
     * Set the time controls for self-play games
     * 
     * @param base_time_ms Base time in milliseconds
     * @param increment_ms Increment per move in milliseconds
     */
    void setTimeControls(int base_time_ms, int increment_ms);

    /**
     * Enable opponent modeling during self-play
     * 
     * @param enable Whether to enable opponent modeling
     */
    void enableOpponentModeling(bool enable);

    /**
     * Enable progressive curriculum learning for RL
     * 
     * @param enable Whether to enable curriculum learning
     */
    void enableCurriculum(bool enable);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nags

#endif // NAGS_RL_TRAINER_H 