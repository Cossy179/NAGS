#ifndef NAGS_DUAL_TRAINING_PIPELINE_H
#define NAGS_DUAL_TRAINING_PIPELINE_H

#include "supervised_trainer.h"
#include "rl_trainer.h"
#include "../nn/gnn_evaluator.h"
#include "../meta/meta_learner.h"

#include <string>
#include <memory>

namespace nags {

/**
 * Dual training pipeline that combines supervised learning and reinforcement learning
 * First trains on GM games, then refines with self-play
 */
class DualTrainingPipeline {
public:
    DualTrainingPipeline();
    ~DualTrainingPipeline();

    /**
     * Initialize the training pipeline
     * 
     * @param pgn_path Path to the PGN file containing GM games
     * @param save_dir Directory to save checkpoints and models
     * @return True if initialization successful
     */
    bool initialize(const std::string& pgn_path, const std::string& save_dir);

    /**
     * Run the complete dual training pipeline
     * 
     * @param run_supervised Whether to run the supervised learning phase
     * @param run_rl Whether to run the reinforcement learning phase
     * @return True if training successful
     */
    bool runTraining(bool run_supervised = true, bool run_rl = true);

    /**
     * Configure the supervised learning phase
     * 
     * @param batch_size Batch size for training
     * @param learning_rate Learning rate for optimization
     * @param epochs Number of epochs to train
     * @param validation_split Ratio of data to use for validation
     */
    void configureSupervisedLearning(int batch_size, double learning_rate, 
                                    int epochs, double validation_split);

    /**
     * Configure the reinforcement learning phase
     * 
     * @param num_selfplay_games Number of self-play games per iteration
     * @param num_iterations Number of training iterations
     * @param learning_rate Learning rate for optimization
     * @param time_control_ms Base time in milliseconds for self-play
     */
    void configureReinforcementLearning(int num_selfplay_games, int num_iterations, 
                                       double learning_rate, int time_control_ms);

    /**
     * Enable curriculum learning for both training phases
     * 
     * @param enable Whether to enable curriculum learning
     */
    void enableCurriculum(bool enable);

    /**
     * Load existing model to continue training
     * 
     * @param model_path Path to the model
     * @return True if loading successful
     */
    bool loadCheckpoint(const std::string& model_path);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nags

#endif // NAGS_DUAL_TRAINING_PIPELINE_H 