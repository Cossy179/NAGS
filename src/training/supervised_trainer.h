#ifndef NAGS_SUPERVISED_TRAINER_H
#define NAGS_SUPERVISED_TRAINER_H

#include "../core/board.h"
#include "../core/engine.h"
#include "../meta/meta_learner.h"
#include "../nn/gnn_evaluator.h"

#include <string>
#include <vector>
#include <memory>

namespace nags {

/**
 * Supervised trainer class that learns from GM games in PGN format
 */
class SupervisedTrainer {
public:
    SupervisedTrainer();
    ~SupervisedTrainer();

    /**
     * Initialize the trainer with configuration parameters
     * 
     * @param pgn_path Path to the PGN file containing GM games
     * @param batch_size Batch size for training
     * @param learning_rate Learning rate for optimization
     * @param epochs Number of epochs to train
     * @return True if initialization successful
     */
    bool initialize(const std::string& pgn_path, int batch_size = 512, double learning_rate = 1e-4, int epochs = 10);

    /**
     * Train the model using supervised learning on GM games
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
     * Load a trained model from disk
     * 
     * @param path Path to load the model from
     * @return True if load successful
     */
    bool loadModel(const std::string& path);

    /**
     * Set the validation split ratio
     * 
     * @param ratio Ratio of data to use for validation (0.0-1.0)
     */
    void setValidationSplit(double ratio);

    /**
     * Apply curriculum learning stages
     * 
     * @param enable Whether to enable curriculum learning
     */
    void enableCurriculum(bool enable);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nags

#endif // NAGS_SUPERVISED_TRAINER_H 