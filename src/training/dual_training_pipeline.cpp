#include "dual_training_pipeline.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <thread>

namespace nags {

// Implementation of DualTrainingPipeline
struct DualTrainingPipeline::Impl {
    Impl() 
    : evaluator_(nullptr)
    , meta_learner_(nullptr)
    , supervised_batch_size_(512)
    , supervised_learning_rate_(1e-4)
    , supervised_epochs_(10)
    , supervised_validation_split_(0.1)
    , rl_num_selfplay_games_(1000)
    , rl_num_iterations_(20)
    , rl_learning_rate_(5e-5)
    , rl_time_control_ms_(60000)
    , enable_curriculum_(true)
    , has_checkpoint_(false)
    , pgn_path_("")
    , save_dir_("./models")
    {}

    bool initialize(const std::string& pgn_path, const std::string& save_dir) {
        // Initialize evaluator and meta-learner
        evaluator_ = std::make_unique<GNNEvaluator>();
        meta_learner_ = std::make_unique<MetaLearner>();

        // Initialize trainers
        supervised_trainer_ = std::make_unique<SupervisedTrainer>();
        rl_trainer_ = std::make_unique<RLTrainer>();

        // Set paths
        pgn_path_ = pgn_path;
        save_dir_ = save_dir;

        // Create save directory if it doesn't exist
        std::filesystem::create_directories(save_dir_);

        // Initialize meta-learner with dataset
        if (!meta_learner_->initialize(pgn_path_)) {
            std::cerr << "Failed to initialize meta-learner with dataset: " << pgn_path_ << std::endl;
            return false;
        }

        // Initialize trainers
        if (!supervised_trainer_->initialize(pgn_path_, supervised_batch_size_, 
                                            supervised_learning_rate_, supervised_epochs_)) {
            std::cerr << "Failed to initialize supervised trainer with dataset: " << pgn_path_ << std::endl;
            return false;
        }

        if (!rl_trainer_->initialize(rl_num_selfplay_games_, rl_num_iterations_, 
                                    rl_learning_rate_, supervised_batch_size_)) {
            std::cerr << "Failed to initialize RL trainer" << std::endl;
            return false;
        }

        // Set curriculum learning if enabled
        supervised_trainer_->enableCurriculum(enable_curriculum_);
        rl_trainer_->enableCurriculum(enable_curriculum_);

        // Set validation split for supervised learning
        supervised_trainer_->setValidationSplit(supervised_validation_split_);

        // Set time controls for RL
        rl_trainer_->setTimeControls(rl_time_control_ms_, 1000);

        return true;
    }

    bool runTraining(bool run_supervised, bool run_rl) {
        std::cout << "Starting dual training pipeline..." << std::endl;
        
        // Track start time
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Run supervised learning if enabled
        if (run_supervised) {
            std::cout << "=== Phase 1: Supervised Learning from GM Games ===" << std::endl;
            if (!supervised_trainer_->train(evaluator_.get(), meta_learner_.get())) {
                std::cerr << "Supervised training failed" << std::endl;
                return false;
            }
            
            // Save supervised learning checkpoint
            std::string sl_model_path = save_dir_ + "/supervised_model.bin";
            if (!supervised_trainer_->saveModel(sl_model_path)) {
                std::cerr << "Failed to save supervised model" << std::endl;
                return false;
            }
            
            std::cout << "Supervised learning completed, model saved to: " << sl_model_path << std::endl;
        }
        
        // Run reinforcement learning if enabled
        if (run_rl) {
            std::cout << "=== Phase 2: Reinforcement Learning through Self-Play ===" << std::endl;
            
            // Load supervised model if we skipped supervised learning
            if (!run_supervised && has_checkpoint_) {
                std::string sl_model_path = save_dir_ + "/supervised_model.bin";
                if (!rl_trainer_->loadPretrainedModel(sl_model_path)) {
                    std::cerr << "Failed to load supervised model for RL" << std::endl;
                    return false;
                }
            }
            
            if (!rl_trainer_->train(evaluator_.get(), meta_learner_.get())) {
                std::cerr << "RL training failed" << std::endl;
                return false;
            }
            
            // Save final model
            std::string rl_model_path = save_dir_ + "/rl_model.bin";
            if (!rl_trainer_->saveModel(rl_model_path)) {
                std::cerr << "Failed to save RL model" << std::endl;
                return false;
            }
            
            std::cout << "Reinforcement learning completed, model saved to: " << rl_model_path << std::endl;
        }
        
        // Calculate and display total training time
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::hours>(end_time - start_time);
        std::cout << "Dual training pipeline completed in " << duration.count() << " hours" << std::endl;
        
        return true;
    }

    bool loadCheckpoint(const std::string& model_path) {
        // Load checkpoint for continuing training
        has_checkpoint_ = true;
        
        // Load model into evaluator
        if (!evaluator_->loadModel(model_path)) {
            std::cerr << "Failed to load model checkpoint" << std::endl;
            return false;
        }
        
        // Load meta-learner state
        std::string meta_path = model_path + ".meta";
        if (!meta_learner_->loadModel(meta_path)) {
            std::cerr << "Warning: Could not load meta-learner state, using defaults" << std::endl;
        }
        
        return true;
    }

    // Configuration methods
    void configureSupervisedLearning(int batch_size, double learning_rate, 
                                    int epochs, double validation_split) {
        supervised_batch_size_ = batch_size;
        supervised_learning_rate_ = learning_rate;
        supervised_epochs_ = epochs;
        supervised_validation_split_ = validation_split;
    }

    void configureReinforcementLearning(int num_selfplay_games, int num_iterations, 
                                       double learning_rate, int time_control_ms) {
        rl_num_selfplay_games_ = num_selfplay_games;
        rl_num_iterations_ = num_iterations;
        rl_learning_rate_ = learning_rate;
        rl_time_control_ms_ = time_control_ms;
    }

    void enableCurriculum(bool enable) {
        enable_curriculum_ = enable;
        
        // Update trainers if they're already initialized
        if (supervised_trainer_) {
            supervised_trainer_->enableCurriculum(enable);
        }
        
        if (rl_trainer_) {
            rl_trainer_->enableCurriculum(enable);
        }
    }

    // Member variables
    std::unique_ptr<GNNEvaluator> evaluator_;
    std::unique_ptr<MetaLearner> meta_learner_;
    std::unique_ptr<SupervisedTrainer> supervised_trainer_;
    std::unique_ptr<RLTrainer> rl_trainer_;
    
    // Supervised learning config
    int supervised_batch_size_;
    double supervised_learning_rate_;
    int supervised_epochs_;
    double supervised_validation_split_;
    
    // RL config
    int rl_num_selfplay_games_;
    int rl_num_iterations_;
    double rl_learning_rate_;
    int rl_time_control_ms_;
    
    // General config
    bool enable_curriculum_;
    bool has_checkpoint_;
    std::string pgn_path_;
    std::string save_dir_;
};

// DualTrainingPipeline implementation (public interface)
DualTrainingPipeline::DualTrainingPipeline() : impl_(new Impl()) {}

DualTrainingPipeline::~DualTrainingPipeline() {}

bool DualTrainingPipeline::initialize(const std::string& pgn_path, const std::string& save_dir) {
    return impl_->initialize(pgn_path, save_dir);
}

bool DualTrainingPipeline::runTraining(bool run_supervised, bool run_rl) {
    return impl_->runTraining(run_supervised, run_rl);
}

void DualTrainingPipeline::configureSupervisedLearning(int batch_size, double learning_rate, 
                                                     int epochs, double validation_split) {
    impl_->configureSupervisedLearning(batch_size, learning_rate, epochs, validation_split);
}

void DualTrainingPipeline::configureReinforcementLearning(int num_selfplay_games, int num_iterations, 
                                                        double learning_rate, int time_control_ms) {
    impl_->configureReinforcementLearning(num_selfplay_games, num_iterations, learning_rate, time_control_ms);
}

void DualTrainingPipeline::enableCurriculum(bool enable) {
    impl_->enableCurriculum(enable);
}

bool DualTrainingPipeline::loadCheckpoint(const std::string& model_path) {
    return impl_->loadCheckpoint(model_path);
}

} // namespace nags 