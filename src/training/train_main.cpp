#include "dual_training_pipeline.h"
#include <iostream>
#include <string>
#include <chrono>
#include <ctime>

using namespace nags;

void printUsage() {
    std::cout << "NAGS Training Utility\n";
    std::cout << "Usage: nags_train [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --pgn <path>           Path to PGN file with GM games (default: AJ-CORR-PGN-000.pgn)\n";
    std::cout << "  --model-dir <path>     Directory to save models (default: ./models)\n";
    std::cout << "  --supervised-only      Run only supervised learning phase\n";
    std::cout << "  --rl-only              Run only reinforcement learning phase\n";
    std::cout << "  --checkpoint <path>    Load checkpoint from path\n";
    std::cout << "  --sl-batch <size>      Supervised learning batch size (default: 512)\n";
    std::cout << "  --sl-epochs <num>      Supervised learning epochs (default: 10)\n";
    std::cout << "  --sl-lr <rate>         Supervised learning rate (default: 1e-4)\n";
    std::cout << "  --rl-games <num>       Number of self-play games per iteration (default: 1000)\n";
    std::cout << "  --rl-iter <num>        Number of RL iterations (default: 20)\n";
    std::cout << "  --rl-lr <rate>         RL learning rate (default: 5e-5)\n";
    std::cout << "  --rl-time <ms>         Base time control for RL in ms (default: 60000)\n";
    std::cout << "  --disable-curriculum   Disable curriculum learning\n";
    std::cout << "  --help                 Show this help message\n";
}

int main(int argc, char** argv) {
    // Default parameters
    std::string pgn_path = "AJ-CORR-PGN-000.pgn";
    std::string model_dir = "./models";
    std::string checkpoint_path = "";
    bool run_supervised = true;
    bool run_rl = true;
    bool enable_curriculum = true;
    int sl_batch_size = 512;
    int sl_epochs = 10;
    double sl_lr = 1e-4;
    double sl_validation = 0.1;
    int rl_games = 1000;
    int rl_iterations = 20;
    double rl_lr = 5e-5;
    int rl_time_ms = 60000;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            printUsage();
            return 0;
        } else if (arg == "--pgn" && i + 1 < argc) {
            pgn_path = argv[++i];
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--checkpoint" && i + 1 < argc) {
            checkpoint_path = argv[++i];
        } else if (arg == "--supervised-only") {
            run_supervised = true;
            run_rl = false;
        } else if (arg == "--rl-only") {
            run_supervised = false;
            run_rl = true;
        } else if (arg == "--disable-curriculum") {
            enable_curriculum = false;
        } else if (arg == "--sl-batch" && i + 1 < argc) {
            sl_batch_size = std::stoi(argv[++i]);
        } else if (arg == "--sl-epochs" && i + 1 < argc) {
            sl_epochs = std::stoi(argv[++i]);
        } else if (arg == "--sl-lr" && i + 1 < argc) {
            sl_lr = std::stod(argv[++i]);
        } else if (arg == "--rl-games" && i + 1 < argc) {
            rl_games = std::stoi(argv[++i]);
        } else if (arg == "--rl-iter" && i + 1 < argc) {
            rl_iterations = std::stoi(argv[++i]);
        } else if (arg == "--rl-lr" && i + 1 < argc) {
            rl_lr = std::stod(argv[++i]);
        } else if (arg == "--rl-time" && i + 1 < argc) {
            rl_time_ms = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage();
            return 1;
        }
    }
    
    // Print startup information
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);
    std::cout << "NAGS Training started at: " << std::ctime(&time_t_now);
    std::cout << "PGN path: " << pgn_path << std::endl;
    std::cout << "Model directory: " << model_dir << std::endl;
    std::cout << "Training mode: " 
              << (run_supervised && run_rl ? "Full (Supervised + RL)" : 
                  run_supervised ? "Supervised only" : "RL only") << std::endl;
    
    // Initialize the training pipeline
    DualTrainingPipeline pipeline;
    if (!pipeline.initialize(pgn_path, model_dir)) {
        std::cerr << "Failed to initialize training pipeline" << std::endl;
        return 1;
    }
    
    // Configure the pipeline
    pipeline.configureSupervisedLearning(sl_batch_size, sl_lr, sl_epochs, sl_validation);
    pipeline.configureReinforcementLearning(rl_games, rl_iterations, rl_lr, rl_time_ms);
    pipeline.enableCurriculum(enable_curriculum);
    
    // Load checkpoint if specified
    if (!checkpoint_path.empty()) {
        std::cout << "Loading checkpoint from: " << checkpoint_path << std::endl;
        if (!pipeline.loadCheckpoint(checkpoint_path)) {
            std::cerr << "Failed to load checkpoint, starting from scratch" << std::endl;
        }
    }
    
    // Run training
    if (!pipeline.runTraining(run_supervised, run_rl)) {
        std::cerr << "Training failed" << std::endl;
        return 1;
    }
    
    std::cout << "Training completed successfully" << std::endl;
    return 0;
} 