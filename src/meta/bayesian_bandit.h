#pragma once

#include <vector>
#include <memory>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <mutex>

namespace nags {

/**
 * Sophisticated Bayesian Multi-Armed Bandit for meta-learning
 * Uses Thompson Sampling with Gaussian processes for optimal exploration/exploitation
 * This helps the chess engine adaptively choose between different search strategies
 */
class BayesianBandit {
public:
    struct ArmStatistics {
        double mean;
        double variance;
        int pulls;
        double cumulative_reward;
        double last_reward;
        std::vector<double> reward_history;
        
        // Gaussian process hyperparameters
        double length_scale;
        double signal_variance;
        double noise_variance;
    };

    BayesianBandit(int num_arms, double prior_mean = 0.5, double prior_variance = 0.25);
    ~BayesianBandit();
    
    /**
     * Select an arm using Thompson Sampling with Gaussian Process priors
     * @return The index of the selected arm
     */
    int selectArm();
    
    /**
     * Update the statistics for an arm after observing a reward
     * @param arm The arm that was pulled
     * @param reward The observed reward (normalized to [0, 1])
     */
    void updateArm(int arm, double reward);
    
    /**
     * Get the posterior distribution parameters for an arm
     * @param arm The arm index
     * @return pair of (mean, variance)
     */
    std::pair<double, double> getPosterior(int arm) const;
    
    /**
     * Reset the bandit to initial state
     */
    void reset();
    
    /**
     * Get the best arm based on current estimates
     * @return The index of the arm with highest expected reward
     */
    int getBestArm() const;
    
    /**
     * Compute the information value of pulling each arm
     * Used for sophisticated exploration strategies
     * @return Vector of information values for each arm
     */
    std::vector<double> computeInformationValues() const;
    
    /**
     * Get the upper confidence bound for each arm
     * @param confidence_level The confidence parameter (typically sqrt(2 * log(t)))
     * @return Vector of UCB values
     */
    std::vector<double> getUpperConfidenceBounds(double confidence_level) const;
    
    /**
     * Adaptive hyperparameter tuning using empirical Bayes
     */
    void tuneHyperparameters();
    
    /**
     * Get the exploration probability for sophisticated epsilon-greedy
     * Decays over time but increases when uncertainty is high
     */
    double getExplorationProbability() const;
    
    /**
     * Compute the Gittins index for each arm
     * This provides the optimal solution to the exploration-exploitation tradeoff
     * @param discount_factor The discount factor for future rewards
     * @return Vector of Gittins indices
     */
    std::vector<double> computeGittinsIndices(double discount_factor = 0.99) const;

private:
    int num_arms_;
    double prior_mean_;
    double prior_variance_;
    std::vector<ArmStatistics> arms_;
    
    // Random number generation
    mutable std::mt19937 rng_;
    mutable std::normal_distribution<double> normal_dist_;
    
    // Thread safety
    mutable std::mutex mutex_;
    
    // Total number of pulls across all arms
    int total_pulls_;
    
    // Gaussian process kernel function
    double kernelFunction(double x1, double x2, double length_scale) const;
    
    // Compute posterior mean and variance using Gaussian process
    std::pair<double, double> computeGaussianProcessPosterior(int arm) const;
    
    // Information-theoretic utilities
    double computeEntropy(double variance) const;
    double computeKLDivergence(double mean1, double var1, double mean2, double var2) const;
    
    // Sophisticated sampling from posterior
    double sampleFromPosterior(int arm) const;
};

} // namespace nags 