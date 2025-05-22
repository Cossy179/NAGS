#include "bayesian_bandit.h"
#include <limits>
#include <stdexcept>

namespace nags {

BayesianBandit::BayesianBandit(int num_arms, double prior_mean, double prior_variance)
    : num_arms_(num_arms),
      prior_mean_(prior_mean),
      prior_variance_(prior_variance),
      total_pulls_(0),
      rng_(std::random_device{}()),
      normal_dist_(0.0, 1.0) {
    
    arms_.resize(num_arms_);
    for (auto& arm : arms_) {
        arm.mean = prior_mean_;
        arm.variance = prior_variance_;
        arm.pulls = 0;
        arm.cumulative_reward = 0.0;
        arm.last_reward = 0.0;
        arm.reward_history.reserve(1000);
        
        // Initialize GP hyperparameters
        arm.length_scale = 1.0;
        arm.signal_variance = 1.0;
        arm.noise_variance = 0.1;
    }
}

BayesianBandit::~BayesianBandit() = default;

int BayesianBandit::selectArm() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Thompson Sampling with Gaussian Process posterior
    std::vector<double> samples(num_arms_);
    
    for (int i = 0; i < num_arms_; ++i) {
        samples[i] = sampleFromPosterior(i);
    }
    
    // Add exploration bonus based on information value
    auto info_values = computeInformationValues();
    double exploration_weight = getExplorationProbability();
    
    for (int i = 0; i < num_arms_; ++i) {
        samples[i] += exploration_weight * info_values[i];
    }
    
    // Select arm with highest sample
    return std::distance(samples.begin(), std::max_element(samples.begin(), samples.end()));
}

void BayesianBandit::updateArm(int arm, double reward) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (arm < 0 || arm >= num_arms_) {
        throw std::out_of_range("Invalid arm index");
    }
    
    auto& arm_stats = arms_[arm];
    arm_stats.pulls++;
    arm_stats.cumulative_reward += reward;
    arm_stats.last_reward = reward;
    arm_stats.reward_history.push_back(reward);
    
    // Update posterior using Bayesian update rule
    double n = arm_stats.pulls;
    double old_mean = arm_stats.mean;
    double old_variance = arm_stats.variance;
    
    // Online update of mean and variance
    double delta = reward - old_mean;
    arm_stats.mean = old_mean + delta / n;
    
    if (n > 1) {
        double delta2 = reward - arm_stats.mean;
        double sum_sq = (n - 1) * old_variance + delta * delta2;
        arm_stats.variance = sum_sq / n;
    }
    
    // Update using Gaussian process posterior if enough data
    if (n > 5) {
        auto [gp_mean, gp_var] = computeGaussianProcessPosterior(arm);
        // Blend empirical and GP estimates
        double alpha = 0.7; // Weight for GP estimate
        arm_stats.mean = alpha * gp_mean + (1 - alpha) * arm_stats.mean;
        arm_stats.variance = alpha * gp_var + (1 - alpha) * arm_stats.variance;
    }
    
    total_pulls_++;
    
    // Periodically tune hyperparameters
    if (total_pulls_ % 100 == 0) {
        tuneHyperparameters();
    }
}

std::pair<double, double> BayesianBandit::getPosterior(int arm) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (arm < 0 || arm >= num_arms_) {
        throw std::out_of_range("Invalid arm index");
    }
    
    const auto& arm_stats = arms_[arm];
    return {arm_stats.mean, arm_stats.variance};
}

void BayesianBandit::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& arm : arms_) {
        arm.mean = prior_mean_;
        arm.variance = prior_variance_;
        arm.pulls = 0;
        arm.cumulative_reward = 0.0;
        arm.last_reward = 0.0;
        arm.reward_history.clear();
    }
    
    total_pulls_ = 0;
}

int BayesianBandit::getBestArm() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Use Gittins indices for optimal selection
    auto gittins = computeGittinsIndices();
    return std::distance(gittins.begin(), std::max_element(gittins.begin(), gittins.end()));
}

std::vector<double> BayesianBandit::computeInformationValues() const {
    std::vector<double> info_values(num_arms_);
    
    for (int i = 0; i < num_arms_; ++i) {
        // Information value is proportional to uncertainty reduction
        double current_entropy = computeEntropy(arms_[i].variance);
        
        // Estimate entropy reduction from one more sample
        double n = arms_[i].pulls + 1;
        double estimated_new_variance = arms_[i].variance * (n - 1) / n;
        double new_entropy = computeEntropy(estimated_new_variance);
        
        info_values[i] = current_entropy - new_entropy;
        
        // Add bonus for under-explored arms
        if (arms_[i].pulls < 10) {
            info_values[i] *= (10.0 / (arms_[i].pulls + 1));
        }
    }
    
    return info_values;
}

std::vector<double> BayesianBandit::getUpperConfidenceBounds(double confidence_level) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<double> ucb_values(num_arms_);
    
    for (int i = 0; i < num_arms_; ++i) {
        double mean = arms_[i].mean;
        double variance = arms_[i].variance;
        double n = arms_[i].pulls + 1; // Add 1 to avoid division by zero
        
        // UCB with variance adaptation
        double exploration_term = confidence_level * std::sqrt(variance / n);
        ucb_values[i] = mean + exploration_term;
    }
    
    return ucb_values;
}

void BayesianBandit::tuneHyperparameters() {
    // Empirical Bayes hyperparameter optimization
    for (auto& arm : arms_) {
        if (arm.reward_history.size() < 10) continue;
        
        // Estimate noise variance from recent rewards
        double mean = std::accumulate(arm.reward_history.end() - 10, arm.reward_history.end(), 0.0) / 10.0;
        double sq_sum = 0.0;
        for (auto it = arm.reward_history.end() - 10; it != arm.reward_history.end(); ++it) {
            sq_sum += (*it - mean) * (*it - mean);
        }
        arm.noise_variance = sq_sum / 9.0;
        
        // Adapt length scale based on reward variability
        arm.length_scale = 1.0 + 2.0 * std::sqrt(arm.noise_variance);
    }
}

double BayesianBandit::getExplorationProbability() const {
    // Sophisticated exploration schedule
    double base_exploration = 0.1;
    double decay_factor = 0.995;
    double min_exploration = 0.01;
    
    // Decay exploration over time
    double time_factor = std::pow(decay_factor, total_pulls_ / 100.0);
    double exploration = base_exploration * time_factor;
    
    // But increase exploration when uncertainty is high
    double avg_variance = 0.0;
    for (const auto& arm : arms_) {
        avg_variance += arm.variance;
    }
    avg_variance /= num_arms_;
    
    double uncertainty_bonus = 0.1 * std::sqrt(avg_variance);
    exploration += uncertainty_bonus;
    
    return std::max(min_exploration, std::min(exploration, 0.3));
}

std::vector<double> BayesianBandit::computeGittinsIndices(double discount_factor) const {
    std::vector<double> gittins(num_arms_);
    
    for (int i = 0; i < num_arms_; ++i) {
        // Simplified Gittins index approximation
        // In practice, this would use dynamic programming
        double mean = arms_[i].mean;
        double variance = arms_[i].variance;
        double n = arms_[i].pulls + 1;
        
        // Whittle's index approximation
        double exploration_value = std::sqrt(2 * variance * std::log(1.0 / (1.0 - discount_factor)));
        gittins[i] = mean + exploration_value / std::sqrt(n);
    }
    
    return gittins;
}

double BayesianBandit::kernelFunction(double x1, double x2, double length_scale) const {
    double diff = x1 - x2;
    return std::exp(-0.5 * diff * diff / (length_scale * length_scale));
}

std::pair<double, double> BayesianBandit::computeGaussianProcessPosterior(int arm) const {
    const auto& arm_stats = arms_[arm];
    if (arm_stats.reward_history.empty()) {
        return {arm_stats.mean, arm_stats.variance};
    }
    
    // Simplified GP posterior computation
    // In a full implementation, this would use matrix operations
    double weighted_sum = 0.0;
    double weight_sum = 0.0;
    
    int history_size = arm_stats.reward_history.size();
    for (int i = 0; i < history_size; ++i) {
        double time_diff = (history_size - i) / static_cast<double>(history_size);
        double weight = kernelFunction(1.0, time_diff, arm_stats.length_scale);
        weighted_sum += weight * arm_stats.reward_history[i];
        weight_sum += weight;
    }
    
    double gp_mean = weighted_sum / weight_sum;
    double gp_variance = arm_stats.signal_variance + arm_stats.noise_variance / arm_stats.pulls;
    
    return {gp_mean, gp_variance};
}

double BayesianBandit::computeEntropy(double variance) const {
    // Entropy of Gaussian distribution
    constexpr double PI = 3.14159265358979323846;
    constexpr double E = 2.71828182845904523536;
    return 0.5 * std::log(2 * PI * E * variance);
}

double BayesianBandit::computeKLDivergence(double mean1, double var1, double mean2, double var2) const {
    // KL divergence between two Gaussians
    double mean_diff = mean1 - mean2;
    return 0.5 * (std::log(var2 / var1) + var1 / var2 + mean_diff * mean_diff / var2 - 1.0);
}

double BayesianBandit::sampleFromPosterior(int arm) const {
    const auto& arm_stats = arms_[arm];
    double std_dev = std::sqrt(arm_stats.variance);
    return arm_stats.mean + std_dev * normal_dist_(rng_);
}

} // namespace nags 